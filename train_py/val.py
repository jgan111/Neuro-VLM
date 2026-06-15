import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import re
import gc
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn

import unsloth
from unsloth import FastVisionModel
from transformers import AutoProcessor
from peft import PeftModel, PeftConfig


def sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else float("inf")


def get_saved_embedding_rows_from_adapter(checkpoint_dir):
    """
    读取 adapter_model.safetensors / adapter_model.bin 中保存的 embed_tokens/lm_head 行数。
    用于兼容：
    - 旧 checkpoint：embedding/lm_head 可能是 151666
    - 新 no-shrink checkpoint：embedding/lm_head 可能是 152064
    """
    adapter_safe = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(checkpoint_dir, "adapter_model.bin")

    keywords = [
        "embed_tokens.original_module.weight",
        "embed_tokens.modules_to_save",
        "lm_head.original_module.weight",
        "lm_head.modules_to_save",
    ]

    if os.path.exists(adapter_safe):
        try:
            from safetensors import safe_open
            with safe_open(adapter_safe, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if any(k in key for k in keywords):
                        shape = f.get_slice(key).get_shape()
                        if len(shape) >= 2:
                            print(f"Detected saved embedding/head tensor: {key} | shape={shape}")
                            return int(shape[0])
        except Exception as e:
            print(f"Warning: failed to inspect safetensors header: {e}")

    if os.path.exists(adapter_bin):
        try:
            state = torch.load(adapter_bin, map_location="cpu")
            for key, value in state.items():
                if any(k in key for k in keywords):
                    if hasattr(value, "shape") and len(value.shape) >= 2:
                        print(f"Detected saved embedding/head tensor: {key} | shape={tuple(value.shape)}")
                        return int(value.shape[0])
        except Exception as e:
            print(f"Warning: failed to inspect adapter_model.bin: {e}")

    return None


def resize_token_embeddings_for_checkpoint(model, tokenizer, checkpoint_dir):
    """
    验证时必须让 base model 的 embedding 行数与 checkpoint 中保存的
    embed_tokens/lm_head 形状一致，否则 PeftModel.from_pretrained 会报 size mismatch。
    """
    tok_n = len(tokenizer)
    cur_n = model.get_input_embeddings().weight.shape[0]
    saved_rows = get_saved_embedding_rows_from_adapter(checkpoint_dir)

    print(f"Tokenizer vocab size: {tok_n}")
    print(f"Current input embedding rows: {cur_n}")
    print(f"Checkpoint saved embedding/head rows: {saved_rows}")

    if saved_rows is not None:
        if saved_rows != cur_n:
            print(f"Resizing token embeddings from {cur_n} to {saved_rows} to match checkpoint.")
            model.resize_token_embeddings(saved_rows)
        else:
            print(f"Embedding rows already match checkpoint: {cur_n}")
    else:
        # fallback：没有找到 embed/lm_head 形状时，只扩不缩
        if tok_n > cur_n:
            print(f"Tokenizer size {tok_n} > embedding size {cur_n}; expanding embeddings.")
            try:
                model.resize_token_embeddings(tok_n, pad_to_multiple_of=128)
            except TypeError:
                model.resize_token_embeddings(tok_n)
        else:
            print(f"Tokenizer size {tok_n} <= embedding size {cur_n}; skip resize.")

    print("Input embedding shape:", model.get_input_embeddings().weight.shape)


def attach_stage2_placeholder_modules(model):
    """
    stage-2 checkpoint 里 adapter_config 的 modules_to_save 包含 seg_token_mask_projection。
    推理时也必须先挂上同名模块，否则 PEFT 加载 adapter 时找不到这个模块。
    """
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        core_model = model.base_model.model
    else:
        core_model = model

    if not hasattr(core_model, "seg_token_mask_projection"):
        hidden_size = core_model.config.hidden_size
        dtype = core_model.get_input_embeddings().weight.dtype
        device = core_model.get_input_embeddings().weight.device

        core_model.seg_token_mask_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
        ).to(device=device, dtype=dtype)

        print("Attached placeholder seg_token_mask_projection.")

    return model


def maybe_print_adapter_config(checkpoint_dir):
    try:
        peft_cfg = PeftConfig.from_pretrained(checkpoint_dir)
        print("Adapter config loaded from checkpoint.")
        print("Adapter target_modules from checkpoint:", peft_cfg.target_modules)
        print("Adapter r:", getattr(peft_cfg, "r", None))
        print("Adapter lora_alpha:", getattr(peft_cfg, "lora_alpha", None))
        print("Adapter modules_to_save:", getattr(peft_cfg, "modules_to_save", None))
    except Exception as e:
        print(f"Warning: failed to read PEFT adapter_config.json: {e}")


def count_active_lora_params(model, adapter_name="stage2_vqa"):
    """
    打印真正被 checkpoint adapter 加载进来的 LoRA 参数数量。
    注意：这里不是训练参数，只是为了确认推理阶段 adapter 结构。
    """
    lang_tensors = 0
    merger_tensors = 0
    visual_block_tensors = 0
    other_lora_tensors = 0

    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if adapter_name not in name:
            continue

        if "language_model" in name:
            lang_tensors += 1
        elif "visual.merger" in name or "merger" in name:
            merger_tensors += 1
        elif "visual.blocks" in name:
            visual_block_tensors += 1
        else:
            other_lora_tensors += 1

    print(
        f"Loaded active adapter LoRA tensors | "
        f"language_model={lang_tensors} | merger={merger_tensors} | "
        f"visual_blocks={visual_block_tensors} | other={other_lora_tensors}"
    )


def ablate_language_lora_if_needed(model, adapter_name="stage2_vqa"):
    """
    可选消融：只在你想比较“完整 LoRA”与“去掉 language_model LoRA”时使用。

    默认不启用。
    运行前如果设置：
        export EVAL_ABLATE_LANGUAGE_LORA=1

    则会把当前 adapter 中 language_model 的 LoRA 权重置零。
    这样推理时只保留 visual/merger LoRA，能真正观察 language_model LoRA 的贡献。
    """
    flag = os.environ.get("EVAL_ABLATE_LANGUAGE_LORA", "0").strip().lower()
    if flag not in ["1", "true", "yes", "y"]:
        print("EVAL_ABLATE_LANGUAGE_LORA=0, keep full adapter.")
        return model

    zeroed_tensors = 0
    zeroed_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "language_model" in name and "lora_" in name and adapter_name in name:
                param.zero_()
                zeroed_tensors += 1
                zeroed_params += param.numel()

    print(
        f"[ABLATION] Zeroed language_model LoRA of adapter '{adapter_name}': "
        f"{zeroed_tensors} tensors, {zeroed_params} params."
    )
    return model


def load_stage2_model(base_model_path, checkpoint_dir):
    print(f"\nLoading base model from: {base_model_path}")
    model, _ = FastVisionModel.from_pretrained(
        model_name=base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    print(f"Loading processor from: {checkpoint_dir}")
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # 关键修改：
    # 根据 checkpoint 里实际保存的 embed_tokens/lm_head 形状调整 embedding。
    # 这样既能验证旧 checkpoint，也能验证当前 no-shrink 训练得到的新 checkpoint。
    resize_token_embeddings_for_checkpoint(model, tokenizer, checkpoint_dir)

    vocab = tokenizer.get_vocab()
    if "<SEG>" in vocab:
        model.config.seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
    else:
        model.config.seg_token_id = None

    try:
        model.config.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        model.config.im_end_token_id = None

    model = attach_stage2_placeholder_modules(model)

    # 关键修改：
    # 不再手写 target_modules_config，也不再先 get_peft_model 再 load_adapter。
    # 直接让 PEFT 从 checkpoint_dir/adapter_config.json 读取真实 target_modules。
    # 这样推理模型结构严格等于该 checkpoint 训练时保存的 adapter 结构。
    maybe_print_adapter_config(checkpoint_dir)

    print(f"Loading adapter directly from checkpoint: {checkpoint_dir}")
    model = PeftModel.from_pretrained(
        model,
        checkpoint_dir,
        adapter_name="stage2_vqa",
        is_trainable=False,
    )
    model.set_adapter("stage2_vqa")

    count_active_lora_params(model, adapter_name="stage2_vqa")

    # 可选：用于真正比较 language LoRA 是否有贡献。
    model = ablate_language_lora_if_needed(model, adapter_name="stage2_vqa")

    model.eval()

    print("Available adapters:", list(getattr(model, "peft_config", {}).keys()))
    print("Active adapter:", model.active_adapters if hasattr(model, "active_adapters") else "N/A")
    print("Tokenizer vocab size:", len(tokenizer))
    print("Input embedding shape:", model.get_input_embeddings().weight.shape)

    return model, processor, tokenizer


def batch_infer(model, processor, tokenizer, qa_data, output_path, batch_size=1):
    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(qa_data), batch_size)):
        batch_chunk = qa_data[i:i + batch_size]

        batch_images = []
        batch_prompts = []
        valid_items = []

        for item in batch_chunk:
            image_path = item["image"]
            question = item["conversations"][0]["value"].replace("\n<image>", "").replace("<image>\n", "").strip()
            question = question + "\nAnswer with one short phrase only."

            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image not found at {image_path}. Skipping.")
                continue

            try:
                image = Image.open(image_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                batch_images.append(image)
                batch_prompts.append(prompt)
                valid_items.append(item)

            except Exception as e:
                print(f"⚠️ Error processing {image_path}: {e}")
                continue

        if not batch_images:
            continue

        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for item, ans in zip(valid_items, answers):
            results.append({
                "image": item["image"],
                "question": item["conversations"][0]["value"],
                "answer": ans.strip(),
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 保存完成：{output_path}, 共计 {len(results)} 条结果")


def main():
    base_model_path = "/home/zhangxw/Mode/Qwen2.5-VL-32B/"

    # 按当前小参数二阶段训练的输出目录来验证。
    # 如果你的启动脚本 --output_dir 用的是别的路径，可以运行前导出：
    #   export CHECKPOINT_ROOT="/你的输出目录/"
    checkpoint_root = os.environ.get(
        "CHECKPOINT_ROOT",
        "/home/zhangxw/share_data/VQA-RAD/LISA_output/",
    )

    # 如果做 language LoRA 消融，建议结果单独保存，避免覆盖完整模型结果。
    if os.environ.get("EVAL_ABLATE_LANGUAGE_LORA", "0").strip().lower() in ["1", "true", "yes", "y"]:
        output_dir = os.environ.get(
            "VAL_OUTPUT_DIR",
            "/home/zhangxw/share_data/VQA-RAD/val_ablate_language_lora/",
        )
    else:
        output_dir = os.environ.get(
            "VAL_OUTPUT_DIR",
            "/home/zhangxw/share_data/VQA-RAD/val/",
        )

    json_path = "/home/zhangxw/share_data/VQA-RAD/test.json"

    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    model_dirs = sorted(
        [os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root) if d.startswith("checkpoint-")],
        key=lambda x: sort_key(os.path.basename(x)),
    )

    for idx, model_path in enumerate(model_dirs, 1):
        print(f"\n🚀 Running inference with model {os.path.basename(model_path)} (#{idx})")

        model, processor, tokenizer = load_stage2_model(base_model_path, model_path)

        output_path = os.path.join(output_dir, f"result_{os.path.basename(model_path)}.json")
        batch_infer(model, processor, tokenizer, qa_data, output_path, batch_size=1)

        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
