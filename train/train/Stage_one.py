from unsloth import FastVisionModel
import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
from collections import Counter

from QwenLISA_two import build_lisa_modules

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from trainer import replace_qwen2_vl_attention_class, DistillationTrainer
from transformers import HfArgumentParser, AutoProcessor, TrainerCallback
from qwenvl.data.data_LISA import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments

local_rank = None


def rank0_print(*args):
    print(*args)


def get_core_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    保存 HF / PEFT 常规部分：
    - LoRA adapter
    - modules_to_save 里的模块（embed_tokens / lm_head / seg_token_mask_projection）
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def save_lisa_extra_modules_from_model(model, output_dir: str):
    """
    单独保存不走 PEFT adapter 包装的 LISA 额外模块：
    - visual_model.mask_decoder
    - seg_token_mask_projection（冗余保存一份）
    """
    os.makedirs(output_dir, exist_ok=True)
    core_model = get_core_model(model)

    extra_state = {
        "seg_token_mask_projection": {
            k: v.detach().cpu()
            for k, v in core_model.seg_token_mask_projection.state_dict().items()
        },
        "visual_model.mask_decoder": {
            k: v.detach().cpu()
            for k, v in core_model.visual_model.mask_decoder.state_dict().items()
        },
    }

    save_path = os.path.join(output_dir, "lisa_extra_modules.pt")
    torch.save(extra_state, save_path)
    rank0_print(f"Saved extra LISA modules to: {save_path}")


def save_lisa_extra_modules(trainer: transformers.Trainer, output_dir: str):
    save_lisa_extra_modules_from_model(trainer.model, output_dir)


class SaveLISAExtraModulesCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_lisa_extra_modules_from_model(model, checkpoint_dir)
        return control

class EpochSummaryCallback(TrainerCallback):
    def __init__(self, trainer_ref_getter):
        self.trainer_ref_getter = trainer_ref_getter

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_ref_getter()
        if trainer is None:
            return control

        parts = []
        for k in [
            "loss",
            "ce_loss",
            "mask_bce_loss",
            "mask_dice_loss",
            "area_loss",
            "seg_token_loss",
            "seg_token_margin_loss",
        ]:
            cnt = trainer._epoch_loss_counts.get(k, 0)
            if cnt > 0:
                avg = trainer._epoch_loss_sums[k] / cnt
                parts.append(f"avg_{k}={avg:.6f}")

        msg = f"\n========== Epoch {state.epoch:.2f} finished"
        if parts:
            msg += " | " + " | ".join(parts)
        msg += f" | steps_in_epoch={trainer._epoch_step_count} ==========\n"

        print(msg)

        trainer._reset_epoch_stats()
        return control

class LISATrainer(DistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_epoch_stats()

    def _reset_epoch_stats(self):
        self._epoch_step_count = 0
        self._epoch_loss_sums = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "mask_bce_loss": 0.0,
            "mask_dice_loss": 0.0,
            "area_loss": 0.0,
            "seg_token_loss": 0.0,
            "seg_token_margin_loss": 0.0,
        }
        self._epoch_loss_counts = {
            "loss": 0,
            "ce_loss": 0,
            "mask_bce_loss": 0,
            "mask_dice_loss": 0,
            "area_loss": 0,
            "seg_token_loss": 0,
            "seg_token_margin_loss": 0,
        }

    @staticmethod
    def _maybe_to_float(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return float(x.detach().float().mean().item())
        try:
            return float(x)
        except Exception:
            return None

    def _accumulate_epoch_stat(self, name, value):
        if value is None:
            return
        self._epoch_loss_sums[name] += float(value)
        self._epoch_loss_counts[name] += 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if isinstance(loss, torch.Tensor) and loss.device != self.args.device:
            loss = loss.to(self.args.device)

        total_loss_val = self._maybe_to_float(loss)
        self._latest_total_loss = total_loss_val

        if isinstance(outputs, dict):
            self._latest_loss_breakdown = {
                "ce_loss": self._maybe_to_float(outputs.get("ce_loss", None)),
                "mask_bce_loss": self._maybe_to_float(outputs.get("mask_bce_loss", None)),
                "mask_dice_loss": self._maybe_to_float(outputs.get("mask_dice_loss", None)),
                "area_loss": self._maybe_to_float(outputs.get("area_loss", None)),
                "seg_token_loss": self._maybe_to_float(outputs.get("seg_token_loss", None)),
                "seg_token_margin_loss": self._maybe_to_float(outputs.get("seg_token_margin_loss", None)),
            }
        else:
            self._latest_loss_breakdown = {}

        # 统计整轮 epoch 的平均值
        self._epoch_step_count += 1
        self._accumulate_epoch_stat("loss", total_loss_val)

        for k in [
            "ce_loss",
            "mask_bce_loss",
            "mask_dice_loss",
            "area_loss",
            "seg_token_loss",
            "seg_token_margin_loss",
        ]:
            self._accumulate_epoch_stat(k, self._latest_loss_breakdown.get(k, None))

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        merged_logs = dict(logs)

        if hasattr(self, "_latest_loss_breakdown"):
            for k, v in self._latest_loss_breakdown.items():
                if v is not None:
                    merged_logs[k] = v

        super().log(merged_logs, *args, **kwargs)


def print_trainable_layers(model):
    print("\n====== Parameter Trainability Report (LoRA) ======")
    try:
        model.print_trainable_parameters()
    except AttributeError:
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable | {name} | {param.shape}")
        print("---")
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Trainable %: {(100 * trainable_params / total_params):.4f}")
    print("=================================================\n")

def build_manual_device_map():
    """
    手写 Qwen 主干的双卡切分：
    - GPU0: visual + embed + language_model 前 44 层
    - GPU1: language_model 后 20 层 + norm + lm_head
    之所以偏向 GPU0，是因为 SAM/LISA 分支固定在 GPU1。
    """
    device_map = {}

    # ===== Vision / embedding 相关，优先放 GPU0 =====
    candidate_modules_gpu0 = [
        "model.visual",
        "model.visual.patch_embed",
        "model.visual.rotary_pos_emb",
        "model.visual.blocks",
        "model.visual.merger",
        "model.embed_tokens",
        "model.rotary_emb",
    ]
    for name in candidate_modules_gpu0:
        device_map[name] = 0
    device_map["model.language_model.embed_tokens"] = 0
    # ===== Qwen language model 分层 =====
    # 前 44 层 -> GPU0
    for i in range(60):
        device_map[f"model.language_model.layers.{i}"] = 0

    # 后 20 层 -> GPU1
    for i in range(60, 64):
        device_map[f"model.language_model.layers.{i}"] = 1

    # ===== 输出头放 GPU1 =====
    device_map["model.language_model.norm"] = 1
    device_map["model.language_model.rotary_emb"] = 1
    device_map["lm_head"] = 1

    return device_map


def print_hf_device_map(model):
    print("\n" + "=" * 100)
    print("Inspecting hf_device_map ...")

    candidates = [
        ("model", model),
        ("model.model", getattr(model, "model", None)),
        ("model.base_model", getattr(model, "base_model", None)),
        ("model.base_model.model", getattr(getattr(model, "base_model", None), "model", None)),
    ]

    found = False
    chosen_map = None
    chosen_name = None

    for name, obj in candidates:
        if obj is None:
            continue

        hf_map = getattr(obj, "hf_device_map", None)
        if hf_map is not None:
            found = True
            chosen_map = hf_map
            chosen_name = name

            print(f"\n[{name}] hf_device_map found:")
            for k, v in hf_map.items():
                print(f"  {k} -> {v}")

            print("\nModule count by device:")
            print(Counter(hf_map.values()))
            break

    if not found:
        print("No hf_device_map found on checked objects.")
        print("=" * 100 + "\n")
        return

    print("\nKey module placement:")
    key_patterns = [
        "embed_tokens",
        "layers.",
        "norm",
        "lm_head",
        "visual",
        "merger",
        "rotary_emb",
    ]

    for k, v in chosen_map.items():
        if any(p in k for p in key_patterns):
            print(f"  {k} -> {v}")

    print(f"\nUsing hf_device_map from: {chosen_name}")
    print("=" * 100 + "\n")


def train(attn_implementation="flash_attention_2"):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    # 单进程双卡模型并行
    assert torch.cuda.is_available(), "未检测到 CUDA"
    assert torch.cuda.device_count() >= 2, "该方案需要至少 2 张 GPU"

    primary_device = "cuda:0"
    seg_device = "cuda:1"

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    load_in_4bit = True
    manual_device_map = build_manual_device_map()
    rank0_print("Loading model in single-process 2-GPU model-parallel mode...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        load_in_4bit=load_in_4bit,
        torch_dtype=dtype,
        cache_dir=training_args.cache_dir,
        device_map=manual_device_map,
        attn_implementation=attn_implementation,
    )
    print_hf_device_map(model)

    rank0_print("Adding <SEG> token.")
    if hasattr(tokenizer, "tokenizer"):
        inner_tokenizer = tokenizer.tokenizer
    else:
        inner_tokenizer = tokenizer

    if "<SEG>" not in inner_tokenizer.get_vocab():
        inner_tokenizer.add_tokens(["<SEG>"])
        model.resize_token_embeddings(len(inner_tokenizer))

    seg_token_id = inner_tokenizer.convert_tokens_to_ids("<SEG>")
    im_end_token_id = inner_tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.config.seg_token_id = seg_token_id
    model.config.im_end_token_id = im_end_token_id

    rank0_print("Initializing <SEG> embedding.")
    with torch.no_grad():
        ref_token_id = inner_tokenizer.convert_tokens_to_ids("mask")
        if ref_token_id is None or ref_token_id == inner_tokenizer.unk_token_id:
            ref_token_id = inner_tokenizer.convert_tokens_to_ids(".")

        input_embeds = model.get_input_embeddings().weight
        input_embeds[seg_token_id] = input_embeds[ref_token_id].clone()

        output_embeds = model.get_output_embeddings()
        if output_embeds is not None:
            output_embeds.weight[seg_token_id] = output_embeds.weight[ref_token_id].clone()

    sam_checkpoint = getattr(model_args, "sam_checkpoint", "/home/zhangxw/Mode/sam_vit_b_01ec64.pth")
    rank0_print("Injecting LISA modules on secondary GPU.")
    model = build_lisa_modules(
        model,
        projection_dim=256,
        sam_checkpoint=sam_checkpoint,
        seg_device=seg_device,
    )

    # Loss weights
    model.loss_mask_weight = 5.0
    model.loss_dice_weight = 2.0
    model.loss_area_weight = 1.0
    model.ce_loss_weight = 1.0
    model.seg_token_loss_weight = 0.5
    model.seg_token_margin_weight = 0.1
    model.seg_token_margin = 1.0

    rank0_print(
        f"Loss weights | ce={model.ce_loss_weight} | mask_bce={model.loss_mask_weight} | "
        f"dice={model.loss_dice_weight} | area={model.loss_area_weight} | "
        f"seg_ce={model.seg_token_loss_weight} | seg_margin={model.seg_token_margin_weight}"
    )

    rank0_print("Applying LoRA.")
    target_modules_config = ["qkv", "proj", "fc1", "fc2", "merger.mlp.0", "merger.mlp.2"]

    # 只让 Qwen / tokenizer head / projection 进入 PEFT 保存流
    # mask_decoder 不放进 modules_to_save，避免包装破坏 SAM 接口
    model = FastVisionModel.get_peft_model(
        model,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=target_modules_config,
        modules_to_save=[
            "embed_tokens",
            "lm_head",
            "seg_token_mask_projection",
        ],
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        random_state=3407,
    )

    # 参数冻结 / 解冻策略
    for name, param in model.named_parameters():
        if "seg_token_mask_projection" in name:
            param.requires_grad = True
        elif "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
        elif "lora_" in name:
            param.requires_grad = True
        elif "visual_model.image_encoder" in name:
            param.requires_grad = False
        elif "visual_model.prompt_encoder" in name:
            param.requires_grad = False
        elif "visual_model.mask_decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    try:
        data_args.image_processor = model.processor.image_processor
    except AttributeError:
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor

    data_args.model_type = "qwen2.5vl"
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    rank0_print("=== Trainable Parameters ===")
    print_trainable_layers(model)

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=inner_tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=inner_tokenizer, data_args=data_args)

    llm_params, vision_head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "seg_token_mask_projection" in name or "visual_model.mask_decoder" in name:
            vision_head_params.append(param)
        else:
            llm_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": llm_params, "lr": training_args.learning_rate},
            {"params": vision_head_params, "lr": getattr(training_args, "projector_lr", 1e-4)},
        ],
        weight_decay=training_args.weight_decay,
    )

    trainer = LISATrainer(
        model=model,
        teacher_model=None,
        alpha=0.0,
        temperature=2.0,
        distillation_mode="vision",
        processing_class=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        callbacks=[
            SaveLISAExtraModulesCallback(),
        ],
        **data_module,
    )
    epoch_summary_callback = EpochSummaryCallback(lambda: trainer)
    trainer.add_callback(epoch_summary_callback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    try:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save tokenizer. Error: {e}")

    try:
        data_args.image_processor.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save image_processor. Error: {e}")

    # 1) 保存 HF / PEFT 常规内容
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
    )

    # 2) 保存最终根目录的额外模块
    save_lisa_extra_modules(
        trainer=trainer,
        output_dir=training_args.output_dir,
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
