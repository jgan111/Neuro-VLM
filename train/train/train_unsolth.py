from unsloth import FastVisionModel
import os
import logging
import pathlib
import torch
import transformers
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# === 1. 修改导入 ===
from trainer import replace_qwen2_vl_attention_class, DistillationTrainer

from transformers import (
    # Qwen2VLForConditionalGeneration, # <--- 删掉
    # Qwen2_5_VLForConditionalGeneration, # <--- 删掉
    HfArgumentParser,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor

local_rank = None


def rank0_print(*args):
    if local_rank <= 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# === 2. 'set_model' 函数已被删除 ===


def print_trainable_layers(model):
    """
    替代原有的打印函数，用于显示 LoRA 模型的可训练参数。
    """
    print("\n====== Parameter Trainability Report (LoRA) ======")
    try:
        model.print_trainable_parameters()
    except AttributeError:
        # 备用方案，如果 Unsloth 的函数不可用
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable | {name} | {param.shape}")
            else:
                # 不打印所有冻结的层，太多了
                pass
        print("---")
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Trainable %: {(100 * trainable_params / total_params):.4f}")
    print("=================================================\n")


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # === 3. 添加 Unsloth 配置 ===
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    load_in_4bit = True # <--- 强制 4-bit

    # === 4. 修改 "学生" 模型加载 ===
    rank0_print("Loading Student Model (4-bit QLoRA)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        load_in_4bit = load_in_4bit,
        torch_dtype = dtype,
        cache_dir = training_args.cache_dir,
        device_map = "auto",
        attn_implementation = attn_implementation,
    )
    
    # === 5. 添加 LoRA (PEFT) 配置 ===
    rank0_print("Applying LoRA to Student Model...")
    
    # 确保 training_args.gradient_checkpointing 为 True
    # 我们将在下面再次设置它，但 Unsloth 会在这里使用它
    training_args.gradient_checkpointing = True 
    
    lora_target_modules_arg = model_args.lora_target_modules
    if lora_target_modules_arg.lower() == "all-linear":
        target_modules_config = "all-linear"
    else:
        target_modules_config = lora_target_modules_arg.split(",")
    model = FastVisionModel.get_peft_model(
        model,
        r = model_args.lora_r,
        lora_alpha = model_args.lora_alpha,
        lora_dropout = model_args.lora_dropout,
        bias = "none",
        use_gradient_checkpointing = training_args.gradient_checkpointing, 
        target_modules = target_modules_config,
    )
    
    # 自动获取 Image Processor
    try:
        data_args.image_processor = model.processor.image_processor
    except AttributeError:
        rank0_print("Warning: Could not find model.processor.image_processor. Falling back to AutoProcessor.")
        data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
        
    data_args.model_type = "qwen2.5vl" # 假设

    # === 6. 修改 "教师" 模型加载 ===
#    teacher_model = None
 #   if model_args.intermediate_model_path:
 #       rank0_print(f"Loading Teacher Model (4-bit) from: {model_args.intermediate_model_path}")
 #       teacher_model, _ = FastVisionModel.from_pretrained(
 #           model_name = model_args.intermediate_model_path,
 #           load_in_4bit = load_in_4bit,
 #           torch_dtype = dtype,
 #           cache_dir = training_args.cache_dir,
 #           attn_implementation = attn_implementation,
 #       )

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    
    # 注意：PeftModel 会自动处理 use_cache
    # model.config.use_cache = False 

    # === 7. 梯度检查点 ===
    # Unsloth 的 get_peft_model 已经处理了梯度检查点,
    # 但我们保留 transformers 原本的逻辑以防万一
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            if hasattr(model, "get_input_embeddings"):
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ... (tokenizer 加载保持不变, FastVisionModel 已经返回了) ...
    
    # === 8. 'set_model' 调用已被删除 ===

        # 打印 LoRA 可训练参数
    rank0_print("=== Trainable Parameters (Student Model) ===")
    print_trainable_layers(model) # <--- 使用新的打印

    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
    # === 打印确认使用的数据集信息 ===
    print("\n" + "=" * 80)
    print(f"[Qwen-VL] ✅ Using datasets: {data_args.dataset_use}")

    try:
        if hasattr(data_module, "train_dataset"):
            train_ds = data_module["train_dataset"]
            print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        elif isinstance(data_module, dict) and "train_dataset" in data_module:
            train_ds = data_module["train_dataset"]
            print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        else:
            print("[Qwen-VL] ⚠️ Unable to directly get dataset length (custom loader may be used).")

        log_path = os.path.join(training_args.output_dir, "dataset_info.log")
        with open(log_path, "w") as f:
            f.write(f"Using datasets: {data_args.dataset_use}\n")
            try:
                f.write(f"Train dataset size: {len(train_ds)}\n")
            except Exception:
                f.write("Dataset size: unknown\n")
        print(f"[Qwen-VL] Dataset info has been saved to: {log_path}")
    except Exception as e:
        print(f"[Qwen-VL] ⚠️ Failed to retrieve dataset info: {e}")
    print("=" * 80 + "\n")
    
    # === 9. 确保 TrainingArguments 开启梯度检查点 ===
    training_args.gradient_checkpointing = True 
    
    # === 10. DistillationTrainer 实例化 (保持不变) ===
    trainer = DistillationTrainer(
        model=model,                 # 学生模型 (现在是 PeftModel)
        teacher_model=None, # 教师模型 (现在是 4-bit)
        alpha=0.0,                   # 蒸馏权重 (可调)
        temperature=2.0,
        distillation_mode="vision",
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    try:
        data_args.image_processor.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save image_processor. Error: {e}")

    # model.config.use_cache = True # PeftModel 会处理

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
