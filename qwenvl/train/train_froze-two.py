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

from trainer import replace_qwen2_vl_attention_class, DistillationTrainer
from transformers import (
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
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


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


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank if training_args.local_rank is not None else -1
    os.makedirs(training_args.output_dir, exist_ok=True)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    load_in_4bit = True 

    rank0_print("Loading Student Model (4-bit QLoRA)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        load_in_4bit = load_in_4bit,
        torch_dtype = dtype,
        cache_dir = training_args.cache_dir,
        device_map = "auto", 
        attn_implementation = attn_implementation,
    )
    
    rank0_print("Applying LoRA to Student Model...")
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

    # === 【删除】自定义冻结代码块（已移除） ===
    
    try:
        data_args.image_processor = model.processor.image_processor
    except AttributeError:
        rank0_print("Warning: Could not find model.processor.image_processor. Falling back to AutoProcessor.")
        data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
        
    data_args.model_type = "qwen2.5vl" 

    teacher_model = None
    # ... (教师模型加载代码保持注释掉) ...

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            if hasattr(model, "get_input_embeddings"):
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    rank0_print("=== Trainable Parameters (Student Model) ===")
    print_trainable_layers(model)
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
    print("\n" + "=" * 80)
    print(f"[Qwen-VL] ✅ Using datasets: {data_args.dataset_use}")
    try:
        train_ds = data_module["train_dataset"]
        print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        log_path = os.path.join(training_args.output_dir, "dataset_info.log")
        with open(log_path, "w") as f:
            f.write(f"Using datasets: {data_args.dataset_use}\n")
            f.write(f"Train dataset size: {len(train_ds)}\n")
        print(f"[Qwen-VL] Dataset info has been saved to: {log_path}")
    except Exception as e:
        print(f"[Qwen-VL] ⚠️ Failed to retrieve dataset info: {e}")
    print("=" * 80 + "\n")
    
    training_args.gradient_checkpointing = True 
    
    trainer = DistillationTrainer(
        model=model,                 
        teacher_model=None,          
        alpha=0.0,                   
        temperature=2.0,
        distillation_mode="vision",
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    # --- 关键修复 ---
    # 移除有问题的 if/else 逻辑
    # 这一行代码会正确地：
    # 1. 检查 training_args.resume_from_checkpoint (来自 --resume_from_checkpoint) 是否是一个有效路径。
    # 2. 如果是，就从那里加载。
    # 3. 如果不是，就检查 output_dir 是否有 checkpoint。
    # 4. 如果都没有，就从 0 开始。
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # --- 修复结束 ---
        
    trainer.save_state()
    
    try:
        data_args.image_processor.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save image_processor. Error: {e}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")