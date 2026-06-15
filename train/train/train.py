import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
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


# 冻结中间模型
def freeze_intermediate_model(model, intermediate_model_path):
    # 加载冻结的中间模型
    intermediate_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(intermediate_model_path)
    
    # 冻结中间模型的所有参数
    for param in intermediate_model.parameters():
        param.requires_grad = False
    
    return intermediate_model


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def print_trainable_layers(model):
    print("\n====== Parameter Trainability Report ======")
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"{status:<10} | {name}")
    print("===========================================\n")


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    # 加载并冻结中间模型
    intermediate_model_path = "/home/zhangxw/share_data/VLD/mode/checkpoint-1600/"  # 设置为您的冻结中间模型路径
    intermediate_model = freeze_intermediate_model(model, intermediate_model_path)

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # === 打印确认使用的数据集信息 ===
    print("\n" + "=" * 80)
    print(f"[Qwen-VL] ✅ Using datasets: {data_args.dataset_use}")

    # 检查 data_module 中的数据集内容（train_dataset / eval_dataset）
    try:
        if hasattr(data_module, "train_dataset"):
            train_ds = data_module.train_dataset
            print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        elif isinstance(data_module, dict) and "train_dataset" in data_module:
            train_ds = data_module["train_dataset"]
            print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        else:
            print("[Qwen-VL] ⚠️ Unable to directly get dataset length (custom loader may be used).")

        # 写入日志文件
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
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
