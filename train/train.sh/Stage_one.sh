#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

MODEL_PATH="/home/zhangxw/Mode/Qwen2.5-VL-32B/"
SAM_CKPT="/home/zhangxw/Mode/sam_vit_b_01ec64.pth"
OUTPUT_DIR="/home/zhangxw/share_data/LISA_zhao_hong/"

python /home/zhangxw/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/Stage_one.py \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_use my_dataset14 \
  --model_type qwen \
  --lora_r 64 \
  --lora_alpha 128 \
  --sam_checkpoint "$SAM_CKPT" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 80 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 100 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --save_strategy epoch \
  --save_total_limit 10000 \
  --dataloader_num_workers 24 \
  --dataloader_pin_memory True
