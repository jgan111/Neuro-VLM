#!/bin/bash

MODEL_PATH="/home/zhangxw/Mode/Qwen2.5-VL-32B/"
OUTPUT_DIR="/home/zhangxw/share_data/VQA-RAD/output_3/"
CACHE_DIR="./cache"

# Stage-1 LISA adapter is used only as initialization for stage-2 VQA SFT.
# Do not pass this path to --resume_from_checkpoint.
export STAGE1_ADAPTER_PATH="/home/zhangxw/share_data/VLD/SPM-mode/checkpoint-9606/"

# Legacy stage-2 policy: train all LLM LoRA and early visual block LoRA.
export LANGUAGE_LORA_MODE="full"
export TRAIN_LM_LAST_N_LAYERS=64
export QWEN_LM_NUM_LAYERS=64
export FORCE_SHORT_ANSWER_PROMPT=1

# Keep stage-2 adapter compact and avoid disturbing the base language head.
export TRAIN_EMBED_LM_HEAD=0
export LOAD_STAGE1_EMBED_LM_HEAD=0
export TRAIN_VISUAL_BLOCKS_LORA=1
export TRAIN_VISUAL_FIRST_N_BLOCKS=5

# Stage-2 VQA has no mask supervision, so do not load/save LISA segmentation heads.
export LOAD_STAGE1_SEG_PROJECTION=0
export SAVE_STAGE2_SEG_PROJECTION=0
export USE_SEG_TOKEN_IN_STAGE2=0

# Only set this when resuming an interrupted stage-2 run, never for stage-1.
STAGE2_RESUME_FROM_CHECKPOINT=""

CMD=(
    python /home/zhangxw/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/Stage_two.py
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --dataset_use my_dataset4
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv,proj,fc1,fc2,merger.mlp.0,merger.mlp.2
    --max_pixel 112896
    --learning_rate 1e-5
    --weight_decay 0.05
    --per_device_train_batch_size 64
    --gradient_accumulation_steps 1
    --bf16 True
    --num_train_epochs 30
    --model_max_length 4096
    --logging_steps 10
    --save_strategy epoch
    --save_total_limit 1000
    --dataloader_num_workers 24
    --dataloader_pin_memory True
)

if [ -n "$STAGE2_RESUME_FROM_CHECKPOINT" ]; then
    CMD+=(--resume_from_checkpoint "$STAGE2_RESUME_FROM_CHECKPOINT")
fi

"${CMD[@]}"
