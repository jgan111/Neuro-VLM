import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    # tune_mm_llm: bool = field(default=False)
    # tune_mm_mlp: bool = field(default=False)
    # tune_mm_vision: bool = field(default=False)

    freeze_layers: int = field(
        default=30,
        metadata={"help": "冻结前N层 LLM transformer"}
    )

    intermediate_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the intermediate model checkpoint to be loaded and frozen."}
    )

    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA 秩 (rank)"}
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )

    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout"}
    )

    lora_target_modules: Optional[str] = field(
        default="all-linear",
        metadata={"help": "要应用 LoRA 的模块, 逗号分隔"}
    )

    sam_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "SAM 模型权重文件的路径，例如 sam_vit_h_4b8939.pth"}
    )

    loss_mask_weight: float = field(
        default=5.0,
        metadata={"help": "分割掩码 BCE 损失的权重 (lambda_mask)"}
    )

    loss_dice_weight: float = field(
        default=2.0,
        metadata={"help": "分割掩码 DICE 损失的权重 (lambda_dice)"}
    )


@dataclass
class DataArguments:
    model_type: str = field(
        default="qwen2.5-vl",
        metadata={"help": "The type of model being used (e.g., qwen2.5-vl, qwen2-vl, etc.)"}
    )

    dataset_use: str = field(default="")

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data JSON file."}
    )

    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)

    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)

    base_interval: int = field(default=2)

    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch")

    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)

    head_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for embed_tokens and lm_head."}
    )

    merger_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for merger LoRA parameters."}
    )