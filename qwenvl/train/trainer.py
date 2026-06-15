import os
from typing import Dict, List, Optional, Sequence, Any

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration, # 确保这个被导入
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.trainer import (
   # ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import seed_worker

### FIX 1: Import the EmptyLogits class ### 


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches
# Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters


class DistillationTrainer(Trainer):
    """
    自定义的 Trainer，用于实现知识蒸馏。
    
    它会同时处理一个“学生模型”（model）和一个“教师模型”（teacher_model）。
    损失函数 = (1-alpha) * L_hard + alpha * L_soft
    
    新增功能:
    - distillation_mode:
        - 'logits': (默认) 传统的知识蒸馏，匹配最终的文本输出 logits。
        - 'vision': 特征蒸馏，匹配 LLM 输入端的视觉/文本混合特征。
    """
    def __init__(
        self,
        *args,
        teacher_model: Optional[nn.Module] = None,
        alpha: float = 0.5,
        temperature: float = 2.0,
        distillation_mode: str = "logits",  # <--- 新增参数
        **kwargs
    ):
        """
        初始化 DistillationTrainer。
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_mode = distillation_mode # <--- 新增参数

        if self.distillation_mode not in ["logits", "vision"]:
            raise ValueError(f"distillation_mode 必须是 'logits' 或 'vision'，但收到了: {self.distillation_mode}")

        if self.teacher_model is not None:
            # 切换到评估模式（关闭 dropout 等）
            self.teacher_model.eval()
            # 冻结所有参数
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        print(f"--- DistillationTrainer ---")
        print(f"  Mode:  {self.distillation_mode}")
        print(f"  Alpha: {self.alpha}")
        if self.distillation_mode == "logits":
            print(f"  Temp:  {self.temperature}")
        print(f"-----------------------------")


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss 方法来实现蒸馏。
        """
        
        # --- 步骤 1: 计算学生模型的 "Hard Loss" ---
        # output_hidden_states=True 确保我们能拿到特征
        student_outputs = model(**inputs, output_hidden_states=True)

        if not (hasattr(student_outputs, "loss") and student_outputs.loss is not None):
            raise ValueError(
                "学生模型没有返回 'loss'。在蒸馏模式下，"
                "学生模型必须在其 forward 输出中计算并返回自己的 'hard loss'（例如：与真实标签的损失）。"
            )
        
        # loss_hard 是学生自己学习文本输出的部分
        loss_hard = student_outputs.loss

        # --- 步骤 2: 计算蒸馏的 "Soft Loss" ---
        
        if self.teacher_model is None or self.alpha == 0.0:
            return (loss_hard, student_outputs) if return_outputs else loss_hard

        with torch.no_grad():
            # 教师模型也需要输出 hidden_states
            teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)

        
        if self.distillation_mode == "logits":
            # --- 模式 A: Logit 蒸馏 (模仿文本输出) ---
            student_logits = student_outputs.get("logits")
            teacher_logits = teacher_outputs.get("logits")

            if teacher_logits.__class__.__name__ == "EmptyLogits":
                 return (loss_hard, student_outputs) if return_outputs else loss_hard

            if student_logits is None or teacher_logits is None:
                raise ValueError("Logit 蒸馏失败：学生或教师模型没有返回 'logits'。")
            
            if student_logits.shape != teacher_logits.shape:
                raise ValueError(f"Logits 形状不匹配！学生: {student_logits.shape}, 教师: {teacher_logits.shape}")

            # 使用温度 T 平滑 logits 并计算 KL 散度
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)

            loss_soft = F.kl_div(
                input=soft_prob, 
                target=soft_targets, 
                reduction='batchmean'
            ) * (self.temperature ** 2)

        elif self.distillation_mode == "vision":
            # --- 模式 B: Vision/Feature 蒸馏 (模仿视觉特征) ---
            
            # 我们比较 LLM 的输入层 (hidden_states[0])
            # 这一层是词嵌入和投影后的视觉特征合并的地方
            student_features = student_outputs.hidden_states[0]
            teacher_features = teacher_outputs.hidden_states[0]

            if student_features is None or teacher_features is None:
                 raise ValueError("Feature 蒸馏失败：学生或教师模型没有返回 'hidden_states'。")

            if student_features.shape != teacher_features.shape:
                raise ValueError(f"Feature 形状不匹配！学生: {student_features.shape}, 教师: {teacher_features.shape}")

            # 使用均方误差 (MSE) 来让学生特征匹配教师特征
            student_features_fp32 = student_features.to(torch.float32)
            teacher_features_fp32 = teacher_features.to(torch.float32)
            loss_soft_fp32 = F.mse_loss(student_features_fp32, teacher_features_fp32)
            loss_soft = loss_soft_fp32.to(student_outputs.loss.dtype)

        else:
            # 这种情况不应该发生，已在 __init__ 中检查
            raise NotImplementedError

        # --- 步骤 3: 组合损失 ---
        # (1-alpha) * (学生学习标签) + (alpha) * (学生学习教师)
        total_loss = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft
        
        return (total_loss, student_outputs) if return_outputs else total_loss

