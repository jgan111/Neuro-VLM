import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = 2.0 * (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (inter + eps) / (union + eps)).mean()


def compute_seg_token_losses(logits, labels, seg_token_id, seg_sample_mask=None, im_end_token_id=None, margin=1.0):
    """
    只在 label 真正等于 <SEG> 的预测位点上额外施压：
    1) 再做一遍聚焦版 CE（提高 <SEG> 学习强度）
    2) 显式压制 <|im_end|> 的 logit，不让它在第一步轻易赢过 <SEG>
    """
    if logits is None or labels is None:
        return None, None

    if logits.size(1) < 2 or labels.size(1) < 2:
        zero = logits.new_zeros(())
        return zero, zero

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    seg_positions = shift_labels.eq(seg_token_id)

    if seg_sample_mask is not None:
        seg_positions = seg_positions & seg_sample_mask.to(seg_positions.device).view(-1, 1).bool()

    if not seg_positions.any():
        zero = shift_logits.new_zeros(())
        return zero, zero

    seg_logits = shift_logits[seg_positions]
    seg_targets = shift_labels[seg_positions]
    seg_ce_loss = F.cross_entropy(seg_logits.float(), seg_targets.long(), reduction="mean")

    if im_end_token_id is None:
        seg_margin_loss = shift_logits.new_zeros(())
    else:
        target_scores = seg_logits[:, seg_token_id]
        eos_scores = seg_logits[:, im_end_token_id]
        seg_margin_loss = F.relu(eos_scores - target_scores + margin).mean()

    return seg_ce_loss, seg_margin_loss


def build_lisa_modules(model, projection_dim, sam_checkpoint, seg_device="cuda:1"):
    """
    单进程双卡模型并行版本：
    - Qwen 主干由 from_pretrained(device_map=...) 自动分配
    - LISA/SAM 分割分支固定放到 seg_device（默认 cuda:1）
    """
    hidden_size = model.config.hidden_size

    model.seg_device = torch.device(seg_device)

    model.seg_token_mask_projection = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, projection_dim),
        nn.LayerNorm(projection_dim),
    ).to(model.seg_device).to(model.dtype)

    def build_sam_by_name(checkpoint_path):
        name = str(checkpoint_path).lower()
        if "vit_b" in name:
            return build_sam_vit_b(checkpoint=checkpoint_path)
        elif "vit_l" in name:
            return build_sam_vit_l(checkpoint=checkpoint_path)
        else:
            return build_sam_vit_h(checkpoint=checkpoint_path)

    model.visual_model = build_sam_by_name(sam_checkpoint).to(model.seg_device)

    # 冻结 SAM image encoder
    for p in model.visual_model.image_encoder.parameters():
        p.requires_grad = False

    model.ce_loss_weight = getattr(model, "ce_loss_weight", 0.3)
    model.loss_mask_weight = getattr(model, "loss_mask_weight", 4.0)
    model.loss_dice_weight = getattr(model, "loss_dice_weight", 2.0)
    model.loss_area_weight = getattr(model, "loss_area_weight", 1.0)
    model.seg_token_loss_weight = getattr(model, "seg_token_loss_weight", 0.8)
    model.seg_token_margin_weight = getattr(model, "seg_token_margin_weight", 0.2)
    model.seg_token_margin = getattr(model, "seg_token_margin", 0.5)

    if not hasattr(model, "_lisa_base_forward"):
        model._lisa_base_forward = model.forward

    def lisa_forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        position_ids=None,
        masks=None,
        sam_images=None,
        seg_sample_mask=None,
        output_hidden_states=True,
        **kwargs,
    ):
        base_outputs = self._lisa_base_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            output_hidden_states=True,
            **kwargs,
        )

        ce_loss = base_outputs.loss
        hidden_states = base_outputs.hidden_states[-1]
        total_loss = ce_loss * getattr(self, "ce_loss_weight", 1.0) if ce_loss is not None else None

        pred_masks = None
        mask_bce_loss = None
        mask_dice_loss = None
        area_loss = None
        seg_token_loss = None
        seg_token_margin_loss = None

        seg_token_id = getattr(self.config, "seg_token_id", None)
        if seg_token_id is None:
            raise ValueError("model.config.seg_token_id is missing.")
        im_end_token_id = getattr(self.config, "im_end_token_id", None)

        if labels is not None and hasattr(base_outputs, "logits"):
            seg_token_loss, seg_token_margin_loss = compute_seg_token_losses(
                logits=base_outputs.logits,
                labels=labels,
                seg_token_id=seg_token_id,
                seg_sample_mask=seg_sample_mask,
                im_end_token_id=im_end_token_id,
                margin=getattr(self, "seg_token_margin", 1.0),
            )
            aux = (
                getattr(self, "seg_token_loss_weight", 3.0) * seg_token_loss
                + getattr(self, "seg_token_margin_weight", 1.0) * seg_token_margin_loss
            )
            total_loss = aux if total_loss is None else total_loss + aux

        if sam_images is None or masks is None or seg_sample_mask is None:
            return {
                "loss": total_loss if total_loss is not None else base_outputs.loss,
                "pred_masks": pred_masks,
                "ce_loss": ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "area_loss": area_loss,
                "seg_token_loss": seg_token_loss.detach() if isinstance(seg_token_loss, torch.Tensor) else seg_token_loss,
                "seg_token_margin_loss": seg_token_margin_loss.detach() if isinstance(seg_token_margin_loss, torch.Tensor) else seg_token_margin_loss,
                "text_outputs": base_outputs,
            }

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        seg_sample_mask = seg_sample_mask.to(hidden_states.device).bool()
        if seg_sample_mask.ndim != 1 or seg_sample_mask.shape[0] != hidden_states.shape[0]:
            raise ValueError(f"seg_sample_mask shape mismatch: {seg_sample_mask.shape}, batch={hidden_states.shape[0]}")

        valid_batch_indices = torch.nonzero(seg_sample_mask, as_tuple=False).squeeze(1)
        if valid_batch_indices.numel() == 0:
            return {
                "loss": total_loss if total_loss is not None else base_outputs.loss,
                "pred_masks": pred_masks,
                "ce_loss": ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "area_loss": area_loss,
                "seg_token_loss": seg_token_loss.detach() if isinstance(seg_token_loss, torch.Tensor) else seg_token_loss,
                "seg_token_margin_loss": seg_token_margin_loss.detach() if isinstance(seg_token_margin_loss, torch.Tensor) else seg_token_margin_loss,
                "text_outputs": base_outputs,
            }

        seg_queries = []
        used_indices = []

        for b in valid_batch_indices.tolist():
            token_pos = torch.empty(0, device=hidden_states.device, dtype=torch.long)

            if labels is not None:
                token_pos = torch.nonzero(labels[b] == seg_token_id, as_tuple=False).squeeze(1)

            if token_pos.numel() == 0:
                token_pos = torch.nonzero(input_ids[b] == seg_token_id, as_tuple=False).squeeze(1)

            if token_pos.numel() == 0:
                continue

            # 把 query hidden state 搬到分割卡
            seg_queries.append(hidden_states[b, token_pos[-1], :].to(self.seg_device))
            used_indices.append(b)

        if len(seg_queries) == 0:
            return {
                "loss": total_loss if total_loss is not None else base_outputs.loss,
                "pred_masks": pred_masks,
                "ce_loss": ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "area_loss": area_loss,
                "seg_token_loss": seg_token_loss.detach() if isinstance(seg_token_loss, torch.Tensor) else seg_token_loss,
                "seg_token_margin_loss": seg_token_margin_loss.detach() if isinstance(seg_token_margin_loss, torch.Tensor) else seg_token_margin_loss,
                "text_outputs": base_outputs,
            }

        used_indices_cpu = used_indices
        used_indices = torch.tensor(used_indices, device=self.seg_device, dtype=torch.long)
        seg_queries = torch.stack(seg_queries, dim=0)

        seg_prompts = self.seg_token_mask_projection(seg_queries).to(
            self.visual_model.mask_decoder.iou_prediction_head.layers[0].weight.dtype
        )

        sam_images_used = sam_images[used_indices_cpu].to(self.seg_device).to(
            self.visual_model.image_encoder.patch_embed.proj.weight.dtype
        )
        image_embeddings = self.visual_model.image_encoder(sam_images_used)

        low_res_masks_list = []
        for i in range(seg_prompts.shape[0]):
            sparse_prompt_embeddings = seg_prompts[i:i + 1].unsqueeze(1)

            dense_prompt_embeddings = self.visual_model.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(
                1,
                -1,
                self.visual_model.prompt_encoder.image_embedding_size[0],
                self.visual_model.prompt_encoder.image_embedding_size[1],
            ).to(self.seg_device)

            low_res_mask_i, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                image_pe=self.visual_model.prompt_encoder.get_dense_pe().to(self.seg_device),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
            )
            low_res_masks_list.append(low_res_mask_i)

        low_res_masks = torch.cat(low_res_masks_list, dim=0)

        target_mask_size = 512
        pred_masks = F.interpolate(
            low_res_masks,
            size=(target_mask_size, target_mask_size),
            mode="bilinear",
            align_corners=False,
        )

        gt_masks = masks[used_indices_cpu].to(self.seg_device).float()
        gt_masks = F.interpolate(
            gt_masks,
            size=(target_mask_size, target_mask_size),
            mode="nearest",
        )

        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks.float(), gt_masks.float())
        mask_dice_loss = dice_loss_from_logits(pred_masks.float(), gt_masks.float())

        pred_probs = torch.sigmoid(pred_masks.float())
        gt_area_ratio = gt_masks.float().mean(dim=(1, 2, 3))
        pred_area_ratio = pred_probs.mean(dim=(1, 2, 3))
        area_loss = F.l1_loss(pred_area_ratio, gt_area_ratio)

        mask_loss = (
            getattr(self, "loss_mask_weight", 4.0) * mask_bce_loss
            + getattr(self, "loss_dice_weight", 2.0) * mask_dice_loss
            + getattr(self, "loss_area_weight", 1.0) * area_loss
        )

        if total_loss is None:
            total_loss = mask_loss
        else:
            total_loss = total_loss.to(mask_loss.device) + mask_loss
        if isinstance(ce_loss, torch.Tensor):
            total_loss = total_loss.to(ce_loss.device)
        elif input_ids is not None and isinstance(input_ids, torch.Tensor):
            total_loss = total_loss.to(input_ids.device)

        return {
            "loss": total_loss,
            "pred_masks": pred_masks,
            "ce_loss": ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            "mask_bce_loss": mask_bce_loss.detach(),
            "mask_dice_loss": mask_dice_loss.detach(),
            "area_loss": area_loss.detach(),
            "seg_token_loss": seg_token_loss.detach() if isinstance(seg_token_loss, torch.Tensor) else seg_token_loss,
            "seg_token_margin_loss": seg_token_margin_loss.detach() if isinstance(seg_token_margin_loss, torch.Tensor) else seg_token_margin_loss,
            "text_outputs": base_outputs,
        }

    model.forward = types.MethodType(lisa_forward, model)
    return model