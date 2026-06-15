import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from unsloth import FastVisionModel
import json
import cv2
import sys
import re
import gc
import csv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import AutoProcessor

from QwenLISA_two import build_lisa_modules

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ====== 默认路径；也可以在命令行用参数覆盖 ======
BASE_MODEL = "/home/zhangxw/Mode/Qwen2.5-VL-32B/"
# 单个 adapter/checkpoint 路径；不传 --adapter-root 时使用
ADAPTER_PATH = "/home/zhangxw/share_data/LISA_mix/checkpoint-15168/"
# 多个 adapter/checkpoint 的父目录；命令行传 --adapter-root 后会自动扫描里面的 checkpoint-*
ADAPTER_ROOT = None
SAM_CHECKPOINT = "/home/zhangxw/Mode/sam_vit_b_01ec64.pth"

# 单图模式默认值（不传 --input-json 时仍可用）
IMAGE_PATH = "/home/zhangxw/share_data/VLD/image/zhao_hong/zhao_hong_Axial_T1_Slice1.png"
MASK_PATH = "/home/zhangxw/share_data/VLD/image/zhao_hong/zhao_hong_Axial_T1_Slice1_L35.png"
OUTPUT_DIR = "/home/zhangxw/share_data/LISA_output_lora/verify/"
USER_QUERY = "Please output the segmentation mask for the Brain Stem in this image."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 32

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0


# -----------------------------
# 参数解析
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # 数据与输出
    parser.add_argument("--input-json", type=str, default=None, help="JSON 文件路径，内容为样本列表或单个样本对象")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="总结果保存目录")

    # 单模型 / 多模型选择
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH, help="单个 adapter/checkpoint 路径；未传 --adapter-root 时使用")
    parser.add_argument("--adapter-root", type=str, default=ADAPTER_ROOT, help="包含多个 checkpoint/adapters 的父文件夹，例如 /home/zhangxw/share_data/LISA_output")
    parser.add_argument("--checkpoint-pattern", type=str, default="checkpoint-*", help="在 --adapter-root 下匹配模型文件夹的模式，默认 checkpoint-*")
    parser.add_argument("--checkpoint-names", type=str, default=None, help="只验证指定子文件夹名，逗号分隔，例如 checkpoint-100,checkpoint-200")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 --adapter-root 下的 checkpoint 文件夹")
    parser.add_argument("--keep-loaded-adapters", action="store_true", help="验证完一个 adapter 后不卸载；默认会尝试卸载以节省显存")

    # 验证参数
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for pred_prob")
    parser.add_argument(
        "--seg-index",
        type=int,
        default=1,
        help="Which <SEG> to use for final saved output (1-based). If > num_segs, use last one.",
    )
    parser.add_argument(
        "--keep-first-seg-only",
        action="store_true",
        help="After generation, truncate output at the first <SEG> (inclusive).",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="批处理时，遇到单条失败继续后续样本；否则直接抛错终止。",
    )
    parser.add_argument(
        "--skip-failed-model",
        action="store_true",
        help="多模型验证时，某个模型加载或整体验证失败后继续验证下一个模型；否则直接终止。",
    )
    return parser.parse_args()


args = parse_args()
THRESHOLD = args.threshold
SEG_INDEX = args.seg_index
KEEP_FIRST_SEG_ONLY = args.keep_first_seg_only
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# 工具函数
# -----------------------------
def build_sam_image(image_path: str):
    raw_image = Image.open(image_path).convert("RGB")
    w, h = raw_image.size
    scale = 1024 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = raw_image.resize((new_w, new_h))
    img_tensor = to_tensor(img_resized) * 255.0

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    img_tensor = (img_tensor - pixel_mean) / pixel_std
    sam_image = F.pad(img_tensor, (0, 1024 - new_w, 0, 1024 - new_h))
    sam_image = sam_image.unsqueeze(0)   # 先留在 CPU，后面按模块设备再搬
    return raw_image, sam_image


def load_gt_mask(mask_path: str, out_hw=(1024, 1024)):
    if mask_path is None or str(mask_path).strip() == "" or not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path)
    if mask is None:
        return None
    binary_mask = ((mask[:, :, 2].astype(np.int16) - mask[:, :, 0].astype(np.int16)) > 20).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    gt_mask = (clean_mask > 0).astype(np.float32)
    gt_mask = cv2.resize(gt_mask, out_hw, interpolation=cv2.INTER_NEAREST)
    return gt_mask


def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray):
    pred = pred_binary.astype(np.bool_)
    gt = gt_binary.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    iou = inter / (union + 1e-6)
    dice = 2 * inter / (pred_sum + gt_sum + 1e-6)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "pred_area": int(pred_sum),
        "gt_area": int(gt_sum),
        "intersection": int(inter),
        "union": int(union),
    }


def overlay_mask_on_image(image_rgb: np.ndarray, mask_binary: np.ndarray):
    overlay = image_rgb.copy()
    red = np.zeros_like(image_rgb)
    red[:, :, 2] = 255
    alpha = 0.35
    overlay[mask_binary > 0] = (
        (1 - alpha) * overlay[mask_binary > 0] + alpha * red[mask_binary > 0]
    ).astype(np.uint8)
    return overlay


def get_core_model(m):
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model
    return m


def get_module_device(module):
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    raise RuntimeError(f"Cannot infer device for module: {module.__class__.__name__}")


def print_key_module_devices(core_model):
    try:
        print("[Device] seg_token_mask_projection ->", get_module_device(core_model.seg_token_mask_projection))
    except Exception as e:
        print("[Device] seg_token_mask_projection -> ERROR:", e)

    try:
        print("[Device] visual_model.image_encoder ->", get_module_device(core_model.visual_model.image_encoder))
    except Exception as e:
        print("[Device] visual_model.image_encoder -> ERROR:", e)

    try:
        print("[Device] visual_model.mask_decoder ->", get_module_device(core_model.visual_model.mask_decoder))
    except Exception as e:
        print("[Device] visual_model.mask_decoder -> ERROR:", e)

    try:
        print("[Device] visual_model.prompt_encoder ->", get_module_device(core_model.visual_model.prompt_encoder))
    except Exception as e:
        print("[Device] visual_model.prompt_encoder -> ERROR:", e)


def load_processor(adapter_path: str, base_model: str):
    candidates = [adapter_path, str(Path(adapter_path).parent), base_model]
    for candidate in candidates:
        try:
            processor = AutoProcessor.from_pretrained(candidate, local_files_only=True)
            print(f"Loaded processor from: {candidate}")
            return processor
        except Exception as e:
            print(f"Processor load failed from {candidate}: {e}")
    raise RuntimeError("Could not load processor from adapter/output/base path.")


def load_lisa_extra_modules(core_model, adapter_path: str):
    extra_path = os.path.join(adapter_path, "lisa_extra_modules.pt")
    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"lisa_extra_modules.pt not found: {extra_path}")

    ckpt = torch.load(extra_path, map_location="cpu")

    # seg_token_mask_projection 已经在 adapter 里通过 modules_to_save 恢复，
    # 这里不要再手动 load，否则会和 PEFT 的 ModulesToSaveWrapper key 冲突。
    if "visual_model.mask_decoder" in ckpt:
        core_model.visual_model.mask_decoder.load_state_dict(
            ckpt["visual_model.mask_decoder"], strict=True
        )
        print("Loaded visual_model.mask_decoder from lisa_extra_modules.pt")
    else:
        print("Warning: visual_model.mask_decoder not found in lisa_extra_modules.pt")

    print(f"Loaded extra LISA modules from: {extra_path}")


def first_step_debug(model, inputs, seg_token_id, im_end_token_id, tokenizer, k=8):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=False,
            )
    logits = outputs["text_outputs"].logits if isinstance(outputs, dict) else outputs.logits
    next_logits = logits[0, -1].float()
    probs = torch.softmax(next_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    top_tokens = []
    for tid, prob in zip(top_ids.tolist(), top_probs.tolist()):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        top_tokens.append({"token_id": tid, "token_text": token_str, "prob": float(prob)})
    return {
        "seg_token_id": int(seg_token_id),
        "seg_prob": float(probs[seg_token_id].item()),
        "im_end_token_id": int(im_end_token_id),
        "im_end_prob": float(probs[im_end_token_id].item()),
        "topk": top_tokens,
    }


def run_single_seg_inference(core_model, hidden_states, pos, sam_image, gt_mask, threshold):
    # 1) seg query 先送到 seg_token_mask_projection 所在设备
    seg_proj_device = get_module_device(core_model.seg_token_mask_projection)
    seg_query = hidden_states[0, pos, :].unsqueeze(0).to(seg_proj_device, dtype=torch.float32)

    with torch.no_grad():
        seg_prompt = core_model.seg_token_mask_projection(seg_query).float()

    # 2) SAM 相关模块统一到同一设备
    sam_device = get_module_device(core_model.visual_model.image_encoder)

    sparse_prompt_embeddings = seg_prompt.unsqueeze(1).to(sam_device, dtype=torch.float32)
    sam_image = sam_image.to(sam_device, dtype=torch.float32)

    with torch.no_grad():
        image_embeddings = core_model.visual_model.image_encoder(sam_image).float()

        image_pe = core_model.visual_model.prompt_encoder.get_dense_pe().to(
            sam_device, dtype=torch.float32
        )

        dense_prompt_embeddings = core_model.visual_model.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(
            1,
            -1,
            core_model.visual_model.prompt_encoder.image_embedding_size[0],
            core_model.visual_model.prompt_encoder.image_embedding_size[1],
        ).to(sam_device, dtype=torch.float32)

        low_res_masks, _ = core_model.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )

        pred_masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

    pred_logits = pred_masks[0, 0].detach().float().cpu().numpy()
    pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))
    pred_binary = (pred_prob > threshold).astype(np.uint8)

    result = {
        "token_pos": int(pos),
        "pred_prob": pred_prob,
        "pred_binary": pred_binary,
        "pred_area": int(pred_binary.sum()),
    }

    if gt_mask is not None:
        result["metrics"] = compute_metrics(pred_binary, gt_mask)

    return result


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "sample"
    return text[:max_len]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def natural_sort_key(path_or_text):
    """让 checkpoint-2 排在 checkpoint-10 前面。"""
    text = str(Path(path_or_text).name if isinstance(path_or_text, (str, Path)) else path_or_text)
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]


def is_adapter_dir(path: Path) -> bool:
    """判断一个文件夹是否像 PEFT adapter/checkpoint。"""
    if not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weight = (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
        or any(path.glob("adapter_model-*.safetensors"))
        or any(path.glob("adapter_model-*.bin"))
    )
    return has_config and has_weight


def discover_adapter_dirs(adapter_root: str, checkpoint_pattern: str = "checkpoint-*", checkpoint_names: str = None, recursive: bool = False):
    """
    从父目录中找出所有可验证的 adapter/checkpoint 文件夹。

    支持两种情况：
    1) adapter_root 本身就是一个 checkpoint；
    2) adapter_root 下面有多个 checkpoint-* 子目录。
    """
    root = Path(adapter_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--adapter-root 不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--adapter-root 不是文件夹: {root}")

    # 如果 root 本身就是一个 adapter/checkpoint，则直接返回它
    if is_adapter_dir(root):
        candidates = [root]
    else:
        iterator = root.rglob(checkpoint_pattern) if recursive else root.glob(checkpoint_pattern)
        candidates = [p for p in iterator if is_adapter_dir(p)]

        # 如果默认 checkpoint-* 没找到，则兜底扫描所有直接子目录
        if len(candidates) == 0 and not recursive:
            candidates = [p for p in root.iterdir() if is_adapter_dir(p)]

    if checkpoint_names is not None and checkpoint_names.strip() != "":
        wanted = {x.strip() for x in checkpoint_names.split(",") if x.strip()}
        candidates = [p for p in candidates if p.name in wanted]
        missing = sorted(wanted - {p.name for p in candidates}, key=natural_sort_key)
        if missing:
            print(f"[Warning] 以下指定 checkpoint 没有找到或不是有效 adapter: {missing}")

    candidates = sorted(set(candidates), key=natural_sort_key)
    if len(candidates) == 0:
        raise RuntimeError(
            f"在 {root} 中没有找到有效 adapter/checkpoint。\n"
            f"要求目录内至少包含 adapter_config.json 和 adapter_model.safetensors/bin。"
        )
    return [str(p) for p in candidates]


def write_all_models_csv(all_model_summaries, csv_path: str):
    """额外输出一个 CSV，方便快速比较不同 checkpoint 的 IoU/Dice。"""
    rows = []
    for model_summary in all_model_summaries:
        model_name = model_summary.get("model_name")
        adapter_path = model_summary.get("adapter_path")
        model_success = model_summary.get("success", False)
        for item in model_summary.get("results", []):
            row = {
                "model_name": model_name,
                "adapter_path": adapter_path,
                "model_success": model_success,
                "sample_index": item.get("sample_index"),
                "sample_name": item.get("sample_name"),
                "sample_success": item.get("success", False),
                "contains_seg_token": item.get("contains_seg_token"),
                "num_generated_seg_tokens": item.get("num_generated_seg_tokens"),
                "chosen_seg_index": item.get("chosen_seg_index"),
                "image_path": item.get("image_path"),
                "mask_path": item.get("mask_path"),
                "pred_mask_binary": item.get("pred_mask_binary"),
                "pred_overlay": item.get("pred_overlay"),
                "report_path": item.get("report_path"),
                "error": item.get("error"),
            }
            metrics = item.get("metrics_1024") or {}
            row.update({
                "iou": metrics.get("iou"),
                "dice": metrics.get("dice"),
                "pred_area": metrics.get("pred_area"),
                "gt_area": metrics.get("gt_area"),
                "intersection": metrics.get("intersection"),
                "union": metrics.get("union"),
            })
            rows.append(row)

    fieldnames = [
        "model_name", "adapter_path", "model_success",
        "sample_index", "sample_name", "sample_success",
        "iou", "dice", "pred_area", "gt_area", "intersection", "union",
        "contains_seg_token", "num_generated_seg_tokens", "chosen_seg_index",
        "image_path", "mask_path", "pred_mask_binary", "pred_overlay", "report_path", "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def make_sample_name(sample: dict, index: int) -> str:
    # 优先使用显式 id / sample_id / name
    for key in ["id", "sample_id", "name", "case_id"]:
        if key in sample and sample[key] not in [None, ""]:
            return sanitize_filename(sample[key])

    image_path = sample.get("image_path") or sample.get("image") or sample.get("img_path") or sample.get("path")
    if image_path:
        stem = Path(image_path).stem
        return f"{index:04d}_{sanitize_filename(stem)}"
    return f"{index:04d}_sample"


def extract_query_from_conversations(conversations):
    if not isinstance(conversations, list):
        return None

    # 优先取 human / user 的问题
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).strip().lower()
        value = turn.get("value", turn.get("text", turn.get("content", None)))
        if value is None:
            continue
        if role in ["human", "user"]:
            text = str(value).replace("<image>", "").strip()
            if text:
                return text

    # 如果没有明确 human/user，就退化为取第一条有文本的内容
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        value = turn.get("value", turn.get("text", turn.get("content", None)))
        if value is None:
            continue
        text = str(value).replace("<image>", "").strip()
        if text:
            return text

    return None


def resolve_sample_fields(sample: dict):
    image_path = (
        sample.get("image_path")
        or sample.get("image")
        or sample.get("img_path")
        or sample.get("path")
    )
    if image_path is None:
        raise KeyError("样本缺少 image_path/image/img_path/path 字段。")

    user_query = (
        sample.get("question")
        or sample.get("query")
        or sample.get("prompt")
        or sample.get("text")
    )

    if user_query is None:
        user_query = extract_query_from_conversations(sample.get("conversations"))

    if user_query is None:
        raise KeyError("样本缺少 question/query/prompt/text，且 conversations 中也没有可解析的问题文本。")

    mask_path = sample.get("mask_path") or sample.get("mask") or sample.get("gt_mask")
    return str(image_path), str(user_query), (None if mask_path in [None, ""] else str(mask_path))


def load_samples_from_json(input_json: str):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        # 兼容 {"data": [...]} / {"samples": [...]} / 单个样本对象
        if isinstance(data.get("samples"), list):
            samples = data["samples"]
        elif isinstance(data.get("data"), list):
            samples = data["data"]
        elif isinstance(data.get("items"), list):
            samples = data["items"]
        else:
            samples = [data]
    else:
        raise ValueError("JSON 内容必须是 list 或 dict。")

    if len(samples) == 0:
        raise ValueError("输入 JSON 中没有可处理的样本。")
    return samples


def process_one_sample(
    sample_idx,
    sample,
    model,
    core_model,
    processor,
    inner_tokenizer,
    seg_token_id,
    im_end_token_id,
    output_root,
):
    image_path, user_query, mask_path = resolve_sample_fields(sample)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    sample_name = make_sample_name(sample, sample_idx)
    sample_out_dir = os.path.join(output_root, sample_name)
    ensure_dir(sample_out_dir)

    print("\n" + "=" * 100)
    print(f"[Sample {sample_idx}] {sample_name}")
    print(f"Image: {image_path}")
    print(f"Query: {user_query}")
    print(f"Mask : {mask_path}")
    print(f"Save : {sample_out_dir}")

    raw_image, sam_image = build_sam_image(image_path)
    image_for_qwen = raw_image.convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]},
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image_for_qwen], return_tensors="pt")
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    print("\n[Step 0] Inspect first generated-token distribution...")
    first_token_stats = first_step_debug(model, inputs, seg_token_id, im_end_token_id, inner_tokenizer, k=8)
    print("P(<SEG>)=", first_token_stats["seg_prob"])
    print("P(<|im_end|>)=", first_token_stats["im_end_prob"])
    print("Top-k next tokens:")
    for item in first_token_stats["topk"]:
        print(f"  id={item['token_id']:<8d} prob={item['prob']:.6f} text={repr(item['token_text'])}")

    print("\n[Step 1] Generate text to check whether model learned to emit <SEG>...")
    if not hasattr(core_model, "_lisa_base_forward"):
        raise RuntimeError("core_model does not have _lisa_base_forward; current QwenLISA patch is not loaded.")

    orig_forward = core_model.forward
    core_model.forward = core_model._lisa_base_forward
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
    finally:
        core_model.forward = orig_forward

    prompt_len = inputs["input_ids"].shape[1]
    generated_only = gen_ids[:, prompt_len:]

    seg_rel_pos = torch.nonzero(generated_only[0] == seg_token_id, as_tuple=False).squeeze(1)
    num_seg_generated = int(seg_rel_pos.numel())

    if KEEP_FIRST_SEG_ONLY and num_seg_generated > 0:
        first_seg_rel = seg_rel_pos[0].item()
        gen_ids = gen_ids[:, : prompt_len + first_seg_rel + 1]
        generated_only = gen_ids[:, prompt_len:]
        seg_rel_pos = torch.nonzero(generated_only[0] == seg_token_id, as_tuple=False).squeeze(1)
        num_seg_generated = int(seg_rel_pos.numel())

    generated_text = processor.batch_decode(
        generated_only,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]

    print("Generated:", generated_text)
    print("Contains <SEG>:", "<SEG>" in generated_text)
    print("Number of generated <SEG>:", num_seg_generated)

    print("\n[Step 2] Run segmentation branch manually...")
    full_attention_mask = torch.ones_like(gen_ids, device=gen_ids.device)
    pixel_values = inputs.get("pixel_values", None)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_outputs = core_model._lisa_base_forward(
                input_ids=gen_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=True,
            )

    hidden_states = base_outputs.hidden_states[-1]
    seg_pos = torch.nonzero(gen_ids[0] == seg_token_id, as_tuple=False).squeeze(1)
    if seg_pos.numel() == 0:
        raise ValueError("模型生成结果里没有 <SEG> token，无法提取 segmentation query。")

    gt_mask = load_gt_mask(mask_path)
    all_results = []

    for idx, pos in enumerate(seg_pos.tolist(), start=1):
        result = run_single_seg_inference(
            core_model=core_model,
            hidden_states=hidden_states,
            pos=pos,
            sam_image=sam_image,
            gt_mask=gt_mask,
            threshold=THRESHOLD,
        )
        result["seg_index"] = idx
        all_results.append(result)

    print("\nResults for each <SEG>:")
    for r in all_results:
        msg = f"  seg_index={r['seg_index']} token_pos={r['token_pos']} pred_area={r['pred_area']}"
        if "metrics" in r:
            msg += (
                f" iou={r['metrics']['iou']:.6f}"
                f" dice={r['metrics']['dice']:.6f}"
                f" inter={r['metrics']['intersection']}"
                f" union={r['metrics']['union']}"
            )
        print(msg)

    chosen_idx = min(max(SEG_INDEX, 1), len(all_results)) - 1
    chosen = all_results[chosen_idx]
    print(f"\nUsing seg_index={chosen['seg_index']} for final saved outputs.")

    pred_prob = chosen["pred_prob"]
    pred_binary = chosen["pred_binary"]

    image_np = np.array(raw_image)
    orig_w, orig_h = raw_image.size
    pred_binary_orig = cv2.resize(pred_binary * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    pred_prob_vis = (pred_prob * 255).clip(0, 255).astype(np.uint8)
    pred_prob_vis = cv2.resize(pred_prob_vis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    overlay = overlay_mask_on_image(image_np, pred_binary_orig > 0)

    pred_mask_path = os.path.join(sample_out_dir, f"pred_mask_binary_seg{chosen['seg_index']}.png")
    pred_prob_path = os.path.join(sample_out_dir, f"pred_mask_prob_seg{chosen['seg_index']}.png")
    overlay_path = os.path.join(sample_out_dir, f"pred_overlay_seg{chosen['seg_index']}.png")
    cv2.imwrite(pred_mask_path, pred_binary_orig)
    cv2.imwrite(pred_prob_path, pred_prob_vis)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    report = {
        "sample_index": sample_idx,
        "sample_name": sample_name,
        "image_path": image_path,
        "mask_path": mask_path,
        "query": user_query,
        "generated_text": generated_text,
        "contains_seg_token": "<SEG>" in generated_text,
        "num_generated_seg_tokens": num_seg_generated,
        "threshold": THRESHOLD,
        "keep_first_seg_only": KEEP_FIRST_SEG_ONLY,
        "requested_seg_index": SEG_INDEX,
        "chosen_seg_index": chosen["seg_index"],
        "first_token_stats": first_token_stats,
        "all_seg_results": [
            {
                "seg_index": r["seg_index"],
                "token_pos": r["token_pos"],
                "pred_area": r["pred_area"],
                "metrics": r.get("metrics", None),
            }
            for r in all_results
        ],
        "pred_mask_binary": pred_mask_path,
        "pred_mask_prob": pred_prob_path,
        "pred_overlay": overlay_path,
        "pred_foreground_pixels_1024": int(pred_binary.sum()),
    }

    if "metrics" in chosen:
        report["metrics_1024"] = chosen["metrics"]
        print("\nChosen metrics:")
        for k, v in chosen["metrics"].items():
            print(f"  {k}: {v}")
    else:
        print("\nNo GT mask provided; skipped IoU/Dice evaluation.")

    report_path = os.path.join(sample_out_dir, f"verify_report_seg{chosen['seg_index']}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(" -", pred_mask_path)
    print(" -", pred_prob_path)
    print(" -", overlay_path)
    print(" -", report_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "sample_index": sample_idx,
        "sample_name": sample_name,
        "image_path": image_path,
        "mask_path": mask_path,
        "query": user_query,
        "success": True,
        "contains_seg_token": "<SEG>" in generated_text,
        "num_generated_seg_tokens": num_seg_generated,
        "chosen_seg_index": chosen["seg_index"],
        "pred_mask_binary": pred_mask_path,
        "pred_mask_prob": pred_prob_path,
        "pred_overlay": overlay_path,
        "report_path": report_path,
    }
    if "metrics" in chosen:
        summary["metrics_1024"] = chosen["metrics"]
    return summary


# -----------------------------
# 模型加载：基础模型只加载一次
# 多个 checkpoint/adapter 依次加载验证
# -----------------------------
def load_base_model_once():
    print("Loading base model.")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if hasattr(tokenizer, "tokenizer"):
        inner_tokenizer = tokenizer.tokenizer
    else:
        inner_tokenizer = tokenizer

    if "<SEG>" not in inner_tokenizer.get_vocab():
        inner_tokenizer.add_tokens(["<SEG>"])

    seg_token_id = inner_tokenizer.convert_tokens_to_ids("<SEG>")
    im_end_token_id = inner_tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.resize_token_embeddings(len(inner_tokenizer))
    model.config.seg_token_id = seg_token_id
    model.config.im_end_token_id = im_end_token_id

    model = build_lisa_modules(model, projection_dim=256, sam_checkpoint=SAM_CHECKPOINT)

    target_modules_config = ["qkv", "proj", "fc1", "fc2", "merger.mlp.0", "merger.mlp.2"]
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules_config,
        modules_to_save=["embed_tokens", "lm_head", "seg_token_mask_projection"],
        use_gradient_checkpointing=False,
        random_state=3407,
    )

    # 验证阶段全部冻结
    for _, param in model.named_parameters():
        param.requires_grad = False

    model.eval()
    core_model = get_core_model(model)
    return model, inner_tokenizer, seg_token_id, im_end_token_id, core_model


def load_samples_for_eval():
    if args.input_json is not None:
        samples = load_samples_from_json(args.input_json)
        print(f"\nLoaded {len(samples)} samples from JSON: {args.input_json}")
    else:
        samples = [
            {
                "image_path": IMAGE_PATH,
                "mask_path": MASK_PATH,
                "question": USER_QUERY,
                "id": "single_demo",
            }
        ]
        print("\nNo --input-json provided, fallback to single-image mode.")
    return samples


def run_all_samples_for_one_adapter(
    model_idx,
    adapter_path,
    samples,
    model,
    core_model,
    inner_tokenizer,
    seg_token_id,
    im_end_token_id,
):
    adapter_path = str(Path(adapter_path).expanduser().resolve())
    model_name = sanitize_filename(Path(adapter_path).name)
    adapter_name = f"lisa_eval_{model_idx}_{model_name}"
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    ensure_dir(model_output_dir)

    print("\n" + "#" * 120)
    print(f"[Model {model_idx}] Start verifying: {model_name}")
    print(f"Adapter path: {adapter_path}")
    print(f"Model output dir: {model_output_dir}")

    model_summary_path = os.path.join(model_output_dir, "batch_summary.json")
    all_summaries = []
    num_success = 0
    num_failed = 0

    try:
        print(f"Loading adapter from: {adapter_path}")
        model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
        model.set_adapter(adapter_name)
        model.eval()

        # lisa_extra_modules.pt 里的 mask_decoder 会随 checkpoint 变化，所以每个模型都要重新加载一次
        load_lisa_extra_modules(core_model, adapter_path)

        core_model.seg_token_mask_projection = core_model.seg_token_mask_projection.float()
        core_model.visual_model.mask_decoder = core_model.visual_model.mask_decoder.float()
        print_key_module_devices(core_model)

        processor = load_processor(adapter_path, BASE_MODEL)

        for idx, sample in enumerate(samples, start=1):
            try:
                summary = process_one_sample(
                    sample_idx=idx,
                    sample=sample,
                    model=model,
                    core_model=core_model,
                    processor=processor,
                    inner_tokenizer=inner_tokenizer,
                    seg_token_id=seg_token_id,
                    im_end_token_id=im_end_token_id,
                    output_root=model_output_dir,
                )
                summary["model_name"] = model_name
                summary["adapter_path"] = adapter_path
                all_summaries.append(summary)
                num_success += 1
            except Exception as e:
                num_failed += 1
                err_msg = f"{type(e).__name__}: {e}"
                sample_name = make_sample_name(sample, idx) if isinstance(sample, dict) else f"{idx:04d}_sample"
                print("\n" + "!" * 100)
                print(f"[Model {model_name}] [Sample {idx}] FAILED: {sample_name}")
                print(err_msg)
                failed_summary = {
                    "model_name": model_name,
                    "adapter_path": adapter_path,
                    "sample_index": idx,
                    "sample_name": sample_name,
                    "success": False,
                    "error": err_msg,
                    "raw_sample": sample,
                }
                all_summaries.append(failed_summary)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not args.skip_failed:
                    partial_path = os.path.join(model_output_dir, "batch_summary_partial.json")
                    with open(partial_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "model_name": model_name,
                                "adapter_path": adapter_path,
                                "input_json": args.input_json,
                                "output_dir": model_output_dir,
                                "num_total": len(samples),
                                "num_success": num_success,
                                "num_failed": num_failed,
                                "results": all_summaries,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    raise

        model_summary = {
            "model_index": model_idx,
            "model_name": model_name,
            "adapter_path": adapter_path,
            "success": True,
            "input_json": args.input_json,
            "output_dir": model_output_dir,
            "num_total": len(samples),
            "num_success": num_success,
            "num_failed": num_failed,
            "results": all_summaries,
        }
        with open(model_summary_path, "w", encoding="utf-8") as f:
            json.dump(model_summary, f, ensure_ascii=False, indent=2)

        print("\n" + "#" * 100)
        print(f"Model finished: {model_name}")
        print(f"Total   : {len(samples)}")
        print(f"Success : {num_success}")
        print(f"Failed  : {num_failed}")
        print(f"Summary : {model_summary_path}")
        return model_summary

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print("\n" + "!" * 120)
        print(f"[Model {model_idx}] FAILED: {model_name}")
        print(err_msg)
        failed_model_summary = {
            "model_index": model_idx,
            "model_name": model_name,
            "adapter_path": adapter_path,
            "success": False,
            "error": err_msg,
            "input_json": args.input_json,
            "output_dir": model_output_dir,
            "num_total": len(samples),
            "num_success": num_success,
            "num_failed": num_failed,
            "results": all_summaries,
        }
        with open(model_summary_path, "w", encoding="utf-8") as f:
            json.dump(failed_model_summary, f, ensure_ascii=False, indent=2)

        if not args.skip_failed_model:
            raise
        return failed_model_summary

    finally:
        if not args.keep_loaded_adapters:
            try:
                if hasattr(model, "delete_adapter"):
                    model.delete_adapter(adapter_name)
                    print(f"Deleted adapter from memory: {adapter_name}")
                else:
                    print("[Warning] 当前 PEFT/Unsloth 模型没有 delete_adapter 方法，adapter 可能会保留在内存中。")
            except Exception as e:
                print(f"[Warning] delete_adapter failed for {adapter_name}: {type(e).__name__}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    samples = load_samples_for_eval()

    if args.adapter_root is not None and str(args.adapter_root).strip() != "":
        adapter_paths = discover_adapter_dirs(
            adapter_root=args.adapter_root,
            checkpoint_pattern=args.checkpoint_pattern,
            checkpoint_names=args.checkpoint_names,
            recursive=args.recursive,
        )
    else:
        adapter_paths = [str(Path(args.adapter_path).expanduser().resolve())]

    print("\nFound adapters/checkpoints to verify:")
    for i, p in enumerate(adapter_paths, start=1):
        print(f"  [{i}] {p}")

    model, inner_tokenizer, seg_token_id, im_end_token_id, core_model = load_base_model_once()

    all_model_summaries = []
    for model_idx, adapter_path in enumerate(adapter_paths, start=1):
        model_summary = run_all_samples_for_one_adapter(
            model_idx=model_idx,
            adapter_path=adapter_path,
            samples=samples,
            model=model,
            core_model=core_model,
            inner_tokenizer=inner_tokenizer,
            seg_token_id=seg_token_id,
            im_end_token_id=im_end_token_id,
        )
        all_model_summaries.append(model_summary)

    num_model_success = sum(1 for x in all_model_summaries if x.get("success", False))
    num_model_failed = len(all_model_summaries) - num_model_success

    all_models_summary_path = os.path.join(OUTPUT_DIR, "all_models_summary.json")
    all_models_csv_path = os.path.join(OUTPUT_DIR, "all_models_metrics.csv")

    all_models_summary = {
        "adapter_root": args.adapter_root,
        "adapter_path": args.adapter_path if args.adapter_root is None else None,
        "checkpoint_pattern": args.checkpoint_pattern,
        "checkpoint_names": args.checkpoint_names,
        "input_json": args.input_json,
        "output_dir": OUTPUT_DIR,
        "num_models": len(all_model_summaries),
        "num_model_success": num_model_success,
        "num_model_failed": num_model_failed,
        "models": all_model_summaries,
    }
    with open(all_models_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_models_summary, f, ensure_ascii=False, indent=2)

    write_all_models_csv(all_model_summaries, all_models_csv_path)

    print("\n" + "#" * 120)
    print("All model verification finished.")
    print(f"Models total   : {len(all_model_summaries)}")
    print(f"Models success : {num_model_success}")
    print(f"Models failed  : {num_model_failed}")
    print(f"JSON summary   : {all_models_summary_path}")
    print(f"CSV summary    : {all_models_csv_path}")


if __name__ == "__main__":
    main()
