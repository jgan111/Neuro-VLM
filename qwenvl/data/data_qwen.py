import os
import copy
import json
import random
import time
import itertools
from dataclasses import dataclass
from typing import Dict, List
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Keep training prompt consistent with validation prompt.
# Validation currently appends: "Answer with one short phrase only."
# Enabling the same instruction during SFT often improves short-answer VQA metrics.
FORCE_SHORT_ANSWER_PROMPT = os.environ.get("FORCE_SHORT_ANSWER_PROMPT", "1") == "1"
SHORT_ANSWER_PROMPT = os.environ.get(
    "SHORT_ANSWER_PROMPT",
    "Answer with one short phrase only.",
)

local_rank = None


def rank0_print(*args):
    if local_rank is None or local_rank <= 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _ensure_real_tokenizer(tokenizer_or_processor):
    """兼容传入 Qwen2.5-VL AutoProcessor 或真实 tokenizer。"""
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def _normalize_tokenizer_output_ids(encoded):
    """
    统一 tokenizer(...)["input_ids"] 的返回格式：
    - list[int]
    - torch.Tensor
    - list[list[int]]
    都规整为 list[int]
    """
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()
    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


def _encode_text(real_tokenizer, text: str) -> List[int]:
    """兼容 encode(...) 与 tokenizer(text=...) 两种接口。"""
    if hasattr(real_tokenizer, "encode"):
        return real_tokenizer.encode(text, add_special_tokens=False)
    return _normalize_tokenizer_output_ids(real_tokenizer(text=text, add_special_tokens=False))


def _to_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = None,
    grid_thw_video: List = None,
) -> Dict:
    """
    二阶段 VQA/SFT 文本监督：
    - user / system 部分 label = IGNORE_INDEX
    - assistant 内容参与 CE loss
    - 自动把 <image> 替换成 Qwen2.5-VL 需要的 vision token 序列
    """
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    if grid_thw_image is None:
        grid_thw_image = []
    if grid_thw_video is None:
        grid_thw_video = []

    real_tokenizer = _ensure_real_tokenizer(tokenizer)
    real_tokenizer = copy.deepcopy(real_tokenizer)

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for src_idx, source in enumerate(sources):
        if len(source) == 0:
            raise ValueError(f"Empty conversation at source index {src_idx}")

        first_role = source[0].get("from", source[0].get("role"))
        if roles.get(first_role, first_role) != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        system_text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
        system_tokens = _encode_text(real_tokenizer, system_text)
        input_id += system_tokens
        target += [IGNORE_INDEX] * len(system_tokens)

        for conv_idx, conv in enumerate(source):
            role = conv.get("role", conv.get("from"))
            content = conv.get("content", conv.get("value"))
            if content is None:
                content = ""
            role = roles.get(role, role)

            if role == "user":
                if "<image>" in content:
                    num_image_tokens = content.count("<image>")
                    if visual_replicate_index_image + num_image_tokens > len(grid_thw_image):
                        raise ValueError(
                            f"Not enough image grid tokens at source {src_idx}, conv {conv_idx}: "
                            f"need {visual_replicate_index_image + num_image_tokens}, got {len(grid_thw_image)}"
                        )

                    parts = content.split("<image>")
                    new_parts = []
                    for k in range(len(parts) - 1):
                        new_parts.append(parts[k])
                        pad_len = _to_int(grid_thw_image[visual_replicate_index_image])
                        replacement = (
                            "<|vision_start|>"
                            + "<|image_pad|>" * pad_len
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    num_video_tokens = content.count("<video>")
                    if visual_replicate_index_video + num_video_tokens > len(grid_thw_video):
                        raise ValueError(
                            f"Not enough video grid tokens at source {src_idx}, conv {conv_idx}: "
                            f"need {visual_replicate_index_video + num_video_tokens}, got {len(grid_thw_video)}"
                        )

                    parts = content.split("<video>")
                    new_parts = []
                    for k in range(len(parts) - 1):
                        new_parts.append(parts[k])
                        pad_len = _to_int(grid_thw_video[visual_replicate_index_video])
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>" * pad_len
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            if role == "user" and FORCE_SHORT_ANSWER_PROMPT:
                if SHORT_ANSWER_PROMPT not in content:
                    content = content.rstrip() + "\n" + SHORT_ANSWER_PROMPT

            text_string = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            encode_id = _encode_text(real_tokenizer, text_string)
            input_id += encode_id

            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                # 监督 assistant 的完整回复内容，但不监督 role/header 部分。
                target_mask = [IGNORE_INDEX] * len(encode_id)
                prefix_ignore_len = min(3, len(encode_id))
                for idx in range(prefix_ignore_len, len(encode_id)):
                    target_mask[idx] = encode_id[idx]
                target += target_mask

        if len(input_id) != len(target):
            raise AssertionError(f"input/target length mismatch: {len(input_id)} != {len(target)}")

        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return {"input_ids": input_ids, "labels": targets}


class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()
        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")

        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_25 if data_args.model_type == "qwen2.5vl" else get_rope_index_2

        list_data_dict = []
        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            annotations = read_jsonl(data["annotation_path"]) if file_format == "jsonl" else json.load(open(data["annotation_path"], "r", encoding="utf-8"))

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                rank0_print(f"Sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")
        random.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified(self, image_file):
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"Invalid image path: {image_file}")

        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        visual_processed = processor.preprocess(image, return_tensors="pt")

        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]

        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        last_exception = None
        for attempt_idx in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                last_exception = e
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception: {e}")
                time.sleep(1)

        raise RuntimeError(f"Failed to fetch sample {i} after 3 retries") from last_exception

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]

        if "image" in sample:
            image_files = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
        elif "images" in sample:
            image_files = sample["images"] if isinstance(sample["images"], list) else [sample["images"]]
        else:
            raise ValueError(f"Sample {i} has no image/images field")

        images = []
        grid_thw_list = []
        for img_file in image_files:
            img_path = os.path.join(sample["data_path"], img_file)
            img_tensor, grid_thw = self.process_image_unified(img_path)
            images.append(img_tensor)
            grid_thw_list.append(grid_thw)

        if "conversations" not in sample:
            raise ValueError(f"Sample {i} has no conversations field")

        conversations = [sample["conversations"]]
        grid_thw_merged = [
            int(g.prod().item() if isinstance(g.prod(), torch.Tensor) else g.prod())
            // self.data_args.image_processor.merge_size ** 2
            for g in grid_thw_list
        ]

        data_dict = preprocess_qwen_2_visual(
            conversations,
            self.tokenizer,
            grid_thw_image=grid_thw_merged,
            grid_thw_video=[],
        )

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw_list, dim=0),
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"] = torch.cat(images, dim=0)
        data_dict["image_grid_thw"] = torch.cat([g.unsqueeze(0) for g in grid_thw_list], dim=0)

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        real_tokenizer = _ensure_real_tokenizer(self.tokenizer)
        pad_token_id = getattr(real_tokenizer, "pad_token_id", 151643)
        max_len = getattr(real_tokenizer, "model_max_length", 4096)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        # position_ids shape: [3, batch, seq_len]，必须截断最后一个维度。
        position_ids = position_ids[:, :, :max_len]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
            "position_ids": position_ids,
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]

        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(instance["attention_mask"] for instance in instances if "attention_mask" in instance)
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": cumsum_seq_lens,
            "position_ids": position_ids,
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]

        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """
    Unsloth 训练默认使用 2D attention_mask，因此这里继续使用普通 DataCollator。
    如果你要使用 data_flatten / packing，请确认对应 packed data module 与 Qwen2.5-VL 完全兼容。
    """
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
