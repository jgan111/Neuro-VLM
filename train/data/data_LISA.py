import os
import copy
import json
import random
import time
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, List
from collections.abc import Sequence as ABCSequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import transformers
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _ensure_real_tokenizer(tokenizer_or_processor):
    """
    兼容传入 tokenizer 或 processor。
    如果传进来的是 Qwen2_5_VLProcessor / AutoProcessor，则取其 .tokenizer。
    """
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def _normalize_tokenizer_output_ids(encoded):
    """
    统一 tokenizer(...)[\"input_ids\"] 的返回格式：
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


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = None,
    grid_thw_video: List = None,
    seg_sample_flags: List = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    if grid_thw_image is None:
        grid_thw_image = []
    if grid_thw_video is None:
        grid_thw_video = []

    # 兼容误传 processor 的情况
    tokenizer = _ensure_real_tokenizer(tokenizer)
    tokenizer = copy.deepcopy(tokenizer)

    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    if seg_sample_flags is None:
        seg_sample_flags = [False] * len(sources)

    # 保留，后续若你想单独用 <SEG> 监督可继续扩展
    try:
        seg_token_id = tokenizer.convert_tokens_to_ids("<SEG>")
    except Exception:
        seg_token_id = None

    for src_idx, source in enumerate(sources):
        is_seg_sample = bool(seg_sample_flags[src_idx])

        if len(source) == 0:
            raise ValueError(f"Empty conversation at source index {src_idx}")

        if roles.get(source[0].get("from", source[0].get("role")), source[0].get("from", source[0].get("role"))) != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        sys_text = f"<|im_start|>system\n{system_message}<|im_end|>\n"
        sys_encode = _normalize_tokenizer_output_ids(
            tokenizer(text=sys_text, add_special_tokens=False)
        )

        input_id += sys_encode
        target += [IGNORE_INDEX] * len(sys_encode)

        for conv in source:
            role = conv.get("role", conv.get("from"))
            content = conv.get("content", conv.get("value"))

            if content is None:
                content = ""

            role = roles.get(role, role)

            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    num_image_tokens_needed = len(parts) - 1

                    if len(grid_thw_image) < visual_replicate_index_image + num_image_tokens_needed:
                        raise ValueError(
                            f"Not enough image grid tokens: need {visual_replicate_index_image + num_image_tokens_needed}, "
                            f"but got {len(grid_thw_image)}"
                        )

                    for k in range(len(parts) - 1):
                        new_parts.append(parts[k])
                        replacement = (
                            "<|vision_start|>"
                            + "<|image_pad|>" * int(grid_thw_image[visual_replicate_index_image])
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    num_video_tokens_needed = len(parts) - 1

                    if len(grid_thw_video) < visual_replicate_index_video + num_video_tokens_needed:
                        raise ValueError(
                            f"Not enough video grid tokens: need {visual_replicate_index_video + num_video_tokens_needed}, "
                            f"but got {len(grid_thw_video)}"
                        )

                    for k in range(len(parts) - 1):
                        new_parts.append(parts[k])
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>" * int(grid_thw_video[visual_replicate_index_video])
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            text_string = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            encode_id = _normalize_tokenizer_output_ids(
                tokenizer(text=text_string, add_special_tokens=False)
            )
            input_id += encode_id

            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = [IGNORE_INDEX] * len(encode_id)

                # 监督 assistant 的完整回复内容
                # 前 3 个 token: "<|im_start|>", "assistant", "\n" 不参与 loss
                prefix_ignore_len = min(3, len(encode_id))
                target_mask[:prefix_ignore_len] = [IGNORE_INDEX] * prefix_ignore_len
                for idx in range(prefix_ignore_len, len(encode_id)):
                    target_mask[idx] = encode_id[idx]

                target += target_mask

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
            annotations = (
                read_jsonl(data["annotation_path"])
                if file_format == "jsonl"
                else json.load(open(data["annotation_path"], "r"))
            )

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))

            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")
        random.shuffle(list_data_dict)

        # 兼容上游传 processor 的情况
        self.tokenizer = _ensure_real_tokenizer(tokenizer)
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        visual_processed = processor.preprocess(image, return_tensors="pt")

        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]

        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        try:
            return self.video_decord(video_file)
        except Exception:
            return self.video_torchcodec(video_file)

    def video_decord(self, video_file):
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps

        interval = getattr(self.data_args, "base_interval", 4)
        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)

        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        decoder = VideoDecoder(video_file, device="cpu")
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps

        interval = getattr(self.data_args, "base_interval", 4)
        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)

        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length

        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)

        return video_tensor, grid_thw, second_per_grid_ts

    def _load_mask(self, mask_path):
        gt_mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if gt_mask_color is None:
            return torch.zeros((1, 1024, 1024), dtype=torch.float32)

        b_channel = gt_mask_color[:, :, 0].astype(np.int32)
        r_channel = gt_mask_color[:, :, 2].astype(np.int32)
        diff_rb = r_channel - b_channel
        binary_mask = (diff_rb > 20).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        gt_mask = (clean_mask > 0).astype(np.float32)
        gt_mask = cv2.resize(gt_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(gt_mask).unsqueeze(0).float()

    def _load_sam_image(self, img_path):
        try:
            raw_image = Image.open(img_path).convert("RGB")
            w, h = raw_image.size
            scale = 1024 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

            img_resized = raw_image.resize((new_w, new_h))
            img_tensor = to_tensor(img_resized) * 255.0

            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            return F.pad(img_tensor, (0, 1024 - new_w, 0, 1024 - new_h))
        except Exception as e:
            print(f"Error loading SAM image {img_path}: {e}")
            return torch.zeros((3, 1024, 1024), dtype=torch.float32)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        last_exception = None
        for _ in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                last_exception = e
                print(f"Failed to fetch sample {i}: {e}")
                time.sleep(1)

        raise RuntimeError(f"Failed to fetch sample {i} after 3 retries") from last_exception

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        sources = [source]

        grid_thw = None
        grid_thw_merged = None
        video_grid_thw = None
        video_grid_thw_merged = None
        second_per_grid_ts = None

        if "image" in source:
            image_folder = source["data_path"]
            image_file = source["image"]

            if isinstance(image_file, list):
                image_files = [os.path.join(image_folder, f) for f in image_file]
            else:
                image_files = [os.path.join(image_folder, image_file)]

            results = [self.process_image_unified(f) for f in image_files]
            image, grid_thw = zip(*results)
            grid_thw_merged = [
                int(thw.prod().item() if isinstance(thw.prod(), torch.Tensor) else thw.prod())
                // self.data_args.image_processor.merge_size ** 2
                for thw in grid_thw
            ]
        else:
            image = None

        if "video" in source:
            video_folder = source["data_path"]
            video_file = source["video"]

            if isinstance(video_file, list):
                video_files = [os.path.join(video_folder, f) for f in video_file]
            else:
                video_files = [os.path.join(video_folder, video_file)]

            results = [self.process_video(f) for f in video_files]
            video, video_grid_thw, second_per_grid_ts = zip(*results)
            video_grid_thw_merged = [
                int(thw.prod().item() if isinstance(thw.prod(), torch.Tensor) else thw.prod())
                // self.data_args.image_processor.merge_size ** 2
                for thw in video_grid_thw
            ]
        else:
            video = None

        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        has_seg = "mask" in source

        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged is not None else [],
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged is not None else [],
            seg_sample_flags=[has_seg],
        )

        data_dict["seg_sample_mask"] = torch.tensor(has_seg, dtype=torch.bool)

        if has_seg:
            mask_path = os.path.join(source.get("data_path", ""), source["mask"])
            data_dict["masks"] = self._load_mask(mask_path)
        else:
            data_dict["masks"] = torch.zeros((1, 1024, 1024), dtype=torch.float32)

        if "image" in source:
            image_file_for_sam = source["image"][0] if isinstance(source["image"], list) else source["image"]
            img_path = os.path.join(source.get("data_path", ""), image_file_for_sam)
            data_dict["sam_images"] = self._load_sam_image(img_path)

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(list(grid_thw), dim=0) if grid_thw is not None else None,
            video_grid_thw=torch.stack(list(video_grid_thw), dim=0) if video_grid_thw is not None else None,
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts is not None else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if image is not None:
            data_dict["pixel_values"] = torch.cat(list(image), dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0)
        elif video is not None:
            data_dict["pixel_values_videos"] = torch.cat(list(video), dim=0)
            data_dict["video_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in video_grid_thw], dim=0)

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
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

        max_len = getattr(real_tokenizer, "model_max_length", 4096)
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
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

        batch["masks"] = torch.stack([instance["masks"] for instance in instances], dim=0)
        batch["seg_sample_mask"] = torch.stack([instance["seg_sample_mask"] for instance in instances], dim=0)

        if any("sam_images" in instance for instance in instances):
            batch["sam_images"] = torch.stack(
                [
                    instance.get(
                        "sam_images",
                        torch.zeros(3, 1024, 1024, dtype=torch.float32),
                    )
                    for instance in instances
                ],
                dim=0,
            )
        else:
            batch["sam_images"] = None

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
            itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance))
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        batch = {
            "input_ids": torch.cat(input_ids, dim=1),
            "labels": torch.cat(labels, dim=1),
            "position_ids": torch.cat(position_ids, dim=2),
            "attention_mask": cumsum_seq_lens,
            "masks": torch.stack([instance["masks"] for instance in instances], dim=0),
            "seg_sample_mask": torch.stack([instance["seg_sample_mask"] for instance in instances], dim=0),
        }

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance]

        batch["pixel_values"] = torch.cat(images, dim=0) if images else None
        batch["image_grid_thw"] = (
            torch.cat([instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance], dim=0)
            if images else None
        )
        batch["pixel_values_videos"] = torch.cat(videos, dim=0) if videos else None
        batch["video_grid_thw"] = (
            torch.cat([instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance], dim=0)
            if videos else None
        )
        batch["sam_images"] = torch.stack(
            [
                instance.get("sam_images", torch.zeros(3, 1024, 1024, dtype=torch.float32))
                for instance in instances
            ],
            dim=0,
        )
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        return {
            "train_dataset": train_dataset,
            "eval_dataset": None,
            "data_collator": FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer),
        }
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    }


if __name__ == "__main__":
    pass