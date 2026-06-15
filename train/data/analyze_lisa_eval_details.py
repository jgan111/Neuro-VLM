#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总在线验证保存的逐样本结果，找出：
1) 哪些标签/脑区长期低分；
2) 哪些病例/样本长期失败；
3) 每个 checkpoint 的整体走势。

示例：
python analyze_lisa_eval_details.py \
  --details-dir /home/zhangxw/share_data/LISA_zhao_hong/lisa_eval_details \
  --output-dir /home/zhangxw/share_data/LISA_zhao_hong/eval_analysis \
  --low-iou-threshold 0.1
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def read_jsonl_files(details_dir):
    files = sorted(glob.glob(os.path.join(details_dir, "*_eval_details.jsonl")))
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r["details_file"] = fp
                rows.append(r)
    return rows, files


def aggregate(rows, key_fields, low_iou_threshold):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in key_fields)
        groups[key].append(r)

    out = []
    for key, items in groups.items():
        ious = [safe_float(x.get("iou")) for x in items]
        dices = [safe_float(x.get("dice")) for x in items]
        pred_area_ratios = [safe_float(x.get("pred_area_ratio")) for x in items]
        gt_area_ratios = [safe_float(x.get("gt_area_ratio")) for x in items]
        low_count = sum(v < low_iou_threshold for v in ious)
        checkpoints = sorted(set(str(x.get("checkpoint_name", "")) for x in items))

        row = {k: key[i] for i, k in enumerate(key_fields)}
        first = items[0]
        # 补充常用字段，便于人工定位。
        for extra in [
            "label_id", "label_name", "label_key", "patient_id", "case_key",
            "image_path", "mask_path", "image_name", "mask_name", "view", "modality", "slice_index",
        ]:
            if extra not in row:
                row[extra] = first.get(extra)

        row.update({
            "num_records": len(items),
            "num_checkpoints": len(checkpoints),
            "low_iou_count": low_count,
            "low_iou_rate": low_count / max(len(items), 1),
            "mean_iou": sum(ious) / max(len(ious), 1),
            "min_iou": min(ious) if ious else 0.0,
            "max_iou": max(ious) if ious else 0.0,
            "mean_dice": sum(dices) / max(len(dices), 1),
            "min_dice": min(dices) if dices else 0.0,
            "max_dice": max(dices) if dices else 0.0,
            "mean_pred_area_ratio": sum(pred_area_ratios) / max(len(pred_area_ratios), 1),
            "mean_gt_area_ratio": sum(gt_area_ratios) / max(len(gt_area_ratios), 1),
            "checkpoints": ";".join(checkpoints),
        })
        out.append(row)

    out.sort(key=lambda x: (-safe_float(x.get("low_iou_rate")), safe_float(x.get("mean_iou"))))
    return out


def checkpoint_trend(rows, low_iou_threshold):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get("checkpoint_name", ""))].append(r)

    out = []
    for ckpt, items in groups.items():
        ious = [safe_float(x.get("iou")) for x in items]
        dices = [safe_float(x.get("dice")) for x in items]
        low_count = sum(v < low_iou_threshold for v in ious)
        step_vals = [safe_float(x.get("global_step"), -1) for x in items]
        epoch_vals = [safe_float(x.get("epoch"), -1) for x in items]
        out.append({
            "checkpoint_name": ckpt,
            "global_step": max(step_vals) if step_vals else None,
            "epoch": max(epoch_vals) if epoch_vals else None,
            "num_samples": len(items),
            "low_iou_count": low_count,
            "low_iou_rate": low_count / max(len(items), 1),
            "mean_iou": sum(ious) / max(len(ious), 1),
            "mean_dice": sum(dices) / max(len(dices), 1),
        })
    out.sort(key=lambda x: safe_float(x.get("global_step")))
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-dir", required=True, help="lisa_eval_details 目录")
    parser.add_argument("--output-dir", required=True, help="分析结果输出目录")
    parser.add_argument("--low-iou-threshold", type=float, default=0.1, help="低分样本阈值")
    args = parser.parse_args()

    rows, files = read_jsonl_files(args.details_dir)
    if not rows:
        raise RuntimeError(f"No eval detail rows found in: {args.details_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    label_rows = aggregate(rows, ["label_key"], args.low_iou_threshold)
    case_rows = aggregate(rows, ["case_key"], args.low_iou_threshold)
    patient_rows = aggregate(rows, ["patient_id"], args.low_iou_threshold)
    trend_rows = checkpoint_trend(rows, args.low_iou_threshold)

    # 严格长期失败：在 80% 及以上 checkpoint 中 IoU 低于阈值。
    persistent_label_rows = [r for r in label_rows if safe_float(r.get("low_iou_rate")) >= 0.8]
    persistent_case_rows = [r for r in case_rows if safe_float(r.get("low_iou_rate")) >= 0.8]

    write_csv(os.path.join(args.output_dir, "label_low_score_summary.csv"), label_rows)
    write_csv(os.path.join(args.output_dir, "case_low_score_summary.csv"), case_rows)
    write_csv(os.path.join(args.output_dir, "patient_low_score_summary.csv"), patient_rows)
    write_csv(os.path.join(args.output_dir, "checkpoint_trend.csv"), trend_rows)
    write_csv(os.path.join(args.output_dir, "persistent_bad_labels.csv"), persistent_label_rows)
    write_csv(os.path.join(args.output_dir, "persistent_bad_cases.csv"), persistent_case_rows)

    print(f"Loaded detail files: {len(files)}")
    print(f"Loaded sample records: {len(rows)}")
    print(f"Output dir: {args.output_dir}")
    print("Generated:")
    print("  label_low_score_summary.csv")
    print("  case_low_score_summary.csv")
    print("  patient_low_score_summary.csv")
    print("  checkpoint_trend.csv")
    print("  persistent_bad_labels.csv")
    print("  persistent_bad_cases.csv")


if __name__ == "__main__":
    main()
