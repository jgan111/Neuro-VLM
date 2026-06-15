# Neuro-VLM

Official implementation of **Neuro-VLM**, an anatomy-aware vision-language model for brain imaging visual question answering.

This repository provides example data, two-stage training scripts, validation scripts, and core data/model utilities for adapting Qwen2.5-VL to brain imaging VQA.

## Overview

Neuro-VLM is designed for brain imaging question answering. The current codebase includes:

- Stage-one training for anatomy-aware visual representation learning.
- Stage-two VQA fine-tuning with LoRA adapters.
- Example annotations and images for both training stages.
- Validation scripts for base models and fine-tuned checkpoints.
- Qwen2.5-VL data preprocessing, trainer utilities, and model argument definitions.

## Repository Structure

```text
Neuro-VLM/
+-- dataset/
|   +-- Stage_one/
|   +-- Stage_two/
+-- train/
|   +-- data/
|   +-- train/
|   |   +-- Stage_one.py
|   |   +-- Stage_two.py
|   +-- train.sh/
|       +-- Stage_one.sh
|       +-- Stage_two.sh
+-- value_py/
|   +-- val.py
|   +-- val_base.py
+-- README.md
```

## Installation

Create a Python environment and install the main dependencies:

```bash
pip install torch torchvision
pip install transformers accelerate peft pillow tqdm safetensors
pip install unsloth qwen-vl-utils
```

Optional packages such as `deepspeed` and `flash-attn` can be installed according to your CUDA, PyTorch, and GPU environment.

## Data Format

The repository includes small examples under `dataset/Stage_one/` and `dataset/Stage_two/`.

Stage-one examples use image-mask paired annotations with a conversation field:

```json
{
  "image": "/path/to/image.png",
  "mask": "/path/to/mask.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIdentify the target anatomical region."
    },
    {
      "from": "gpt",
      "value": "<SEG>"
    }
  ]
}
```

Stage-two examples use standard VQA annotations:

```json
{
  "image": "/path/to/image.png",
  "conversations": [
    {
      "from": "human",
      "value": "what lobe of the brain is the lesion located in?\n<image>"
    },
    {
      "from": "gpt",
      "value": "right frontal lobe"
    }
  ]
}
```

Dataset aliases and local annotation paths are configured in:

```text
train/data/__init__.py
```

Update the `annotation_path` and `data_path` values before training on your own data.

## Training

The launch scripts are provided in:

```text
train/train.sh/Stage_one.sh
train/train.sh/Stage_two.sh
```

Edit the model path, dataset alias, output directory, batch size, and GPU settings in these scripts before running.

Run stage one:

```bash
bash train/train.sh/Stage_one.sh
```

Run stage two:

```bash
bash train/train.sh/Stage_two.sh
```

The main Python entry points are:

```text
train/train/Stage_one.py
train/train/Stage_two.py
```

Training arguments such as `model_name_or_path`, `dataset_use`, `lora_r`, `lora_alpha`, `lora_dropout`, `model_max_length`, and optimizer settings are defined through Hugging Face argument parsing in:

```text
train/train/argument.py
```

## Validation

Validation scripts are stored in:

```text
value_py/val_base.py
value_py/val.py
```

Use `val_base.py` to evaluate a base Qwen2.5-VL model and `val.py` to evaluate a fine-tuned adapter/checkpoint. Before running either script, update the model path, checkpoint path, input JSON path, and output directory inside the file.

Example:

```bash
python value_py/val_base.py
python value_py/val.py
```

## Notes

- Large model checkpoints and full datasets are not included.
- Several scripts contain local absolute paths and should be edited for your machine.
- The code is intended for GPU execution; memory requirements depend on the selected Qwen2.5-VL backbone and batch size.
- Some scripts set Hugging Face offline environment variables. Disable or adjust those settings if you want to download models or processors automatically.

## Acknowledgements

This project builds on Qwen2.5-VL, Hugging Face Transformers, PEFT/LoRA, Unsloth, and related open-source tooling.
