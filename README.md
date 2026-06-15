# Neuro-VLM

Official implementation of **Neuro-VLM**, an anatomy-aware vision-language model for brain imaging visual question answering.

This repository contains training, evaluation, and demo code built around Qwen2.5-VL and LoRA fine-tuning.

## Highlights

- Brain imaging visual question answering with Qwen2.5-VL backbones.
- Supervised fine-tuning support for image-text and video-text conversations.
- LoRA-based training with configurable rank, alpha, dropout, and target modules.
- Batch evaluation scripts and a Gradio web demo.

## Repository Structure

```text
Neuro-VLM/
+-- qwenvl/
|   +-- data/                  # Dataset loading and Qwen-VL preprocessing
|   +-- train/                 # Training, trainer, and LoRA code
+-- train_py/
|   +-- val.py                 # Validation / checkpoint inference
|   +-- val_base.py            # Base-model validation
|   +-- web_demo_mm.py         # Gradio demo
+-- README.md
```

## Environment

Create a Python environment and install the core dependencies:

```bash
pip install torch torchvision
pip install transformers accelerate peft pillow tqdm gradio qwen-vl-utils
```

Optional dependencies may be needed depending on the training script you use:

```bash
pip install deepspeed flash-attn unsloth safetensors
```

Install optional packages according to your CUDA, PyTorch, and GPU environment.

## Data Preparation

Dataset entries are configured in:

```text
qwenvl/data/__init__.py
```

Update the `annotation_path` and `data_path` fields to match your local dataset paths.

The annotation files are expected to follow a conversation-style VQA format. A typical item looks like:

```json
{
  "image": "/path/to/image.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat abnormality is visible?"
    },
    {
      "from": "gpt",
      "value": "glioma"
    }
  ]
}
```

The data loader also supports JSONL files and video tokens in the Qwen-VL preprocessing path.

## Training

A standard Qwen2.5-VL fine-tuning entry point is:

```bash
python qwenvl/train/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset_use my_dataset \
  --output_dir outputs/neuro-vlm \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16 True \
  --gradient_checkpointing True
```

Additional training variants are available in `qwenvl/train/`. For example:

```bash
python qwenvl/train/train_LISA_two.py \
  --model_name_or_path /path/to/base/model \
  --dataset_use my_dataset \
  --output_dir outputs/neuro-vlm-lisa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16 True
```

Adjust paths, batch sizes, precision settings, and distributed training options for your hardware.

## Evaluation

Batch validation scripts are provided in `train_py/`:

```bash
python train_py/val_base.py
python train_py/val.py
```

Before running evaluation, update the model, checkpoint, dataset, and output paths inside the scripts to match your environment.

## Demo

Launch the Gradio demo with:

```bash
python train_py/web_demo_mm.py \
  --checkpoint-path /path/to/checkpoint \
  --server-name 127.0.0.1 \
  --server-port 7860
```

Add `--share` if you want Gradio to create a public share link.

## Notes

- Large model checkpoints and datasets are not included in this repository.
- Several scripts contain local absolute paths; replace them with paths on your machine before running.
- Some workflows assume CUDA GPUs and may require significant GPU memory.
- If using offline model or dataset caches, set the relevant Hugging Face environment variables before running.

## Acknowledgements

This project builds on Qwen2.5-VL, Hugging Face Transformers, PEFT/LoRA, Unsloth, and Gradio.
