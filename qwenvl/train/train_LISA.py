from unsloth import FastVisionModel
import os
import sys
import json
import re
import torch
import transformers
from pathlib import Path

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

try:
    from peft import set_peft_model_state_dict
except Exception:
    set_peft_model_state_dict = None

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class, DistillationTrainer
from transformers import HfArgumentParser, AutoProcessor
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments

local_rank = None

# ============================================================
# Stage-2 VQA final strategy
# ------------------------------------------------------------
# 1) Start from Qwen2.5-VL base model.
# 2) Load tokenizer/processor from the stage-1 checkpoint so image preprocessing
#    stays consistent. Stage-2 VQA does not need to emit or supervise <SEG>.
# 3) Build a NEW LoRA structure that contains both:
#    - stage-1 visual/merger targets: qkv/proj/fc1/fc2/merger.mlp.0/merger.mlp.2
#    - stage-2 language targets: q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
# 4) Partially load stage-1 adapter weights into this new LoRA structure.
#    Existing visual/merger LoRA weights are inherited.
#    Newly added language_model LoRA weights are randomly initialized and trained.
# 5) Freeze visual encoder LoRA by default to preserve anatomical priors.
# 6) Train language_model LoRA + visual.merger LoRA for VQA/SFT.
# 7) Do NOT load lisa_extra_modules.pt, SAM, or mask_decoder.
# ============================================================

STAGE1_ADAPTER_PATH = os.environ.get(
    "STAGE1_ADAPTER_PATH",
    "/mnt/data/zhangxw/LISA_mix/checkpoint-29187/",
)

# FastVisionModel.get_peft_model usually creates the active adapter named "default".
# We train this adapter after partially loading stage-1 weights into it.
ACTIVE_ADAPTER_NAME = os.environ.get("ACTIVE_ADAPTER_NAME", "default")

STAGE1_PROJECTION_DIM = int(os.environ.get("STAGE1_PROJECTION_DIM", "256"))

# Default: do not train or save embed_tokens/lm_head in stage-2.
# If you really need them, export TRAIN_EMBED_LM_HEAD=1 and set --head_lr very small, e.g. 1e-6.
TRAIN_EMBED_LM_HEAD = os.environ.get("TRAIN_EMBED_LM_HEAD", "0") == "1"

# Default: do not load stage-1 embed_tokens/lm_head into stage-2.
# Reason: stage-2 keeps base-model embedding rows and freezes these two modules by default.
# If the stage-1 tokenizer/vocab size is smaller, loading them can cause shape mismatch.
LOAD_STAGE1_EMBED_LM_HEAD = os.environ.get("LOAD_STAGE1_EMBED_LM_HEAD", "0") == "1"

# Default: do not load or save LISA's segmentation projection in stage-2 VQA.
# It is useful for mask decoding, but VQA-RAD-style SFT has no mask supervision.
LOAD_STAGE1_SEG_PROJECTION = os.environ.get("LOAD_STAGE1_SEG_PROJECTION", "0") == "1"
SAVE_STAGE2_SEG_PROJECTION = os.environ.get("SAVE_STAGE2_SEG_PROJECTION", "0") == "1"

# Default: do not add a new <SEG> token in stage-2 if the selected tokenizer lacks it.
# Stage-2 data is ordinary VQA SFT and should not depend on segmentation tokens.
USE_SEG_TOKEN_IN_STAGE2 = os.environ.get("USE_SEG_TOKEN_IN_STAGE2", "0") == "1"

# Legacy stage-2 policy from train_froze.py:
# train early visual block LoRA and freeze visual.blocks.5+.
TRAIN_VISUAL_BLOCKS_LORA = os.environ.get("TRAIN_VISUAL_BLOCKS_LORA", "1") == "1"
TRAIN_VISUAL_FIRST_N_BLOCKS = int(os.environ.get("TRAIN_VISUAL_FIRST_N_BLOCKS", "5"))

# ============================================================
# Small-data VQA stabilization options
# ------------------------------------------------------------
# Legacy stage-2 compatibility:
# train_froze.py takes LoRA target modules from --lora_target_modules and then
# freezes only visual.blocks.5+. These variables are kept only for reporting and
# for optional experiments; the default launch uses --lora_target_modules
# from ModelArguments, whose default is usually all-linear.
#
# Recommended defaults:
#   export LANGUAGE_LORA_MODE=full
#   export TRAIN_LM_LAST_N_LAYERS=64
#   export QWEN_LM_NUM_LAYERS=64
# ============================================================
TRAIN_LM_LAST_N_LAYERS = int(os.environ.get("TRAIN_LM_LAST_N_LAYERS", "64"))
QWEN_LM_NUM_LAYERS = int(os.environ.get("QWEN_LM_NUM_LAYERS", "64"))
LANGUAGE_LORA_MODE = os.environ.get("LANGUAGE_LORA_MODE", "full").strip().lower()
if LANGUAGE_LORA_MODE not in ["attn_only", "full"]:
    raise ValueError(
        f"Invalid LANGUAGE_LORA_MODE={LANGUAGE_LORA_MODE}. "
        f"Use 'attn_only' or 'full'."
    )


# ----------------------------
# Utilities
# ----------------------------
def rank0_print(*args):
    if local_rank is None or local_rank <= 0:
        print(*args)


def get_real_tokenizer(tokenizer_or_processor):
    """Compatible with AutoProcessor and a real tokenizer."""
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save PEFT adapter and modules_to_save.
    This does not save optimizer/scheduler states except through trainer.save_state().
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def print_trainable_layers(model, max_names: int = 200):
    print("\n====== Parameter Trainability Report ======")
    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append((name, tuple(param.shape)))

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    print("--- Trainable parameter names ---")
    for name, shape in trainable_names[:max_names]:
        print(f"Trainable | {name} | {shape}")
    if len(trainable_names) > max_names:
        print(f"... omitted {len(trainable_names) - max_names} trainable tensors")

    print("---")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    if total_params > 0:
        print(f"Trainable %: {(100 * trainable_params / total_params):.6f}")
    print("==========================================\n")


# ----------------------------
# Stage-1 checkpoint handling
# ----------------------------
def check_stage1_adapter_dir(stage1_dir: str):
    if not os.path.isdir(stage1_dir):
        raise FileNotFoundError(
            f"STAGE1_ADAPTER_PATH does not exist or is not a directory: {stage1_dir}\n"
            f"Please export STAGE1_ADAPTER_PATH=/path/to/checkpoint-29187 or modify this script."
        )

    adapter_config = os.path.join(stage1_dir, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"Missing required adapter_config.json: {adapter_config}")

    adapter_weight = find_adapter_weight_file(stage1_dir)
    if adapter_weight is None:
        raise FileNotFoundError(
            f"Cannot find adapter weights under {stage1_dir}. "
            f"Expected adapter_model.safetensors or adapter_model.bin."
        )

    lisa_extra = os.path.join(stage1_dir, "lisa_extra_modules.pt")
    if os.path.exists(lisa_extra):
        rank0_print(
            f"Found {lisa_extra}, but stage-2 VQA will NOT load it. "
            f"SAM / mask_decoder are intentionally excluded."
        )

    try:
        with open(adapter_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rank0_print("Stage-1 adapter target_modules:", cfg.get("target_modules", None))
        rank0_print("Stage-1 modules_to_save:", cfg.get("modules_to_save", None))
    except Exception as e:
        rank0_print(f"Warning: failed to inspect adapter_config.json: {e}")


def find_adapter_weight_file(stage1_dir: str):
    candidates = [
        os.path.join(stage1_dir, "adapter_model.safetensors"),
        os.path.join(stage1_dir, "adapter_model.bin"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_stage1_adapter_state_dict(stage1_dir: str):
    weight_file = find_adapter_weight_file(stage1_dir)
    if weight_file is None:
        raise FileNotFoundError(f"No adapter weight file found in {stage1_dir}")

    rank0_print(f"Loading stage-1 adapter tensor file: {weight_file}")
    if weight_file.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("safetensors is required to load adapter_model.safetensors")
        return safe_load_file(weight_file)

    return torch.load(weight_file, map_location="cpu")


def _prefilter_stage1_state_for_stage2(model, stage1_state: dict, adapter_name: str = "default"):
    """
    Pre-filter stage-1 adapter tensors for stage-2 VQA SFT.

    Why this is necessary:
    - We no longer shrink token embeddings when the stage-1 tokenizer is shorter than the base model.
    - Therefore stage-1 embed_tokens/lm_head tensors may have a smaller first dimension.
    - strict=False does NOT ignore shape mismatch.
    - Stage-2 freezes embed_tokens/lm_head by default, so these tensors are not needed.
    - LISA segmentation projection / mask decoder weights are not part of VQA SFT.
    """
    model_state = model.state_dict()

    filtered_state = {}
    skipped_head = []
    skipped_seg = []
    skipped_exact_shape = []

    for key, value in stage1_state.items():
        # Stage-2 freezes these by default. Skipping them avoids vocab-size mismatch.
        if ("embed_tokens" in key or "lm_head" in key) and not LOAD_STAGE1_EMBED_LM_HEAD:
            skipped_head.append(key)
            continue

        # Stage-2 VQA does not use mask decoding. Keep the anatomical prior only through
        # Qwen visual/merger LoRA tensors unless explicitly requested for experiments.
        if (
            "seg_token_mask_projection" in key
            or "visual_model" in key
            or "mask_decoder" in key
        ) and not LOAD_STAGE1_SEG_PROJECTION:
            skipped_seg.append(key)
            continue

        # If a key exactly exists in current model state, we can safely check shape here.
        # LoRA adapter keys often need PEFT name rewriting, so many keys will not exact-match;
        # those are still kept and handled by PEFT.
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            skipped_exact_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue

        filtered_state[key] = value

    rank0_print("\n" + "=" * 80)
    rank0_print("Stage-1 adapter pre-filter summary")
    rank0_print(f"Original tensors: {len(stage1_state)}")
    rank0_print(f"Kept tensors before PEFT loading: {len(filtered_state)}")
    rank0_print(f"Skipped embed_tokens/lm_head tensors: {len(skipped_head)}")
    rank0_print(f"Skipped LISA segmentation tensors: {len(skipped_seg)}")
    rank0_print(f"Skipped exact shape-mismatch tensors: {len(skipped_exact_shape)}")

    if skipped_head:
        rank0_print("First several skipped embed/lm_head keys:")
        for key in skipped_head[:10]:
            rank0_print(f"  skipped_head: {key}")
        if len(skipped_head) > 10:
            rank0_print(f"  ... omitted {len(skipped_head) - 10} keys")

    if skipped_seg:
        rank0_print("First several skipped LISA segmentation keys:")
        for key in skipped_seg[:10]:
            rank0_print(f"  skipped_seg: {key}")
        if len(skipped_seg) > 10:
            rank0_print(f"  ... omitted {len(skipped_seg) - 10} keys")

    if skipped_exact_shape:
        rank0_print("First several exact shape-mismatch keys:")
        for key, stage1_shape, model_shape in skipped_exact_shape[:10]:
            rank0_print(f"  {key}: stage1={stage1_shape}, current={model_shape}")
        if len(skipped_exact_shape) > 10:
            rank0_print(f"  ... omitted {len(skipped_exact_shape) - 10} keys")

    rank0_print("=" * 80 + "\n")
    return filtered_state


def _direct_key_candidates(key: str, adapter_name: str = "default"):
    """Generate likely current-model keys for direct fallback loading."""
    candidates = [key]

    # Raw PEFT LoRA checkpoint keys often have .lora_A.weight, while model.state_dict()
    # has .lora_A.default.weight.
    for tag in ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]:
        raw = f".{tag}.weight"
        named = f".{tag}.{adapter_name}.weight"
        if raw in key:
            candidates.append(key.replace(raw, named))

    # ModulesToSaveWrapper can use modules_to_save.weight in checkpoint and
    # modules_to_save.default.weight in current model.
    if ".modules_to_save.weight" in key:
        candidates.append(key.replace(".modules_to_save.weight", f".modules_to_save.{adapter_name}.weight"))

    # Remove duplicates while preserving order.
    out = []
    seen = set()
    for item in candidates:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _fallback_load_state_dict_shape_matched(model, filtered_state: dict, adapter_name: str = "default"):
    """
    Fallback loader used only if PEFT set_peft_model_state_dict fails.
    It maps the most common adapter key patterns and loads only exact shape matches.
    """
    model_state = model.state_dict()
    mapped_state = {}
    skipped_missing = []
    skipped_shape = []

    for key, value in filtered_state.items():
        matched_key = None
        for cand in _direct_key_candidates(key, adapter_name=adapter_name):
            if cand in model_state:
                matched_key = cand
                break

        if matched_key is None:
            skipped_missing.append(key)
            continue

        if tuple(model_state[matched_key].shape) != tuple(value.shape):
            skipped_shape.append((matched_key, tuple(value.shape), tuple(model_state[matched_key].shape)))
            continue

        mapped_state[matched_key] = value

    incompatible = model.load_state_dict(mapped_state, strict=False)

    rank0_print("\n" + "=" * 80)
    rank0_print("Fallback direct partial loading summary")
    rank0_print(f"Mapped and loaded tensors: {len(mapped_state)} / {len(filtered_state)}")
    rank0_print(f"Skipped missing keys: {len(skipped_missing)}")
    rank0_print(f"Skipped shape-mismatch keys: {len(skipped_shape)}")
    rank0_print(f"load_state_dict missing keys: {len(incompatible.missing_keys)}")
    rank0_print(f"load_state_dict unexpected keys: {len(incompatible.unexpected_keys)}")

    if skipped_shape:
        rank0_print("First several fallback shape-mismatch keys:")
        for key, stage1_shape, model_shape in skipped_shape[:10]:
            rank0_print(f"  {key}: stage1={stage1_shape}, current={model_shape}")

    rank0_print("=" * 80 + "\n")


def load_stage1_weights_partially(model, stage1_dir: str, adapter_name: str = "default"):
    """
    Load stage-1 adapter weights into the current PEFT adapter.

    Important fix:
    - Do NOT call PEFT's set_peft_model_state_dict here.
    - In several PEFT versions, set_peft_model_state_dict still expects
      modules_to_save keys such as embed_tokens/lm_head to exist in the incoming
      state_dict. Since stage-2 deliberately skips embed_tokens/lm_head, PEFT may
      raise KeyError, e.g.:
        KeyError: base_model.model.model.language_model.embed_tokens.weight

    Therefore, we use the exact shape-matched fallback loader directly:
    - compatible stage-1 visual/merger LoRA tensors are loaded;
    - stage-1 embed_tokens/lm_head tensors are skipped by default;
    - stage-1 seg_token_mask_projection / mask_decoder tensors are skipped by default;
    - new stage-2 language_model LoRA tensors remain randomly initialized;
    - lisa_extra_modules.pt / SAM / mask_decoder are never loaded in stage-2.
    """
    stage1_state = load_stage1_adapter_state_dict(stage1_dir)
    filtered_state = _prefilter_stage1_state_for_stage2(
        model=model,
        stage1_state=stage1_state,
        adapter_name=adapter_name,
    )

    rank0_print(
        "Using exact shape-matched direct partial loading for stage-1 adapter "
        f"into active adapter: {adapter_name}"
    )
    _fallback_load_state_dict_shape_matched(
        model,
        filtered_state,
        adapter_name=adapter_name,
    )

# ----------------------------
# Tokenizer and compatibility modules
# ----------------------------
def ensure_seg_token_exists(model, tokenizer_or_processor):
    """
    Stable stage-2 tokenizer handling.

    Stage-2 does not train segmentation. If the stage-1 tokenizer already contains
    <SEG>, we keep it for compatibility. If it does not, we do not add it by
    default because VQA-RAD-style SFT never supervises <SEG>.

    The important rule here is: expand embeddings only when explicitly needed,
    never shrink them.

    Why:
    - The base Qwen2.5-VL model may have embedding rows such as 152064.
    - The saved stage-1 tokenizer may have a smaller length such as 151666.
    - Shrinking embeddings can discard base-model vocabulary rows and is unnecessary because
      embed_tokens/lm_head are frozen by default in stage-2.
    """
    inner_tokenizer = get_real_tokenizer(tokenizer_or_processor)

    vocab = inner_tokenizer.get_vocab()
    seg_added = False

    if "<SEG>" not in vocab:
        if not USE_SEG_TOKEN_IN_STAGE2:
            rank0_print(
                "<SEG> is absent from the tokenizer and USE_SEG_TOKEN_IN_STAGE2=0; "
                "stage-2 VQA will continue without adding a segmentation token."
            )
            model.config.seg_token_id = None
            try:
                model.config.im_end_token_id = inner_tokenizer.convert_tokens_to_ids("<|im_end|>")
            except Exception:
                model.config.im_end_token_id = None
            return tokenizer_or_processor, inner_tokenizer

        inner_tokenizer.add_tokens(["<SEG>"])
        seg_added = True
        rank0_print("Added <SEG> token to tokenizer.")

    seg_token_id = inner_tokenizer.convert_tokens_to_ids("<SEG>")
    if seg_token_id is None or seg_token_id < 0:
        raise ValueError("Failed to obtain a valid <SEG> token id from tokenizer.")

    cur_embed_n = model.get_input_embeddings().weight.shape[0]
    tok_n = len(inner_tokenizer)

    if tok_n > cur_embed_n:
        rank0_print(
            f"Tokenizer size {tok_n} > embedding size {cur_embed_n}; expanding token embeddings."
        )
        try:
            model.resize_token_embeddings(tok_n, pad_to_multiple_of=128)
        except TypeError:
            model.resize_token_embeddings(tok_n)
        new_embed_n = model.get_input_embeddings().weight.shape[0]
        rank0_print(f"Expanded token embeddings from {cur_embed_n} to {new_embed_n}.")
    else:
        rank0_print(
            f"Tokenizer size {tok_n} <= embedding size {cur_embed_n}; "
            f"skip resize to avoid shrinking embeddings."
        )

    final_embed_n = model.get_input_embeddings().weight.shape[0]
    if seg_token_id >= final_embed_n:
        raise ValueError(
            f"<SEG> token id {seg_token_id} is outside embedding size {final_embed_n}. "
            f"Tokenizer/model vocab are incompatible."
        )

    model.config.seg_token_id = seg_token_id
    try:
        im_end_token_id = inner_tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_token_id = None
    model.config.im_end_token_id = im_end_token_id

    # Only initialize <SEG> embedding if we added it in this run.
    # If it already exists in the stage-1 tokenizer, do not overwrite anything.
    if seg_added:
        with torch.no_grad():
            ref_token_id = inner_tokenizer.convert_tokens_to_ids("mask")
            if ref_token_id is None or ref_token_id == getattr(inner_tokenizer, "unk_token_id", None):
                ref_token_id = inner_tokenizer.convert_tokens_to_ids(".")

            if ref_token_id is not None and 0 <= ref_token_id < final_embed_n:
                input_embeds = model.get_input_embeddings().weight
                input_embeds[seg_token_id] = input_embeds[ref_token_id].clone()

                output_embeds = model.get_output_embeddings()
                if output_embeds is not None and seg_token_id < output_embeds.weight.shape[0]:
                    output_embeds.weight[seg_token_id] = output_embeds.weight[ref_token_id].clone()

                rank0_print(f"Initialized <SEG> embedding from token id {ref_token_id}.")
            else:
                rank0_print("Warning: could not find a valid reference token to initialize <SEG>.")

    rank0_print(
        f"<SEG> token id: {seg_token_id} | tokenizer length: {tok_n} | "
        f"embedding rows: {model.get_input_embeddings().weight.shape[0]}"
    )

    return tokenizer_or_processor, inner_tokenizer

def attach_stage1_placeholder_modules(model, projection_dim=256):
    """
    Stage-1 adapter_config contains seg_token_mask_projection in modules_to_save.
    Stage-2 does not use it, but the module must exist so adapter weights can be loaded.
    It is frozen later and never participates in the VQA loss.
    """
    hidden_size = model.config.hidden_size
    dtype = model.get_input_embeddings().weight.dtype
    device = model.get_input_embeddings().weight.device

    if not hasattr(model, "seg_token_mask_projection"):
        model.seg_token_mask_projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, projection_dim),
            torch.nn.LayerNorm(projection_dim),
        ).to(device=device, dtype=dtype)
        rank0_print("Attached placeholder seg_token_mask_projection for stage-1 adapter compatibility.")

    return model


# ----------------------------
# LoRA target modules
# ----------------------------
def get_stage2_target_modules(model_args=None):
    """
    Match train_froze.py: LoRA targets come from --lora_target_modules.

    For the current LISA stage-1 checkpoint, "all-linear" is the safest legacy
    setting because it creates both the old VQA LoRA modules and the stage-1
    visual/merger LoRA modules that can be partially inherited.
    """
    if model_args is None:
        if LANGUAGE_LORA_MODE == "full":
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "qkv",
                "proj",
                "fc1",
                "fc2",
                "merger.mlp.0",
                "merger.mlp.2",
            ]
        return ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj", "fc1", "fc2"]

    lora_target_modules_arg = model_args.lora_target_modules
    if lora_target_modules_arg.lower() == "all-linear":
        return "all-linear"
    return [item.strip() for item in lora_target_modules_arg.split(",") if item.strip()]


def get_stage2_modules_to_save():
    """
    Only save non-LoRA modules that are actually trained in stage-2.

    Keeping embed_tokens/lm_head in modules_to_save by default makes the adapter
    very large and can let a small VQA dataset disturb the base language head.
    The segmentation projection is also excluded by default because stage-2 has
    no mask loss.
    """
    modules_to_save = []
    if TRAIN_EMBED_LM_HEAD:
        modules_to_save.extend(["embed_tokens", "lm_head"])
    if SAVE_STAGE2_SEG_PROJECTION:
        modules_to_save.append("seg_token_mask_projection")
    return modules_to_save



def is_lora_param(name: str) -> bool:
    return "lora_" in name


def is_visual_param(name: str) -> bool:
    # Qwen2.5-VL visual tower is usually under .visual.; SAM visual_model is not loaded in stage-2.
    return ".visual." in name or "model.visual." in name or "visual_model" in name


def is_visual_merger_lora(name: str) -> bool:
    return is_lora_param(name) and "merger" in name


def is_visual_block_lora(name: str) -> bool:
    return is_lora_param(name) and is_visual_param(name) and "merger" not in name


def is_language_lora(name: str) -> bool:
    # Anything LoRA that is not the visual tower/merger and not segmentation/head is treated as language-side LoRA.
    if not is_lora_param(name):
        return False
    if is_visual_param(name):
        return False
    if "seg_token_mask_projection" in name:
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    return True


def get_language_layer_index(name: str):
    """
    Extract language_model layer index from a parameter name.

    Example:
      base_model.model.model.language_model.layers.52.self_attn.q_proj.lora_A.default.weight
      -> 52
    """
    match = re.search(r"language_model\.layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def get_visual_block_index(name: str):
    """
    Extract Qwen visual block index from a parameter name.

    Example:
      base_model.model.model.visual.blocks.3.attn.qkv.lora_A.default.weight
      -> 3
    """
    match = re.search(r"visual\.blocks\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def should_train_language_lora_layer(name: str) -> bool:
    """
    Train the selected language_model LoRA layers.

    With the legacy stage-2 default TRAIN_LM_LAST_N_LAYERS=64, this trains all
    Qwen2.5-VL-32B language layers, matching train_froze.py's "All LLM LoRA
    parameters remain trainable" policy. If the layer index cannot be parsed,
    the parameter is kept trainable to avoid accidentally freezing non-layer
    language LoRA.
    """
    layer_idx = get_language_layer_index(name)
    if layer_idx is None:
        return True

    start_layer = max(0, QWEN_LM_NUM_LAYERS - TRAIN_LM_LAST_N_LAYERS)
    return layer_idx >= start_layer


def should_train_visual_block_lora(name: str, train_visual_blocks_lora: bool) -> bool:
    """
    Match the old train_froze.py visual policy:
    train visual.blocks.0-(N-1), freeze visual.blocks.N+.
    """
    if not train_visual_blocks_lora:
        return False

    block_idx = get_visual_block_index(name)
    if block_idx is None:
        # If a visual LoRA is not under visual.blocks and is not merger, keep it
        # trainable to match the old script's broad LoRA behavior.
        return True

    return block_idx < TRAIN_VISUAL_FIRST_N_BLOCKS


def is_head_or_embedding_module(name: str) -> bool:
    return "embed_tokens" in name or "lm_head" in name


def freeze_for_stage2_vqa(model, train_embed_lm_head: bool = False, train_visual_blocks_lora: bool = False):
    """
    Match train_froze.py's custom freezing as closely as possible:
      - keep PEFT's default trainable LoRA set;
      - freeze visual.blocks.N+ LoRA;
      - keep visual.blocks.0-(N-1) LoRA trainable;
      - keep all LLM LoRA trainable.

    The only current-stage adaptation is that LISA segmentation modules are not
    loaded for VQA SFT, and full embed/lm_head modules stay frozen unless the
    user explicitly enables TRAIN_EMBED_LM_HEAD.
    """
    stats = {
        "train_language_lora": 0,
        "train_merger_lora": 0,
        "train_visual_block_lora": 0,
        "train_head": 0,
        "frozen": 0,
    }
    trainable_names = []
    total_params_frozen_visual = 0
    total_params_kept_visual = 0

    for name, param in model.named_parameters():
        if "seg_token_mask_projection" in name:
            param.requires_grad = False
            continue

        if is_head_or_embedding_module(name) and not is_lora_param(name):
            param.requires_grad = bool(train_embed_lm_head)
            continue

        if "visual.blocks." in name and "lora_" in name:
            try:
                block_num_str = name.split("visual.blocks.")[1].split(".")[0]
                block_num = int(block_num_str)
                if train_visual_blocks_lora and block_num < TRAIN_VISUAL_FIRST_N_BLOCKS:
                    param.requires_grad = True
                    total_params_kept_visual += param.numel()
                else:
                    param.requires_grad = False
                    total_params_frozen_visual += param.numel()
            except Exception as e:
                rank0_print(f"Warning: Could not parse block number from '{name}'. Error: {e}")
            continue

        if ("model.layers." in name or "language_model.layers." in name) and "lora_" in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            if is_head_or_embedding_module(name) and not is_lora_param(name):
                category = "train_head"
            elif is_visual_merger_lora(name):
                category = "train_merger_lora"
            elif is_visual_block_lora(name):
                category = "train_visual_block_lora"
            elif is_lora_param(name):
                category = "train_language_lora"
            else:
                category = "train_language_lora"
            stats[category] += param.numel()
            trainable_names.append(name)
        else:
            stats["frozen"] += param.numel()


    trainable_total = (
        stats["train_language_lora"]
        + stats["train_merger_lora"]
        + stats["train_visual_block_lora"]
        + stats["train_head"]
    )

    rank0_print("\n" + "=" * 78)
    rank0_print("FINAL STAGE-2 VQA FINETUNE POLICY")
    rank0_print(f"Stage-1 adapter path: {STAGE1_ADAPTER_PATH}")
    rank0_print(f"Active adapter name: {ACTIVE_ADAPTER_NAME}")
    rank0_print("SAM / mask_decoder / lisa_extra_modules.pt are NOT loaded.")
    rank0_print("Policy copied from train_froze.py custom freezing.")
    rank0_print(f"Freeze visual.blocks.{TRAIN_VISUAL_FIRST_N_BLOCKS}+ LoRA.")
    rank0_print(f"Keep visual.blocks.0-{TRAIN_VISUAL_FIRST_N_BLOCKS - 1} LoRA trainable.")
    rank0_print("All LLM LoRA parameters remain trainable.")
    rank0_print(f"LANGUAGE_LORA_MODE: {LANGUAGE_LORA_MODE}")
    rank0_print(f"TRAIN_LM_LAST_N_LAYERS: {TRAIN_LM_LAST_N_LAYERS}")
    rank0_print(f"QWEN_LM_NUM_LAYERS: {QWEN_LM_NUM_LAYERS}")
    rank0_print(f"TRAIN_EMBED_LM_HEAD: {train_embed_lm_head}")
    rank0_print(f"LOAD_STAGE1_EMBED_LM_HEAD: {LOAD_STAGE1_EMBED_LM_HEAD}")
    rank0_print(f"LOAD_STAGE1_SEG_PROJECTION: {LOAD_STAGE1_SEG_PROJECTION}")
    rank0_print(f"SAVE_STAGE2_SEG_PROJECTION: {SAVE_STAGE2_SEG_PROJECTION}")
    rank0_print(f"TRAIN_VISUAL_BLOCKS_LORA: {train_visual_blocks_lora}")
    rank0_print(f"TRAIN_VISUAL_FIRST_N_BLOCKS: {TRAIN_VISUAL_FIRST_N_BLOCKS}")
    rank0_print(f"Froze visual block LoRA params: {total_params_frozen_visual}")
    rank0_print(f"Kept visual block LoRA params trainable: {total_params_kept_visual}")
    rank0_print(f"Train language LoRA params: {stats['train_language_lora']}")
    rank0_print(f"Train merger LoRA params: {stats['train_merger_lora']}")
    rank0_print(f"Train visual block LoRA params: {stats['train_visual_block_lora']}")
    rank0_print(f"Train embed/lm_head params: {stats['train_head']}")
    rank0_print(f"Frozen params: {stats['frozen']}")
    rank0_print(f"Total trainable params: {trainable_total}")
    rank0_print("=" * 78 + "\n")

    if stats["train_language_lora"] == 0:
        rank0_print(
            "WARNING: train_language_lora is 0. This means language_model LoRA was not created.\n"
            "Check whether Qwen language modules use target names q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj."
        )

    return trainable_names, stats


def validate_stage2_resume_checkpoint(training_args):
    """
    Stage-1 transfer and Trainer resume are intentionally separate.

    Use STAGE1_ADAPTER_PATH for stage-1 initialization. Use
    --resume_from_checkpoint only for continuing an interrupted stage-2 VQA run.
    """
    resume = getattr(training_args, "resume_from_checkpoint", None)
    if not isinstance(resume, str) or not resume.strip():
        return

    resume_path = os.path.abspath(os.path.normpath(resume))
    stage1_path = os.path.abspath(os.path.normpath(STAGE1_ADAPTER_PATH))
    resume_has_lisa_extra = os.path.exists(os.path.join(resume_path, "lisa_extra_modules.pt"))

    if resume_path == stage1_path or resume_has_lisa_extra:
        raise ValueError(
            "Do not pass a LISA stage-1 checkpoint to --resume_from_checkpoint.\n"
            "For stage-2 VQA, set STAGE1_ADAPTER_PATH=/path/to/stage1/checkpoint "
            "and leave --resume_from_checkpoint empty unless you are resuming an "
            "interrupted stage-2 run."
        )


# ----------------------------
# Train
# ----------------------------
def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank if training_args.local_rank is not None else -1
    os.makedirs(training_args.output_dir, exist_ok=True)

    validate_stage2_resume_checkpoint(training_args)
    check_stage1_adapter_dir(STAGE1_ADAPTER_PATH)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    load_in_4bit = True

    rank0_print("Loading base Qwen2.5-VL model for final stage-2 VQA finetuning...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        load_in_4bit=load_in_4bit,
        torch_dtype=dtype,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    # Match train_froze.py: force gradient checkpointing before building LoRA.
    training_args.gradient_checkpointing = True

    tokenizer, inner_tokenizer = ensure_seg_token_exists(model, tokenizer)

    if LOAD_STAGE1_SEG_PROJECTION or SAVE_STAGE2_SEG_PROJECTION:
        model = attach_stage1_placeholder_modules(model, projection_dim=STAGE1_PROJECTION_DIM)
    else:
        rank0_print("Stage-2 VQA will not attach seg_token_mask_projection.")

    target_modules_config = get_stage2_target_modules(model_args)
    modules_to_save_config = get_stage2_modules_to_save()
    rank0_print("Building final stage-2 PEFT wrapper.")
    rank0_print(f"Stage-2 target_modules: {target_modules_config}")
    rank0_print(f"Stage-2 modules_to_save: {modules_to_save_config or 'None'}")

    peft_kwargs = dict(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=target_modules_config,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        random_state=3407,
    )
    if modules_to_save_config:
        peft_kwargs["modules_to_save"] = modules_to_save_config

    model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    # Make sure the adapter name is active. Most PEFT/Unsloth models use "default" here.
    try:
        model.set_adapter(ACTIVE_ADAPTER_NAME)
    except Exception as e:
        rank0_print(f"Warning: could not set adapter {ACTIVE_ADAPTER_NAME}: {e}")
        rank0_print("Continuing with the currently active adapter.")

    # Partial inheritance from stage-1.
    load_stage1_weights_partially(model, STAGE1_ADAPTER_PATH, adapter_name=ACTIVE_ADAPTER_NAME)

    # Apply final stage-2 train/freeze policy.
    trainable_names, train_stats = freeze_for_stage2_vqa(
        model,
        train_embed_lm_head=TRAIN_EMBED_LM_HEAD,
        train_visual_blocks_lora=TRAIN_VISUAL_BLOCKS_LORA,
    )

    try:
        data_args.image_processor = model.processor.image_processor
    except AttributeError:
        rank0_print("Warning: Could not find model.processor.image_processor. Falling back to AutoProcessor.")
        data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()

    if training_args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            if hasattr(model, "get_input_embeddings"):
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    rank0_print("=== Trainable Parameters (Final Stage2 VQA) ===")
    print_trainable_layers(model)

    if data_args.data_packing:
        rank0_print(
            "WARNING: data_packing=True. Make sure data_qwen_packed is compatible "
            "with Qwen2.5-VL and your VQA JSON format."
        )
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print("\n" + "=" * 80)
    print(f"[Qwen-VL] Using datasets: {data_args.dataset_use}")
    try:
        train_ds = data_module["train_dataset"]
        print(f"[Qwen-VL] Train dataset loaded with {len(train_ds)} samples.")
        log_path = os.path.join(training_args.output_dir, "dataset_info.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Using datasets: {data_args.dataset_use}\n")
            f.write(f"Train dataset size: {len(train_ds)}\n")
            f.write(f"Stage1 adapter init: {STAGE1_ADAPTER_PATH}\n")
            f.write(f"Active adapter name: {ACTIVE_ADAPTER_NAME}\n")
            f.write(f"Stage2 target_modules: {target_modules_config}\n")
            f.write(f"Stage2 modules_to_save: {modules_to_save_config or 'None'}\n")
            f.write(f"LANGUAGE_LORA_MODE: {LANGUAGE_LORA_MODE}\n")
            f.write(f"TRAIN_LM_LAST_N_LAYERS: {TRAIN_LM_LAST_N_LAYERS}\n")
            f.write(f"QWEN_LM_NUM_LAYERS: {QWEN_LM_NUM_LAYERS}\n")
            f.write(f"TRAIN_EMBED_LM_HEAD: {TRAIN_EMBED_LM_HEAD}\n")
            f.write(f"LOAD_STAGE1_EMBED_LM_HEAD: {LOAD_STAGE1_EMBED_LM_HEAD}\n")
            f.write(f"LOAD_STAGE1_SEG_PROJECTION: {LOAD_STAGE1_SEG_PROJECTION}\n")
            f.write(f"SAVE_STAGE2_SEG_PROJECTION: {SAVE_STAGE2_SEG_PROJECTION}\n")
            f.write(f"TRAIN_VISUAL_BLOCKS_LORA: {TRAIN_VISUAL_BLOCKS_LORA}\n")
            f.write(f"TRAIN_VISUAL_FIRST_N_BLOCKS: {TRAIN_VISUAL_FIRST_N_BLOCKS}\n")
            f.write(f"Train stats: {train_stats}\n")
            f.write("Trainable names preview:\n")
            for n in trainable_names[:200]:
                f.write(n + "\n")
        print(f"[Qwen-VL] Dataset info has been saved to: {log_path}")
    except Exception as e:
        print(f"[Qwen-VL] Failed to retrieve dataset info: {e}")
    print("=" * 80 + "\n")

    trainer = DistillationTrainer(
        model=model,
        teacher_model=None,
        alpha=0.0,
        temperature=2.0,
        distillation_mode="vision",
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_state()

    try:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save tokenizer/processor. Error: {e}")

    try:
        data_args.image_processor.save_pretrained(training_args.output_dir)
    except Exception as e:
        rank0_print(f"Warning: Could not save image_processor. Error: {e}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
