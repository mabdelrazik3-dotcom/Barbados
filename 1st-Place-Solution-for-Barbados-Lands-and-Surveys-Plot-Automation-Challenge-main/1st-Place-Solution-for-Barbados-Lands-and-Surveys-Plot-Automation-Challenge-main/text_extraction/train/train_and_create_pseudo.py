"""Fine-tune Qwen3-VL on corrected labels and generate pseudo labels for test data."""

import os
import sys
from pathlib import Path

os.environ["BNB_CUDA_VERSION"] = ""
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

# Import Unsloth first


from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

1  # - To ensure transformers imported after unsloth
from trl import SFTConfig, SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ast
import json
import re

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from text_extraction.prompts import LABEL_CORRECTION_PROMPT, TRAINING_PROMPT
from utils.base_utils import load_config

BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")


ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

patched_df = pd.read_csv(DATA_DIR / "patched_1024.csv")
corr_df = pd.read_csv(DATA_DIR / "label_corrections.csv")
patched_df = patched_df.merge(
    corr_df[
        [
            "ID",
            "Land Surveyor_corr",
            "Surveyed For_corr",
            "Certified date_corr",
            "Total Area_corr",
            "Unit of Measurement_corr",
            "Address_corr",
            "LT Num_corr",
        ]
    ],
    on="ID",
    how="left",
)

train_df = patched_df[patched_df.split == "train"].reset_index(drop=True)
test_df = patched_df[patched_df.split == "test"].reset_index(drop=True)

QWEN3VL8B_CHECKPOINT = MODELS_DIR / TEXT_CONFIG.qwen3vl8b.repo_id.split("/")[-1]


model, tokenizer = FastVisionModel.from_pretrained(
    str(QWEN3VL8B_CHECKPOINT),
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0.35,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# --- Helpers ---
def _open_rgb(path):
    """Load an image path as RGB for prompt construction."""
    with Image.open(path) as img:
        return img.convert("RGB")


def _row_to_answer_json(row):
    """Format corrected labels into the multi-line answer expected during training."""
    # Return a plain multi-line string (your previous braces created a set -> not JSON-serializable)
    return (
        f"Land Surveyor: {row['Land Surveyor_corr']}\n"
        f"Land Surveyor2: {row['Land Surveyor']}\n"
        f"Surveyed For: {row['Surveyed For_corr']}\n"
        f"Certified date: {row['Certified date_corr']}\n"
        f"Total Area: {row['Total Area_corr']}\n"
        f"Unit of Measurement: {row['Unit of Measurement_corr']}\n"
        f"Address: {row['Address_corr']}\n"
        f"Address2: {row['Address']}\n"
        f"Parish: {row['Parish']}\n"
        f"LT Num: {row['LT Num_corr']}"
    )


def _row_to_conversation(row):
    """Create Unsloth chat conversation with user prompt and gold answer."""
    images = [
        {"type": "image", "image": _open_rgb(row["full_path"])},
        {"type": "image", "image": _open_rgb(row["bottom_path"])},
        {"type": "image", "image": _open_rgb(row["bl_path"])},
        {"type": "image", "image": _open_rgb(row["br_path"])},
        {"type": "image", "image": _open_rgb(row["top_path"])},
        {"type": "image", "image": _open_rgb(row["tl_path"])},
        {"type": "image", "image": _open_rgb(row["tr_path"])},
    ]
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": TRAINING_PROMPT}, *images],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": _row_to_answer_json(row)}],
            },
        ]
    }


# --- Build Unsloth-style datasets from your DataFrame ---
train_dataset = [_row_to_conversation(r) for _, r in train_df.iterrows()]


FastVisionModel.for_training(model)  # Enable for training!

MAX_LENGTH = 11200  # Must be large enough to hold all images + text
tokenizer.truncation_side = "left"  # keep the tail (images) if truncated
tokenizer.model_max_length = MAX_LENGTH


trainer = SFTTrainer(
    model=model,
    data_collator=UnslothVisionDataCollator(
        model, tokenizer, max_seq_length=MAX_LENGTH, resize=512
    ),  # Must use!
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        max_steps=80,  # 80
        seed=2025,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        neftune_noise_alpha=5,
        packing=False,
    ),
)

trainer_stats = trainer.train()


FastVisionModel.for_inference(model)
model.eval()

device = next(model.parameters()).device
float_dtypes = [p.dtype for p in model.parameters() if p.is_floating_point()]
model_dtype = float_dtypes[0] if float_dtypes else torch.float16

JSON_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object. "
    "No prose, no markdown, no code fences. "
    "Quote all string values (e.g., '7A' must be \"7A\")."
)


def _extract_first_json_object(s: str) -> str | None:
    """Return the first balanced JSON substring from generated text."""
    s = s.strip()
    # Strip code fences
    s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    # Find first balanced {...} ignoring braces in strings
    start = None
    depth = 0
    in_str = False
    esc = False
    quote = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s[start : i + 1]
    # Fallback: return whole string if it at least looks like a dict
    if "{" in s and "}" in s:
        i, j = s.find("{"), s.rfind("}")
        if i < j:
            return s[i : j + 1]
    return None


def _repair_jsonish(s: str) -> str:
    """Normalize slightly-invalid JSON text so it can be parsed."""
    # Normalize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove trailing commas in objects/arrays
    s = re.sub(r",\s*(?=[}\]])", "", s)
    # Convert True/False/None to JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # Quote unquoted keys: {key: ...} or , key: ...
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*)(\s*):", r'\1"\2"\3:', s)
    # Ensure keys don't accidentally swallow a quote-less colon scenario like "http://"
    return s


def _parse_json(text: str) -> dict:
    """Parse model output into a JSON dict, falling back to key:value parsing."""
    cand = _extract_first_json_object(text) or text.strip()
    # Try strict JSON
    try:
        return json.loads(cand)
    except Exception:
        pass
    # Try python literal
    try:
        val = ast.literal_eval(cand)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    # Try repaired JSON-ish
    repaired = _repair_jsonish(cand)
    try:
        return json.loads(repaired)
    except Exception:
        pass
    # Last resort: parse "Key: Value" lines
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "*")):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip().rstrip(",")
            if k:
                data[k] = v
    return data


pred_rows = []
all_keys = []

for i, row in tqdm(
    enumerate(test_df.itertuples(index=True), start=0), total=len(test_df)
):
    try:
        images = [
            _open_rgb(getattr(row, "full_path")),
            _open_rgb(getattr(row, "bottom_path")),
            _open_rgb(getattr(row, "bl_path")),
            _open_rgb(getattr(row, "br_path")),
            _open_rgb(getattr(row, "top_path")),
            _open_rgb(getattr(row, "tl_path")),
            _open_rgb(getattr(row, "tr_path")),
        ]

        messages = [
            {
                "role": "user",
                "content": (
                    [
                        {
                            "type": "text",
                            "text": TRAINING_PROMPT + "\n\n" + JSON_INSTRUCTION,
                        }
                    ]
                    + [{"type": "image"} for _ in range(7)]
                ),
            }
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                v = v.to(device)
                if torch.is_floating_point(v):
                    v = v.to(model_dtype)
                inputs[k] = v

        with torch.inference_mode():
            if device.type == "cuda":
                try:
                    with torch.autocast(device_type="cuda", dtype=model_dtype):
                        out = model.generate(
                            **inputs,
                            max_new_tokens=384,
                            do_sample=False,
                            use_cache=True,
                        )
                except TypeError:
                    with torch.cuda.amp.autocast(dtype=model_dtype):
                        out = model.generate(
                            **inputs,
                            max_new_tokens=384,
                            do_sample=False,
                            use_cache=True,
                        )
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    use_cache=True,
                )

        gen_only = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        pred = _parse_json(text)

        # Ensure dict
        if not isinstance(pred, dict):
            pred = {}

        pred_rows.append(pred)
        for k in pred.keys():
            if k not in all_keys:
                all_keys.append(k)

    except Exception:
        pred_rows.append({})  # keep alignment

# Build pred_df with union of all keys, aligned to test_df.index
pred_df = pd.DataFrame.from_records(pred_rows, index=test_df.index)
pred_df = pred_df.reindex(columns=all_keys)
pred_df["ID"] = test_df["ID"].values
pred_df = pred_df.merge(
    patched_df[["ID"] + [col for col in patched_df.columns if col.endswith("_path")]],
    how="left",
    on="ID",
)
print(pred_df.head())
pred_df.to_csv(DATA_DIR / "pseudo_df.csv", index=False)
