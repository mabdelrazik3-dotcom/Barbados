"""Infer survey metadata with the fine-tuned Qwen3-VL Unsloth adapter."""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BNB_CUDA_VERSION"] = ""
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import ast
import json
import re

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
# Import Unsloth first
from unsloth import FastVisionModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from text_extraction.prompts import TRAINING_PROMPT
from text_extraction.train.patchify_images import patchify_and_resize
from utils.base_utils import load_config

BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")


ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
FINETUNED_MODEL_ADAPTER_DIR = Path("outputs") / "qwen3vl8b_finetuned_lora"

TEST_IMAGES_DIR = BASE_CONFIG["test_images_dir"]
TEST_CSV_PATH = BASE_CONFIG["test_csv_path"]

TEST_PATCHED_IMAGES_DIR = DATA_DIR / "test_patched_images"
TEST_PATCHED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

images_df = pd.Series(list(Path(TEST_IMAGES_DIR).glob("*"))).to_frame(name="image_path")
images_df["ID"] = images_df["image_path"].apply(lambda x: x.stem.split("_")[-1])
test_df = pd.read_csv(TEST_CSV_PATH)
test_df = test_df.merge(images_df, on="ID", how="left")


test_patched_df = patchify_and_resize(
    df=test_df,
    images_outp_dir=TEST_PATCHED_IMAGES_DIR,
    csv_path=DATA_DIR / "test_patched_1024.csv",
    image_size=1024,
    image_path_col="image_path",
)
print(test_patched_df.head())


# --- Helpers ---
def _open_rgb(path):
    """Load an image path as an RGB PIL image."""
    with Image.open(path) as img:
        return img.convert("RGB")


p = Path(FINETUNED_MODEL_ADAPTER_DIR) / "adapter_config.json"
cfg = json.loads(p.read_text())
cfg["base_model_name_or_path"] = str(
    (Path(ROOT_DIR) / "models" / "Qwen3-VL-8B-Instruct-unsloth-bnb-4bit").resolve()
)
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(cfg, indent=2) + "\n")
t.replace(p)


model, tokenizer = FastVisionModel.from_pretrained(
    model_name=str(FINETUNED_MODEL_ADAPTER_DIR),  # points to LoRA dir
    load_in_4bit=True,
)


# Enable inference
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
    """Return the first balanced JSON object substring if present."""
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
    """Apply lightweight repairs so mildly-broken JSON can be parsed."""
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
    """Parse model output into a dict, tolerating both JSON and JSON-ish replies."""
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
    enumerate(test_patched_df.itertuples(index=True), start=0),
    total=len(test_patched_df),
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
pred_df = pd.DataFrame.from_records(pred_rows, index=test_patched_df.index)
pred_df = pred_df.reindex(columns=all_keys)
pred_df["ID"] = test_patched_df["ID"].values
pred_df.to_csv(DATA_DIR / "text_predictions_df.csv", index=False)
print(pred_df.head())
