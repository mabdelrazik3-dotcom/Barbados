"""Use base Qwen3-VL to clean noisy training metadata labels."""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BNB_CUDA_VERSION"] = ""
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from unsloth import FastVisionModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from text_extraction.prompts import LABEL_CORRECTION_PROMPT
from utils.base_utils import load_config

BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")


ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

df = pd.read_csv(DATA_DIR / "patched_1024.csv")
train_df = (
    df[df["split"] == "train"].drop_duplicates(subset=["ID"]).reset_index(drop=True)
)

QWEN3VL8B_CHECKPOINT = MODELS_DIR / TEXT_CONFIG.qwen3vl8b.repo_id.split("/")[-1]

model, tokenizer = FastVisionModel.from_pretrained(
    str(QWEN3VL8B_CHECKPOINT),
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)
TARGET_IMG_SIZE = TEXT_CONFIG.qwen3vl8b.image_size

tokenizer.image_processor.do_resize = False
tokenizer.image_processor.do_center_crop = False
tokenizer.image_processor.do_pad = False
tokenizer.image_processor.default_to_square = True
tokenizer.image_processor.min_pixels = None
tokenizer.image_processor.max_pixels = None
tokenizer.image_processor.size = {
    "shortest_edge": TARGET_IMG_SIZE,
    "longest_edge": TARGET_IMG_SIZE,
}

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# --- Helpers ---
def _open_rgb(path):
    """Load an image as RGB while avoiding mode mismatches."""
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img


def _row_to_labels_json(row):
    """Serialize noisy labels to JSON for the correction prompt."""
    # No Parish field; these are the noisy labels passed to the model
    obj = {
        "Land Surveyor": str(row["Land Surveyor"]),
        "Surveyed For": str(row["Surveyed For"]),
        "Certified date": str(row["Certified date"]),
        "Total Area": float(row["Total Area"]),
        "Unit of Measurement": str(row["Unit of Measurement"]),
        "Address": str(row["Address"]),
        "LT Num": str(row["LT Num"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def _row_to_request(row):
    """Build a VLM request payload containing images plus noisy labels."""
    # Image order: full, bottom, bottom_left, bottom_right, top, top_left, top_right
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
                "content": [
                    {"type": "text", "text": LABEL_CORRECTION_PROMPT},
                    {"type": "text", "text": _row_to_labels_json(row)},
                    *images,
                ],
            }
        ]
    }


# --- Build requests (correction/inference) from your DataFrame ---
requests = [_row_to_request(r) for _, r in train_df.iterrows()]


FastVisionModel.for_inference(model)

FIELDS = [
    "Land Surveyor",
    "Surveyed For",
    "Certified date",
    "Total Area",
    "Unit of Measurement",
    "Address",
    "LT Num",
]


def _extract_json(txt: str):
    """Best-effort extraction of a JSON object from the model's text."""
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _to_device_dtype(batch, device, model_dtype):
    """Move tokenizer outputs to the correct device/dtype combination."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(model_dtype)
        out[k] = v
    return out


df_out = train_df.copy()
for f in FIELDS:
    df_out[f + "_corr"] = None

device = next(model.parameters()).device
float_dtypes = [p.dtype for p in model.parameters() if p.is_floating_point()]
model_dtype = float_dtypes[0] if float_dtypes else torch.float16
use_amp = device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)

BATCH_SIZE = 10  # tune to VRAM
CSV_PATH = DATA_DIR / "label_corrections.csv"
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)


for start in tqdm(range(0, len(requests), BATCH_SIZE)):
    batch = requests[start : start + BATCH_SIZE]

    texts = []
    images_batch = []
    for req in batch:
        msgs = req["messages"]
        texts.append(
            tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        )
        content = msgs[0]["content"]
        images_batch.append(
            [
                c["image"]
                for c in content
                if isinstance(c, dict) and c.get("type") == "image"
            ]
        )

    inputs = tokenizer(
        text=texts,
        images=images_batch,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = _to_device_dtype(inputs, device, model_dtype)

    with torch.no_grad():
        if use_amp:
            try:
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=384,
                        do_sample=False,
                        temperature=0.001,
                        use_cache=True,
                    )
            except TypeError:
                with torch.cuda.amp.autocast(dtype=model_dtype):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=384,
                        do_sample=False,
                        temperature=0.001,
                        use_cache=True,
                    )
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                temperature=0.001,
                use_cache=True,
            )

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lens = [inputs["input_ids"].shape[1]] * len(batch)

    for i in range(len(batch)):
        seq = out[i]
        ilen = int(input_lens[i])
        gen_ids = seq[ilen:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = _extract_json(gen_text)

        df_idx = df_out.index[start + i]
        for f in FIELDS:
            df_out.at[df_idx, f + "_corr"] = pred.get(f, None)

    # write this batch’s rows to CSV (append mode)
    written = df_out.iloc[start : start + len(batch)].copy()
    written["Total Area_corr"] = pd.to_numeric(
        written["Total Area_corr"], errors="coerce"
    )
    written.to_csv(
        CSV_PATH,
        mode="a",
        header=not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0,
        index=False,
    )
