"""Fine-tune Qwen3-VL on corrected + pseudo labels and save the LoRA adapter."""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BNB_CUDA_VERSION"] = ""
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

# Import Unsloth first
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

1  # - To ensure transformers imported after unsloth
import ast
import json
import re

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer

from text_extraction.prompts import LABEL_CORRECTION_PROMPT, TRAINING_PROMPT
from utils.base_utils import load_config

sys.path.append(str(Path(__file__).resolve().parent.parent))


BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")


ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

pseudo_df = pd.read_csv(DATA_DIR / "pseudo_df.csv")
corr_df = pd.read_csv(DATA_DIR / "label_corrections.csv")
corr_df = corr_df.drop(
    columns=[
        "Surveyed For",
        "Certified date",
        "Total Area",
        "Unit of Measurement",
        "LT Num",
    ]
)
corr_df = corr_df.rename(
    columns={
        "Land Surveyor_corr": "Land Surveyor",
        "Land Surveyor": "Land Surveyor2",
        "Surveyed For_corr": "Surveyed For",
        "Certified date_corr": "Certified date",
        "Total Area_corr": "Total Area",
        "Unit of Measurement_corr": "Unit of Measurement",
        "Address_corr": "Address",
        "Address": "Address2",
        "LT Num_corr": "LT Num",
    }
)

aug_df = pd.concat([corr_df, pseudo_df], ignore_index=True)

test_df = aug_df[aug_df.split == "test"].reset_index(drop=True)

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
    """Load images as RGB so each tile is prompt-friendly."""
    with Image.open(path) as img:
        return img.convert("RGB")


def _row_to_answer_json(row):
    """Serialize labels into the multi-line assistant response format."""
    # Return a plain multi-line string (your previous braces created a set -> not JSON-serializable)
    return (
        f"Land Surveyor: {row['Land Surveyor']}\n"
        f"Land Surveyor2: {row['Land Surveyor2']}\n"
        f"Surveyed For: {row['Surveyed For']}\n"
        f"Certified date: {row['Certified date']}\n"
        f"Total Area: {row['Total Area']}\n"
        f"Unit of Measurement: {row['Unit of Measurement']}\n"
        f"Address: {row['Address']}\n"
        f"Address2: {row['Address2']}\n"
        f"Parish: {row['Parish']}\n"
        f"LT Num: {row['LT Num']}"
    )


def _row_to_conversation(row):
    """Build a chat example with user prompt + assistant answer + all tiles."""
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
train_dataset = [_row_to_conversation(r) for _, r in aug_df.iterrows()]


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

# Save LoRA + tokenizer
ADAPTER_DIR = Path("outputs") / "qwen3vl8b_finetuned_lora"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print("Training complete and model saved to ", ADAPTER_DIR)
