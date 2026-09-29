"""
Download base VLM checkpoints from HuggingFace for Unsloth fine-tuning.

Downloads Qwen3-VL-8B and Qwen3-VL-32B models to models/ directory.
Uses HF transfer for faster downloads.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.base_utils import load_config

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")

ROOT_DIR = Path(BASE_CONFIG["root_dir"])
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

repo_ids = [
    TEXT_CONFIG.qwen3vl8b.repo_id,
    TEXT_CONFIG.qwen3vl32b.repo_id,
]

# Download both models with resumable downloads
for repo_id in repo_ids:
    local_dir = MODELS_DIR / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # keep real files in your models dir
        resume_download=True,  # resumable
        max_workers=16,  # increase if you have bandwidth/IO
    )
    print(f"downloaded: {repo_id} -> {path}")
