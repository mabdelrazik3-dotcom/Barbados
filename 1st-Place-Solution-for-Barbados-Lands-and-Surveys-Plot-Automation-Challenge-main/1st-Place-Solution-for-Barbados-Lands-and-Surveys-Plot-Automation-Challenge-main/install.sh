#!/bin/bash
# Bootstrap a reproducible training/inference environment with pinned GPU stacks.
set -euo pipefail

# 1) Create base virtualenv that all downstream scripts expect.
python3.11 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip and installing uv..."
pip install --upgrade pip==25.2
pip install uv==0.8.4

# Optional: clear bitsandbytes override warning you saw
export BNB_CUDA_VERSION=

# Prevent Unsloth from touching fast-gen paths on import
export UNSLOTH_DISABLE_FAST_GENERATION=1

echo "Installing PyTorch (CUDA 12.4, pinned)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# *** Fix 1: add matching Zoo; Fix 2: pin TRL to a version Unsloth 2025.10.3 expects ***
uv pip install unsloth==2025.10.3 unsloth_zoo==2025.10.3
uv pip install transformers==4.57.0
uv pip install --no-deps trl==0.22.2
uv pip install huggingface_hub==0.35.3

echo "Installing libraries (pinned where versions provided)..."
uv pip install scikit-learn==1.7.1
uv pip install python-box==7.3.2
uv pip install ipywidgets==8.1.7
uv pip install ipykernel==6.30.0

echo "Done! Environment is ready."
