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

# Prevent Unsloth from touching vLLM/fast-gen paths on import
export UNSLOTH_DISABLE_FAST_GENERATION=1

echo "Installing PyTorch (CUDA 12.4, pinned)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "Core training libs..."
uv pip install pytorch-lightning==2.5.2 fastai==2.8.2 timm==1.0.19

# *** Fix 1: add matching Zoo; Fix 2: pin TRL to a version Unsloth 2025.10.3 expects ***
uv pip install unsloth==2025.10.3 unsloth_zoo==2025.10.3
uv pip install transformers==4.57.0
uv pip install --no-deps trl==0.22.2
uv pip install huggingface_hub==0.35.3

# *** Fix 3: align vLLM with Torch 2.5.1 (or upgrade Torch to 2.8 if you want vLLM==0.11) ***
uv pip install vllm==0.10.0 --torch-backend=auto

echo "Installing libraries (pinned where versions provided)..."
uv pip install autogen-agentchat==0.7.5 autogen-ext[openai]==0.7.5
uv pip install segmentation-models-pytorch==0.5.0
uv pip install geopandas==1.1.1
uv pip install scikit-learn==1.7.1
uv pip install scikit-image==0.25.2
uv pip install python-box==7.3.2
uv pip install seaborn==0.13.2
uv pip install albumentations==2.0.8
uv pip install opencv-python==4.12.0.88
uv pip install ipywidgets==8.1.7
uv pip install ipykernel==6.30.0
uv pip install pyarrow==21.0.0

echo "Done! Environment is ready."
