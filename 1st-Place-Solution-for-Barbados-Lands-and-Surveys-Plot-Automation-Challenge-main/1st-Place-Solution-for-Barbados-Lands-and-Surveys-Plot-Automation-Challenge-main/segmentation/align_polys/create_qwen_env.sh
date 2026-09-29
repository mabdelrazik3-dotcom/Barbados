#!/usr/bin/env bash
# Bootstrap a disposable environment tailored for Qwen vLLM workflows.
set -euo pipefail

# Create an isolated env just for Qwen/vLLM
python3.11 -m venv .qvenv
source .qvenv/bin/activate

python -m pip install --upgrade pip
pip install -U uv

echo "[1/5] Install vLLM 0.11.0 (pulls matching Torch/CUDA wheels)"
uv pip install "vllm==0.11.0" \
  --extra-index-url "https://wheels.vllm.ai/0.11.0/" \
  --torch-backend=auto

echo "[2/5] Qwen VL utilities + HF basics"
uv pip install "qwen-vl-utils==0.0.14"
uv pip install "transformers>=4.46,<5" "huggingface_hub>=0.24,<1" "accelerate>=1.0,<2"

echo "[3/5] AutoGen (AgentChat + OpenAI client)"
uv pip install "autogen-agentchat==0.7.5" "autogen-ext[openai]==0.7.5"

echo "[4/5] Install other libraries (pinned where versions provided)"
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

echo "[5/5] Sanity check: print versions and import Qwen3-VL + AutoGen"
python - <<'PY'
import torch, vllm
from transformers import Qwen3VLMoeForConditionalGeneration
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", vllm.__version__)
print("OK: Qwen3VLMoeForConditionalGeneration import")
print("OK: AutoGen imports (AssistantAgent, OpenAIChatCompletionClient)")
PY

echo "Done! Environment is ready."
