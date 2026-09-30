#!/usr/bin/env bash
# Create the Barbados-Brainiac environment (CUDA 12.4 wheels, like the 1st-place setup).
#   bash install.sh            # core
#   WITH_UNSLOTH=1 bash install.sh   # + Unsloth/TRL for the 1st-place training preset
#   WITH_VLLM=1 bash install.sh      # + vLLM backend
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip uv

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

if [[ "${WITH_UNSLOTH:-0}" == "1" ]]; then
  export UNSLOTH_DISABLE_FAST_GENERATION=1
  uv pip install unsloth==2025.10.3 unsloth_zoo==2025.10.3
  uv pip install --no-deps trl==0.22.2
fi
if [[ "${WITH_VLLM:-0}" == "1" ]]; then
  uv pip install vllm
fi

echo "Environment ready: source $SCRIPT_DIR/.venv/bin/activate"
