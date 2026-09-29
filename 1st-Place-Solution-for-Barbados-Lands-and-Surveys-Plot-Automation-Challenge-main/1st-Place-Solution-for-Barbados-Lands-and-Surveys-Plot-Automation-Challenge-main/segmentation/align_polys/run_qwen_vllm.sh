#!/usr/bin/env bash
set -euo pipefail

# Detect project root
if ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  ROOT_DIR="$SCRIPT_DIR"
fi

MODEL_PATH="$ROOT_DIR/models/Qwen3-VL-30B-A3B-Thinking"
SERVED_NAME="qwen3-vl-30b-a3b-thinking"
HOST="0.0.0.0"
PORT="8000"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Tear down any stale server before relaunching.
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 20000 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --chat-template-content-format openai \
  --allowed-local-media-path "$ROOT_DIR/data/survey_plans" \
  --disable-log-stats
  # Optional:
  # --reasoning-parser qwen3

echo "[INFO] ROOT_DIR: $ROOT_DIR"
echo "[INFO] MODEL_PATH: $MODEL_PATH"
echo "[INFO] Allowed media path: $ROOT_DIR/data/survey_plans"
echo "[INFO] vLLM server ready at http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
