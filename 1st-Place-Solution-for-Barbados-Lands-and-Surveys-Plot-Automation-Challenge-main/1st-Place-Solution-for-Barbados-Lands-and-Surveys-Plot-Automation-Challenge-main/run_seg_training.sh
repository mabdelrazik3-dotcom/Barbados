#!/usr/bin/env bash
# Orchestrate preprocessing + segmentation model + bias embedding training.
set -euo pipefail

# Detect project root
if ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  ROOT_DIR="$SCRIPT_DIR"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

CREATE_DATASET="$ROOT_DIR/preprocessing/create_dataset.py"
TRAIN_SEG="$ROOT_DIR/segmentation/train/train_seg.py"
TRAIN_BIAS="$ROOT_DIR/segmentation/train/train_bias.py"

[[ -f "$CREATE_DATASET" ]] || { echo "Missing: $CREATE_DATASET"; exit 1; }
[[ -f "$TRAIN_SEG" ]] || { echo "Missing: $TRAIN_SEG"; exit 1; }
[[ -f "$TRAIN_BIAS" ]] || { echo "Missing: $TRAIN_BIAS"; exit 1; }

cd "$ROOT_DIR"
# 1) Materialize fold assignments, image paths, and pre-aligned polygons.
"$PYTHON_BIN" "$CREATE_DATASET"
# 2) Train Lightning segmentation model with configured fold split.
"$PYTHON_BIN" "$TRAIN_SEG"
# 3) Fit bias embeddings used to regularize surveyor preferences at inference.
"$PYTHON_BIN" "$TRAIN_BIAS"
