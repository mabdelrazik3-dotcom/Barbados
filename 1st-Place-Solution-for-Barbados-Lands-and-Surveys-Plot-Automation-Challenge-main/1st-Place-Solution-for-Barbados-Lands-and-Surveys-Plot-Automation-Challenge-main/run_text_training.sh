#!/usr/bin/env bash
# Pipeline for OCR model preparation, pseudo-labeling, and fine-tuning.
set -euo pipefail

# Detect project root
if ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  ROOT_DIR="$SCRIPT_DIR"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

DOWNLOAD_MODELS="$ROOT_DIR/text_extraction/train/download_models.py"
PATCHIFY_IMAGES="$ROOT_DIR/text_extraction/train/patchify_images.py"
LABEL_CORRECTION="$ROOT_DIR/text_extraction/train/automated_label_correction.py"
PSEUDO_LABELS="$ROOT_DIR/text_extraction/train/train_and_create_pseudo.py"
TRAIN_WITH_PSEUDOS="$ROOT_DIR/text_extraction/train/train_with_pseudos.py"


[[ -f "$DOWNLOAD_MODELS" ]] || { echo "Missing: $DOWNLOAD_MODELS"; exit 1; }
[[ -f "$PATCHIFY_IMAGES" ]] || { echo "Missing: $PATCHIFY_IMAGES"; exit 1; }
[[ -f "$LABEL_CORRECTION" ]] || { echo "Missing: $LABEL_CORRECTION"; exit 1; }
[[ -f "$PSEUDO_LABELS" ]] || { echo "Missing: $PSEUDO_LABELS"; exit 1; }
[[ -f "$TRAIN_WITH_PSEUDOS" ]] || { echo "Missing: $TRAIN_WITH_PSEUDOS"; exit 1; }

cd "$ROOT_DIR"
# 1) Download base checkpoints required for Unsloth fine-tuning.
"$PYTHON_BIN" "$DOWNLOAD_MODELS"
# 2) Produce tiled crops that drive both correction and training prompts.
"$PYTHON_BIN" "$PATCHIFY_IMAGES"
# 3) Correct noisy labels, bootstrap pseudo labels, then fine-tune the VLM.
"$PYTHON_BIN" "$LABEL_CORRECTION"
"$PYTHON_BIN" "$PSEUDO_LABELS"
"$PYTHON_BIN" "$TRAIN_WITH_PSEUDOS"
