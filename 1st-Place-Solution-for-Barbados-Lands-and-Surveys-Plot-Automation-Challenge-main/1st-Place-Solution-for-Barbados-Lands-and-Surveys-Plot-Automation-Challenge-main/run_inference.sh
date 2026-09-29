#!/usr/bin/env bash
# Run full inference pipeline: OCR extraction, cleanup, then polygon prediction.
set -euo pipefail

# Detect project root
if ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
  ROOT_DIR="$SCRIPT_DIR"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

TEXT_INF="$ROOT_DIR/text_extraction/infer/text_inference.py"
CLN="$ROOT_DIR/text_extraction/infer/clean_text_preds.py"
SEG_INF="$ROOT_DIR/segmentation/infer/seg_inference.py"

[[ -f "$TEXT_INF" ]] || { echo "Missing: $TEXT_INF"; exit 1; }
[[ -f "$CLN" ]] || { echo "Missing: $CLN"; exit 1; }
[[ -f "$SEG_INF" ]] || { echo "Missing: $SEG_INF"; exit 1; }

cd "$ROOT_DIR"
# Text pipeline produces metadata CSVs consumed by segmentation inference.
"$PYTHON_BIN" "$TEXT_INF"
"$PYTHON_BIN" "$CLN"
# Segmentation writes final submission by fusing geometry with metadata.
"$PYTHON_BIN" "$SEG_INF"
