#!/usr/bin/env bash
# Run the whole Barbados-Brainiac pipeline: every stage listed in the config, one after another.
#   bash run_all.sh                                   # configs/brainiac.yaml
#   CONFIG=configs/dry_run.yaml bash run_all.sh       # wiring check with the mock backend
#   bash run_all.sh --from-stage b_pool               # extra arguments go to brainiac.run
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/brainiac.yaml}"

"$PYTHON_BIN" -m brainiac.run --config "$CONFIG" "$@"
