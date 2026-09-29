# work/ — full-data forensics, evaluation toolkit, notebook build + tests

| File | Purpose |
|---|---|
| `road_metric.py` | Official leaderboard metric `0.5(1−WE/12)+0.5(1−CE/55)` + notebook-metric comparison + bootstrap |
| `road_eval.py` | Shared toolkit: official score, MBR (official cost), word-ROVER, error forensics, paired bootstrap |
| `01_folds.py` | Rebuilds barbados-2's exact Fold-0 split → `fold_assignments.csv` |
| `02_image_forensics.py` | ~70 measurements for ALL 6,159 images + exact Qwen processor geometry → `image_features.csv`, thumbnails |
| `03_label_forensics.py` | Character inventory, lengths, words/n-grams, markup, variants, near-duplicate labels → `label_report.json` |
| `04_joint_analysis.py` | Batches, image↔label correlations, clusters, pHash duplicates, adversarial shift, fold-leakage → `joint_report.json` |
| `06_tokenizer_forensics.py` | Every label through the three Qwen tokenizers → `tokenizer_report.json` |
| `build_enhanced_notebook.py` | Generates `../barbados-2-enhanced.ipynb` (cells kept as strings so they can be tested) |
| `dryrun_notebook.py` | CPU end-to-end run of the notebook with tiny random Qwen2.5-VL / Qwen3-VL (`tiny/`). Env: `DRY_MODE=full`, `DRY_DO_TRAIN=0 DRY_ADAPTERS=…`, `DRY_PSEUDO=…`; run with `PYTHONPATH=work/tf500/pkg` for transformers 5.0.0 |
| `analyze_trials.py` | **Phase-20 loop:** `python work/analyze_trials.py <downloaded /kaggle/working dir>` → `TRIAL_ANALYSIS.md` |

`hf/` holds the official Qwen tokenizer/processor configs (no weights); `tf500/` holds transformers 5.0.0 (the Kaggle version) for compatibility tests.
