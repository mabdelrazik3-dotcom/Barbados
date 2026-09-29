# Barbados-Brainiac

Line transcription for the R.O.A.D. Barbados deed books (1639–1710): one handwritten line in, its
transcription in the collection's conventions out, scored by the leaderboard metric

```
score = 0.5 · (1 − word_edits / 12) + 0.5 · (1 − char_edits / 55)      (edits per line, averaged)
```

Two approaches share one candidate pool:

- **Approach A** ports the **1st-place solution** of the Barbados Lands & Surveys challenge
  (text-extraction part) to line crops: multi-view inputs, VLM label correction, dual-target
  LoRA fine-tuning, pseudo labels, output cleaning.
- **Approach B** implements the **generate → score → pick → stabilise → gates** workflow:
  several fine-tuned VLMs and a CTC reader write candidates, VLM judges, the CTC reader and a
  character LM score them, and a GBDT ranker picks one per line.

**Fusion.** Approach A's models add their candidates to B's pool, and the ranker chooses among
all of them. Every method also writes its own submission.

Inputs: `Train.csv`, `Test.csv`, `SampleSubmission.csv`, the line images, and the two prompts
`mega_prompt.md` (v1) and `mega_prompt_v2.md` (v2). Backbones: **Qwen2.5-VL-7B-Instruct**
and **Qwen3-VL-8B-Instruct** (you can add more in `configs/models.yaml`).

Workflow diagrams (end to end, approach A, approach B, scopes): [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

---

## Quick start

```bash
bash install.sh                  # CUDA 12.4 venv; WITH_UNSLOTH=1 / WITH_VLLM=1 for the optional backends
source .venv/bin/activate
bash run_all.sh                  # every stage of configs/brainiac.yaml, one after another
```

The final submission is written to `runs/<experiment>/submission.csv`. Every method's own
submission is in `runs/<experiment>/submissions/`.

```bash
python -m brainiac.run --list                          # the stage order
python -m brainiac.run --stages b_features,b_rank,fuse # selected stages
python -m brainiac.run --from-stage b_pool             # resume
python -m brainiac.run --force --stages b_rank         # recompute
python -m brainiac.run seed=7 approach_b.ranker.threads=8 backend=vllm   # overrides
CONFIG=configs/dry_run.yaml bash run_all.sh            # wiring check: mock VLM, 60 rows per split
```

---

## Stages

Stages run strictly in sequence in one process, with one model on the GPU at a time. A stage
skips work whose outputs already exist unless you pass `--force`.

| # | stage | what it does | output (below `runs/<experiment>/`) |
|---|---|---|---|
| 1 | `prepare` | line table: split (train/test/extra), family A/B, near-duplicate groups, folds | `data/lines.csv` |
| 2 | `views` | model input images per line (see *Views*) | `views/<spec>/` |
| 3 | `label_correction` | **A** · VLM checks every label against its image; optional external corrections | `data/label_corrections.csv` |
| 4 | `a_train_first` | **A** · first LoRA fine-tune (per backbone, per scope) | `models/a1_*` |
| 5 | `a_pseudo` | **A** · pseudo labels for `Test` + unlabelled images | `data/pseudo_a_*.csv` |
| 6 | `a_train_final` | **A** · final LoRA fine-tune on labels + pseudo labels | `models/a2_*` |
| 7 | `a_infer` | **A** · candidates for the holdout fold and test; approach-A submission | `candidates/a_*`, `submissions/a_*.csv` |
| 8 | `b_train_generators` | **B** · generator LoRAs (several seeds per soup) | `models/gen_*_s<seed>` |
| 9 | `b_soups` | **B** · average seed LoRAs into one adapter | `models/gen_*_soup` |
| 10 | `b_generate` | **B** · ~14 candidates per generator and line | `candidates/<gen>__*.jsonl` |
| 11 | `b_readers` | **B** · CRNN CTC reader: train, beam-8 strings; external readers | `models/crnn_*.pt`, `candidates/ctc__*` |
| 12 | `b_pool` | **A + B** · union of all candidates per line (fusion) | `pool/pool__*.jsonl` |
| 13 | `b_judges` | **B** · six judges: forced-decode log-probs of every candidate | `judges/<judge>__*.jsonl` |
| 14 | `b_features` | **B** · F0–F3 + reader blocks; target = exact metric loss | `features/features__*.csv` |
| 15 | `b_baseline` | **B** · judge mix → MBR → CTC overlay | `selection/baseline*`, `submissions/baseline.csv` |
| 16 | `b_rank` | **B** · GBDT ranker, nested CV, 3 members × 5 seeds, gates | `selection/ranker*`, `submissions/ranker.csv` |
| 17 | `fuse` | per-method submissions, fallbacks, primary pick by the gates | `submissions/*.csv`, `submission.csv` |
| 18 | `report` | holdout-fold score of every source, the oracle and each method | `report.txt`, `report.json` |

### Scopes: honest training data for the ranker

Every trainable component (LoRAs, CTC reader, char LM) is built twice:

- **`oof`** is trained on every fold except the holdout fold (`data.holdout_fold`, about 800
  lines). Its candidates and scores on the holdout fold are the ranker's training data, and
  none of those lines was seen in training.
- **`full`** is trained on all labelled lines. It serves the test set.

Parallel copies of the same formula and duplicate crops of the same physical line share a
`group` and therefore a fold, so a copy of a training line never lands in the holdout fold.

---

## Approach A — the 1st-place pipeline, adapted

| 1st place (survey plans) | here (deed-book lines) | code |
|---|---|---|
| `download_models.py` | `models.yaml` `hf_id` / `local_path` (downloaded on first use) | `vlm/registry.py` |
| `patchify_images.py`: 7 crops per plan, letterboxed to 1024 px | `views.multi`: the whole line plus overlapping left/right parts, rescaled by height so letters fill the vision patch grid (28 px Qwen2.5-VL, 32 px Qwen3-VL) | `data/views.py` |
| `automated_label_correction.py`: base Qwen3-VL corrects noisy labels | the full mega prompt plus a correction header: the VLM checks each `Target` word by word. Corrections are accepted only within `max_char_edits`; rows the VLM rejects wholesale become *noisy* and are dropped from training (capped at `max_noisy_rows`). Optional external corrections (e.g. the `opus_label` study) are applied first | `pipeline/approach_a.py` |
| answer = corrected field (`Land Surveyor`) + raw field (`Land Surveyor2`); the raw-style field is submitted | `dual_target`: `{"ink": corrected line, "transcription": annotator-style line}`; `transcription` is submitted, and `ink` joins the pool | `text/prompts.py` |
| `train_and_create_pseudo.py` / `train_with_pseudos.py` (Unsloth, r16, dropout 0.35, NEFTune 5, paged 8-bit AdamW) | `a_train_first` → `a_pseudo` (test + unlabelled images, confidence filter) → `a_train_final`; preset `first_place` (Unsloth, 1 epoch) or `default` (transformers + peft) | `vlm/sft.py` |
| `text_inference.py` + `clean_text_preds.py` | greedy + beam n-best; cleaning maps outputs to the collection's character set (long s → s, `+` → `&`, superscripts → `^`) | `text/clean.py` |
| metric: TargetSurvey WER, exact matches | ROAD metric (edit counts / 12 and / 55) | `metric.py` |

The segmentation half of the 1st-place solution is not used (it was also removed from the
reference copy in this repository).

## Approach B — generate, score, pick, stabilise, gates

| diagram | here |
|---|---|
| Generate: several fine-tuned VLMs, ~14 candidate lines each | `approach_b.generators`, each with greedy + beam 8 (7 returned) + 6 samples |
| 8B soup — Qwen3-VL-8B, 3 LoRAs averaged | generator `q3_soup3`: 3 seeds, `soup: cat`. The soup is the exact average of the three weight updates |
| 9B P3 — fold-0 + full-data models | every generator has an `oof` (fold-0-excluded) and a `full` model |
| 9B soup3d3 — LoRAs on the data3 set | generator `q25_soup_d3`: 3 seeds on `corrected+pseudo` data |
| CTC beam-8 strings | CRNN reader, prefix beam search (width 8, optional char-LM fusion) |
| Six VLM judges, forced-decode log-probs | `approach_b.judges`: a judge is a generator's adapter, optionally with another prompt, scoring each candidate's answer JSON by teacher forcing |
| CONFIG B7b_r64 + q25j mix → MBR → CTC overlay | `b_baseline`: weighted judge mix → posterior → minimum expected metric loss → + CTC nudge |
| Independent readers: CRNN · PP-OCR · char n-gram | CRNN (trained here), `external_readers` (`paddleocr`, `trocr`, `csv`), Kneser-Ney char n-gram |
| Features F0–F3 + reader blocks | `pipeline/features.py` (see its docstring for every column) |
| GBDT ranker, target = exact metric loss, nested CV on 800 fold-0 lines | `b_rank`: outer folds give the out-of-fold score, an inner split sets the boosting rounds; CatBoost (QueryRMSE / YetiRank / RMSE), LightGBM and sklearn members available |
| Stabilise: 3 CatBoost members × 5 seeds, pinned env | per-line z-scores averaged over seeds and members; `ranker.threads` pinned; `deterministic: true` |
| Gates: twin → Tier 1 → Holm → K5 → seed flip ≤ 3 % | `approach_b.gates`: twin-seed noise floor, paired bootstrap vs baseline, Holm over all tested configurations, better in ≥ 4 of 5 folds, ≤ 3 % test picks flipping between seeds |
| Final picks P1 + gblmx | `submissions/ranker.csv` and `submissions/blend.csv` (ranker + MBR + char LM + CTC) |

---

## Views

| spec | images per line | use |
|---|---|---|
| `single` | the whole line, height 112 (family A) / 192 (B) | generators, judges |
| `multi` | whole line + 2 overlapping halves at a larger height | approach A, label correction |
| `composite` | the halves stacked into one image | cheaper alternative to `multi` |

Family A covers the small strips of the 1639–1668 books (≤ 2000 × 180 px); family B covers
the large 1669–1710 crops.

## Prompts

`prompt: {version: v1|v2, mode: full|compact|sections|short, sections: [...]}`.

| mode | contents |
|---|---|
| `full` | the whole mega prompt |
| `compact` | ROLE, SCOPE, the encoding quick reference and the OUTPUT section |
| `sections` | any sections you list, by number (`"4"`) or title word (`"SPELLING"`) |
| `short` | a compact built-in instruction |

The chat layout is identical in training and inference: *user: [prompt, images, instruction] →
assistant: answer JSON*. Each adapter stores its prompt/view/answer format in
`brainiac_meta.json`, and inference reads it back from there.

## Configuration map

| file | what it holds |
|---|---|
| `configs/brainiac.yaml` | stages, seed, backend (`hf` / `vllm` / `mock`), scopes, folds, metric, view specs |
| `configs/paths.yaml` | data files, prompts, image directory, run directory, optional external corrections |
| `configs/models.yaml` | backbones, LoRA presets, training presets (`default`, `first_place`) |
| `configs/approach_a.yaml` | backbones, views, prompt, dual target, label correction, pseudo labels, decoding |
| `configs/approach_b.yaml` | generators, pool, CTC reader, external readers, char LM, judges, baseline, features, ranker, gates |
| `configs/fusion.yaml` | primary submission policy, blend weights, fallback order |
| `configs/dry_run.yaml` | mock backend and tiny settings for a wiring check |

**Backends.** `hf` (transformers) runs anywhere. `vllm` is much faster for generation and
judging, and caches the shared prompt prefix. Soups made with `cat` have rank = the sum of
their members' ranks, so `vllm.max_lora_rank` must be at least that sum. Approach A's
adapters also tune the vision tower (the 1st-place LoRA setting), and vLLM's LoRA support
skips those layers. So A models use `approach_a.infer_backend` (default `hf`) even when the
global backend is `vllm`.

**Memory.** A 7B/8B model in bf16 needs about 17–19 GB for inference. LoRA training with
gradient checkpointing needs more. Set `load_in_4bit: true` on a model for QLoRA / 4-bit
inference, or use the Unsloth preset.

## Code map

```
brainiac/
  config.py            YAML includes, ${...} interpolation, key=value overrides
  run.py               stage runner
  metric.py            ROAD metric, line loss, loss matrices, paired bootstrap
  utils.py             seeding (determinism), logging, run paths, JSONL
  data/dataset.py      line table, families, near-duplicate groups, folds, scopes
  data/views.py        model input views (patchify for lines)
  text/prompts.py      mega-prompt sections, correction prompt, answer JSON, robust parsing
  text/clean.py        output cleaning to the collection's character set
  vlm/registry.py      model specs and loading (Qwen2.5-VL, Qwen3-VL, any AutoModelForImageTextToText)
  vlm/hf_backend.py    generation (greedy / beam / sampling) and forced-decode scoring
  vlm/vllm_backend.py  the same on vLLM (prefix caching, prompt log-probs)
  vlm/mock_backend.py  fake model for wiring checks
  vlm/sft.py           LoRA fine-tuning (transformers + peft, or Unsloth)
  vlm/soup.py          LoRA soups (exact `cat`, approximate `linear`)
  readers/crnn.py      CRNN CTC reader: training, beam search, candidate scoring
  readers/external.py  PaddleOCR / TrOCR / CSV readers
  lm/charlm.py         Kneser-Ney character n-gram LM
  pipeline/            stages: approach_a, approach_b, features, selection, fusion, common
```
