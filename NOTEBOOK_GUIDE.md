# `barbados-2-enhanced.ipynb` — user guide

This notebook fine-tunes Qwen-VL models with LoRA to transcribe the R.O.A.D. Barbados handwriting lines. It measures every change with the **official competition score** on a fixed validation set and writes a Zindi submission. It is a rebuilt version of `barbados-2.ipynb`.

> **Status:** every mode below has been run end to end on CPU with tiny random-weight Qwen2.5-VL and Qwen3-VL models, on transformers 5.0.0 (the Kaggle version) and 5.13. It has **not yet been run on a GPU**, so no real scores exist yet.

---

## 1. Quick start

1. Attach the same three Kaggle inputs as `barbados-2.ipynb`:
   - the wheels dataset (`haniagamal/teleco-wheels`);
   - the competition data (`haniagamal/road-barbados`);
   - the Qwen model(s) listed in `CFG.MODEL_PATHS`.
2. Open **Cell 3 (CFG)**. The defaults run trial **EXP_002**:
   - realistic augmentation for every model;
   - the 32B model at learning rate 1e-4;
   - three decoding variants;
   - cross-model ensembles;
   - per-class routing.
3. *Run All*.
4. Download `/kaggle/working` and run the offline analysis on your PC:

   ```bash
   python work/analyze_trials.py <downloaded_folder>
   ```

To run a different experiment, change only the CFG settings named in §8 and give the run a new `TRIAL_ID`.

---

## 2. The official score

```
score = 0.5 × (1 − word_edits_per_line / 12) + 0.5 × (1 − char_edits_per_line / 55)     (higher is better)
```

- One wrong word costs as much as **4.58 wrong characters**.
- Text is compared after trimming and collapsing spaces. It is case- and punctuation-sensitive.
- Each trial also prints corpus WER and CER for reference.
- On 819 lines, differences smaller than about **0.005 are noise**. That is why every comparison comes with a bootstrap confidence interval and P(better).

## 3. Validation protocol (Fold 0)

- **The split:** `KFold(5, shuffle=True, seed 42)` over Train.csv (shuffled with seed 42), minus the 5 `CORRUPT_IDS`. Fold 0 is identical to `barbados-2.ipynb`.
- **What is scored:** all **819** Fold-0 validation lines, every time.
- **Checkpoint selection:** uses an **inner 8 % slice of the training part** (262 lines), never the validation lines. An assertion stops the run if any validation line leaks into training, checkpoint selection or pseudo-labels.
- **Final submission:** `RUN_MODE = "full"` retrains on all 4,093 lines, and there is no validation.

## 4. The data classes (detected from pixels only)

| Group | Rule | Train lines | What is different |
|---|---|---:|---|
| **B** | image height > 150 px | 1,043 | 1670s–1710s documents; about 77 px per character; 2× `^` superscripts, 4× `:` abbreviations; spellings `heyres`, `saide` |
| **A1** | height ≤ 58 px | 1,669 | tight crops; heavy use of `&`, `ye` and `yt` |
| **A2** | taller, parchment grey ≤ 194 | 383 | dark parchment; larger hand; almost no `&` or `ye` |
| **A3** | taller, parchment grey > 194 | 1,003 | loose crops that often show neighbouring lines; longest lines |

- A1, A2 and A3 together form **class A** (1630s–1660s documents).
- The rule uses only the image, so it works identically on test images. The same function (`image_group`) is used in training and inference.
- Details are in `FULL_DATA_AUDIT.md` §6b.

---

## 5. Cell-by-cell

| Cell | What it does |
|---|---|
| 0 (markdown) | Overview, a table of changes vs the original notebook, how to choose the one-model or specialist setup, and the class table |
| 1 | Offline setup: environment variables and wheels folder discovery |
| 2 | `pip install` from the offline wheels, same as the original |
| 3 | **Imports**, with a pure-Python edit-distance fallback if `rapidfuzz` is missing |
| 4 | **CFG**: `TRIAL_ID`, `TRIAL_DESCRIPTION` and every setting (§7) |
| 5 | Resolves the CSV, image and model paths; creates `trial_predictions/` and `trial_configs/` |
| 6 | **Utilities:** label cleaning, the image-resize ladder (same for training and inference), the **realistic augmentation profile**, **class/group detection**, prompt construction (with the optional class tag), the augmentation stack, model naming |
| 7 | **Model and processor loaders.** Pixel limits (`MIN_PIXELS`, `MAX_PIXELS`) are forced on every transformers version and **checked on a probe image**; the run stops if they are not applied |
| 8 | **Scoring toolkit:** official metric, MBR selection, word/char ROVER, paired bootstrap, error forensics, scores per group |
| 9 | Data: builds the samples, the Fold-0 split and the inner checkpoint slice; detects every image's group; loads test and the **687 unlisted images**; optionally loads pseudo-labels (filtered per class) |
| 10 | **Collator:** builds the chat prompt (plus class tag), masks everything except the transcription, and applies a **fresh augmentation every time a line is seen** |
| 11 | **Generation:** batched beam search returning the N best transcriptions with their log-probabilities; retries one image at a time on out-of-memory |
| 12 | **Trial harness:** the standard trial report, `trial_results.csv`, prediction and config files, champion tracking |
| 13 | **Main loop,** one model at a time (§6) |
| 14 | **Ensembles and decisions:** specialist candidates, cross-model ensembles, per-class routing with the acceptance rule, final ranking, `submission.csv`, `pseudo_labels.csv`, trial status bookkeeping |
| 15 | **Error forensics** of the best system vs the baseline: edit types, confusions, word error classes, per-class deltas, most-worsened lines |
| 16 (markdown) | Next trials |

## 6. What the main loop (Cell 13) does for each model

For each model in `CFG.MODEL_PATHS`, one at a time (the GPU holds only one model):

1. **Load the processor and check** the pixel limits on a probe image.
2. **Apply per-model settings** from `MODEL_OVERRIDES`, e.g. 32B: learning rate 1e-4.
3. **Train the shared LoRA.** It is trained on the Fold-0 training lines plus augmented copies. `CLASS_AUG_COPIES` and `CLASS_SAMPLING_T` add extra copies of smaller or harder classes. The best checkpoint by loss on the inner slice is kept, and the adapter is saved to `final/<model>/fold0/`.
   - With `DO_TRAIN = False`, the saved adapters are loaded instead.
4. **Decode all 819 validation lines with every decoding variant** (§9). Each variant becomes one trial row, and D2 also gets a "single-model MBR" row.
5. **Decode test** with this model's best variant plus D2. **Decode the unlisted images** with D2, for pseudo-labels.
6. **Specialists** (if `SPECIALISTS` is set and this model is in `SPECIALIST_MODELS`). For each key:
   - build the specialist (§10);
   - decode **only that class's** validation and test lines with it;
   - report its gain against the shared model on those lines;
   - save it to `final/<model>/fold0/spec_<key>/`.
7. **Free the GPU** and move on to the next model.
   - The session stops starting new models after `TIME_BUDGET_H` hours. Everything already finished is saved.

## 7. Configuration reference (Cell 3)

### Run control
| Setting | Default | Meaning |
|---|---|---|
| `TRIAL_ID` / `TRIAL_DESCRIPTION` | `"EXP_002"` | The name used in every output file and results row |
| `MODEL_PATHS` | 8B, 7B, 32B | Models run this session, in order |
| `RUN_MODE` | `"fold0"` | `"fold0"` = research benchmark; `"full"` = retrain on all lines and write the final submission |
| `DO_TRAIN` / `DO_INFER` / `PREDICT_TEST` | True | `DO_TRAIN=False` loads adapters from `ADAPTERS_DIR` (fast path) |
| `ADAPTERS_DIR` | None | Folder with `<model>/fold0/` adapters. Also recognises the original notebook's folder names |
| `CKPT_SELECTION` / `INNER_CKPT_FRAC` | `"inner_loss"` / 0.08 | Checkpoint selection on an inner slice of the training part (`"last"` = no selection) |
| `TIME_BUDGET_H` | 11 | Skip remaining models after this many hours |
| `PREV_RESULTS_DIRS` | [] | Earlier sessions' outputs, used to track the champion across sessions |

### Training
| Setting | Default | Meaning |
|---|---|---|
| `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, `PER_DEVICE_TRAIN_BATCH_SIZE` | 2e-4, 1, 4 | Original recipe |
| `R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_VISION` | 32, 64, 0.05, True | LoRA on all language layers plus the vision tower |
| `MODEL_OVERRIDES` | `{"32b": {"LEARNING_RATE": 1e-4}}` | Per-model changes. Keys are size tags from the model path. Allowed fields: `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, `R`, `LORA_ALPHA`, batch size, gradient accumulation |
| `MIN_PIXELS` / `MAX_PIXELS` | 200,704 / 2,000,000 | Image-size limits, enforced and checked |

### Augmentation
| Setting | Default | Meaning |
|---|---|---|
| `AUG_PROFILE` | `"realistic"` | Strengths matched to measured archive statistics. `"original"` = the barbados-2 values |
| `AUG_COPIES` | 1 | Augmented copies per line |
| `AUG_RESAMPLE` | True | A new random augmentation each time a line is seen, instead of one fixed copy |

| Effect | Realistic | Original | Measured in the real data |
|---|---|---|---|
| rotation | ±1.0° | ±4° | 95 % of lines within ±1° |
| noise σ | 3.1 grey levels | 25.5 | 1.8–4.2 |
| JPEG quality | 70–92 | 40–80 | about 85 |
| vignette | up to 12 % | up to 35 % | illumination CV ≤ 0.09 |
| stroke | thicken only, p 0.15 | erode or dilate, p 0.35 | strokes 2–4 px wide (thinning can erase them) |
| fill of new pixels | the page's parchment colour | white | parchment, never white |

### Classes, hints and specialists
| Setting | Default | Meaning |
|---|---|---|
| `CLASS_HEIGHT_SPLIT`, `GROUP_A1_MAX_HEIGHT`, `GROUP_A2_MAX_PAPER` | 150, 58, 194 | The pixel rule of §4 |
| `CLASS_HINT` / `HINT_LEVEL` | False / `"class"` | Adds `" Scan type: A."` or `" Scan type: B."` (or A1/A2/A3/B) to the prompt, in training **and** inference |
| `CLASS_SAMPLING_T` | None | Class-balanced sampling; e.g. 2 means each A2 line is seen about 2×, A3 and B about 1.3× (extra copies are fresh augmentations) |
| `CLASS_AUG_COPIES` | None | Fixed extra copies per class or group, e.g. `{"B": 2}` |
| `SPECIALISTS` | `{}` | Keys `"A"`, `"B"`, `"A1"`, `"A2"`, `"A3"`; value = specialist settings (§10). Empty = one model for all |
| `SPECIALIST_MODELS` | `("8b",)` | Which base models get specialists; `()` = all |
| `SPEC_DEFAULTS` | `init="continue", lr=1e-4, epochs=1, replay=0.15, warmup_steps=10` | Defaults for every specialist |
| `TRAIN_SPECIALISTS` | True | With `DO_TRAIN=False`: train specialists on top of the loaded shared adapters (False = load saved specialists) |
| `CLASS_ROUTING` / `ROUTING_LEVEL` | True / `"class"` | Per-class decision between the champion and challengers; `"group"` decides per A1/A2/A3/B |
| `ROUTE_P_BETTER`, `ROUTE_MIN_OVERALL_GAIN`, `ROUTE_MAX_CANDIDATES`, `ROUTE_SPLITS` | 0.9, 0.002, 3, 30 | The acceptance rule (§11) |

### Decoding, ensembles, final run and pseudo-labels
| Setting | Default | Meaning |
|---|---|---|
| `DECODE_VARIANTS` | D0, D1, D2 | §9 |
| `BASELINE_VARIANT` | `"D0_orig"` | What every delta is measured against |
| `ENSEMBLE_METHODS` | MBR over all candidates, MBR over each model's top answer, word-ROVER, char-ROVER | §9 |
| `MBR_TEMPERATURE` | 1.0 | Sharpness of the weights on each model's candidates |
| `CHAMPION_DECODE` / `CHAMPION_ENSEMBLE` / `ROUTING_MAP` | `"D2_b5nb"` / `"mbr_nbest"` / None | Used only by `RUN_MODE="full"`. `ROUTING_MAP` may be the path of `routing_map.json` |
| `PREDICT_UNLISTED` | True | Decode the 687 unlisted images for pseudo-labels |
| `PSEUDO_LABEL_CSV` / `PSEUDO_KEEP_FRAC` / `PSEUDO_PER_CLASS` | None / 0.7 / True | Add pseudo-labelled test and unlisted lines to training, keeping the lowest-risk 70 % **per class** |

### The prompt
The prompt is unchanged from barbados-2, so it matches the training prompt of your earlier adapters:

```text
This is not modern English, Transcribe the handwriting exactly. Keep the wrong spelling, abbreviations, and marks (^, ff, unusual letters).Do not modernize or correct anything.
```

- The text comes before the image in the user turn.
- Qwen2.5-VL also adds its default system line, "You are a helpful assistant."
- With `CLASS_HINT=True`, ` Scan type: B.` (or A / A1 / A2 / A3) is added at the end.
- The model is trained only on the transcription followed by `<|im_end|>`.

---

## 8. Run recipes

Change only what is listed. Use a new `TRIAL_ID` each time, and **run the 8B model alone** (`MODEL_PATHS = [<8B path>]`) for the one-change experiments, about 1 hour each.

| Goal | CFG changes |
|---|---|
| **EXP_002** (default): realistic augmentation, 32B at 1e-4, decoding and ensembles | none |
| **EXP_003**: one model with class hints | `CLASS_HINT=True`, `CLASS_SAMPLING_T=2` |
| **EXP_004**: shared model + class-B specialist | `SPECIALISTS={"B": {}}` |
| **EXP_005**: shared model + A1 specialist | `SPECIALISTS={"A1": {}}` |
| **EXP_006**: specialist per class | `SPECIALISTS={"A": {}, "B": {}}` |
| **EXP_007**: specialist per sub-class | `SPECIALISTS={"A1": {}, "A2": {}, "A3": {}, "B": {}}`, `ROUTING_LEVEL="group"` |
| **EXP_008**: fully separate class-B model | `SPECIALISTS={"B": {"init": "separate", "replay": 0}}` |
| **Specialists without retraining the shared model** | `DO_TRAIN=False`, `ADAPTERS_DIR=<earlier final/ folder>`, `SPECIALISTS={...}`, `TRAIN_SPECIALISTS=True` |
| **Re-decode with saved adapters and specialists only** | `DO_TRAIN=False`, `TRAIN_SPECIALISTS=False`, `ADAPTERS_DIR=...` |
| **EXP_009**: pseudo-labels | `PSEUDO_LABEL_CSV=<pseudo_labels.csv from an earlier run>` |
| **FINAL submission** | `RUN_MODE="full"`, the promoted training settings, the same `SPECIALISTS` keys, `ROUTING_MAP="<path>/routing_map.json"` |

Notes:
- The "re-decode" setting is also the fastest way to test decoding changes on adapters from the original notebook. Those adapters are found by their original folder names.
- The class hint changes the prompt. Only use `CLASS_HINT=True` with adapters that were **trained** with it.

## 9. Decoding variants and ensembles

| Variant | Settings | Purpose |
|---|---|---|
| `D0_orig` | beams 3, `repetition_penalty=1.2`, `no_repeat_ngram_size=6` | Exactly the original decoding (the baseline) |
| `D1_norep` | beams 3, penalty off, n-gram block off | One change: the penalty hits 14.5 % of correct tokens (` the`, ` and`, `,`, `^` are in the prompt) |
| `D2_b5nb` | beams 5, returns the 5 best | Top answer, plus a "+mbr" row that picks among the 5 |

**MBR (minimum Bayes risk).** Among the candidate transcriptions, MBR picks the one with the lowest *expected official loss* against all the others. Each candidate is weighted by the model's probability for it.

| Ensemble | How it works |
|---|---|
| `mbr_nbest` | MBR over all models' 5-best lists together |
| `mbr_top1` | the "medoid": MBR over each model's top answer, equal weights |
| `word_rover` / `char_rover` | align every model's top answer and take a vote per word (or per character) |

Every ensemble and every specialist variant is scored as its own trial.

## 10. Specialists

A specialist is an extra LoRA adapter kept in the same model as the shared adapter. It is trained on one class or sub-class, for example `"B"` or `"A2"`.

| `init` | What it is | Research expectation |
|---|---|---|
| `"continue"` (default) | A **copy of the shared adapter**, trained further on that class's lines plus **15 % replay** lines from other classes, at learning rate 1e-4 (constant after 10 warm-up steps), for 1 pass | The only kind with published gains: a small gain on distinct classes (B), about 0 on others |
| `"separate"` | A **new LoRA from the base model**, trained on that class only | Expected to lose, since it throws away the other classes' data |

Per-specialist settings override `SPEC_DEFAULTS`, e.g. `{"B": {"lr": 5e-5, "max_steps": 400, "replay": 0.2}}`.

**At inference:**
- A specialist decodes only the lines of its own class. Every other line keeps the shared model's output, so each specialist is a complete, full-length candidate.
- Three candidates are created per specialist:
  - the specialist alone, `<model>@<key>__<variant>`;
  - its 5-best MBR;
  - the cross-model MBR with the specialist added as an extra voter, `ENS[8b+7b+32b+8b@B]__D2_b5nb__mbr_nbest`.
- The last one is the "soft routing" option, which limits the damage if a specialist overfits.

## 11. Per-class routing and the acceptance rule

After all candidates exist, Cell 14 decides class by class (or group by group):

1. **Champion** = the candidate with the best overall validation score. On each class it is compared with up to 3 challengers: at most 2 of that class's specialist candidates first, then the best other systems.
2. The class **switches** to a challenger only if **all four** conditions hold on that class's validation lines:
   - paired bootstrap **P(better) ≥ 0.9**;
   - the **empirical-Bayes-shrunk gain > 0**. The class gain is pulled toward the challenger's overall gain; if the data show no real class-specific effects, the notebook prints "one model for all";
   - the gain has **the same sign in both halves** of most random split-halves;
   - **test share × gain ≥ 0.002** overall.
3. It prints a decision table: class, challenger, delta, P, shrunk gain, sign rate, overall value, accepted or not.
4. The **honest score** of the whole procedure (choose on one half of validation, score on the other, 30 repetitions) is reported next to the champion's score. The honest number is the one to trust.
5. The decision is saved to **`routing_map.json`**, which `RUN_MODE="full"` reproduces on test.

This rule exists because picking the per-class best of several systems on 75–311 lines inflates validation by about 0.005 even when nothing is truly better.

## 12. What each trial prints

```
TRIAL: EXP_002__qwen_3_vl_transformers_8b_instruct_1__D1_norep
Official validation score : 0.9xxxx
Word edits / line (LB WER) / Char edits / line (LB CER)
Delta vs baseline : +0.00xxx  CI95 [.., ..]  P(better)=..  improved=.. worsened=..  word/char edits saved=..
Delta vs champion / Runtime / Peak GPU RAM / Train & validation examples / Empty & truncated predictions
By class (A<=150px, B>150px) : ...        By group (A1/A2/A3/B) : ...        By label length : ...
```

Promote a change only if P(better) ≥ 0.9 **and** the lower CI95 bound is above −0.001.

## 13. Files written to `/kaggle/working`

| File | Contents |
|---|---|
| `trial_results.csv` | One row per trial. Columns include: score, word/char edits per line, corpus WER/CER, delta vs baseline with CI and P(better), delta vs champion, per-class `score_A` / `score_B`, runtime, GPU memory, learning rate, augmentation profile, hint setting, specialist fields, status |
| `trial_predictions/<trial>__val.csv` | Every validation line: ID, ground truth, prediction. Single-model and specialist trials also include log-prob, token count, truncated flag and the 5-best list |
| `trial_predictions/<trial>__test.csv`, `__unlisted.csv` | The same for test and unlisted images |
| `trial_configs/<trial>.json` | The full CFG and decoding settings of the trial |
| `log_history__<model>.csv`, `log_history__<model>__spec_<key>.csv` | Training logs |
| `qwen-vl-ocr/final/<model>/fold0/` (+ `spec_<key>/`) | Shared and specialist adapters |
| `submission.csv` + `submission_source.json` | Submission from the best validated candidate that has test predictions |
| `routing_map.json` | Per-class decision for the final run |
| `pseudo_labels.csv` | MBR transcription and risk for every test and unlisted line (input to pseudo-labelling) |
| `fold0_val_ids.csv` | The 819 validation IDs |

## 14. After a run

1. Download `/kaggle/working`.
2. Run `python work/analyze_trials.py <folder>`. It writes `TRIAL_ANALYSIS.md` with:
   - the trial table and controlled deltas;
   - where the loss is, by class, visual cluster, contrast, sharpness, length and template phrases;
   - word and character confusion tables;
   - confidence calibration;
   - oracle bounds of the model and 5-best;
   - suspected mislabelled lines.
3. Keep or reject changes with the rule in §12, then pick the next single change from §8.

## 15. Runtime and memory (estimates, not measured)

On the RTX PRO 6000 (96 GB):
- **Full EXP_002 with 3 models:** about 3–4.5 h, mostly the 32B.
- **8B only:** about 1 h.
- **Each continued specialist:** about 0.1–0.4 of a shared training run, plus decoding its class.

32B training uses gradient checkpointing. If a decoding batch runs out of memory, it is retried one image at a time.

## 16. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `pixel limits NOT applied` | The processor ignored `MIN_PIXELS` / `MAX_PIXELS`. The run stops on purpose; tell me the transformers version |
| `no adapter for <model> fold0` | With `DO_TRAIN=False`, `ADAPTERS_DIR` must contain `<model>/fold0/adapter_config.json`, under the new or the original folder name |
| `specialist ... missing — add 'B' to SPECIALISTS` | A final-run `ROUTING_MAP` refers to a specialist that was not built. Use the same `SPECIALISTS` keys as the Fold-0 run |
| `validation leak!` / `pseudo-labels must not cover validation lines` | A data-protocol violation; don't bypass it |
| `[time] budget ... reached` | Remaining models were skipped. Their outputs are missing, but everything completed is saved |

## 17. Companion notebook: `barbados-2-classwise.ipynb` (separate model per class)

Built by `work/build_classwise_notebook.py`. It reuses the tested infrastructure cells above (Cells 1–11) and replaces only the configuration, training, ensemble and forensics cells.

**What it does**
- For each base model in `MODEL_PATHS` (default: 8B only), it trains a **brand-new LoRA per class from the base model**, using **only that class's lines**. The adapters are stored as `class_<key>` and saved to `final/<model>/fold0/class_<key>/`.
- Every class has **its own prompt** (`CLASS_PROMPTS`), used both in training and in inference:
  - B: 1670s–1710s documents; `^` superscripts, `:` suspensions, ff-, heyres/saide.
  - A: 1630s–1660s documents; `&`, ye/yt, sd, heires/assignes/publique.
  - A1, A2, A3: the A prompt plus a note on crop and parchment.
- Every validation, test and unlisted line is decoded by the adapter of **its own class**, which is detected from pixels.

**Main settings**

| Setting | Default | Meaning |
|---|---|---|
| `CLASS_SCHEME` | `"class"` | `"class"` trains 2 models (A, B); `"subclass"` trains 4 (A1, A2, A3, B) |
| `CLASS_REPLAY` | 0 | Share of rows drawn from other classes; 0 = fully separate |
| `CLASS_HP` | {} | Per-class overrides, e.g. `{"A2": {"NUM_TRAIN_EPOCHS": 2}}` |
| `BASELINE_PRED_CSV` | None | A shared-model validation predictions file; enables paired per-class comparison |
| `DO_TRAIN=False` + `ADAPTERS_DIR` | — | Reuse saved class adapters without training |
| `RUN_MODE="full"` | — | Retrain all class models on 4,093 lines and write the submission |

**What it prints**
- Per-class and per-group scores for every trial.
- The per-class paired difference against the shared baseline.
- An error breakdown for the best class-wise system.

**Outputs:** file names are compatible with `work/analyze_trials.py` and `work/08_build_pseudo_labels.py`.

**Checks:** CPU dry-runs on transformers 5.0.0 passed:
- class scheme;
- sub-class scheme with replay;
- reuse of saved adapters, which reproduces identical scores;
- full refit;
- the tokenized training text carries each class's own prompt.

## 18. What changed vs `barbados-2.ipynb`

**Scoring and validation**
- Official metric instead of `0.5·jiwer.wer + 0.5·jiwer.cer`.
- All 819 validation lines scored, with checkpoint selection moved to an inner slice of the training part.

**Robustness**
- Pixel limits forced and checked on every transformers version.
- Exact-name adapter lookup; the old lookup could load the 32B adapter for the 8B.

**Decoding and ensembles**
- Repetition-penalty and 5-best decoding compared on the same adapter.
- MBR / ROVER ensembles, which existed but were never called.
- A submission writer, which was missing.

**Training**
- Realistic augmentation with parchment fill and fresh augmentation per visit.
- Per-model learning rates (32B at 1e-4).

**Classes and experiments**
- Class and sub-class detection.
- Class hints and class-balanced sampling.
- Per-class and per-sub-class specialists.
- Per-class routing with the acceptance rule.
- Pseudo-labels filtered per class.
- A trial harness with bootstrap confidence intervals.
- Dead code removed.

The evidence behind each change is in `FULL_DATA_AUDIT.md`, `RESEARCH_PLAN.md` and `reports/Per class specialist HTR models.md`.

## 19. MEGA notebook: `barbados-2-mega.ipynb` (shared + class-wise + sub-class-wise, pooled)

Built by `work/build_mega_notebook.py`. It reuses Cells 1–11 above and adds five new cells (12–16).

**What one run does**
1. **Trains every system in `CFG.SYSTEMS`.** Systems on the same base model share one loaded base; each class or sub-class model is its own LoRA adapter.

   | Default system | Adapters | Prompt |
   |---|---|---|
   | `8b_shared` | 1 (all lines) | original prompt |
   | `8b_class` | A, B (fresh, own class only) | era prompt |
   | `8b_subclass` | A1, A2, A3, B (fresh, own sub-class only) | sub-class prompt |
   | `7b_shared` | 1 (all lines) | original prompt |

   Optional keys per system:
   - `keys=["B"]`: a partial system that votes only on B lines;
   - `init="from:8b_shared"`: continue from a shared adapter;
   - `replay`;
   - `hp`, `class_hp`.
2. **Validates every `EVAL_EVERY=100` optimizer steps, and always at the last step.** The check uses the inner 8% slice of TRAIN (the adapter's own class) and prints one `[VALID …]` line:
   - training loss;
   - teacher-forced inner loss;
   - official score of greedy transcriptions;
   - word and char edits per line.

   The best step by `CKPT_METRIC` (`inner_official` by default) is kept in RAM and restored at the end (`[CKPT …] RESTORED step …`).
   With `MONITOR_VAL_N>0`, Fold-0 val lines are also decoded at each check. This is printed only and never used for selection.
3. **Decodes val, test and unlisted** with every system: beam 5, 5-best, loop guard. Each system gets its own trial rows (top-1 and single-system MBR, per class and per group, compared with `BASELINE_SYSTEM`).
4. **Rescoring (`RESCORE=True`).** Every system computes the exact log-probability of every pooled candidate, using its own prompt and its own class adapter. A sanity line checks that the forced log-prob equals the beam log-prob of the system's own top-1.
5. **Ensembles.** Each one is a trial on the 819 lines, with a paired bootstrap against `pool_mbr_quality`:

   | Method | Evidence | Selection |
   |---|---|---|
   | `pool_mbr_equal` / `pool_mbr_quality` | each system's own 5-best posteriors (equal / 1-over-lost-points weights) | MBR |
   | `pool_mbr_quality_median` | same | MBR + greedy word-level median search |
   | `rescore_mbr` | every system's posterior over the **whole pool** | MBR |
   | `rescore_mbr_median` | same | MBR + median search |
   | `rescore_map` | weighted sum of forced log-probs | argmax (product of experts) |

6. **Reports and writes.**
   - Prints a ranking with A/B/A1/A2/A3 columns, the pool oracle and the share of the headroom captured, and leave-one-system-out deltas (which systems help).
   - Writes the error forensics of the champion and the inner-validation curves.
   - Writes `submission.csv` from the best candidate that has test predictions.
   - Writes `mega_champion.json` (method + system weights) and `pseudo_labels.csv` (champion posterior, risk = expected official cost).

**Why rescoring** (measured offline on EXP_002, `work/selection_experiments.py`, 5-fold CV over lines):
- Tuning the temperature, per-class weights, a character-LM prior or an OOV penalty gave −0.0005 … 0.
- Median search gave +0.0003 (P = 0.78); the earlier reranker gave −0.0028.
- The pool oracle is 0.9497 against 0.9175 for pooled MBR. In ~85% of the lines where MBR misses, the best candidate is in only one model's 5-best, so the other model gives it zero weight.

Rescoring adds that missing evidence. The two other levers are more systems, and more diverse ones.

**Final run:** `RUN_MODE="full"`, the same `SYSTEMS` minus any system whose removal raised the score, and `FINAL_CHAMPION_JSON=<Fold-0 output>/mega_champion.json`.

**Outputs:**
- `trial_results.csv`, `mega_ranking_val.csv`, `mega_champion.json`;
- `eval_curves.csv`, `adapter_summary.csv`, `log_history__<adapter>.csv`;
- `trial_predictions/MEGA_001__<system>__D2_b5nb__<split>.csv`, readable by `work/offline_ensembles.py`, `work/analyze_trials.py` and `work/08_build_pseudo_labels.py`;
- `rescore__<split>.jsonl`, `pseudo_labels.csv`, `run_log.txt` (every printed line).

**Runtime** (estimate from EXP_002 timings): ~4–5 h for the 4 default systems. `TIME_BUDGET_H` skips whole base models, and the rescoring pass, when time runs short.

**Checks** (CPU dry runs on transformers 5.0.0 with tiny Qwen2.5-VL / Qwen3-VL models, `work/dryrun_notebook.py`):
- **Fold-0 run with 5 systems:** shared, class, sub-class, a B-only system continued from shared, and a second base model.
  - Every adapter printed its validation lines.
  - 8 of 9 adapters had their best step restored.
  - Decode, rescoring, all 6 ensembles, the leave-one-out lines, the champion JSON, the submission and the pseudo-labels all ran.
- **`DO_TRAIN=False` reload of the saved adapters:** all 14 validation scores were reproduced exactly (difference 0.0).
- **`RUN_MODE="full"` with `FINAL_CHAMPION_JSON`:** the `rescore_mbr` submission was written.
- **Rescoring unit test:** the forced log-prob equals an independent forward-pass reference (|Δ| ≤ 1e-5) and does not change with batch padding. It also equals the generation score of the same tokens.
