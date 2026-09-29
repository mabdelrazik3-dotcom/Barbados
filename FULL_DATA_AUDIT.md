# FULL_DATA_AUDIT — R.O.A.D. Barbados Historic Handwriting Challenge

*Scope:* 100 % of the data. That is all 4,098 training labels, all 6,159 images (4,098 train, 1,374 test, 687 unlisted), and every label through all three Qwen tokenizers. No sampling was used. Where heavy computation was needed, the images were processed in batches.
*Reproduce:* `work/01_folds.py`, `02_image_forensics.py`, `03_label_forensics.py`, `04_joint_analysis.py`, `06_tokenizer_forensics.py`. The outputs sit next to them in `work/` (`image_features_plus.csv`, `train_joint.csv`, `label_report.json`, `joint_report.json`, `tokenizer_report.json`, …).
*Status of claims:* everything here is **measured** on the data or the code, unless it is marked *(forum)*, *(literature)* or *(hypothesis)*. **No GPU model was trained or scored in this work.** No validation number is claimed.

---

## 0. Headline findings

1. **The metric.** The leaderboard score is `0.5·(1 − word_edits_per_line/12) + 0.5·(1 − char_edits_per_line/55)`, and higher is better. So one word edit costs as much as **4.58 character edits**.
   - The notebook optimised `0.5·jiwer.wer + 0.5·jiwer.cer`. That is a different weighting (5.5×) and it is not the leaderboard number.
2. **Validation was on a 491-line subset.** The notebook took 328 of the 819 Fold-0 validation lines for `eval_loss` checkpoint selection and scored only the remaining 491. Their standard error is about 0.006 score. The top 50 on the leaderboard are within 0.013 of each other.
3. **The pixel-limit override is version-fragile.** `ip.max_pixels = …` works on transformers 5.0.0 (the Kaggle run) but is **silently ignored on ≥5.1x**. With it ignored, a 1060×57 line drops from 276 to 76 visual tokens (2 token rows). The enhanced notebook enforces the limits and asserts them.
4. **`repetition_penalty=1.2` penalises the correct answer.** It touches 14.5 % of all correct target tokens, on 91 % of lines. The most-affected tokens are ` the`, ` and`, `,`, `^`, which are in the prompt itself. In beam search it acts on log-probs (p → p^1.2), so it flips near-ties rather than confident tokens.
5. **Two scan batches.** A normal batch (≈1092×56 px, x-height ≈20 px) and a tall batch (≈4251×353 px, x-height ≈148 px) make up **25 %** of the images. The tall batch is **not** under-resolved after preprocessing. It *is* a different handwriting and language regime: 2× carets, 4× colon suspensions, 1.5× hapax words, and darker, stained parchment.
6. **No train/test/unlisted shift, no duplicate images.** An adversarial classifier gets AUC 0.49–0.53. Only one pHash pair is at Hamming distance ≤ 6. Fold-0 val→train nearest-neighbour distances match test→train (KS p = 0.76). **Random KFold is representative of the test split, not optimistic.**
7. **Label noise sets a floor.** Case is not predictable from word identity for 6.2 % of word tokens. 22 % of repeated legal-formula occurrences have a different surface form ("People"/"people"). The forum reports about 1.5 % mislabelled lines, which carry about 18 % of character edits *(forum)*.
8. **The notebook's ensemble code never ran.** The ROVER and medoid functions exist but are never called, and no `submission.csv` is written. The augmentations are far outside the measured archive statistics (rotation ±4° vs measured ±1°, noise σ 25.5 vs measured 2.4–3.0, JPEG q40–80 vs measured q85).

---

## 1. Metric audit

| Item | Finding |
|---|---|
| Official definition | Zindi: "weighted WER (0.5) + weighted CER (0.5); longer references weigh more; empty or missing predictions are penalised". Weighting a per-line rate by reference length turns it into a per-line **edit count**. |
| Normalisation | The leaderboard columns are mean edits per line, normalised by the benchmark's x_max: `score = 0.5(1 − WE/12) + 0.5(1 − CE/55)`. This reproduces the Benchmark row to 2e-10 and submissions to nine decimals *(forum thread 34861, credited to J0NNY)*. Public LB: top 0.9399, 50th 0.9269, Benchmark 0.0304. |
| Direction | **Higher is better** (a normalised accuracy). |
| Text handling | Assumed to match jiwer defaults: strip and collapse whitespace, case- and punctuation-sensitive, space counts as a character. 98 raw labels have leading or trailing spaces and 55 have double spaces; normalising both sides makes these irrelevant. |
| Starter pack | The data page lists a 28.5 KB `Starters.zip` "Starter OCR Pack", which needs a login. **It was not downloaded; please verify `work/road_metric.py` against its scorer if it contains one.** |
| Notebook metric | `comp_metric` = 0.5·corpus WER + 0.5·corpus CER. Per line this weights a word edit 0.5/11.28 and a char edit 0.5/62.2, a 5.5:1 ratio against the official 4.58:1. It is also not on the leaderboard scale. `_xscore` (never called) averages per-line *rates*, which over-weights short lines. |
| Implementation | `work/road_metric.py` and `work/road_eval.py`, embedded in Cell 7 of the notebook. Self-tests pass. |

Per-line loss = WE/24 + CE/110. 0.1 fewer word edits per line is worth +0.0042; 0.5 fewer char edits per line is worth +0.0045.

## 2. Validation audit

- **Fold 0 reproduced exactly:** 4,093 samples after 5 `CORRUPT_IDS`, fold sizes 819/819/819/818/818, label length p50/p95/p99/max = 62/85/92/108. All match the Kaggle log. IDs are in `work/fold_assignments.csv`.
- **Original protocol:** 819 val lines split into 328 early-stop lines (for `load_best_model_at_end` on `eval_loss`) and 491 scored lines. The scored set is 40 % smaller and checkpoint selection used validation labels.
- **Repaired protocol:** all 819 val lines are scored. Checkpoint selection uses an inner 8 % slice of Fold-0 **train** (262 lines, seed 1042). There is an assertion that no val ID is in train, the inner slice or the pseudo-labels.
- **Noise level:** the forum measured sd ≈ 0.0048 on about 820 lines. The enhanced harness reports a paired bootstrap CI and P(better) for every trial, so a change is promoted only when its interval excludes 0.
- **Fold leakage / optimism (secondary diagnostic):**
  - No image duplicates or re-crops were found: pHash ≤ 6 gives 1 pair; 32×320 thumbnail cosine similarity peaks at 0.941.
  - Near-duplicate *labels* (166 lines with a neighbour at ratio ≥ 90) are recurring legal formulae from different documents, not the same line twice.
  - Fold-0 val→train nearest-neighbour distance (mean 3.446) is indistinguishable from test→train (mean 3.441, KS p = 0.76).
  - **Conclusion: random KFold is not optimistic relative to the test set.** The primary benchmark is unchanged, and no group split is needed.

## 3. Dataset inventory

| Split | Images | Normal batch | Tall batch (H>150) |
|---|---:|---:|---:|
| Train | 4,098 | 3,055 | 1,043 (25.5 %) |
| Test | 1,374 | 1,029 | 345 (25.1 %) |
| Unlisted (in `images.zip`, in neither CSV) | 687 | 534 | 153 (22.3 %) |

- **All images:** RGB JPEG, 4:2:0, baseline encoding, no ICC profile, and luma quantisation ≈ q85 (14 images at q80). JPEG metadata therefore carries **no batch signal**.
- **Unlisted images:** exactly half the test count and statistically identical to train and test (adversarial AUC 0.49 vs train, 0.53 vs test). They are usable for automated self-training, which is allowed under thread 34459.
- **`CORRUPT_IDS` (checked visually):** 79tMUV… is blank; F8DYDD… and JU7lRw… are image/label mismatches. VmrEAL… is a 2-line crop labelled with the top line only. t0UrAS… is a 2-line crop whose label covers both lines, so it is a correct label. Excluding all five is kept, so the Fold-0 IDs stay stable.

## 4. Image forensics (all 6,159 images)

### 4.1 Geometry

| | 5 % | median | 95 % |
|---|---:|---:|---:|
| width px (all) | 937 | 1,118 | 4,901 |
| height px (all) | 43 | 64 | 405 |
| aspect (normal / tall) | — | 19.0 / 12.4 | — |
| x-height px, raw (normal / tall) | — | 20 / 148 | — |
| skew ° | −1.0 | 0.0 | +1.0 |
| slant (shear) | −0.20 | 0.45 | 0.80 |
| baseline slope ° | −0.78 | −0.06 | +0.58 |

- **Multi-line content:** the smoothed row profile shows ≥ 2 peaks in 28 % of normal and **64 % of tall** images, because neighbouring ascenders and descenders or whole neighbouring lines are visible. Only 8 images have aspect < 6.
- **Ink touching the top or bottom edge** is common (median 6–7 % of edge pixels), so crops are tight and clip neighbours.

### 4.2 Degradation

| | normal (median) | tall (median) |
|---|---:|---:|
| background grey | 206 | 172 (darker parchment) |
| parchment tint R−B | 51 | 68 |
| ink/paper contrast | 61 | 72 |
| Fisher separability | 1.08 | 1.73 |
| mid-tone ("bleed / faded") fraction | 3.8 % | 7.3 % (95th pct 20 %) |
| illumination CV | 0.016 | 0.047 |
| sensor noise σ (grey levels) | 2.4 | 3.0 |
| stroke width px | 2.7 | 3.8 |

**The augmentations do not match these statistics.** Rotation ±4° vs measured skew within ±1° for 95 % of lines. Added noise σ = 25.5 vs measured σ 1.8–4.2. JPEG q40–80 vs measured q85. 1-px erode/dilate at p=0.35 on strokes 2–4 px wide. `AUG_PROFILE="realistic"` in the notebook matches the magnitudes to the measured values.

### 4.3 Visual clusters

GMM BIC keeps decreasing up to k=12. The k=8 KMeans clusters on 42 handcrafted features split first by scan batch, then by tint, contrast and stroke width. Every cluster is present in train, test and unlisted in the same proportions. The clusters carry **document-type vocabulary** (log-odds of words):

- cluster 0: *manifest, ship, owners, france, capt:* (shipping protests)
- cluster 4: *adminstrs, guiftes, servantes, myselfe* (wills/estates)
- cluster 3: *s:^t, s:^d, adm:^rs, parrish* (the tall batch, abbreviation-heavy)

## 5. Label forensics (all 4,098 labels)

- **Size:** 255,049 chars; 46,209 word tokens; 6,750 types, 62 % of them hapax. Hapax tokens are 9.1 % of tokens.
- **Line length:** 36–120 chars (p50 62), 4–22 words (p50 11). There are no 1–2-word lines.
- **Character set:** 81 distinct characters, **all ASCII**. 78.2 % lower, 3.5 % upper, 16.5 % space, 1.4 % punctuation/markup, 0.35 % digits.
  - Rare characters (< 20 occurrences): `Y U + * | " Q \ # ?`
- **Markup:**

  | Mark | Lines | Occurrences | Examples |
  |---|---:|---:|---|
  | `^` superscript | 688 | 872 | W^m, y^e, M^r, Adm^rs, w^th, w^ch |
  | `:` suspension | 370 | 470 | Tho:, Exec:^rs, S:^d |
  | `&` | 562 | 757 | — |
  | ff-initial words | — | — | ffebruary, ffrancis |

  There are also `~`/`_` line fillers.
- **Formulae:**
  - Top 4-grams: "the Island of Barbados" (43), "my hand and seale" (34), "To have and to" (33).
  - 784 lines have ≥ 50 % of their words inside a 4-gram that occurs ≥ 3 times elsewhere.
  - 738 spelling-variant pairs at edit distance 1: said/saide/sayd, heires/heyres/heirs, this/theis.
- **Casing:**
  - 28.7 % of line-initial words are capitalised, and 18.5 % of mid-line words.
  - **6.2 % of word tokens differ from their type's majority casing.**
  - In the 75 formula 5-grams that occur ≥ 8 times, 22 % of occurrences use a minority surface form ("To all Christian People to" 12 vs "people" 29).
  - So casing is only predictable from the ink, not from a lexicon. This agrees with the forum finding that recasing loses.
- **Duplicates:** 23 rows share their exact text with another row, i.e. 13 surplus copies (4,085 unique texts). These are formula lines on different images. 44 near-duplicate groups at ratio ≥ 90, the largest with 21 members ("By this publique Act and Instrument of protest…").

## 6. Image ↔ label relationships

- **Length vs geometry is weak.** The best features (ink width in x-height units, aspect, sharpness) reach Spearman ρ ≈ 0.42–0.45 with character count. A Huber length model has median relative error 11–14 % (p90 27–32 %). The biggest outliers, checked visually, are correct labels on densely written lines.
  - **So a geometry-based length prior is weaker than the model's own length control.** The forum reports 96.5 % of predictions within 10 % of the label length. Branch J is rejected on this evidence.
  - **Update 2026-09-28.** A gradient-boosted model on *all* pixel features plus a thumbnail embedding predicts character count much better: cross-validated error 5.0 characters, Spearman **0.867** (`work/07_issue_taxonomy.py`). It is now used as a sanity filter for pseudo-labels (`PSEUDO_LABEL_METHOD.md`). Its value as a decoding prior is still unmeasured.
- **Batch ↔ language (new relationship):**

  | per line | normal | tall |
  |---|---:|---:|
  | carets | 0.17 | 0.35 |
  | colons | 0.06 | 0.27 |
  | uppercase | 3.3 % | 4.8 % |
  | hapax words | 8.5 % | 12.9 % |
  | 4-gram template coverage | 0.23 | 0.14 |
  | characters | 63.8 | 57.7 |

  The tall scan batch is a different set of hands and documents: later formal legal hands, more abbreviations, rarer words. That is a coherent explanation for the forum's result that it scores 0.823 vs 0.900 and that matched-resolution fixes do not help.
- **Visual cluster ↔ vocabulary:** see §4.3. Document type is visible in the paper and ink statistics.

### 6b. The data classes (added 2026-09-28)

**Two classes.** A Gaussian mixture on log(pixels per character) drops in BIC from 8,447 at k=1 to 2,399 at k=2, with means of 17 and 77 px/char. The split is exactly image height > 150 px.

| | Class A (≤150 px) | Class B (>150 px) |
|---|---|---|
| Train / test / unlisted share | 75 / 75 / 78 % | 25 / 25 / 22 % |
| Years written in labels | 1630s–1660s (117 mentions, 83 in the 1640s) | 1670s–1710s (28 of 30 mentions) |
| `ye`/`yt` · `sd` | 10.8 % · 4.7 % of lines | 0.3 % · 0.5 % |
| `^` · `:` · `ff-` | 14 % · 5 % · 0.6 % | 25 % · 17 % · 4.5 % |
| `heires` / `heyres` · `saide` | 8.2 / 0.1 % · 0 % | 3.0 / 3.5 % · 1.1 % |
| Validation words unseen in Fold-0 train | 9.8 % | 14.4 % |

So the two classes are two eras of scribes with different spelling conventions, not merely two scanners.

**Sub-groups inside class A.** A GMM on (scale, crop tightness) finds three. A pixel-only rule reproduces them with ~90 % agreement, and the notebook's detector matches the forensic rule on 100 % of the 4,098 training images.

| Group | Rule (pixels only) | Train lines | Character |
|---|---|---:|---|
| A1 | height ≤ 58 | 1,669 | tight crops; `&` 0.28/line, `ye` 0.18/line |
| A2 | taller, parchment grey ≤ 194 | 383 | dark parchment, larger hand, `&` 0.03/line, short lines |
| A3 | taller, parchment grey > 194 | 1,003 | loose crops (35 % show neighbouring lines), longest lines |
| B | height > 150 | 1,043 | see above |

**What this implies:**
- The model must read the era from the ink to pick the right conventions. A stronger language prior (bigger model, more epochs) pushes the other way.
- The groups are equally represented in train, Fold-0 val, test and unlisted.
- The notebook therefore:
  - scores every trial per class and group;
  - offers class-routed ensembling;
  - filters pseudo-labels per class;
  - makes two training options available: a class tag in the prompt and class-B oversampling.
- Full correlation table: `work/image_label_correlations.csv`. Length outliers: `work/length_outliers.csv`.

## 7. Processor geometry (Research Q3/Q4)

Computed exactly from the `load_image` ladder followed by `smart_resize`, for every image:

| | Qwen2.5-VL (28 px/token) | Qwen3-VL (32 px/token) |
|---|---|---|
| visual tokens, Kaggle run (MIN_PIXELS=200,704 honoured) | median 304 (5–95 %: 264–511), max 1,022 | median 236 (201–384) |
| token rows | 4 rows: 54 %, 5: 26 %, 6+: 19 % | 3: 21 %, 4: 52 %, 5+: 26 % |
| if the override is ignored (transformers ≥5.1x) | median 82, **2 token rows for 55 % of lines** | — |
| x-height after preprocessing, normal / tall | 42 / 70 px | — |
| px per character after preprocessing, normal / tall | 30.3 / 35.4 | — |

- **The ladder downscales 25 % of images, all from the tall batch**, by a median factor of 0.48. They still end up with **more** pixels and tokens per character than the normal batch. Long strips are not disproportionately downscaled.
- Tokens per character: 4.7 normal, 7.7 tall. That is roughly 15–20 visual tokens per text token, well inside the regime where vision-token compression does not limit OCR *(literature: DeepSeek-OCR)*.
- **Q17, horizontal tiling:** not justified at this resolution. Also negative in the literature for lines of this length.

## 8. Tokenizer forensics (all labels × 3 tokenizers)

- **Qwen2.5-VL-7B, Qwen3-VL-8B and Qwen3-VL-32B tokenize every label identically.** They share the Qwen2 BPE and differ only in special tokens, so switching between them adds no tokenizer diversity.
- **Per label:** mean 15.6 tokens (max 34); 3.98 characters per token; 1.37 tokens per word.
- **Where it fragments:**
  - Frequent words (≥ 20 occurrences): 1.14 tokens per word.
  - Hapax words: 2.27 tokens per word.
  - Caret words: **3.55** (` W` `^` `m`).
  - Colon words: 2.98.
  - Four-digit years: 5 (digits are split one by one).
- `^` becomes its own token 77 % of the time, and `:^` merges.
- Nothing is lossy: BPE can emit every character. The fragmentation raises the per-word decision count on exactly the abbreviation-heavy tall batch, which is a mild argument for a complementary character-level CTC model (§9, EXP_006). It is not an argument against the VLM.
- **Chat templates:**
  - Qwen2.5 adds the system prompt "You are a helpful assistant."; Qwen3 does not.
  - The assistant span trained is `LABEL<|im_end|>\n`. This was verified by decoding the unmasked labels in the dry run, with 0 masking failures.

## 9. Notebook audit (`barbados-2.ipynb`)

| Area | Finding | Status in `barbados-2-enhanced.ipynb` |
|---|---|---|
| Metric | Corpus WER/CER mix, not the official score | Official metric, WE/CE per line, bootstrap |
| Validation | 491/819 lines scored; 328 val lines used for checkpoint selection | All 819 scored; inner slice of train for selection; leak assertion |
| Pixel limits | Override applied only through `hasattr`; silently ignored on transformers ≥5.1x | Applied via `size=`, attributes and load kwargs, and asserted on a probe image |
| Decoding | `repetition_penalty=1.2` counts prompt tokens and legitimately repeated words; `no_repeat_ngram_size=6` makes 4 labels impossible to produce | D0 (original) vs D1 (off) vs D2 (beam-5, 5-best) on the same adapter |
| Confidence | Per-character confidence code (`_char_confidences`, `_clean_with_conf`, `_sequence_confidences`) is computed by `transcribe_paths`, which is **never called** | Replaced by the beam `sequences_scores` N-best (length normalisation undone) |
| Ensemble | `CFG.ENSEMBLE`, `_xalign`/`_xvote`/`_xscore` and the markdown's "Cell 11 medoid" / "soup" are **never executed**; no submission file is written | MBR (official cost) over pooled N-best, medoid, word-ROVER and char-ROVER, each scored as a trial; submission from the best validated configuration |
| TTA | `USE_TTA_RES` with `[1.0]` does nothing | Removed |
| Dead code | `_otsu`, `_blur`/`AUG_CBLUR`, `predict()`, `load_image_train/infer` aliases, `RUN_OOF_CV`, `BASE_MODEL_PATH`, `DEBUG` | Removed |
| Augmentation | One static augmented copy per line, magnitudes outside the measured data (§4.2) | Kept as the baseline; `AUG_PROFILE="realistic"` queued as EXP_002 |
| Train/infer consistency | Same `load_image` ladder in both | Kept |
| Adapter lookup (inference-only) | `_resolve_adapter` fallback could match the wrong model (`…8b_instruct_1` vs `…32b_instruct_1`) | Exact-name lookup, unit-tested |
| Batched beam search | — | Batched (left-padded, mixed sizes) N-best verified **identical** to batch-size-1 on both families |
| Kaggle version compatibility | — | Full CPU dry run on transformers **5.0.0 and 5.13**: fold0, inference-only, pseudo-label and full-refit modes all complete |

## 10. Error hypotheses to verify with the first GPU run

These cannot be measured without predictions. `work/analyze_trials.py` measures each one automatically.

1. Word edits dominate the loss (a word edit is 4.58×). Expect about 40–45 % of word substitutions to be 1–2-character misspellings, and casing to be about 13 % *(forum)*.
2. The tall batch carries about 42 % of char edits from about 27 % of lines *(forum)*. Its errors should concentrate on caret and colon abbreviations and hapax names.
3. About 2 % of lines (mislabelled) should carry about 18 % of loss. Identify them as lines where all systems agree with each other but disagree with the ground truth.
4. `&` over-emission (forum ratio 1.09) should shrink under D1, since ` and` is penalised by the prompt under D0.
5. The three Qwen models' errors are correlated (same data, same tokenizer), so expect ensemble gains at the low end, 5–15 % relative *(literature)*. The N-best oracle and the system oracle bound the headroom.
