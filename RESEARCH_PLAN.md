# RESEARCH_PLAN: fan-in, ranked experiment queue, Fold-0 protocol

**Benchmark:** Fold 0 of `KFold(5, shuffle, 42)` over the 4,093 usable lines. Validation is all 819 lines. Checkpoint selection uses an inner 8 % slice of Fold-0 train. The metric is the official leaderboard score (higher is better). Every trial must beat the champion on a paired bootstrap before it is promoted.
**Evidence base:** `FULL_DATA_AUDIT.md` (full-data measurements), four independent literature branches (A+O dedicated HTR, B+L+M VLMs and fine-tuning, C/D/E/F/J preprocessing and augmentation, H/I/K/O/P/Q decoding, ensembling and self-training), and the competition forum.
**Deadline:** the competition closes **4 Oct 2026**. The queue is ordered so each Kaggle session answers the most questions per GPU-hour.

> **No trial below has been run.** This environment has no GPU. EXP_001 is **READY_FOR_GPU**: `barbados-2-enhanced.ipynb` runs it with *Run All* and needs no code edits. It was dry-run end to end on CPU with tiny random-weight Qwen2.5-VL and Qwen3-VL models on transformers 5.0.0 and 5.13.

---

## 1. Fan-in: surviving hypotheses (scored 0–5)

Columns: Data = data evidence, Nov = novelty, ΔCER / ΔWER = expected improvement, HTR = historical-HTR relevance, Comp = complementarity, Feas = compute feasibility, Risk = implementation safety (5 = safe).

| ID | Hypothesis | Data | Nov | ΔCER | ΔWER | HTR | Comp | Feas | Risk | Σ | Tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H05 | Cross-model **MBR over pooled N-best** under the official cost | 4 | 4 | 3 | 4 | 4 | 5 | 5 | 5 | **34** | 1 |
| H02 | **repetition_penalty 1.2 → 1.0**, no n-gram block | 5 | 4 | 2 | 3 | 3 | 3 | 5 | 5 | **30** | 1 |
| H04 | Single-model MBR over beam-5 N-best | 3 | 3 | 2 | 3 | 3 | 3 | 5 | 5 | 27 | 1 |
| H11 | **Pseudo-label test + 687 unlisted** (MBR-risk filtered) | 4 | 3 | 3 | 3 | 4 | 3 | 3 | 4 | 27 | 3 |
| H07 | Realistic augmentation (magnitudes matched to measured stats) | 5 | 3 | 3 | 3 | 4 | 2 | 4 | 3 | 27 | 2 |
| H06 | Word / char ROVER, medoid (confidence vs plain) | 3 | 2 | 2 | 3 | 4 | 4 | 5 | 4 | 27 | 1 |
| H08 | 2 epochs (vs 1) | 3 | 1 | 3 | 3 | 4 | 1 | 4 | 4 | 23 | 2 |
| H10 | Full-data refit (100 % of lines) for the submission | 4 | 0 | 2 | 2 | 3 | 1 | 4 | 5 | 21 | 2 |
| H12 | PP-OCRv6 (kraken, Apache-2.0, staff-approved) CTC as ensemble member / N-best rescorer | 3 | 5 | 2 | 2 | 5 | 5 | 3 | 2 | 27 | 2–3 |
| H13 | GRPO with the official line score as reward (ReMax baseline) | 2 | 4 | 2 | 3 | 3 | 2 | 2 | 1 | 19 | 3 |
| H14 | Multi-seed souping / seed ensemble of the champion | 2 | 2 | 2 | 2 | 3 | 3 | 2 | 4 | 20 | 3 |
| H15 | Epsilon-sampling candidates (ε=0.02, 8 per model) added to the MBR pool | 3 | 3 | 2 | 2 | 3 | 3 | 4 | 4 | 24 | 1 |
| H18 | Drop cross-fitted label-noise rows (all systems agree, GT disagrees) | 4 | 2 | 2 | 2 | 3 | 1 | 3 | 3 | 20 | 2 |
| H17 | Vision-tower LoRA LR at 0.1–0.5× language LR | 1 | 2 | 2 | 1 | 2 | 1 | 4 | 4 | 17 | 2 |

**How to read the expected gains.** Going from 0.906 to 0.94 means cutting 36 % of edits.

| Hypothesis | Expected Δ | Basis |
|---|---|---|
| H05 | +0.004 to +0.012 | MBR in ASR gives 5–19 % relative; diverse-OCR voting gives 24–45 %, but our three Qwens are correlated |
| H02 | 0 to +0.004 | Beam search applies the penalty to log-probs (p→p^1.2), so it only flips near-ties |
| H11 | +0.002 to +0.008 | The unlabelled pool is 0.5× the labelled set |
| H07 | 0 to +0.005 | Gwalther 16th-c. Latin ablation: aggressive augmentations cost up to +20 % CER |

All of these are **hypotheses until EXP_001 returns**.

## 2. Rejected with evidence (not run)

| Branch | Idea | Why rejected |
|---|---|---|
| J | Geometry length prior | Full data: best geometric feature predicts character count only at ρ≈0.42–0.45 (median error 11–14 %). The forum reports 96.5 % of predictions already within 10 % of label length. Keep it only as a possible MBR tie-break. |
| D | Horizontal tiling / multi-crop | After preprocessing: 4.7–7.7 visual tokens per character and 30–35 px per character. Literature gains appear only for text longer than the training width. Stitch points cost word edits. |
| C | Width-anchored or fixed-height resize for the tall batch | Measured: the tall batch already gets **more** px/char (35.4 vs 30.3) and tokens/char (7.7 vs 4.7). The forum's matched-resolution and ink-band crop tests gave +0.0003 and −0.009 LB. |
| H | Character/word n-gram LM rescoring on training text | The autoregressive decoder already models the same text; 62 % of word types are hapax; forum OOV snapping gave −0.012. At most a 0.1–0.3 feature inside MBR. |
| — | Lexicon recasing | 6.2 % of tokens and 22 % of formula occurrences break the majority casing (measured); forum: loses at every threshold. |
| F | Curriculum ordering | No measured gain when fine-tuning pretrained VLMs; hard-example upweighting would amplify the ~1.5 % mislabelled lines. |
| K | Prompt sweeps | After LoRA the prompt must match training; wording effects are second-order. Kept fixed so trials stay comparable. |
| B | Switching to Qwen3.5/3.6/3.8-27B, Gemma 4, InternVL3.5 | Qwen3-VL-8B already has the best open aggregate on handwriting benchmarks (OmniHandwritingOCR 2026). New hybrid architectures need unreleased transformers builds, too risky with 7 days left. CHURRO-3B, Qwen2.5-VL-3B and Nanonets-OCR2-3B are licence-blocked. |
| — | Tokenizer-motivated model swap | All three Qwens tokenize every label identically (measured). |
| — | Group-aware CV | No duplicate or re-crop images; val→train distance equals test→train distance (KS p=0.76). Random KFold is representative. |
| N | Tall-batch specialist | 1,043 lines. The difficulty is language regime plus stains, not resolution. A shared model sees both regimes. |

## 2b. Update 2026-09-28: user results and the class finding

**User results:** larger models and more epochs both scored **worse**. That is consistent with the data (audit §6b):
- noisy casing labels;
- 66 % of validation lines contain an unseen word;
- two eras with different spelling conventions;
- a static augmented copy repeated every epoch;
- one learning rate (2e-4) used for every model size.

**The notebook now defaults to EXP_002:**
- realistic augmentation, parchment-coloured fill, fresh augmentation per visit;
- 32B learning rate 1e-4;
- per-class and per-group scoring;
- a class-routed ensemble (honest split-half estimate), with `routing_map.json` written for the full refit.

**Class-aware training options** are off by default and are tested one at a time:
- EXP_003: class tag in the prompt;
- EXP_004: class-B oversampling;
- EXP_005: per-class pseudo-labels.

## 2c. Update: specialists per class and sub-class

The notebook now supports every option through CFG. The full evidence is in `reports/Per class specialist HTR models.md`.

| Option | CFG | Research expectation |
|---|---|---|
| One model | `SPECIALISTS = {}` | baseline |
| One model + class hint (+ balanced sampling) | `CLASS_HINT=True`, `CLASS_SAMPLING_T=2` | do first (E1) |
| Shared + B specialist (continued) | `SPECIALISTS={"B": {}}` | small gain possible (E2) |
| Shared + specialist per class | `SPECIALISTS={"A": {}, "B": {}}` | A ≈ 0 |
| Shared + specialist per sub-class | `SPECIALISTS={"A1": {}, "A2": {}, "A3": {}, "B": {}}`, `ROUTING_LEVEL="group"` | no gain expected; A2 cannot be verified on 75 lines |
| Fully separate model per class | `{"init": "separate", "replay": 0}` | expected loss |
| Specialists on existing shared adapters | `DO_TRAIN=False`, `ADAPTERS_DIR=...`, `TRAIN_SPECIALISTS=True` | fast path |

**How a class decides between the champion and a specialist.** Routing replaces plain per-class argmax with the acceptance rule. A class switches only if all four hold:
- P(better) ≥ 0.9;
- the empirical-Bayes-shrunk gain is > 0;
- the sign repeats in split-halves;
- π·Δ ≥ 0.002.

The procedure is capped at ≤ 3 challengers per class, and its honest score is estimated from 30 repeated split-halves. `routing_map.json` then drives the `RUN_MODE="full"` submission.

## 3. Ranked experiment queue

| Trial | One controlled change | Models | Tier | Needs | Status |
|---|---|---|---|---|---|
| **EXP_001** | Corrected protocol + original recipe (baseline D0) · D1 no repetition penalty · D2 beam-5 5-best · D2+single-model MBR · cross-model MBR / medoid / word-ROVER / char-ROVER | 8B, 7B, 32B | 1 | — | **READY_FOR_GPU** |
| EXP_002 | `AUG_PROFILE="realistic"` (rotation ±1°, noise σ≈3, JPEG q70–92, stroke p 0.2) | 8B | 2 | EXP_001 champion decoding | PLANNED |
| EXP_003 | `NUM_TRAIN_EPOCHS=2` | 8B | 2 | EXP_002 verdict (keep the winning augmentation) | PLANNED |
| EXP_004 | `PSEUDO_LABEL_CSV=<EXP_001 pseudo_labels.csv>`, `PSEUDO_KEEP_FRAC=0.7` | 8B (then all) | 3 | EXP_001 outputs | PLANNED |
| EXP_005 | Epsilon-sampling candidates in the MBR pool (add variant `do_sample`, ε=0.02, 8 samples) | champion set | 1 | EXP_001 | PLANNED |
| EXP_006 | PP-OCRv6-medium (kraken ≥7.1, `ketos train --arch ppocrv6 --load <zenodo 21788410>`, 128 px lines, `-q fixed -N 30`) → CTC log-likelihood of each Qwen N-best candidate as an MBR weight | CTC + Qwens | 2–3 | Kaggle env (conflicting pins, see audit) | PLANNED |
| FINAL | `RUN_MODE="full"`, `CHAMPION_DECODE` / `CHAMPION_ENSEMBLE` = the promoted configuration, all promoted training changes | champion set | 2 | ≥ EXP_001 | PLANNED |

### EXP_001: exactly what runs (default *Run All* of `barbados-2-enhanced.ipynb`)

- **Train:** each of Qwen3-VL-8B, Qwen2.5-VL-7B and Qwen3-VL-32B is trained on 3,012 Fold-0 train lines plus 1 augmented copy each. Hyperparameters are the original ones: LoRA r32/α64 including the vision tower, LR 2e-4, cosine schedule, 1 epoch, batch 4. Checkpoint selection is best `eval_loss` on the 262-line inner slice.
- **Decode:** all **819** val lines with D0, D1 and D2. D2 also yields the single-model MBR row. Then test with each model's best variant plus D2, and unlisted with D2.
- **Log:** the harness writes about 22 trial rows to `trial_results.csv`:
  - 4 per model (D0, D1, D2, D2+MBR);
  - cross-model ensembles for each variant (medoid, word-ROVER, char-ROVER, plus MBR-N-best for D2).
  - Each row has the score, WE/CE per line, paired bootstrap vs D0, runtime and peak GPU.
- **Write:** `submission.csv` from the best validated configuration, and `pseudo_labels.csv` (test + unlisted, with MBR risk) for EXP_004.
- **Questions answered:** Q1, Q12, Q13, Q14 (by batch, cluster and length), Q16 (32B vs 8B), Q18 (inner-loss checkpoint vs final, from `log_history`), Q20 (confidence-weighted MBR vs plain medoid/ROVER), Q9 (system oracle and correlation). Q2/Q3/Q4/Q7/Q10/Q11/Q17 are already answered from the data (audit §2, §6–8).
- **Runtime estimate:** 2.5–4.5 h on the RTX PRO 6000. The 32B model dominates. `TIME_BUDGET_H=11` skips remaining models rather than losing outputs.
- **If GPU time is short:** set `MODEL_PATHS` to the 8B only (~1 h). That still answers D0/D1/D2/MBR. Or use the **fast path**: `DO_TRAIN=False` with `ADAPTERS_DIR` pointing at the adapters from the earlier `barbados-2` run. The decoding deltas stay valid because both arms share the adapter. Absolute scores on the 328 former early-stop lines are then slightly optimistic.

### After each session (Phase 20 loop)

1. Download `/kaggle/working/trial_results.csv`, `trial_predictions/`, `trial_configs/`, `log_history__*.csv` and `pseudo_labels.csv`.
2. Run `python work/analyze_trials.py <downloaded_dir>`. It writes `TRIAL_ANALYSIS.md` with:
   - the trial table and controlled paired deltas;
   - the loss share by scan batch, visual cluster, row peaks, contrast, sharpness, length, template-ness, caret, digit and hapax;
   - word/char confusion tables and the `&`/`^`/`:` emission ratios;
   - confidence calibration;
   - the inter-system loss correlation, system oracle and N-best oracle;
   - label-noise suspects.
3. **Promote or reject:** promote only if P(better) ≥ 0.9 **and** the CI95 lower bound is > −0.001. With the 0.0048 sd, smaller "wins" are noise.
4. **Choose the next fan-out from the dominant remaining error mode:**

   | If this dominates the remaining loss | Try next |
   |---|---|
   | Word substitutions of 1–2 chars | Pooled MBR + sampling (EXP_005) |
   | The tall batch or abbreviation words | Realistic augmentation (EXP_002) and a CTC complement (EXP_006) |
   | Lines where all systems agree but the ground truth differs | EXP_004 pseudo-labels, plus H18 |
   | An N-best oracle far above top-1 | Better candidate scoring: teacher-forced rescoring with another model, CTC scores |

## 4. Answers to the Phase-21 questions so far

| # | Question | Answer (evidence) |
|---|---|---|
| 1 | Is the local metric the competition metric? | **No.** Replaced by the official per-line form (audit §1). |
| 2 | Is random KFold optimistic? | **No measurable optimism** vs test (KS p=0.76, no duplicates). |
| 3 | Does resizing destroy glyph resolution for long strips? | **No** on transformers 5.0.0: the tall batch gets more px/char. **Yes** on transformers ≥5.1x without the fix (2 token rows). |
| 4 | Does visual token count vary badly with aspect? | Tokens grow with the tall batch's resolution (median 438 vs 296). Normal lines get 4 token rows (Qwen2.5). |
| 5 | Fixed-height vs thumbnail? | Deprioritised by measurement and forum evidence (see §2). |
| 6 | Which augmentations match the degradation? | Photometric tint/illumination/bleed are in range. Rotation, noise and JPEG are far out of range (audit §4.2) → EXP_002. |
| 7 | Is BPE poor for historical symbols? | Caret words fragment 3.5× but it is lossless. Identical across models, so no diversity. |
| 8 | Does a dedicated HTR model beat the VLMs? | Literature: unlikely standalone (1.2–2× the VLM's CER). EXP_006 tests it as a complement. |
| 9 | Is HTR complementary? | Literature: CTC errors are glyph-local, VLM errors are fluent substitutions. Measure in EXP_006. |
| 10 | Can geometry predict length for decoding? | **Only weakly** (ρ≈0.42). Rejected as a standalone prior. |
| 11 | Can a training-text LM reduce WER without raising CER? | Not expected (§2). Not queued. |
| 12–14 | Dominant characters, words and clusters | Computed automatically from EXP_001 (`analyze_trials.py`). |
| 15 | Does vision LoRA help? | Not isolated yet. Candidate for after EXP_003 (H17). |
| 16 | Does 32B beat 8B? | EXP_001 trains both under an identical protocol. |
| 17 | Horizontal tiling? | Not needed at the measured resolution. |
| 18 | Is `eval_loss` checkpoint selection aligned? | EXP_001 logs `log_history`. EXP_00x can compare `CKPT_SELECTION="last"`. |
| 19 | Are 1 epoch and LR 2e-4 right? | EXP_003 (epochs). LR is left at the literature-typical 1e-4 to 2e-4. |
| 20 | Is confidence-weighted voting better than plain? | EXP_001 scores posterior-weighted MBR vs medoid vs ROVER on the same predictions. |
