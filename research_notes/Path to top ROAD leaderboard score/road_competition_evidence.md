# R.O.A.D. Barbados Historic Handwriting Challenge: measured evidence from participants (as of 28 Sep 2026)

Scope and method. I read all 24 discussion threads on the competition's discussion page, which has one page and runs from launch thread 33702 on 3 Jul to thread 34861. I read them in a browser with `document.body.innerText`. I also scraped the live leaderboard (pages 1–12, about 600 rows, including the hidden WER/CER columns), the Info/Rules and Data pages, every public GitHub repository returned by GitHub repo search for "barbados handwriting / htr / ocr / road barbados / zindi handwriting", and every Hugging Face model returned by `?search=barbados`, plus `mlai-dante/road-model-public` and the `ModarIbrahim/road-qwen25vl-*` repos. No login, no posting, no submissions.

Three kinds of numbers appear below:
- **Participant-reported numbers**: quoted from the source.
- **Numbers I verified or computed**: explicitly labelled "(my computation)".
- **Vague or unmeasured claims**: listed separately.

Thread URL pattern: `https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/<id>`

---

## Q1. What measured numbers and recipes have participants posted in the Zindi discussions?

### Takeaway
Only three threads contain real measured modelling numbers:
- **34861** (Damilare06): a dense negative-results post from 820 held-out lines.
- **34350** (dantebhai): "slightly above .91" with Qwen + LoRA + GRPO + beam search. No ablation was given.
- **33847** (J0NNY): the metric formula.

No one in the forum has disclosed a recipe that reaches 0.93 or higher. The two posters who did give recipes now sit at about 0.915 on the public leaderboard (Damilare06 #209 at 0.9162, dantebhai #222 at 0.9145). The loudest unmeasured claim, that image cleaning gives 0.96, comes from a user ranked #360 at 0.877.

### Cited Findings

**Thread 34861, "Five things that DON'T work (measured…)", Damilare06, 17 Sep 2026.** At posting the team was "around 0.906"; it is now #209, 0.916218, 13 submissions ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)). All numbers come from 820 held-out lines of a 5-fold split ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).

- **Noise floor:** sampling sd = **0.0048** on 820 lines, so differences below about 0.005 are "indistinguishable from luck". Tuning on a few hundred lines to chase 0.003 differences "cost us one bad decision" ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **Metric:** public = 0.5·(1 − word_edits_per_line/12) + 0.5·(1 − char_edits_per_line/55). It reproduces the Benchmark row to 2e-10 and their own submissions to nine decimals. One word edit costs as much as **4.58** character edits ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **(1) Recasing from a corpus lexicon: fails.**
  - Casing is about **13%** of their word errors, and a perfect casing oracle would be worth **+0.014**.
  - Lexicon recasing lost at every confidence threshold from 0.60 to 1.00, by **−0.016 to −0.0006**. Casing errors are symmetric: they write lowercase where the truth is uppercase about as often as the reverse ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **(2) Markup fixes (^ : &): negligible.**
  - The model emits *more* ampersands than the truth (ratio **1.09**).
  - Removing the &/and confusion entirely is worth **−0.0002**. A perfect caret would be worth **+0.0027** and a perfect colon **+0.0016**; both are inside the noise band.
  - All markup slips together are **3%** of word edits ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **(3) OOV snapping to the nearest training-vocabulary word (edit distance 1): harmful.**
  - Net **−0.0120**: it fixes 67 words and breaks 79.
  - About **43%** of wrong words have both prediction and truth in the training vocabulary, so an OOV filter cannot see them. The rare words it does reach are disproportionately names and places.
  - Confidence-gating "might rescue it" but was not tested ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **(4) Over-transcription guard: dead code for a well fine-tuned model.**
  - **0** held-out lines are more than 2× the label length, and **96.5%** are within 10% of it.
  - Only about **0.1%** of images have an aspect ratio below 6 ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **(5) Per-batch resolution normalisation: fails.**
  - The high-resolution batch (height >150 px) scores **0.823 vs 0.900** for the rest. It is about **27%** of lines but about **42%** of character edits.
  - Those crops get fewer output pixels per character (**35 vs 47**).
  - Cropping to the ink band brings them to parity on pixels per character but buys only **+0.0003**. At matched pixels per character the gap is still about **0.05** in every bucket, and the same at matched line length.
  - The team lost **0.009 on the LB** to an earlier band-crop preprocessing change ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **Label noise:**
  - About 1 line in 70 appears mislabelled. **12 of 820 (1.5%)** held-out lines carry **18%** of all their character edits.
  - Most of the 21 IDs posted in 33891 are *correct* centre-line labels on tall multi-line crops. "Don't drop all 21 blindly" ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- **Replies:**
  - hammertoe (17 Sep): best about **0.83** after "large models, small models, OCR-tuned models, generic vision models … ensembles". hammertoe is now #395 at 0.865512.
  - vaultguard (21 Sep): "focus on cleaning the images. ull get +0.9", with no numbers ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861); [LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Thread 34350, "OCR models, VLM, or custom?", zoro_x, 9 Aug 2026.** zoro_x asked what people above 0.91 use.
- dantebhai (11 Aug): "I am slightly above 91. I did qwen + lora + grpo and beam search. However, I think some of the data might be bad and I shouldn't use all the data."
- zoro_x (20 Aug) asked for the score without GRPO. There was no answer.
- vaultguard (21 Sep) claims "ull hit 0.96" if you clean "very dirty" images, with no measurement ([34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350)).
- Current positions: dantebhai **#222, 0.914454, 20 submissions, last about 2 months ago**; vaultguard **#360, 0.877189**; zoro_x #388, 0.867670 ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Thread 33847, "Help Please", Sadallah_Software, 12 Jul 2026.**
- J0NNY (13 Jul) posted the metric: WER_weighted = mean word-level Levenshtein distance per line, CER_weighted = mean char-level distance per line, score = 1 − 0.5·(WER_w/12 + CER_w/55), with rapidfuzz code ([33847](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)).
- J0NNY's team "Ynnoj" is #19 at 0.930777, 111 submissions ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).
- Sadallah (33702 reply, 12 Jul): LB WER = 2.89, CER = 6.82, while local jiwer on a held-out split gave WER about 0.27 and CER about 0.11, which shows rate-based local metrics do not match ([33702](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33702)).

**Thread 34304, mirrage, 5 Aug to 17 Sep.**
- mirrage says external same-century English data from Transkribus "jumped" a TrOCR model to "98+" (from "CER 75%"). This is unquantified and not used, since it is disallowed.
- mirrage relays: "maybe qwen and ensemble with GRPO will do same as Jhonny mentioned". That is a second-hand reference to J0NNY using Qwen + ensemble + GRPO; I found no post by J0NNY saying this in the visible threads ([34304](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34304)).
- mirrage is #387 at 0.867776 ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Thread 33891, Alpcan_Cepik, 14–18 Jul.**
- Alpcan found about 20 corrupted training rows in CV and posted **21 IDs**: 79tMUVyfIdy3GzkG, rh8o7bdCGOIBFPwH, F8DYDDp2AvW9Dytw, mfQxfOmeRmwBh0g8, EcxuqKeZl7OQexfB, PO7QQLWIFOT65BTz, yNyf3Tp0zc7DFj5F, JU7lRwk3jKkus24Z, R6iYPb7MHFiHtXH6, VmrEALeZiP1Y6nF9, t0UrASljcgzvBAnO, KH5g92Q3DA5Bo6xi, N78M3v6GKmUzF7sk, 0CCrVKAom8EK53jj, 3ZxOeKcOr5wUYyk0, PJbM7Q1SrblrSWt6, WwlTCykxjP3c4kfo, baTY3OlGskirWgFc, u3b4JNo5bqpfE7Js, MfT9S5oghk9ywNSC, 8H2ITJSWZhAD6eh0 ([33891](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)).
- MPWARE flagged three of them: 79tMUVyfIdy3GzkG is empty or unreadable; F8DYDDp2AvW9Dytw and JU7lRwk3jKkus24Z do not match their labels ([33891](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)).
- Current positions: Alpcan **#44, 0.927377, only 22 submissions, last about 2 months ago**; MPWARE **#18, 0.930920, 132 submissions** ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Thread 34089, youssefilo, 22 Jul.**
- Most multi-line crops are labelled with only the centre line, but some carry full multi-line text ([34089](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34089)).
- gilgamesh (21 Aug) listed examples: ptuXstzPGsZ5p9Wl, Xf53GrwovECETF4H, 259Ksw56mJJlnipt, 2KW2WCEcogSZEaxB ([34089](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34089)).
- Current positions: gilgamesh #63, 0.926462; youssefilo #328, 0.888615 ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Thread 34321 (mz000, 6 Aug)** asks about aspect-preserving resize and padding. The replies say it is "standard practice", with no organizer answer and no numbers ([34321](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34321)).

**Threads with no measured modelling content:**
- Rules and licence threads: 34734, 34712, 34612, 34514, 34481, 34468, 34459, 34367, 34294, 34110, 34053, 33930, 33870.
- Off-topic threads: 34692, 34337.
- Download help: 34194. Starting with `curl -C - -O -L https://storage.googleapis.com/road-handwriting/images.zip` works around the stalled download.
- Launch thread 33702: ashwinraju101 asked about GPU/VRAM limits for top-10 verification of 32B/72B models; the question is unanswered ([34194](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34194); [33702](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33702)).

### Inferences
- The forum holds **no disclosed recipe from anyone in the top 50**. Every disclosed recipe with numbers belongs to people now at 0.915 or below. The 0.93+ teams have not published methods.
- The pattern in 34861 (casing symmetric, markup negligible, OOV snapping harmful, tall-batch gap intrinsic) suggests the remaining 0.906→0.94 gap is **vision/decoder accuracy on ordinary in-vocabulary words**, not post-processing.
  - About 43% of wrong words are in-vocabulary on both sides.
  - At the top, the word term dominates the loss (see Q3).
- dantebhai's "Qwen + LoRA + GRPO + beam" gave only about 0.914 as of 2 months ago. GRPO alone is not shown to reach 0.93.

### Gaps
- No posted ablation for GRPO, beam width, ensembling or pseudo-labelling in the forum.
- J0NNY's alleged "Qwen + ensemble + GRPO" statement (relayed in 34304) could not be found in any visible thread; it may be deleted or in another channel.
- No organizer response on multi-line test labelling convention (34089) or on top-10 hardware (33702).

---

## Q2. What public code, models and notebooks exist, and what numbers do they report?

### Takeaway
No public artefact reports a score of **0.92 or higher**.
- **Best public, documented VLM result:** Qwen2.5-VL-7B LoRA r32 for 2 epochs at height 256, with val 0.900 and LB 0.904.
- **Qwen2.5-VL-32B LoRA r32:** val predictions score **0.9018** on 210 lines (my computation), with eval loss minimum at epoch 2 and overfitting after.
- **Best documented TrOCR/CRNN pipelines:** plateau at **0.846–0.870** on the LB.

Two gated Hugging Face repos show what 0.927–0.931-level teams trained: olmOCR-2, PaddleOCR-VL-1.6, Qwen3-VL-8B, Gemma, InternVL, 5-fold, GRPO, distill plus pseudo-labels. Attribution of the first repo to J0NNY's team (Ynnoj) is inferred and unconfirmed. The file contents are gated, so no numbers are available from them.

### Cited Findings

**VLM artefacts with numbers**

- **ModarIbrahim/road-qwen25vl-lora, -e1, -e3 (Hugging Face).** All three cards are identical:
  - Qwen/Qwen2.5-VL-7B-Instruct, LoRA r=32, α=64, dropout 0.05, all-linear, bf16.
  - Competition data only, **2 epochs**, image **height 256**, max_pixels 28·28·2048, min_pixels 256·28·28 in usage.
  - Held-out **WER 0.1551 / CER 0.0441 (rates) / score 0.9004**; **public LB 0.90413** ([e3](https://huggingface.co/ModarIbrahim/road-qwen25vl-lora-e3); [lora](https://huggingface.co/ModarIbrahim/road-qwen25vl-lora); [e1](https://huggingface.co/ModarIbrahim/road-qwen25vl-lora-e1)).
  - The -e3 name suggests 3 epochs, but its card says 2, so the card was likely copy-pasted.
  - A likely-same user, Zindi "Modar18224", is now **#207 at 0.916469, 37 submissions** ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).
- **EYEDOL/barbados-qwen_32b_hr_adapter (Hugging Face, public).**
  - Recipe: Qwen/Qwen2.5-VL-32B-Instruct, LoRA r=32, α=64, dropout 0.05, targets q/k/v/o/gate/up/down ([adapter_config](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter/blob/main/adapter_config.json)).
  - Trainer state: 6-epoch schedule stopped at epoch 5. Eval loss by epoch: **1: 0.450, 2: 0.416 (best), 3: 0.466, 4: 0.539, 5: 0.645**. It overfits after epoch 2. About 243 steps per epoch, eval on 210 samples ([trainer_state](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter/blob/main/last-checkpoint/trainer_state.json)).
  - **(My computation)** Its public `val_predictions_qwen_32b_hr.csv` (210 lines, ID/Target/Pred) scores **WER_w 1.729, CER_w 2.881, score 0.9018** under the Zindi formula ([file](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter/blob/main/val_predictions_qwen_32b_hr.csv)).
- **mlai-dante/road-model-public (Hugging Face).**
  - Recipe: Qwen/Qwen3-VL-8B-Instruct, LoRA **r=8, α=16**, dropout 0.05, all attention and MLP projections. **1 epoch**, lr 2e-4, batch 2 × grad-accum 4, warmup 0.03, min_pixels 100352, max_pixels 802816, val_size 0.15, max_new_tokens 128 ([run_manifest](https://huggingface.co/mlai-dante/road-model-public/blob/main/run_manifest.json); [adapter_config](https://huggingface.co/mlai-dante/road-model-public/blob/main/adapter_config.json)).
  - Reported val rates: **CER 0.0790, WER 0.2227**, "score" 0.1509 (their mean error rate, not the Zindi score) ([README](https://huggingface.co/mlai-dante/road-model-public)).
  - **(My computation)** Its `val_predictions.csv` (615 lines) scores **WER_w 2.517, CER_w 4.925, Zindi score 0.8503** ([file](https://huggingface.co/mlai-dante/road-model-public/blob/main/val_predictions.csv)).
  - The repo listing also shows about 77 directories named churro3b, qwen, gemma, mistral, llava and similar ([tree](https://huggingface.co/api/models/mlai-dante/road-model-public/tree/main)).
  - Possibly the same person as Zindi "dantebhai"; unverified.
- **ageraustine/OCR-ROAD-BARBADOS (GitHub, pushed 28 Sep 2026).**
  - Qwen3-VL-4B or 8B LoRA with asymmetric ranks: vision tower r=64/α=128, LM r=16/32, rsLoRA.
  - lr 5e-5, effective batch 16, max sequence 3584, early-stop patience 5.
  - **No scores reported** ([repo](https://github.com/ageraustine/OCR-ROAD-BARBADOS)).
- **keystats/historical_barbados_ocr_qwen3vl8b, _lora, barbados_handwriten_ocr(_v2) (Hugging Face).**
  - Full Qwen3-VL-8B weights (8.77B params, 4 shards). The cards are empty templates with no metrics ([API](https://huggingface.co/api/models/keystats/historical_barbados_ocr_qwen3vl8b); [lora card](https://huggingface.co/keystats/historical_barbados_ocr_lora)).
  - keystats and nymfree form team "Legends", **#33 at 0.928449, 148 submissions** ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Gated repos: file lists only, contents require approval**

- **vanijonny/*.** Repos: barbados_htr_model, barbados_htr_models, barbados-htr-sweeps, barbados-htr-newfold, rolling-ckpts, tests_md, PaddleOCR-VL-1.6-htr. All are gated "manual" ([author list](https://huggingface.co/api/models?author=vanijonny)).
  - Directory names show the experiments:
    - **olmOCR-2** at height 112 with LoRA r32/α128, r64/α256 and r128/α512, 5-fold, K=10 folds, 3 seeds of light augmentation, **beam4** test decoding.
    - **PaddleOCR-VL-1.6** (r128/α256, 3 or 6 epochs, "res300k"/"hires200k").
    - **Qwen3-VL-8B** h128 r64/α256, 5-fold, and **Qwen3-VL-32B** 3 folds.
    - **gemma26_a4b / gemma4_e4b** with greedy vs beam3 vs beam4 decode studies and a "g0gate".
    - **InternVL3.5-38B/8B** and **Qwen3.5-9B**.
    - **grpo_single_olmocr2 / grpo_stacked_olmocr2** (GRPO from an SFT init).
    - **distill / distill_pl** (distillation plus pseudo-labels), "deedmix", "ctcaux", "synth", "mcsample", HTR-VT and ViT-CTC sweeps, a 442 MB "rerank" model, "pseudo_una", and a MacBERTh orthography analysis CSV.
  - Sources: [tree](https://huggingface.co/api/models/vanijonny/barbados_htr_model/tree/main); [sweeps](https://huggingface.co/api/models/vanijonny/barbados-htr-sweeps/tree/main); [decode dir](https://huggingface.co/api/models/vanijonny/barbados_htr_model/tree/main/_research/decode); [models](https://huggingface.co/api/models/vanijonny/barbados_htr_models); [newfold](https://huggingface.co/api/models/vanijonny/barbados-htr-newfold); [rolling](https://huggingface.co/api/models/vanijonny/rolling-ckpts); [tests_md](https://huggingface.co/api/models/vanijonny/tests_md).
  - `result.json` files exist, for example `_research/decode/dq_e4b_decode_beam4/result.json`, but return "Access … is restricted".
- **CalebE/olmocr-barbados.** Gated "manual".
  - Directory names: **olmocr-5fold (+ preds_tta)**, **qwen3vl-5fold (+ preds_tta)**, granite-5fold, paddlevl-5fold, qwen25vl-5fold, qwen25-kf-mp512, olmocr-mp512, olmocr-e3, paddlevl-e3, trocr-5fold, gemma4-5fold, gemma4b-5fold, htrvt-judge(-full), and a **byt5-corrector** with a result.json ([API](https://huggingface.co/api/models/CalebE/olmocr-barbados)).
  - Probably Zindi "CalebEmelike", **#53 at 0.926927, 36 submissions**; unverified ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Non-VLM pipelines with LB numbers**

- **jdaltonll02/road_barbados_challenge (GitHub).** TrOCR pipeline, best **LB 0.863875** (WER_w 2.245, CER_w 4.684). That is a 27-model vote ensemble: 9 augmentations × 3 seeds, trocr-large-handwritten, full-data retrained ([README](https://github.com/jdaltonll02/road_barbados_challenge)). Measured steps:
  - Squash beat aspect-preserving padding "decisively". Foxing/mold patch-occlusion augmentation won among 12 augmentations.
  - **Base → large: 0.847 → 0.861** on LB, the biggest lever. Large needed lr 1e-5 rather than 3e-5.
  - Mixing base and large: 0.861 vs 0.861, a wash.
  - **Lexicon rescoring: 0.847 → 0.842** (harmful at ensemble scale).
  - **Multi-seed ×3: 0.861 → 0.864**. Fold-0 metric: 3.696 vs 3.924 for the best single seed.
  - **ROVER word voting at N=27:** WER 2.215 vs 2.245 but CER 4.769 vs 4.684, net worse (3.492 vs 3.465). LB change +0.0005.
  - **Diverse beam search:** worse at N=3 and N=9.
  - **Qwen2-VL-2B LoRA:** fold-0 final_score 5.055–5.312 vs TrOCR-large 4.0–4.1, where lower is better. It kept modernising spelling. The starter trainer uses `save_strategy="no"` and silently ships the last, not the best, checkpoint.
  - **Kraken CNN+BiLSTM+CTC:** 9.741 vs about 4.0.
  - **Masked-image-modelling encoder DAPT:** base improved 4.629 → 4.463 but LB 0.847 → 0.8477 (a wash); on large, 4.001 → 4.175 (worse).
  - Sources: [README](https://github.com/jdaltonll02/road_barbados_challenge); [WORKFLOW](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md).
- **AmanDeva/Barbados-Historic-Handwriting-Challenge (GitHub).** Best **LB 0.869784** (WER_w 2.19592, CER_w 4.25910): 5-fold TrOCR-large (15 epochs per fold, max_length 256, CIELAB preprocessing) plus MBR consensus. Full scorecard ([Context.md](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)):

  | Sub | Method | LB | Note |
  |---|---|---|---|
  | 1 | CRNN (ResNet34 + BiLSTM + CTC) | 0.807875 | |
  | 2 | 5-fold frame-level CTC probability averaging | **0.222338** | Failure: out-of-phase blanks |
  | 3 | Text-level voting | 0.819862 | |
  | 4 | Qwen2-VL-2B, LoRA r16, 3 epochs | 0.804273 | CER_w 6.93; expands abbreviations |
  | 6 | Qwen2-VL-2B, 8 epochs, r32/α64 | 0.810792 | |
  | 7 | TrOCR-large, greedy | **0.857697** | |
  | 8 | TrOCR-large, beam4 | 0.855388 | Beam slightly worse |
  | 9 | Single fold on 80% of data | 0.853193 | |
  | 10 | Sliding-window tiles | **−2.01327** | |
  | 11 | 50k synthetic-font pretraining | 0.826855 | Domain shift, worse |
  | 12 | 5-fold + MBR | 0.869784 | +0.0121 |

  The same file records "Rank #1 eszn: 0.92777385 (WER 1.297158, CER 1.999564)" at an unstated date, with the repo pushed 24 Aug ([Context.md](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)).
- **Alherma7/R.O.A.D.-Barbados-Historic-Handwriting (GitHub).** Final **LB 0.845943** (WER_w 2.590, CER_w 5.074): 3-seed residual CRNN + BiLSTM + CTC, beam 10, char 4-gram LM with α=0.35 ([repo](https://github.com/Alherma7/R.O.A.D.-Barbados-Historic-Handwriting)).
  - Single model LB 0.82659.
  - On a 120-row val set: ensemble +0.0145, beam + 4-gram LM +0.030, greedy→beam alone +0.001, elastic distortion +0.027, augmentation +0.049, residual blocks +0.029.
  - Failures:
    - TrOCR zero-shot 0.382.
    - TrOCR full fine-tune 0.122.
    - TrOCR LoRA 0.609.
    - PaddleOCR zero-shot 0.393.
    - Cosine LR −0.028.
    - Lexicon or confusion post-processing gave noise-level gains or corrupted proper nouns.
- **guelmbaye/barbados-htr (GitHub).** A TrOCR-large ×2 seeds + char 8-gram LM + gated lexicon design.
  - The vision half was "**never trained or benchmarked**". The gated-prior numbers are on simulated noise only: ungated post-processing "injected 0.84 char edits per line into already-correct text", and **11%** of held-out words are OOV but correct.
  - Its claim that the public score is a cross-participant normalisation, and that W_i = L_i^0.5 weighting applies, is **contradicted**: the per-line edit-count formula reproduces leaderboard rows to about 1e-9 (my verification, Q3) ([README](https://github.com/guelmbaye/barbados-htr)).
  - It recorded an earlier LB snapshot: eszn #1 (CER_w 1.950, WER_w 1.264), sdv #2 (1.930 / 1.279), MPWARE #3 (1.985 / 1.281) ([README](https://github.com/guelmbaye/barbados-htr)).
- **MusaNyaks/trocr-barbados-fold0–4 (Hugging Face).** trocr-base-handwritten, 15 epochs, lr 3e-5, effective batch 16. Val loss 0.858 (fold 0) and 0.887 (fold 4); no score ([fold0](https://huggingface.co/MusaNyaks/trocr-barbados-fold0); [fold4](https://huggingface.co/MusaNyaks/trocr-barbados-fold4)).
- **muhammadqasimshabbir3-art/Zindi-…-Corpus (GitHub).** External-corpus lexicon: 162 sources, about 2.2M lines. **95.8%** word-token coverage of train targets, mean 3–5-gram coverage **20.8%**. No LB effect reported ([repo](https://github.com/muhammadqasimshabbir3-art/Zindi-R.O.A.D.-Barbados-Historic-Handwriting-Challenge-Corpus)).
  - The author (Zindi MuhammadQasimShabbeer) is **#74 at 0.924890** ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).
- **Not relevant:** hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4 is a text-only newspaper DAPT, not HTR ([card](https://huggingface.co/hammertoe/Qwen3-Omni-30B-A3B-Barbados-LoRA-v4)).
- **No numbers:** other GitHub repos (derrick4411, danielmuthama23, cdjarabe07, l-munkanta, Ghorbanpoor, peter-njoro) are starter copies or empty ([GitHub search](https://api.github.com/search/repositories?q=barbados+handwriting)).
- **No results:** the woahwhattheheck/commons issue #16021 explicitly makes "No leaderboard claims" ([issue](https://github.com/woahwhattheheck/commons/issues/16021)).
- **Kaggle:** I found no Kaggle notebook for this competition ([search result set](https://www.kaggle.com/code/hayriyigit/handwriting-recognition) was generic).

### Inferences
- **Architecture:** every documented non-VLM pipeline (CRNN, TrOCR, Kraken) plateaus at 0.846–0.870 on the LB. Every documented 7B+ VLM LoRA reaches about 0.90 on validation or LB with a single model (Qwen2.5-VL-7B at 0.904 LB, Qwen2.5-VL-32B at 0.902 val). Tiny VLMs (Qwen2-VL-2B) sit at 0.80–0.81 because they normalise spelling.
- **Epoch budget:** EYEDOL's eval-loss curve (best at epoch 2, then rising) and Modar's 2-epoch card suggest **about 2 epochs** is the sweet spot for 7B–32B LoRA on 4k lines. This is loss-based evidence; jdalton notes that loss and the metric do not track each other closely.
- **What 0.927–0.931 teams appear to use:** judging from gated artefacts, they run large sweeps of document-OCR VLM families (olmOCR-2, PaddleOCR-VL, Qwen3-VL, Gemma, InternVL) with **5-fold or 10-fold CV, multiple seeds, beam-4 decoding, TTA, cross-model reranking or "judge" models, distillation plus pseudo-labels, and GRPO**. They also try low image heights (112–140 px).
- **Post-processing:** in the Barbados data it has repeatedly measured null or negative results at scale: lexicon rescoring −0.005, OOV snapping −0.012, recasing −0.0006 to −0.016.

### Gaps
- vanijonny/* and CalebE/* `result.json` and OOF files are gated, so there is no access to their measured numbers without requesting access, which I did not do.
- Attribution of vanijonny to J0NNY/Ynnoj (#19) and CalebE to CalebEmelike (#53) is inferred from names only.
- ModarIbrahim's later path from 0.904 to 0.916 is undocumented.
- No public artefact describes the #1–#15 methods.

---

## Q3. What does the current leaderboard look like?

### Takeaway
The top of the public LB is extremely compressed.
- #1 YoussefTrabelsi **0.939889** is 0.0043 clear of #2 (0.935549).
- #3 to #24 are packed between 0.9339 and 0.9302.
- 124 entries are at 0.92 or higher.

At the top, the word term is about 3× the character term in lost score, so reducing word edits per line is the main lever. The public split is only about 275–412 lines, so ranks 2–50 are within roughly 1–1.5 sampling sd of each other.

### Cited Findings

**Snapshot of 28 Sep 2026** from [the leaderboard](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard): public score, hidden WER_w/CER_w columns, and number of submissions.

| # | User | Public | WER_w | CER_w | Subs | Last |
|---|---|---|---|---|---|---|
| 1 | YoussefTrabelsi | 0.939888678 | 1.0943 | 1.5966 | 25 | 1 day |
| 2 | Brainiac | 0.935549361 | 1.1628 | 1.7602 | 81 | 1 day |
| 3 | PHs | 0.933948709 | 1.1873 | 1.8239 | 37 | 5 days |
| 4 | wahaym | 0.933267278 | 1.2052 | 1.8169 | 96 | ~21 h |
| 5 | Miscusi | 0.933087497 | 1.2056 | 1.8348 | 38 | ~4 h |
| 6 | Kiera | 0.932681981 | 1.1979 | 1.9144 | 25 | 6 days |
| 7 | eszn | 0.932322192 | 1.2160 | 1.8713 | 123 | 13 days |
| 8 | K63 | 0.932237963 | 1.2063 | 1.9249 | 101 | 20 days |
| 9 | Ecommer | 0.93212785 | 1.2085 | 1.9271 | 27 | 10 days |
| 10 | 4F (team) | 0.932086262 | 1.2088 | 1.9300 | 39 | 21 days |
| 11 | Konolavov | 0.931877391 | 1.2129 | 1.9344 | 85 | 21 days |
| 12 | Minchan | 0.931820198 | 1.2122 | 1.9438 | 62 | 22 days |
| 13 | GIrum | 0.931761376 | 1.2386 | 1.8295 | 45 | 2 days |
| 14 | hiennamkakabi | 0.931692302 | 1.2145 | 1.9472 | 85 | 22 days |
| 15 | hydan | 0.931590847 | 1.2163 | 1.9501 | 55 | 25 days |
| 16 | hhh022 | 0.931424263 | 1.2205 | 1.9492 | 73 | 24 days |
| 17 | Nailiw | 0.931255474 | 1.2233 | 1.9550 | 14 | 15 days |
| 18 | MPWARE | 0.930919808 | 1.2440 | 1.8973 | 132 | ~20 h |
| 19 | Ynnoj (team, J0NNY) | 0.930776868 | 1.2491 | 1.8894 | 111 | 5 days |
| 20 | MoonCake | 0.930669298 | 1.2328 | 1.9760 | 84 | 24 days |

Further rows from the same snapshot ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)):

| # | User | Public | WER_w | CER_w | Subs |
|---|---|---|---|---|---|
| 21 | 3man | 0.930381 | | | |
| 22 | AIMS bois | 0.930289 | | | |
| 23 | Arnaud_TOE | 0.930171 | | | 9 |
| 24 | mrSharafi | 0.930163 | | | |
| 25 | BigZ | 0.929985 | | | |
| 27 | sdv | 0.929175 | 1.2787 | 1.9302 | 99 (last ~1 month ago) |
| 33 | Legends | 0.928449 | | | |
| 44 | Alpcan_Cepik | 0.927377 | 1.2929 | 2.0629 | 22 |
| 50 | EGY_TEAM | 0.927013 | 1.3203 | 1.9773 | 169 |
| 100 | mocto | 0.922289 | | | |
| 150 | jessicaa | 0.918661 | | | |
| 250 | JacoCronje | 0.908740 | | | |
| 500 | leocd | 0.799302 | | | |

- **Benchmark:** 0.030428445 (WER_w 11.529, CER_w 53.812) ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).
- **Distribution** over about 600 listed rows: **24 at ≥0.93, 124 at ≥0.92, 244 at ≥0.91, 282 at ≥0.90**. The page header shows 1,894 joined and 671 active ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).
- **(My computation)** The formula 1 − 0.5·(WER_w/12 + CER_w/55) reproduces #1, #2, #13, #50 and the Benchmark to **≤1e-9**.
  - #1 loses 0.0456 from the word term and 0.0145 from the char term; #50 loses 0.0550 and 0.0180.
  - Cutting 0.1 word edits per line is worth **+0.0042**; cutting 0.1 char edits per line is worth **+0.0009** ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard); formula from [33847](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)).
- **Trajectory:**
  - eszn was #1 at 0.92777 (WER_w 1.297, CER_w 2.000) per AmanDeva's notes ([Context.md](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)).
  - Later, per guelmbaye's snapshot, eszn #1 had 1.264 / 1.950, which is 0.9296 by my calculation ([guelmbaye](https://github.com/guelmbaye/barbados-htr)).
  - eszn is now 0.93232 at #7, and the new #1 is 0.93989 ([LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

### Inferences
- **#1 vs #2:** #1 has about 0.068 fewer word edits and 0.164 fewer char edits per line than #2. **#1 vs #50:** about 0.226 fewer word edits and 0.381 fewer char edits per line.
  - Most of the 0.013 gap from #50 to #1 (about 73%) comes from the word term.
- **Low-submission leaders:** #1 (25 subs), #6 Kiera (25) and #9 Ecommer (27) reached the top with few submissions. That points to a strong offline CV loop and a model or recipe step-change rather than LB probing.
- **Shake-up risk:**
  - The public LB covers about 30% of 1,374 lines (≈412), or 20% (≈275) per the full rules.
  - Scaling Damilare's sd of 0.0048 at 820 lines gives an absolute-score sd of about 0.007–0.008 on the public split.
  - The whole #2–#50 band (0.9270–0.9355) is only about 1.2 sd wide. A private-LB shake-up among ranks 2–50 is likely.
  - #1's lead of 0.0043 over #2 is not decisive either. Paired differences between two similar models on the same lines have less noise than this.

### Gaps
- The hidden "Multi Score" column exists in the table header but is not populated.
- Dates of the eszn snapshots in the AmanDeva and guelmbaye repos are not stated.
- Entries beyond about row 600 were not scraped.

---

## Q4. What have the organizers said about the test set, label noise, scoring and allowed techniques?

### Takeaway
Organizers (meganomaly, Zindi staff) have ruled on:
- **Transductive pseudo-labelling:** allowed if fully automated and reproducible.
- **StackMix-style recombination of training crops:** allowed.
- **Excluding clearly corrupted training rows:** allowed, with documentation.
- **Competition data only** for any training or adaptation.
- **Commercial-licence models only:** CHURRO-3B rejected, PP-OCRv6 medium approved.
- **Rented GPUs:** fine.

They have **not** ruled on the multi-line labelling convention or on hardware limits for top-10 verification. The rules text contradicts itself on the public/private split (30/70 vs 20/80).

### Cited Findings

**Scoring**
- The Info page says longer references are weighted more heavily, and that missing, empty or invalid predictions are penalised as incorrect ([Info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)).
- The leaderboard WER/CER are mean edit counts per line, and the metric is reproducible exactly ([33847](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847); [34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- The starter `eval_metrics.py` imports `wer`/`cer` modules from a missing "evaluations" folder ([33930](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33930)).

**Split and selection**
- The rules summary says the public LB is "approximately **30%**" and private 70%. The full rules say public "approximately **20%**" and private 80% ([Info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)).
- You choose 2 submissions for private scoring; otherwise the 2 best public submissions are used.
- Limits: 5 submissions per day, 200 in total.
- The top 10 must submit code within 48 hours. If the code does not reproduce, rank is adjusted, so seeds must be set ([Info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)).

**Data**
- About 6K cropped line images from 18th–19th-century records. Images can contain one or more handwritten words ([Data](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/data)).
- Images: `https://storage.googleapis.com/road-handwriting/images.zip` ([Data](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/data)).

**Pseudo-labelling (17 Aug)**
- Transductive pseudo-labelling on test images is allowed if fully automated. You may fold all or part of the test predictions back into training.
- Automated, generalisable post-processing rules are allowed, but no hard-coded per-image fixes.
- The full pipeline must be documented and reproducible ([34459](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459)).

**Training data (16 Jul)**
- Excluding clearly corrupted training rows is allowed if the original data stays unchanged and the rows and reasons are documented. The decision must use no test data or external labels.
- Optuna is allowed ([33891](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)).
- A participant reply in 34459 claims manual relabelling of train is not allowed, and the rules forbid manual labelling generally ([34459](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459); [Info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)).

**StackMix augmentation (24 Aug)**
- Segmenting character crops from Train images and recombining them into new labelled lines is permitted if automated and reproducible, using no test or external images and no manually annotated crops ([34514](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34514)).

**External data and pretrained models (11 Sep, the definitive ruling)**
- Competition data only may be used to train, fine-tune, pseudo-label "or otherwise adapt" a model.
- Pretrained base models must carry a licence permitting commercial use, modification and deployment by the host. Research-only or non-commercial weights are banned.
- The upstream training-data licences of a model need not be verified unless the model licence carries them through.
- All weights and inference code must be provided ([34734](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734)).

**Specific model rulings**
- **stanford-oval/churro-3B** (Qwen Research License) was **not permitted** (24 Aug) ([34481](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481); [34468](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34468)).
- **PP-OCRv6 medium** (Apache-2.0) was **approved** (28 Aug) ([34481](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481)).
- The TrOCR licence question (34712, 8 Sep) has no direct answer ([34712](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34712)).

**Compute (16 Jul)**
- Renting GPUs (RunPod, AWS, Nebius) is permitted if data stays private and no proprietary model API is used ([33870](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33870)).

**Multi-line labels (5 Aug)**
- Staff asked for example IDs and promised investigation. No follow-up ruling is visible ([34089](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34089)).

### Inferences
- The 11 Sep ruling ("competition data only … or otherwise adapt") very likely bars external-corpus lexicons or LMs, such as the 2.2M-line corpus above, for post-correction or pseudo-labelling. It also removes non-commercial checkpoints (CHURRO) from any final ensemble.
- Under 34734, TrOCR (MIT weights) is probably acceptable, because upstream-data provenance need not be checked unless the model licence carries it. This is not an explicit ruling.
- Pseudo-labelling on the 1,374 test lines plus the 687 unlisted images is explicitly legal and is being used by at least one top-20-plausible team (vanijonny "distill_pl"; inferred). It is a sanctioned lever.
- Because the public share is small, the final rank will depend heavily on robust choice of the 2 selected submissions (CV-driven rather than LB-driven).

### Gaps
- No organizer statement on the fraction of noisy labels in test, or on how multi-line test crops are labelled.
- No clarification of the 20% vs 30% public-split discrepancy.
- No stated hardware or VRAM budget for top-10 reproduction (question in 33702 unanswered).

---

## Q5. Which techniques go with 0.93–0.94, and which were measured to fail?

### Takeaway
There is **no public, measured recipe at 0.93 or higher**.

**Associated with the ~0.90–0.93 band (public evidence):**
- 7B–32B document/VLM backbones fine-tuned with LoRA for about 2 epochs, giving about 0.90 single-model.
- Plus k-fold, multi-seed ensembles and beam search.
- At the 0.927–0.931 level, olmOCR-2, PaddleOCR-VL, Qwen3-VL and Gemma families with low image heights, GRPO, reranking/judge models and distill + pseudo-label loops (gated artefacts, inferred attribution).

**Measured to fail or to be noise-level on this data:**
- **Casing/lexicon/OOV post-processing:** recasing −0.0006 to −0.016; OOV snapping −0.012; lexicon rescoring 0.847→0.842.
- **Preprocessing:** markup fixes within noise; ink-band cropping +0.0003 locally and −0.009 on the LB.
- **Decoding and ensembling tricks:** frame-level CTC averaging (0.22); diverse beam; ROVER at scale; beam vs greedy for TrOCR (−0.002).
- **Pretraining:** synthetic-font pretraining (−0.031); encoder MIM-DAPT at ensemble scale (0); TrOCR full fine-tune (0.12).
- **Too-small models:** 2B VLMs (0.80–0.81).

### Cited Findings

**Single-model VLM results**
- Qwen2.5-VL-7B LoRA r32, 2 epochs, h256: val 0.9004, **LB 0.90413** ([HF](https://huggingface.co/ModarIbrahim/road-qwen25vl-lora-e3)).
- Qwen2.5-VL-32B LoRA r32: val 0.9018 on 210 lines (my computation). Eval loss is best at epoch 2 of 5 ([HF](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter)).
- Qwen3-VL-8B LoRA r8, 1 epoch: 0.8503 (my computation) ([HF](https://huggingface.co/mlai-dante/road-model-public)).
- Qwen2-VL-2B: 0.804–0.811 LB ([AmanDeva](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)), and worse than TrOCR-large ([jdalton](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)).

**Ensembling gains measured on this data**
- k-fold + MBR: +0.0121 LB ([AmanDeva](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)).
- 3 seeds: +0.003 LB ([jdalton](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)).
- 3-seed CRNN: +0.0145 val ([Alherma7](https://github.com/Alherma7/R.O.A.D.-Barbados-Historic-Handwriting)).

**Qwen + LoRA + GRPO + beam (dantebhai):** "slightly above .91", no ablation, now 0.9145 LB ([34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350); [LB](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)).

**Negative results**
- All five items in 34861 ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).
- jdalton phases 7, 10 and 11 ([WORKFLOW](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)).
- AmanDeva subs 2, 8, 10 and 11 ([Context.md](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)).
- Alherma7's list of TrOCR and PaddleOCR zero-shot and fine-tune failures ([repo](https://github.com/Alherma7/R.O.A.D.-Barbados-Historic-Handwriting)).

**Unmeasured claims**
- "clean the images → 0.96" (vaultguard, LB 0.877) ([34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350)).
- "some data might be bad" (dantebhai) ([34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350)).
- CIELAB ink isolation, deskew and morphological "stroke healing" was used in AmanDeva's 0.8698 pipeline, but its ablation is confounded with the TrOCR max_length=256 fix ([Context.md](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)).

### Inferences
- **Starting point:** a single well-tuned 7B–32B VLM LoRA at about 0.90 is the documented floor.
- **The ~0.03–0.04 gap to 0.93–0.94** most plausibly comes from:
  - (a) stronger document-OCR backbones, with olmOCR-2 and PaddleOCR-VL prominent in top-team artefacts;
  - (b) k-fold × multi-seed ensembling with candidate-level selection (MBR, reranker or judge) rather than character-level averaging;
  - (c) test-time pseudo-labelling or distillation;
  - (d) possibly GRPO on an edit-distance reward.
- None of (a)–(d) has a published ablation for this competition. Treat them as **hypotheses consistent with the artefacts, not measured facts**.
- **Where the gain must come from:** the word term dominates the loss at the top, and 43% of wrong words are in-vocabulary on both sides. Gains must come from better reading of ordinary words (vision/decoder quality, ensembling), not from dictionaries.
- **Validation discipline** (from 34861 and jdalton): use at least 800 lines of CV, and validate ensemble changes at deployment scale. Several tricks that won at N=3 or on a single model reversed at N=27 or on the LB.

### Gaps
- No measured delta for GRPO, pseudo-labelling, TTA, image height, or specific backbones (olmOCR-2, PaddleOCR-VL-1.6, Gemma) on this competition's data is publicly available. The artefacts that would show them are gated.
- No top-10 participant has posted any method.
