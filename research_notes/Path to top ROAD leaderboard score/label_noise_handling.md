# Detecting and Handling Mislabelled Training Lines in HTR / OCR / Seq2seq Training (Measured Effects)

Scope note: the effect sizes below are measured numbers from papers, each given with its dataset, noise rate and baseline -> result. Recommendations and the Kaggle recipe are in the "Inferences" subsections and are my own synthesis. They were not measured on the ROAD data. The competition facts (4,093 lines, ~1.5% swapped labels, 12/820 held-out lines = 18% of character edits, 6.2% casing breaks, multi-line crops) come from the assignment brief and are not independently sourced.

## Q1. Measured effect of label noise on HTR/OCR/seq2seq, and of cleaning it

### Takeaway
At low noise rates (about 1-5%), wrong-pairing noise costs little when the training set is large and the model is strong. Examples: NMT loses 0.7 BLEU when misaligned pairs equal to 5% of the corpus are added, and training-set cleaning on Arabic HTR gives "negligible" extra gain on sets with about 1% errors. In small historical HTR sets (100-350 lines), removing real misaligned lines is worth up to about 3 CER points, and re-pairing them is worth more. In every study the larger and more reliable measured effect is on the evaluation number: cleaning test sets moved CER by 0.3-1.6 pp, and noisy test sets flip model rankings.

### Cited Findings

**HTR, historical, small data: Aradillas, Murillo-Fuentes & Olmos (arXiv 2012.02544, 2020), ICFHR 2018 READ datasets, CRNN-CTC with transfer learning. The "Corrupted Label Purging" (CLP) algorithm uses N-fold out-of-fold CER and drops lines with CER above a threshold τ.**
- They name three real error types in historical HTR ground truth: (1) mislabelled characters, (2) *label misalignment*, where an image is paired with another line's transcription (seen "several times" in the READ/ICFHR18 **Ricordi** set; the ground-truth text is "quite close to the model output" for a *different* line), and (3) special inline annotations, such as IAM's "#" for crossed-out words or bracketed expansions like `R[icchezz]a M[obil]e`. A line of type (3) had about 35% CER even though the model output matched the handwriting. — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- Simulated character noise (20% of lines corrupted, 30% of characters in each corrupted line randomised, Konzil): "the impact of labeling errors in the CER value is more dramatic for small training sets", while the learning-curve slope stays roughly unchanged. — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **Real misalignment noise, Ricordi, no synthetic noise:**
  - 4 pages (88 lines): CER 21.1 -> 18.2 after dropping 16 lines (τ=0.5 or 0.7), and -> 17.4 when the flagged lines are *re-aligned* to the best-matching model output instead of dropped (−3.7 pp total).
  - 12 pages (295 lines): 9.7 -> 9.4 after dropping 38 lines (~13%), and -> 8.9 with re-alignment.
  — [Aradillas et al. 2020, Tables 5 and 7](https://arxiv.org/abs/2012.02544)
- **Removal hurts on clean data (R=0, τ=0.5):**
  - Konzil 4 pages: 7.6 -> 8.5 with 31 of 116 lines removed.
  - Schiller 4 pages: 13.27 -> 14.72 with 12 lines removed.
  - Patzig 4 pages: 18.3 -> 18.93 with 7 lines removed.
  - Konzil 12 pages: 4.6 -> 5.3 with 1 line removed. This suggests run-to-run variance of that size.
  - The stricter τ=0.7 mostly avoided the damage (e.g. Konzil 7.6 -> 7.9, Schiller 13.27 -> 13.61, Patzig 18.3 -> 18.32).
  — [Aradillas et al. 2020, Table 5](https://arxiv.org/abs/2012.02544)
- **With synthetic noise (10% of lines corrupted, 30-50% of their characters)**, CLP helped:
  - Konzil 4 pages, 30% character noise: 8.7 -> 7.82.
  - Schiller 12 pages, 50% character noise: 12.75 -> 10.51.
  - Washington, 150 lines: improvements of 0.8 / 0.63; 325 lines: 0.4 / 0.5, "and no deterioration over the original dataset".
  — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- The authors say that large training sets tolerate some mislabelled samples, "however, when a limited set of annotated lines of a specific writer is available ... mislabeled lines induce an overfitting to transcripts with errors, quite hard to tackle via regularization." — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)

**HTR, Arabic script: Al-azzawi, Barney & Liwicki (arXiv 2601.16713, 2026), "CER-HV".** The detector ranks by CRNN per-sample CER; flagged lines then go to human verification. Numbers below come from the HTML version via a summariser, so spot-check them against the PDF.
- Measured label-error prevalence per dataset (train split):
  - KHATT 0.9%
  - PHTI 1.3%
  - Muharaf 6.4%
  - Ajami 8.5%
  - NUST-UHWR <0.5%
  - PHTD none

  Val and test splits carry similar rates, e.g. Muharaf test 9.4% and Ajami test 9.2%. Error types include transcription, segmentation, orientation, script mismatch and non-text content. — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- Cleaning the evaluation split alone changed test CER by:
  - KHATT −0.34 pp
  - Muharaf −1.64 pp
  - PHTI −0.80 pp
  - NUST-UHWR −0.28 pp
  - Ajami −1.17 pp

  — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- Cleaning train and test together (retrain + clean eval):
  - Muharaf 10.11% -> 8.26%
  - PHTI 8.22% -> 7.33%
  - Ajami 10.59% -> 8.78%

  Training-set cleaning gave "negligible additional improvement" on the clean datasets (KHATT, NUST-UHWR) but meaningful gains on Ajami (about 8.5% errors). The abstract says "up to 1.8 percentage points". — [Al-azzawi et al. 2026](https://arxiv.org/abs/2601.16713)

**NMT, misaligned pairs (the direct analogue of swapped HTR labels): Khayrallah & Koehn (WMT/NMT workshop 2018), De–En.** Noise is *added* at 5/10/20/50/100% of the clean corpus size.
- Misaligned sentences: NMT BLEU 27.2 (clean) -> 26.5 / 26.5 / 26.3 / 26.1 / 25.3, i.e. −0.7 at 5% up to −1.9 at 100%. SMT: 24.0 -> 23.4 at worst (−0.6). — [Khayrallah & Koehn 2018](https://aclanthology.org/W18-2709/)
- For contrast, noise that creates a *shortcut* is catastrophic. Untranslated target (target = copy of the source) at 5%: NMT 27.2 -> 17.6 (−9.6); at 100%: -> 3.2. — [Khayrallah & Koehn 2018](https://aclanthology.org/W18-2709/)
- Raw web-crawled Paracrawl was 41% misaligned pairs. Adding it raw gave SMT +1.2 BLEU but NMT −9.9 BLEU. — [Khayrallah & Koehn 2018](https://aclanthology.org/W18-2709/)

**NMT data filtering by model cross-entropy: Junczys-Dowmunt (WMT 2018), "dual conditional cross-entropy".** It scores each pair with two inverse translation models trained on clean data and penalises disagreement between them.
- Paracrawl 100M-word subsets, dev test16/test17 BLEU:
  - Random selection: 16.2/14.1
  - Language-ID filter + random: 26.6/23.3
  - Plus dual cross-entropy (adq): 35.1/30.2
  - Plus domain score: 36.0/31.0

  All of these beat the clean WMT18-full baseline of 33.9/29.0. — [Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/)
- Ablations: dropping the agreement term |H_A − H_B| costs about 1 BLEU. Dropping the averaged cross-entropy weighting costs about 3 BLEU. So agreement between two independent models is a useful noise signal on top of raw loss. — [Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/)

**NLG, loss truncation: Kang & Hashimoto (ACL 2020).**
- On Gigaword, 35% of reference titles require hallucinated facts. Their average log loss is more than 1.7x that of entailed titles, and more than 80% of examples with log loss above 40 need hallucination. So high loss mostly marks invalid references *in that dataset*. — [Kang & Hashimoto 2020](https://aclanthology.org/2020.acl-main.66/)
- Method: hot-start with normal log-loss, then drop the top c-fraction of per-example losses using a running quantile. They used c = 0.6 on Gigaword.
- Results:
  - E2E BLEU 0.64 -> 0.72.
  - Truncation + rejection sampling factuality 3.79 vs beam 3.51 vs human 3.63.
  - ROUGE-L "not substantially" improved.
- Hot-starting matters: truncating from the start "will result in dropping valid examples"; after hot-start truncation "primarily drops invalid references". — [Kang & Hashimoto 2020](https://aclanthology.org/2020.acl-main.66/); code: [loss_dropper](https://github.com/ddkang/loss_dropper)

**Error Norm Truncation: Li, Xu, Koehn, Khashabi & Murray (ICLR 2024).** Pure loss-based truncation "filter[s] out valid but challenging training examples". ENT uses the whole output distribution, not only target-token NLL, and gains more than 2 BLEU over MLE in MT with up to 50% injected noise. — [Li et al. 2024](https://arxiv.org/abs/2310.00840)

**ASR, corrupted transcripts: Bataev (arXiv 2504.06963, 2025), LibriSpeech 960h, RNN-T.**
- With 50% of utterances corrupted, WER was:
  - Deletions: 81.4%
  - Substitutions: 23.8%
  - Insertions: 13.5%

  Clean baseline test-other is 5.1-6.0% in the HTML summary; a search snippet gave "6.8% on original data", so there is a conflict in the baseline figure.
- A search snippet also reports that small corruption rates degrade WER by 2.9-3.5% absolute. — [Bataev 2025](https://arxiv.org/html/2504.06963)
- Noise-robust transducer losses recovered 94.4% (Star-Transducer, deletions), 61.2% (Bypass, insertions) and 71.4% (mixed errors) of quality. — [Bataev 2025](https://arxiv.org/html/2504.06963)

**LLM fine-tuning, high noise rates only (arXiv 2604.12469, 2026).** Models: GPT-2 full fine-tuning, Qwen2-0.5B QLoRA, Llama-2-7B QLoRA. Label-flip noise at 20-40% was the most harmful type; at 40% flips sentiment accuracy fell from ~94% to 75-85%. Typographic noise was mild or even regularising. They did not test 1-5% noise. — [Analyzing the Effect of Noise in LLM Fine-tuning 2026](https://arxiv.org/html/2604.12469v1)

**Test-set noise: Northcutt, Athalye & Mueller (2021).**
- At least 3.3% label errors on average across 10 benchmark test sets, and at least 6% in the ImageNet validation set.
- A 6% rise in mislabelled test prevalence makes ResNet-18 beat ResNet-50 on ImageNet; a 5% rise makes VGG-11 beat VGG-19 on CIFAR-10.

— [Northcutt et al. 2021](https://arxiv.org/abs/2103.14749)

### Inferences
- The ROAD "swapped label" noise (~1.5%) is the HTR analogue of NMT *misaligned sentences*. At comparable rates that noise type cost only about 0.7 BLEU in a large-data NMT system. It does not create a shortcut the way copy noise does. The Arabic HTR study found training-set cleaning negligible at about 1% error prevalence. So training-side exclusion on ROAD (4,093 lines, strong pretrained 7-8B VLM, LoRA, 1 epoch, each noisy line seen once) will probably give a **small** gain, plausibly a few tenths of a CER point or less. That is an estimate, not a measurement, and it may be within fold-0 noise. Measure it with a paired bootstrap on fold 0 rather than a single CER difference.
- Small-data HTR (Aradillas) showed larger gains (0.3-2.9 pp from removal alone). ROAD's ~4k lines sit between their 88-782-line regime and large-corpus NMT. Expect gains closer to the small end.
- Re-pairing a swapped label to its correct image beat dropping it in Aradillas (Ricordi 18.2 -> 17.4 and 9.4 -> 8.9). Under the Zindi rule ("excluding clearly corrupted rows is allowed"), re-pairing labels between training rows may count as *changing labels*. **Exclusion is the rule-safe default. Re-pairing needs explicit staff confirmation.**
- The largest measurable effect of the ROAD noise is on *validation*: 1.5% of lines carry 18% of character edits, so any fold-0 CER contains a sizeable noise floor that the test set shares.

### Gaps
- I found no study that measures label-noise effects for **VLM/LLM-based HTR** (Qwen-VL, TrOCR, Florence) at 1-2% swap noise. All HTR evidence is CRNN-CTC.
- I could not open Kaggle solution write-ups (Bengali.AI Speech, RUKOPYS handwriting) to extract measured gains from noisy-label removal. The pages are JS-rendered and returned only titles. No Kaggle numbers are included.
- The Aradillas Table 6 (Washington/Parzival) layout was garbled in extraction, so only the text-stated gains are used.
- The casing-inconsistency noise (6.2% of tokens) has no direct measured study. It is systematic annotation variance, not a wrong pairing, and the test set shares it.

## Q2. Which detection signals work best for sequence data

### Takeaway
The best-evidenced automatic signal for line-level HTR is **per-line CER between a held-out (out-of-fold) model prediction and the label**. It reached 71-90% precision in the top-50 flagged lines on Arabic HTR, and CLP improved Ricordi CER. It becomes much more specific when combined with (a) **cross-line matching**, where the label matches a *different* image's prediction (the swapped-label fingerprint), and (b) **agreement between independent models** that disagree with the label. Raw per-sample training loss is weaker for sequences: it is confounded by length and difficulty, and it drops hard-but-correct lines.

### Cited Findings
- **Out-of-fold CER vs label.**
  - CLP splits the training set into N folds, fine-tunes on N−1, and scores the held-out fold by CER. On Ricordi (known misalignments) the CER histogram showed a distinct mode near 0.8.
  - The number of lines removed on clean data at τ=0.7 "is quite an indicator of the dataset containing errors".

  — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **Converged-model per-sample CER with a threshold.**
  - Method: train a CRNN to early stopping, compute CER(ŷ, y) per sample, rank, and flag CER > 0.25. "Samples with CER below 0.25 were overwhelmingly correctly labeled".
  - Precision in the top 50 flagged lines per train split:
    - Muharaf 75%
    - PHTI 88%
    - Ajami 71%

    Val/test splits scored 68-90%.

  — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- **Loss / confidence signals for sequences.**
  - Al-azzawi et al. chose not to use loss ranking. They argue "loss-based ranking is less informative for noise detection in HTR" (CTC alignment uncertainty) and that sequence confidence is "miscalibrated in practice", so Confident Learning / O2U-Net were not evaluated. — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
  - Loss-based truncation removes valid but challenging examples; accounting for the non-target distribution (ENT) filters better. — [Li et al. 2024](https://arxiv.org/abs/2310.00840)
  - Loss truncation *after hot-start* primarily drops invalid references. Without hot-start it drops valid ones. — [Kang & Hashimoto 2020](https://aclanthology.org/2020.acl-main.66/)
- **Token-level Confident Learning (cleanlab) for sequences.** Among 11 scoring methods on CoNLL-2003 real errors, self-confidence per token, aggregated per sentence by the **minimum** over tokens, gave the best sentence-level precision/recall. — [Wang & Mueller 2022](https://arxiv.org/abs/2210.03920); [cleanlab token-classification tutorial](https://docs.cleanlab.ai/stable/tutorials/token_classification.html)
- **Agreement of two models.**
  - Penalising disagreement between two inverse translation models was worth about 1 BLEU in filtering quality, and cross-entropy weighting about 3 BLEU. — [Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/)
  - Co-teaching / joint training uses two networks and small-loss selection, on the principle that "small-loss instances are more likely to be clean". This is classification evidence only. — [JoCoR, Wei et al. 2020](https://arxiv.org/pdf/2003.02752); [small-loss criterion analysis](https://arxiv.org/pdf/2106.09291)
- **Swap / misalignment matching.** For datasets with many misaligned lines, "searching within the outputs of the DNN model for the whole target dataset, the transcript best fitting every annotation in the GT" aligns labels back to images. This recovered further CER (Ricordi 18.2 -> 17.4). — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **Timing / memorisation.** DNNs "learn simple patterns first, then gradually memorize all samples" (the basis of small-loss selection). — [Towards Understanding ... Small-Loss Criterion](https://arxiv.org/pdf/2106.09291). A theoretical LoRA paper claims gradient descent on LoRA learns clean patterns early and memorises noisy labels later, with the separation depending on noise rate and rank. — [Why LoRA Resists Label Noise (2026)](https://arxiv.org/html/2602.00084)

### Inferences
My ranking of signals for ROAD's swapped labels, from most to least specific:
1. **Cross-line match.** The label of line i matches the OOF prediction of another line j, especially a neighbour on the same page, much better than its own prediction. This is near-proof of a swap and is cheap to compute.
2. **Two-model consensus against the label.** Qwen2.5-VL and Qwen3-VL predictions agree with each other (CER < 0.2) but both are far from the label (CER > 0.5).
3. **High OOF CER alone** (> 0.5-0.7). Good recall, but some real hard lines appear here.
4. **Length ratio** len(pred)/len(label) outside about [0.5, 2]. Mostly catches multi-line crops and truncations, not swaps.
5. **Per-token mean NLL from teacher forcing.** Cheapest (no generation), but most confounded by rare names, abbreviations and special notation.

Other points:
- For the 1-epoch LoRA, in-sample predictions on training lines are probably usable, since there is little memorisation after a single exposure. OOF predictions are still safer and are what the cited HTR methods use.
- Cleanlab's sequence variant needs per-token probabilities aligned to label tokens. That works for a VLM under teacher forcing (min-over-tokens of p(label token)). However, it measures the same thing as NLL, and no HTR/VLM evidence for it was found.

### Gaps
- No head-to-head comparison of loss vs OOF-CER vs multi-model agreement vs cross-line matching on the same HTR dataset was found.
- No published precision numbers exist for swap detection by cross-line matching, beyond Aradillas' qualitative confirmation by visual inspection.
- No evidence was found on how well per-token NLL from a large VLM (rather than a CTC model) separates mislabelled lines.

## Q3. Recommended thresholds and measured risks (removing hard-but-correct lines, rare names/abbreviations)

### Takeaway
Measured thresholds in HTR are CER ≥ 0.5-0.7 on out-of-fold predictions for *automatic* removal (Aradillas) and CER > 0.25 only when a human verifies (Al-azzawi). Aggressive removal on data without real errors **increased** CER by 0.3-1.5 pp in small-data HTR. A conservative threshold plus corroborating signals, capped at about 1-2% of lines, is supported by the evidence.

### Cited Findings
- **Threshold choice.** τ=0.7 was better with 4 pages of training data and τ=0.5 with 12 pages. The paper warns: "The selection for τ should not lead to the deletion of healthy lines, otherwise the overall CER would raise." — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **Measured harm from over-removal (no real noise, τ=0.5):**
  - Konzil 4 pages: 7.6 -> 8.5 CER with 31/116 lines (27%) removed.
  - Schiller: 13.27 -> 14.72 with 12/84 removed.
  - Patzig: 18.3 -> 18.93 with 7/156 removed.
  - At τ=0.7 the damage was smaller: 7.6 -> 7.9, 13.27 -> 13.61, 18.3 -> 18.32.

  — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **Hard-but-correct lines score high CER.** A line with special annotations had about 35% CER while the model output matched the handwriting. Special annotations were described as "perhaps, the most common source of error". — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- **The CER > 0.25 threshold is not precise enough for automatic removal.** Its precision in the top 50 was 68-90%, so 10-32% of the top-ranked candidates were not errors, and the framework relies on human verification. — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- **Truncation fractions.**
  - Loss truncation used c = 0.6 on a dataset where 35% of references are invalid, and the authors report the method is "not sensitive to the hyperparameter c on automated metrics". — [Kang & Hashimoto 2020](https://aclanthology.org/2020.acl-main.66/)
  - Loss-based truncation removes valid-but-hard examples. — [Li et al. 2024](https://arxiv.org/abs/2310.00840)
- **Selection-based filtering can be very aggressive when noise is very high.** A language-ID filter alone removed about 70% of Paracrawl candidates and still helped, because raw Paracrawl is only 23% "okay". — [Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/); [Khayrallah & Koehn 2018](https://aclanthology.org/W18-2709/)

### Inferences
- For ROAD the known noise rate is about 1.5%. A defensible **automatic** removal budget is about 1-2% of the 4,093 lines (~40-80 lines), and never the 13-27% removal fractions that hurt in Aradillas.
- **Use OOF CER ≥ 0.5 as a necessary condition,** plus at least one corroborating signal (cross-line match or two-model consensus). Do not remove on CER alone in the 0.25-0.5 band. Per Aradillas, that band is where special annotations, rare names, abbreviations and Latin passages live.
- **Casing:** do not use casing mismatches as a removal signal. Casing-only differences would inflate CER on otherwise-correct lines. Compute the detection CER on case-folded, whitespace-normalised strings, so that only content-level mismatches are flagged.
- **Multi-line crops** (label covers one line): keep them in training unless the model's prediction is itself wildly longer and the label matches a *sub-span* of it. The test set has the same crop style, so the model must learn which line the label refers to. Removing them could hurt test CER; this is inferred, not measured.
- **Down-weighting as an alternative to hard exclusion:** with only 1 epoch, Kang & Hashimoto-style in-batch truncation has no real hot-start phase. It is simpler to pass per-sample loss weights from a previous run's flags, e.g. weight 0 for "tier A" lines and 0.3-0.5 for "tier B". The weight values are untested heuristics.

### Gaps
- No measured study was found of threshold sensitivity for VLM/LLM HTR or at noise rates as low as 1.5%.
- No measured evidence was found on how often rare proper names or abbreviations are wrongly flagged at CER ≥ 0.5 in historical HTR. Only the single 35% CER annotation example exists.

## Q4. Evaluate with or without the suspected noisy validation lines?

### Takeaway
Report **both**. Use the full fold (all 819 lines) as the leaderboard proxy, because the hidden test set has the same noise. Use a fixed "clean" subset, with flagged lines excluded, for model selection and ablations, because noisy eval labels distort CER by up to about 1.6 pp and can flip rankings.

### Cited Findings
- Cleaning only the evaluation split changed HTR CER by 0.28-1.64 pp across five Arabic datasets. The authors state that "label errors in benchmark test sets can disproportionately affect model evaluation and comparison", and they released cleaned eval splits as revised benchmarks. — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- A 5-6% rise in mislabelled test prevalence flipped ResNet-18/50 and VGG-11/19 rankings. The authors recommend judging models on correctly labelled test sets. — [Northcutt et al. 2021](https://arxiv.org/abs/2103.14749)
- Reference-based metrics "encourage the models to fully imitate the data distribution — including invalid ... examples", so they under-reward cleaner models. — [Kang & Hashimoto 2020](https://aclanthology.org/2020.acl-main.66/)

### Inferences
- On ROAD's fold 0, 12 lines carry 18% of character edits (brief). The clean-subset CER is therefore about 18% lower (relative) than the full-fold CER.
  - For the full fold, the error on swapped lines is roughly model-independent: any correct reading of the image scores near 100% CER against the wrong label. That adds a near-constant floor that also exists on the test set.
  - Model differences therefore show up more clearly on the clean subset.
- **Freeze the exclusion list once** (derived from training-data-only signals) and use the same list for every experiment. Otherwise the metric shifts with the detector.
- **Use a paired bootstrap** over lines (same lines, two systems) to judge whether a gain is real. With 819 lines and a heavy-tailed per-line CER distribution, single-number differences of a few tenths of a point can be noise. This is my inference; the fold-0 variance was not measured.
- **Keep leaderboard expectation-setting on the full-fold number.** Test labels cannot be changed, so noise in the test set is an irreducible floor.
- **Document both numbers in the competition write-up.** The Zindi rule requires exclusion to be documented and based only on the training data. The validation fold is part of the training data, so flagging validation lines with OOF signals is within the rule.

### Gaps
- No HTR/OCR competition write-up was found that explicitly reports "clean vs full" validation numbers.
- No study was found that quantifies ranking flips specifically for CER under a 1.5% swap rate.

## Q5. A concrete, automatable detection recipe that fits a Kaggle notebook

### Takeaway
Use a 2- or 5-fold OOF CER screen, then a swap test (cross-line matching with rapidfuzz), then two-model consensus. Exclude only the lines with corroborated evidence, capped at about 2%. Everything here is automatic and uses training data only. It is derived from the methods cited above (CLP out-of-fold CER, CER-HV thresholding, dual-model agreement, alignment-by-best-match) and has not been validated on ROAD.

### Cited Findings
- The out-of-fold CER screen with N=2 folds was sufficient in CLP (train on one half, score the other, swap). — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- A CER threshold of 0.25 as a *candidate* filter reached 71-90% precision in the top 50. — [Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)
- The best-match search of labels against all model outputs identifies misaligned pairs. — [Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)
- Agreement between two independently trained models improves noise filtering. — [Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/)

### Inferences (the recipe; untested on ROAD)
1. **Get predictions for every training line from a model that did not train on it.**
   - Preferred: the existing 5-fold CV, run as OOF for all 4,093 lines.
   - Cheaper: 2 folds (2 × 1-epoch LoRA runs on ~2k lines each).
   - Fallback: greedy predictions from the fold-0 model on all lines. In-sample predictions are less trustworthy, but memorisation is limited after 1 epoch.
   - If budget allows, repeat with a second model family (Qwen2.5-VL-7B vs Qwen3-VL-8B) for the consensus signal.
2. **Normalise both strings before scoring:** NFC, case-fold, collapse whitespace, and optionally strip punctuation. Casing noise (6.2% of tokens) must not trigger flags.
3. **Compute the following per line:**
   - `cer_own` = normalised Levenshtein(pred_i, label_i) / len(label_i)
   - `len_ratio` = len(pred_i) / len(label_i)
   - `cer_models` = CER(pred_i^A, pred_i^B), if two models are available
4. **Swap test (cross-line matching)** for candidates with `cer_own ≥ 0.5`. Compute `best_j = argmin_j CER(label_i, pred_j)` over all j ≠ i, and use `rapidfuzz.process.cdist` with `workers=-1`. For 4,093 × ~100 candidates this takes seconds.
   - Also check page neighbours j = i±1, i±2 first, since an off-by-one pairing is the likely mechanism.
   - Flag as a **swap** if `CER(label_i, pred_best_j) ≤ 0.3` and it is ≤ 0.5 × `cer_own`.
   - Additionally flag **exact duplicate labels** attached to different images.
5. **Tiers:**
   - **Tier A, auto-exclude from training:** `cer_own ≥ 0.5` AND (swap test positive OR (`cer_models ≤ 0.2` AND both models' CER to the label ≥ 0.5)).
   - **Tier B, keep or down-weight:** `cer_own ≥ 0.5` with no corroboration, or `len_ratio` outside [0.5, 2] (likely multi-line crop or truncation).
   - **Cap:** if Tier A exceeds about 2% of lines, raise the thresholds rather than remove more.
   - Log every excluded id with its signal values for the write-up.
6. **Retrain:**
   - Retrain on all lines minus Tier A.
   - Optionally add a variant with Tier B at loss weight ~0.5 (custom `compute_loss` in the HF Trainer multiplying per-sample loss).
   - Compare against the unfiltered baseline on fold-0 **clean subset** and **full**, using a paired bootstrap (e.g. 1,000 resamples of the 819 lines).
   - Keep the filter only if the clean-subset gain is positive with the 95% CI excluding 0, and the full-fold CER does not get worse.
7. **Do not re-pair** swapped labels to their matching image unless Zindi staff confirm that it counts as "excluding corrupted rows" rather than relabelling. Aradillas measured re-pairing as better than dropping, so it is worth asking.

Minimal code sketch (for the report writer; untested):

```python
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process
import unicodedata, re, numpy as np

norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).casefold()).strip()
L = [norm(x) for x in labels]; P = [norm(x) for x in oof_preds]
cer_own = np.array([Levenshtein.distance(p, l) / max(len(l), 1) for p, l in zip(P, L)])
cand = np.where(cer_own >= 0.5)[0]
D = process.cdist([L[i] for i in cand], P, scorer=Levenshtein.normalized_distance, workers=-1)
for k, i in enumerate(cand):
    D[k, i] = np.inf                        # exclude own image
best_j = D.argmin(1); best_d = D.min(1)     # normalized_distance ~ CER-like, 0..1
swap = (best_d <= 0.3) & (best_d <= 0.5 * cer_own[cand])
```

### Gaps
- None of this recipe's thresholds (0.5 / 0.3 / 0.2 / 2% cap) have been measured on ROAD or on any VLM HTR model. They are extrapolated from CRNN-CTC HTR studies and NMT filtering.
- Compute cost on Kaggle was not measured. OOF generation for 4,093 lines with a 7-8B VLM (4-bit, greedy) is likely around an hour or more per model on a T4/P100 without vLLM. This is an estimate, not a measured figure.
