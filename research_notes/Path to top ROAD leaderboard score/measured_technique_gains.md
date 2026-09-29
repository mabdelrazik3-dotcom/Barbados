# Measured gains of techniques for fine-tuning VLM / transformer OCR on small-data handwritten & historical text lines

Scope reminder: target is a ~35–40% relative cut in edits (CER ~4.5% -> ~2.7%, WER ~15.5% -> ~9.5%) for a Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3-VL-32B LoRA system trained on 4,093 labelled 17th–18th c. English legal lines. "Rel." percentages marked (computed) were calculated by me from the numbers the source reports; everything else is as reported. Anything not directly measured is in Inferences.

## Q1. RL / preference fine-tuning after SFT with edit-distance (or unit-test) rewards

### Takeaway
Every measured case of RL after SFT for OCR comes from document parsing with 100k+ training samples. None is on handwritten text lines. In those cases GRPO after SFT cuts text edit distance by about 11–35% relative with a plain edit-distance reward, and by up to 65% with a shaped reward. Benchmark gains are +1.4 to +3.9 points on olmOCR-Bench, and one of the extra benefits is fewer repetition loops. I found no measured GRPO, DPO or minimum-risk-training result on line-level HTR with 1K–10K lines.

### Cited Findings
- **DocPO (Qwen2.5-VL-3B, OmniDocBench).** After SFT on 490k pages (1 epoch, constant LR 3e-5), text NED was 0.0358.
  - GRPO with a plain edit-distance reward brought text NED to 0.0238, a 34% relative drop (computed).
  - Their step-aware annealed reward brought it to 0.0125, a 65% drop vs SFT.
  - RL data was 612k element patches, including 210k text blocks. RL used LR 1e-6, batch 128 and 8 rollouts.
  - — [DocPO, arXiv 2608.00536](https://arxiv.org/html/2608.00536v2)
- **DocPO vs its non-annealed variant.** DocPO lowers edit distance further by 0.0349 on fuzzy scans, 0.0447 on watermark pages, 0.0319 on colourful textbooks and 0.0341 on exam papers. The authors say the gains are largest where visual corruption makes fine-grained discrimination important. — [DocPO](https://arxiv.org/html/2608.00536v2)
- **FD-RL (Qwen3-VL-4B, OmniDocBench).** SFT on 566k samples gave text edit distance 0.055. Adding GRPO gave 0.049, an 11% relative drop (computed); overall score went 87.13 -> 90.41.
  - RL alone without SFT gave only 46.06 -> 49.37 overall.
  - Filtering the RL pool to keep only high-entropy samples mattered. Filtering 0% of samples gave 88.47 overall, 50% gave 90.41, and 75% gave 88.58.
  - RL used LR 1e-6, 8 rollouts, temperature 1 and no KL term.
  - — [FD-RL, arXiv 2601.08834](https://arxiv.org/pdf/2601.08834)
  - In FD-RL's reward ablation (Table 6), one configuration shows text edit distance 0.044, lower than the final model's 0.049. That suggests the unified string-match reward was best for plain text. The row labels were garbled in PDF extraction, so treat this as tentative.
- **olmOCR 2 (Qwen2.5-VL-7B).** The combined step "synthetic data + GRPO RLVR + souping 6 seeds" moved olmOCR-Bench from 78.5±1.1 to 82.4±1.1.
  - The "old scans" category went 43.9 -> 47.7.
  - RL setup: 1 epoch over 2,186 synthetic pages with 30,381 unit tests, 28 completions per page, KL β=0.01.
  - The effects of RL and souping are not reported separately.
  - — [olmOCR 2, arXiv 2510.19817](https://arxiv.org/abs/2510.19817)
- **LightOnOCR.** RLVR after SFT moved LightOnOCR-2-1B from 81.8±0.9 to 83.2±0.9 on olmOCR-Bench, and LightOnOCR-1B from 76.1 to 77.7 with GRPO. RLVR also cut "loopy" (repetition) generations from 1.14% to 0.50%. — [LightOnOCR, arXiv 2601.14251](https://arxiv.org/html/2601.14251)
- **Jina-OCR-v1.** Starting from DeepSeek-OCR at 76.0 on olmOCR-Bench, the full post-training stack (instruction alignment, robustness FT and GRPO with dense verifiable rewards, 8 rollouts, ReMax baseline) reached 83.4. The paper has no ablation isolating GRPO. — [Jina-OCR-v1, arXiv 2609.03181](https://arxiv.org/html/2609.03181v1)
- **Minimum-risk training (speech analogue, not OCR).** Minimum word error rate (MWER) training on N-best lists improved WER by up to 8.2% relative over a cross-entropy baseline. — [Prabhavalkar et al., arXiv 1712.01818](https://arxiv.org/pdf/1712.01818)

### Inferences
- The consistent pattern is that RL helps most when SFT leaves residual "structural" failures: loops, truncation, hallucinated or normalised tokens.
  - The user's errors include historical-spelling normalisation, abbreviation and superscript conventions, and casing. These are exactly the token-level behaviours a CER reward can push on.
  - A plausible but unmeasured expectation for line HTR is 10–30% relative CER reduction.
- RL with 4k lines is data-poor compared with every measured case (2k–600k RL samples). DocPO and FD-RL both used LR 1e-6 and 8 rollouts. FD-RL showed that filtering out low-entropy (already-easy) samples matters.
  - For a 4k-line set, RL on the lines the SFT model gets wrong or is uncertain about is the closest analogue.
- The ~1.5% mislabelled lines give wrong reward targets. Speculative mitigation: exclude lines whose SFT CER is extreme.

### Gaps
- I found no measured GRPO, DPO or minimum-risk result on handwritten or historical line-level HTR with 1K–10K lines.
- No isolated RL-only number for olmOCR 2 (it is confounded with souping and synthetic data).

## Q2. Ensembling, voting, ROVER, MBR, checkpoint averaging and model soups

### Takeaway
Output-level voting of several independently trained line recognisers is the best-evidenced small-data gain. It reduces CER by 14–34% relative in the measured HTR/OCR cases (TrOCR on 3.6k lines; CTC models on 150–1,000 lines). Sampling several outputs from one VLM (self-consistency) did not help on average. Weight souping is used by olmOCR 2 and LightOnOCR, but its effect was never isolated.

### Cited Findings
- **Gwalther TrOCR ensemble.** TrOCR-base on 16th-c. Latin, 3,603 training lines (4,037 total), trained 20 epochs at LR 3e-5.
  - Single baseline CER was 1.93. Voting all 11 models gave 1.66, and the top-5 vote gave 1.60.
  - That is a 14% and 17% relative reduction vs the paper's own baseline (computed).
  - The members were the same model trained with different augmentations. The best single augmentation (elastic or rotation) gave only 1.86 (−3.6%).
  - — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)
- **Cross-fold 5-model sequence voting on early printed books (Calamari/OCRopus).** CER went from the fold average to the voted output as follows (relative drops computed):
  - Deep net at 1,000 training lines: 0.86 -> 0.65 (−24%), 0.72 -> 0.50 (−31%), 1.52 -> 1.23 (−19%).
  - Shallow net at 1,000 lines: 1.55 -> 1.11 (−28%), 1.17 -> 0.77 (−34%), 1.84 -> 1.44 (−22%).
  - Deep net at 250 lines: 2.26 -> 1.75 (−23%), 1.81 -> 1.29 (−29%), 2.33 -> 1.91 (−18%).
  - Confidence voting added a further 5–10% over sequence voting.
  - Voting helped less for the deeper, less diverse models.
  - — [Wick, Reul & Puppe, arXiv 1802.10033](https://arxiv.org/pdf/1802.10033)
- **Earlier cross-fold training plus confidence voting** cut errors "by up to 50% and more" on seven early printed books. — [Reul et al., arXiv 1711.09670](https://arxiv.org/abs/1711.09670)
- **Voting across commercial OCR engines** (Rice et al., 1996, as cited): CER went from 1.17–9.90% for single engines to 0.85% voted. — [cited in arXiv 1802.10033](https://arxiv.org/pdf/1802.10033)
- **Multi-VLM consensus vs self-consistency (CE-OCR, CVPR 2026).**
  - Multi-VLM consensus selection is reported at +8.2% on average on OCR benchmarks. Sampling the same VLM several times (self-consistency) averaged −2.8%.
  - Extracted internal numbers: OCRBench single best 888, CE-ensemble 902, CE-OCR with routing 922.
  - — [Consensus Entropy, arXiv 2504.11101](https://arxiv.org/abs/2504.11101)
- **Souping in OCR VLMs.**
  - olmOCR 2 soups 6 RL runs with different seeds. The authors report it as "beneficial", but no separate number is given. — [olmOCR 2](https://arxiv.org/abs/2510.19817)
  - LightOnOCR souped its last 5 checkpoints to form its base model. A task-arithmetic merge (α=0.4) scored 82.4. — [LightOnOCR](https://arxiv.org/html/2601.14251)
- **Related decoding lever: n-gram LM decoding in PyLaia.** Averaged over datasets it gave −12% CER and −13% WER. On historical NorHand it gave −17.5% CER and −24.3% WER. It gave little or negative benefit where CER < 2%. — [PyLaia 2024, arXiv 2404.18722](https://arxiv.org/abs/2404.18722)

### Inferences
- The measured HTR ensembles are all voting ensembles of differently trained members, with diversity from augmentation, data fold or architecture.
  - The user already has three model families (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3-VL-32B), which is the diverse-member setting.
  - Character-level ROVER or MBR selection over their outputs is the lowest-risk path to roughly 15–30% relative fewer edits. This is extrapolated, not measured on VLMs.
- Choosing the hypothesis with minimum summed edit distance to the others (MBR with a CER utility) is conceptually what CE-OCR does. Its positive multi-model result and negative same-model result suggest diversity across models matters more than sampling.
- Averaging LoRA weights across seeds (souping) is cheap but unquantified for OCR.

### Gaps
- No measured result for character-level ROVER or MBR across fine-tuned VLMs on handwriting.
- No isolated soup-vs-single-model number for any OCR VLM.

## Q3. Self-training / pseudo-labelling (AT-ST, noisy student, Wolf & Fink, SoftCTC)

### Takeaway
Self-training gives large CER cuts (up to 55%) when the unlabelled pool is 100x the labelled set. The measured cases are AT-ST with 1.14M unlabelled vs 9.2k labelled lines, and SoftCTC on Bentham. Confidence filtering (keeping roughly the top 10%) or soft pseudo-labels matter. I found no measurement where the unlabelled pool is smaller than the labelled set, which is the user's case (~2,061 unlabelled vs 4,093 labelled).

### Cited Findings
- **AT-ST.** Handwritten target domain: 9,198 annotated lines plus 1,141,566 unannotated lines, with a 189,805-line related-domain seed set. CER results by setup:

  | Setup | Seed model | After self-training | With language model |
  |---|---|---|---|
  | "Small" | 6.43 | 4.41 | 3.27 |
  | "Small-2" | 4.41 | 2.92 | 2.88 |
  | "Big" | 2.58 | 2.07 | 1.94 |

  - Overall that is up to a 55% CER reduction on handwriting and 38% on print.
  - Pseudo-labels were selected by transcription posterior probability.
  - Aggressive masking augmentation alone cut about 10% (READ: 5.28 -> 3.91 with masking, 3.20 with masking plus traditional augmentation).
  - — [AT-ST, arXiv 2104.13037](https://arxiv.org/abs/2104.13037)
- **SoftCTC (Bentham-Full test CER).**
  - AT-ST with no confidence filtering: 4.73.
  - AT-ST keeping the top 10% most confident: 4.41.
  - SoftCTC with soft multi-variant pseudo-labels and no filtering: 4.35.
  - — [SoftCTC, arXiv 2212.02135](https://arxiv.org/abs/2212.02135)
- **Wolf & Fink.** Self-training from a synthetic-trained initial model with confidence-based pseudo-label selection "clearly outperforms" learning-free and adaptation baselines on four word-level datasets. I could not retrieve exact CER/WER numbers. — [Wolf & Fink, IJDAR 2024](https://link.springer.com/article/10.1007/s10032-024-00484-9)
- **Supervised fine-tuning on few target lines (a scale reference for small adaptation sets).**
  - For CTC models, fine-tuning on 16 new-writer lines gave 25% relative CER improvement, and 256 lines gave 50%. — [Kohút & Hradiš, arXiv 2302.06308](https://arxiv.org/abs/2302.06308)
  - For autoregressive transformers, 16 lines gave about 10% and 256 lines about 40%. Confidence-based active learning halved the annotations needed. — [arXiv 2503.19546](https://arxiv.org/html/2503.19546)

### Inferences
- With ~2k unlabelled lines (687 unlabelled plus 1,374 test lines), pseudo-labelling can add at most about 50% more training lines from the same distribution. Transductive training on test lines also adapts to test-page writers.
  - The writer-adaptation numbers (25–50% from 16–256 lines) suggest the transductive part could matter more than the added volume.
  - This is speculative; no source measures it for VLMs.
- If pseudo-labelling is done, the measured best practice is posterior-confidence filtering or ensemble-agreement filtering. Pseudo-labels produced by an ensemble (Q2) are the natural "noisy student" teacher.

### Gaps
- No measured VLM self-training on HTR.
- No curve of gain vs unlabelled-pool size at small ratios.
- Kang et al. (the self-training reference in the brief): I found no retrievable numbers.

## Q4. Input resolution, line height and tiling for VLM OCR

### Takeaway
Resolution effects for VLM OCR are reported only qualitatively or bundled with other changes. Larger inputs "slightly" help; olmOCR raised the longest edge from 1024 to 1288 px and LightOnOCR from 1024 to 1540 px. Upscaling small images beyond the training distribution hurts, and too few vision tokens per text token collapses accuracy. I found no line-height sweep for Qwen-VL on handwritten lines.

### Cited Findings
- **olmOCR 2.** Going from 1024 to 1288 px on the longest edge was chosen from a sweep ("bigger images … slightly better performance across many model families").
  - It was bundled with a trainer change, YAML output and the switch to Qwen2.5-VL, and that bundle gave 75.8 -> 78.5.
  - — [olmOCR 2](https://arxiv.org/abs/2510.19817)
- **LightOnOCR-2.** Raised the maximum longest edge from 1024 to 1540 px for small text; no isolated ablation. — [LightOnOCR](https://arxiv.org/html/2601.14251)
- **Qwen2-VL report.** Raising min_pixels improves perception tasks within a range. On OCRBench, which has many tiny images, a too-high min_pixels causes "severe performance decline" because upscaling pushes images out of the training distribution. — [Qwen2-VL, arXiv 2409.12191](https://arxiv.org/abs/2409.12191)
- **DeepSeek-OCR.** OCR precision is about 97% when text tokens are under 10x the vision tokens, and about 60% at 20x compression. — [DeepSeek-OCR, arXiv 2510.18234](https://arxiv.org/abs/2510.18234)
- **Typhoon OCR.** Resizing to a fixed 1,800 px width "improves both training stability and evaluation performance"; no numbers given. — [Typhoon OCR, arXiv 2601.14722](https://arxiv.org/html/2601.14722)
- **TrOCR on the Gwalther lines.** Adding a "Resize" augmentation raised CER from 1.93 to 2.31 (+20%). — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)
- **Zero-shot Qwen2.5-VL-7B on Arabic manuscript lines.** Sharpening was the only preprocessing that helped (CER 70.4 -> 67.2). Binarisation, denoising and contrast enhancement hurt. — [arXiv 2608.22366](https://arxiv.org/abs/2608.22366)
- **Specialised HTR reference.** HTR-VT uses a fixed 64x512 input and reaches IAM CER 4.7. — [HTR-VT, arXiv 2409.08573](https://arxiv.org/abs/2409.08573)

### Inferences
- For Qwen-VL on text lines, the lever is the number of vision tokens per character: height after resize, and min/max_pixels.
  - For Qwen2.5-VL a patch covers 28x28 pixels after merging; for Qwen3-VL it is 32x32.
  - Small superscripts ('^') and suspension marks (':') are the likeliest victims of low resolution.
  - A sweep, for example line height 56/84/112 px with train and test matched, is cheap. The size of its gain is unmeasured.

### Gaps
- No measured line-height or min_pixels sweep for fine-tuned Qwen-VL on handwriting.

## Q5. Model choice and size after fine-tuning (3B / 7B / 32B; Qwen2.5-VL vs Qwen3-VL vs others)

### Takeaway
After fine-tuning, a small VLM often matches or beats much larger ones. CHURRO-3B beats zero-shot 72B; Typhoon 3B roughly equals 7B; Qwen3-VL-2B with 2x the data beats Qwen2.5-VL-7B. Data and recipe dominate model size. Zero-shot, Qwen2.5-VL-72B and Qwen3-VL-8B are about equal on English handwriting. OCR-specialist VLMs (PaddleOCR-VL, DeepSeek-OCR, MinerU) are far worse zero-shot on handwriting and prone to repetition loops. I found no controlled fine-tuned 7B vs 32B comparison on handwriting.

### Cited Findings
- **CHURRO (Qwen2.5-VL-3B fine-tuned on 97,151 historical pages).**
  - Handwritten normalised Levenshtein similarity was 70.1%, vs 63.6% for zero-shot Gemini 2.5 Pro, 54.5% for zero-shot Qwen2.5-VL-72B and 42.9% for the base 3B.
  - On printed text it scored 82.3%, vs 67.8% for the base 3B.
  - — [CHURRO, arXiv 2509.19768](https://arxiv.org/abs/2509.19768)
- **Typhoon OCR (full fine-tuning of Qwen2.5-VL, 3 epochs, 77k samples).** Levenshtein distance for 3B vs 7B:
  - Financial reports: 0.08 vs 0.07.
  - Government forms: 0.05 vs 0.08.
  - Books: 0.32 vs 0.31.
  - v1.5 on Qwen3-VL-2B with 155k samples averaged 0.251 vs v1's 0.332, but that comparison is confounded by doubling the data.
  - — [Typhoon OCR](https://arxiv.org/html/2601.14722)
- **18th-c. printed English lines (~195k training lines, ~10k test lines).**
  - PEFT-tuned Qwen2.5-VL-7B with the vision encoder frozen: CER 0.60%, WER 2.48%.
  - Fine-tuned TrOCR-large: CER 1.05%, WER 3.89%.
  - Qwen showed "orthographic normalization" of historical spellings (e.g. "Antient" -> "Ancient").
  - — [Error Patterns in Historical OCR, arXiv 2602.14524](https://arxiv.org/abs/2602.14524)
- **Manchu (LoRA r64, LR 1e-4, 60k synthetic word images).** Real handwritten-test CER:

  | Model | Epochs | Real-handwriting CER |
  |---|---|---|
  | LLaMA-3.2-11B | 5 | 0.0219 |
  | Qwen2.5-VL-7B | 10 | 0.254 |
  | Qwen2.5-VL-3B | 15 | 0.368 |

  - The Qwen models overfit the synthetic data.
  - — [Manchu VLM OCR, arXiv 2507.06761](https://arxiv.org/abs/2507.06761)
- **Handwritten math (CROHME 2014/16/19 ExpRate), 1-epoch full fine-tuning.**
  - Vanilla fine-tuned Qwen2.5-VL-3B: 72.62 / 64.26 / 69.14.
  - Zero-shot Qwen2.5-VL-72B: 59.74 / 54.32 / 55.13.
  - Zero-shot Qwen2.5-VL-7B: 55.98 / 50.92 / 49.62.
  - — [Uni-MuMER, arXiv 2505.23566](https://arxiv.org/abs/2505.23566)
- **OmniHandwritingOCR (2026), zero-shot English handwriting CER.** Lowest (best) first:

  | Model | CER | WER |
  |---|---|---|
  | Qwen2.5-VL-72B | 16.01 | 28.21 |
  | Qwen3-VL-8B | 16.44 | 28.43 |
  | Nanonets-OCR2-3B | 19.41 | — |
  | GPT-4o | 20.05 | — |
  | MonkeyOCR-pro-1.2B | 23.64 | — |
  | DeepSeek-OCR | 39.83 | — |
  | PaddleOCR-VL | 47.48 | — |
  | MinerU2.5 | 93.43 | — |

  — [OmniHandwritingOCR, arXiv 2608.18586](https://arxiv.org/abs/2608.18586)
- **ExpertHTR.**
  - Fully fine-tuned Qwen3.5-0.8B (3 epochs, LR 4e-6, 7 HTR sources) reached IAM paragraph CER 2.92, vs 11.55 zero-shot.
  - On Bentham it reached 17.57, while zero-shot LightOnOCR reached 7.24.
  - Several released OCR-VLM checkpoints entered repetition loops on handwriting, producing CER > 100%.
  - — [ExpertHTR, arXiv 2609.12705](https://arxiv.org/abs/2609.12705)
- **Zero-shot LLM HTR benchmark.** Qwen2-VL-7B scored CER 2.92% on IAM and 28.08% on Bentham. — [arXiv 2503.15195](https://arxiv.org/abs/2503.15195)
- **Specialised HTR trained from scratch (HTR-VT).**
  - IAM (6,482 lines): CER 4.7, WER 14.9.
  - READ2016: CER 3.9.
  - LAM: CER 2.8, vs 3.6 for TrOCR.
  - — [HTR-VT](https://arxiv.org/abs/2409.08573)
- **VLM grounding.** Under character-level perturbations, VLMs (Qwen3-VL-2B/8B, olmOCR-2-7B and others) diverge from the image more than traditional OCR, i.e. they "guess" from language priors. — [Reading or Guessing?, arXiv 2605.27750](https://arxiv.org/abs/2605.27750)

### Inferences
- The user saw 32B score worse than 7B/8B. That matches the measured "size is not the main lever after fine-tuning" pattern (Typhoon, CHURRO, Uni-MuMER).
- Different families make errors in different places, as the CE-OCR, Calamari and Rice results suggest. So the models' value is higher as ensemble members than as single-model replacements.
- The language-prior normalisation documented for Qwen, and "guessing" under weak visual evidence, fits the user's historical-spelling and abbreviation error class.

### Gaps
- No controlled Qwen2.5-VL-7B vs Qwen3-VL-8B vs Qwen3-VL-32B fine-tuned comparison on handwritten lines.
- No fine-tuned numbers for PaddleOCR-VL, dots.ocr or Florence-2 on historical handwriting.

## Q6. Training schedule: epochs, LR, LoRA rank, LoRA vs full fine-tuning, vision tower

### Takeaway
Measured OCR evidence says LoRA rank matters little between r16 and r512; one study found a spread of 70.1–73.0. Full fine-tuning can beat attention-only LoRA by a lot in one domain-shift study (86.5 vs ≤73.0), but not in a scarce-data scene-text study. Most strong OCR VLMs use 1 epoch of SFT (olmOCR, DocPO, Uni-MuMER, QARI) or 2–3 (Typhoon, ExpertHTR). I found no measured vision-tower-tuning ablation for Qwen-VL OCR.

### Cited Findings
- **Phi-3.5-Vision on 21,442 Indian ID documents** (field extraction, mean of exact-match and partial-match).
  - Base model: 40.45.
  - LoRA on attention projections only, ranks 16–512: 70.13–73.03. Best was r512/α256 at 73.03; r32/α256 scored 72.51.
  - Full fine-tuning: 86.48.
  - — [arXiv 2602.16430](https://arxiv.org/abs/2602.16430)
- **Low-light scene-text recognition.** LoRA gave CER 52.84, while both frozen and fully fine-tuned models gave 53.16. The authors conclude full FT offers no benefit on scarce data. — [arXiv 2604.23685](https://arxiv.org/pdf/2604.23685)
- **Autoregressive HTR adaptation on limited lines.**
  - Fine-tuning only the encoder was best for known transcription styles; full-model fine-tuning was best for unseen styles or languages; decoder-only was worst.
  - About 20 epochs at LR 5e-5 was optimal.
  - — [arXiv 2503.19546](https://arxiv.org/html/2503.19546)
- **Epoch and LR settings in the SFT recipes found:**
  - olmOCR: 1 epoch SFT. [olmOCR 2](https://arxiv.org/abs/2510.19817)
  - DocPO: 1 epoch, constant LR 3e-5. [DocPO](https://arxiv.org/html/2608.00536v2)
  - Uni-MuMER: 1 epoch, full fine-tuning. [Uni-MuMER](https://arxiv.org/abs/2505.23566)
  - ExpertHTR: 3 epochs, LR 4e-6, full fine-tuning. [ExpertHTR](https://arxiv.org/abs/2609.12705)
  - Typhoon: full fine-tuning, 3 epochs (v1) and 2 epochs (v1.5). [Typhoon](https://arxiv.org/html/2601.14722)
  - Manchu: LoRA r64, LR 1e-4, 15/10/5 epochs for 3B/7B/11B. [Manchu](https://arxiv.org/abs/2507.06761)
- **Frozen vision tower.** The PEFT-tuned Qwen2.5-VL-7B in the 18th-c. printed-English study kept the vision encoder frozen and still reached 0.60% CER. — [arXiv 2602.14524](https://arxiv.org/abs/2602.14524)
- **Regularisation ablations (HTR-VT, READ2016).**
  - Sharpness-aware minimisation (SAM): CER 4.8 -> 4.5.
  - Span masking: 5.1 -> 4.5.
  - — [HTR-VT](https://arxiv.org/abs/2409.08573)

### Inferences
- The user already found 1 epoch at 2e-4 best, consistent with published 1-epoch recipes.
- The full-FT advantage above was measured against attention-only LoRA. The user's LoRA r32 with vision layers is a stronger configuration, so the expected remaining gain from switching to full FT is smaller. That estimate is unmeasured.

### Gaps
- No LoRA-vs-full or vision-tower ablation on handwritten lines for Qwen-VL.
- No LR sweep for OCR LoRA.

## Q7. Test-time augmentation (TTA) for OCR

### Takeaway
The only measured HTR TTA result is on a CNN-BiLSTM: 1,616 shear/rotation variants scored with a language model cut IAM CER 4.80 -> 4.37 (−9%) and WER 13.85 -> 12.03 (−13%). Same-model VLM sampling (self-consistency) was net negative in CE-OCR. I found no TTA measurement for fine-tuned VLM OCR.

### Cited Findings
- **CNN-BiLSTM on IAM**, trained on about 2.5M synthetic images plus IAM training data.
  - TTA: CER 4.80 -> 4.37 and WER 13.85 -> 12.03.
  - Case-insensitive, no punctuation: CER 4.38 -> 3.59 and WER 12.00 -> 9.44.
  - Variants were selected by optical score plus a 4-gram LM score.
  - — [arXiv 2307.00664](https://arxiv.org/abs/2307.00664)
- **Self-consistency (3 samples from one VLM) on OCR benchmarks** averaged −2.8%, vs +8.2% for multi-model consensus. — [CE-OCR](https://arxiv.org/abs/2504.11101)
- **Test-time training (DocTTT, WACV 2025)** adapts visual parameters per input with a masked-autoencoder loss and reports beating SOTA. I could not retrieve exact numbers. — [DocTTT, arXiv 2501.12898](https://arxiv.org/abs/2501.12898)
- **In-context writer adaptation** with an 8M-parameter model: IAM CER 4.22 -> 3.92 and RIMES 2.91 -> 2.34 using 9 context lines. — [arXiv 2603.29450](https://arxiv.org/abs/2603.29450)

### Inferences
- For VLMs, TTA variants such as small rescales, padding or crops, combined by character-level voting or MBR, behave like a weak ensemble. The expected gain is below that of cross-model ensembling. This is speculative.

### Gaps
- No measured TTA for fine-tuned Qwen-VL or any VLM on handwriting.

## Q8. Synthetic data made from the training set itself (StackMix, line recombination, word crops)

### Takeaway
StackMix, which builds new lines by stitching together character or word image pieces from the training set, cut CER by 10–45% relative over standard augmentation for CTC models trained from scratch on 1.4k–45k lines. The line text came from large external corpora (180k–3M lines), not from the training set. Blot augmentation added 10–20%. I found no measurement for VLM fine-tuning.

### Cited Findings
- **StackMix CER.** "augs" = standard augmentation; "all" = augs + blots + StackMix. Relative drops vs augs (computed) in brackets.

  | Dataset (train lines) | No aug | Augs | Augs + blots | Augs + StackMix | All |
  |---|---|---|---|---|---|
  | BenthamR0 | 2.99 | 2.85 | 2.27 | 2.08 (−27%) | 1.73 (−39%) |
  | IAM-B (6,482) | 5.80 | 5.43 | 4.59 | 4.90 (−10%) | 3.77 (−31%) |
  | IAM-D | 4.55 | 4.38 | 3.70 | 3.77 | 3.01 |
  | Digital Peter, 18th c. (6,237) | 4.44 | 4.15 | 3.39 | 3.40 (−18%) | 2.50 (−40%) |
  | HKR (45,559) | 6.71 | 6.67 | 7.40 | 3.69 (−45%) | 3.49 |
  | Saint Gall (~1,410) | 4.71 | 4.49 | 4.08 | 4.06 (−10%) | 3.65 (−19%) |

  - Models were trained for 1,000 epochs.
  - Text sources: Digital Peter used about 3M lines of 17th–18th c. text; IAM/Bentham used a 560k-line Kaggle corpus.
  - — [StackMix, arXiv 2108.11667](https://arxiv.org/abs/2108.11667)
- **Masking augmentation (AT-ST)** cut CER by about 10% (READ 5.28 -> 3.91). — [AT-ST](https://arxiv.org/abs/2104.13037)
- **Label-noise handling.** Transfer learning plus augmentation plus corrupted-label purging (CLP) gave "up to 6%" CER reduction on historical sets. — [arXiv 2012.02544](https://arxiv.org/abs/2012.02544)

### Inferences
- Under the competition rule (only competition data), StackMix would have to draw its text from training transcriptions: reshuffled words and n-grams from the 4,093 lines. That weakens the "new text" benefit StackMix measured.
  - It still directly targets rare abbreviations and superscripts by recombining their crops.
  - The VLM effect is unmeasured. VLM decoders may learn implausible word sequences from shuffled text, which is a risk.

### Gaps
- No measurement of StackMix or line recombination for fine-tuning a VLM or TrOCR.

## Q9. Cross-technique ranking toward a 35–40% relative edit reduction

### Takeaway
No single measured technique reliably delivers 35–40% on its own at this data scale. The evidence-weighted ranking is:
1. Cross-model output ensembling (ROVER/MBR/voting): 14–34% measured on small-data HTR.
2. RL with an edit-distance reward after SFT: 11–35% measured, but only on large document-parsing sets.
3. Confidence-filtered pseudo-labelling with transductive test adaptation: up to 55%, but only with huge unlabelled pools.
4. StackMix-style synthesis: 10–45% for CTC models; unmeasured for VLMs.
5. Resolution tuning, TTA, and model size / LoRA-rank changes: small or unmeasured.

### Cited Findings
- Ensemble voting: −17% (TrOCR, 3.6k lines) — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1); −18% to −34% (5-fold, 250–1,000 lines) — [arXiv 1802.10033](https://arxiv.org/pdf/1802.10033).
- RL after SFT: −11% text edit distance — [FD-RL](https://arxiv.org/pdf/2601.08834); −34% with a plain edit-distance reward and −65% with a shaped reward — [DocPO](https://arxiv.org/html/2608.00536v2).
- Self-training with 1.14M unlabelled lines — [AT-ST](https://arxiv.org/abs/2104.13037):
  - "Big" setup: −20% (2.58 -> 2.07), or −25% with a language model (1.94).
  - "Small" setup, one round: −31% (6.43 -> 4.41).
  - "Small" setup, two rounds plus a language model: −55% (6.43 -> 2.88).
- StackMix, with or without blots: −10% to −45% over standard augmentation — [StackMix](https://arxiv.org/abs/2108.11667).
- TTA: −9% CER — [arXiv 2307.00664](https://arxiv.org/abs/2307.00664).

### Inferences
- The only well-evidenced route to ~35–40% is stacking roughly independent levers. For example, a diverse-model ensemble (≈15–25%) combined with RL or pseudo-label-refreshed models (≈10–20%) could compound to ≈25–40%. This is speculative arithmetic, not measured.
- Because the user's errors concentrate on abbreviations, superscripts, casing and historical spelling, ensembles and edit-distance RL act on exactly those token-level choices. Resolution and model size do not obviously address them.

### Gaps
- No study combines these levers on a VLM for handwritten lines, so how the gains compound is unknown.
