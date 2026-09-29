# Winning Solutions of Handwriting / Historical Text Recognition Competitions (2018–2026) and Their Measured Ablations

Scope note: this file only records measured numbers (baseline -> result) with the competition, metric and data size wherever the source gives them. Target transfer setting: Zindi R.O.A.D. Barbados (line-level, ~4K training lines, 17th–18th-c. English legal hands, currently a fine-tuned Qwen-VL LoRA).

## Q1. Which competitions are comparable to R.O.A.D. (line-level, few-thousand-line, early-modern handwriting)?

### Takeaway
The closest match is AI Journey 2020 "Digital Peter": line-level, 6,237 training lines, early-18th-c. manuscripts, CER-ranked, ~1,000 teams, and both winning write-ups are public. Other useful comparisons are ICFHR 2018 READ (few-shot, per-document adaptation), the Gwalther 16th-c. Latin benchmark (3,603 training lines, the same size as R.O.A.D.), and ICDAR 2026 CMMHWR, the first HTR competition found where a Qwen-based VLM was entered against dedicated HTR models. The ICDAR 2024/2025 handwriting competitions (Indic HWD, BRESSAY, Ciphers, Handwritten Notes QA) are less comparable, and their per-team methods are poorly documented publicly.

### Cited Findings
- **Digital Peter (AI Journey 2020, Sber):** 9,694 line images from Peter the Great's manuscripts, 1709–1713, with 265,788 characters and ~50,998 words. Split: 6,237 train / 1,930 val / 1,527 test pairs. Nearly 1,000 teams took part. Ranking was by CER, with WER then string accuracy as tie-breakers. The organizers' baseline (7 CNN layers + 2 BiGRU + CTC) scored CER 10.5 / WER 44.4 / ACC 21.7. — [Potanin et al., Digital Peter paper, arXiv 2103.09354](https://arxiv.org/abs/2103.09354)
- The Digital Peter GitHub reports its own baseline at CER 9.786 / WER 44.222 / ACC 21.532. The final leaderboard is only published as an image. — [ai-forever/digital_peter_aij2020](https://github.com/ai-forever/digital_peter_aij2020)
- **ICFHR 2018 Competition on Automated Text Recognition on a READ Dataset:** 22 heterogeneous single-writer documents (Italian, modern and medieval German), ~25 pages per document. The general set has 17 documents; 5 documents are held out for adaptation. Entries were submitted after adapting on 0, 1, 4 or 16 document-specific pages, and each test document has 15 pages. — [Yousef et al., arXiv 1812.11894 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1812.11894)
- **ICDAR 2026 Competition on Multilingual Medieval Handwriting Recognition (CMMHWR26):**
  - Task 1: 8 Romance languages (incl. Latin), test data similar to training.
  - Task 2: Occitan, not present in training.
  - Task 3: Czech, a non-Romance language.
  - Ranking uses the unweighted average of per-language CER/WER. The results deadline was 3 April 2026 and results were presented in Vienna, 31 Aug–2 Sep 2026.
  - Sources: [CMMHWR26 site](https://cmmhwr26.inria.fr/); [Zenodo test set](https://zenodo.org/records/19884972)
- **ICDAR 2024 Competition on Handwriting Recognition of Historical Ciphers:** low-resource HTR on ~600 cipher pages. Best reported CERs were 1.62% (Task 2B, >100 symbols) and 5.61% (Task 3B, the Ramanacoil cipher, few training pages). Participant methods included a transformer that detects characters in parallel, and a seq2seq CRNN (conv + LSTM encoder, attention decoder). — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-70552-6_20); [RRC overview](https://rrc.cvc.uab.es/?ch=27&com=introduction) (figures taken from search-result summaries; the full paper was unreachable)
- **ICDAR 2024 BRESSAY (Brazilian Portuguese essays; line, paragraph and page levels):** 14 participants; 4 groups submitted 11 proposals. The summary credits "preprocessing techniques, synthetic data approaches, and advanced deep learning models", but per-team numbers were not accessible. — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-70552-6_21)
- **ICDAR 2024 Recognition and VQA on Handwritten Documents (Indic):**
  - Task A (isolated words): won by TSNUK with 98.00% CRR / 94.26% WRR.
  - Task B (page level): won by IndependentOCR with 76.32% PCRR / 62.57% PWRR.
  - Winner methods were not retrieved.
  - Source: [Springer PDF](https://link.springer.com/content/pdf/10.1007/978-3-031-70552-6_26.pdf)
- **AI Journey 2021 Fusion Brain:** included an HTR sub-task on 94,130 Russian+English image–text pairs from school notebooks. 41 teams made 513 submissions. It is a multitask competition, so it is not a pure HTR comparison. — [Fusion Brain overview, arXiv 2111.10974](https://arxiv.org/abs/2111.10974)
- **ICDAR 2025 Handwritten Notes Understanding:** evidence-grounded QA over 2,000+ STEM handwritten-note images, with 6 valid submissions. This is VQA, not transcription, so it is not comparable. — [Competition report](https://cvit.iiit.ac.in/images/ConferencePapers/2025/ICDAR_2025_Competition_Report_HTRU.pdf)
- **Kaggle Kuzushiji Recognition (2019):** character detection and classification on classical Japanese pages, scored by F1. It is a different task type (not line transcription), but its ensembling and pseudo-label deltas are documented. — [lopuhin 2nd-place repo](https://github.com/lopuhin/kaggle-kuzushiji-2019)
- **Gwalther dataset (not a competition, but the closest size match):** 16th-c. Latin, 4,037 lines total with 3,603 for training. Augmentation and ensembling are ablated in full. — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)

### Inferences
- Digital Peter is the best proxy for R.O.A.D. Both are line-level, the training sets are similar in size (6.2K vs ~4K lines), both are 17th–18th-c. hands with archaic orthography, the metric is CER, and the competition was large enough for the winning margins to mean something.
- ICFHR 2018 READ and Gwalther show what adapting to a specific document or writer, and ensembling, are worth at this data scale.

### Gaps
- No Zindi or DrivenData HTR/OCR competition with published winner write-ups was found. The only Zindi HTR competition surfaced is R.O.A.D. itself.
- Per-team method tables for BRESSAY 2024, the ICDAR 2024 Ciphers competition and ICDAR 2024 HWD could not be retrieved: the full papers were paywalled, or the servers refused or timed out.
- The AI Journey 2021 HTR sub-task winners were not identified.

## Q2. What did the winners do, and what was each component measured to contribute?

### Takeaway
In the most comparable competition (Digital Peter, 6.2K lines), the public-LB winner reached CER 2.53 against 4.86 for 2nd place. Its authors' paper attributes the gain to synthetic line generation (StackMix) plus strike-through "blot" augmentation, which on its own took a single CTC model from 4.44 to 2.50 CER (−44% relative).

Other measured deltas at this data scale:
- A character 6-gram LM with beam search cut CER 5.7 -> 3.7 (−35%) on a CRNN. A larger interpolated LM brought a further 3.7 -> 3.5.
- A 3-model ensemble gave −3.3%.
- ICFHR 2018 READ was won by a single CTC fully convolutional network (no LM, no ensemble), with strong augmentation and document-specific fine-tuning (CER 25.35 -> 5.82 going from 0 to 16 pages).

### Cited Findings

#### Digital Peter (AIJ 2020): 1st place on the public LB (team Shonenkov, Karachev, Smolin, Novopoltsev)
- **Leaderboard:** public LB CER 2.531%, WER 13.5%, ACC 62.107% (32 s inference). Holdout: CER 2.382%, WER 13.001%, ACC 64.557%. The repository contains `ctc_labeling.py` and `flip.py`. — [lolpa1n/digital-peter-ocrv](https://github.com/lolpa1n/digital-peter-ocrv)
- **Method paper by the same authors:** "StackMix and Blot Augmentations for HTR" (Shonenkov, Karachev et al.).
  - Architecture: ResNet-34 (first 3 blocks) -> 3 BiLSTM layers -> 2 FC layers with GELU, trained with CTC.
  - Input: 128 × 2048 px.
  - Training length: Base 300 epochs, StackMix 1,000 epochs.
  - Single model; no LM or ensemble is reported.
  - Source: [arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)
- **StackMix mechanism:** tokenizes an external text corpus, matches tokens to character crops cut from the training lines, and stacks the crops into new synthetic line images. The Digital Peter corpus was ~3M lines of 17th–18th-c. texts. HandWritten Blots draw Bézier-curve strike-throughs (proba 0.5, 1–11 blots, intensity 0.9, transparency 0.95). — [arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)
- **Measured ablation, Digital Peter test (1,930 lines), same model:**

  | Setting | CER | WER | ACC |
  |---|---|---|---|
  | Base | 4.44 | 24.3 | 43.7 |
  | + standard augs (CLAHE, JPEG, Rotate) | 4.15 | 23.0 | 45.7 |
  | Blots only | 3.39 | 19.3 | 51.9 |
  | StackMix only | 3.40 | 19.2 | 51.6 |
  | All combined | 2.50 | 14.6 | 60.8 |

  Source: [arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)
- **Same ablation on other historical sets (CER, Base -> All):**
  - Bentham: 2.99 -> 1.73 (standard augs alone: 2.85)
  - IAM-B: 5.80 -> 3.77
  - IAM-D: 4.55 -> 3.01
  - Saint Gall: 4.71 -> 3.65
  - HKR: 6.71 -> 3.49. Here Blots alone hurt (7.40); StackMix gave most of the gain (3.69).
  - Source: [arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)

#### Digital Peter: 2nd place (vadimtimakin / t0efL)
- **Setup:** an ensemble of 3 models built on an "OCR-transformer pipeline" (CNN backbone + transformer):
  - DenseNet161 with smart resize
  - ResNeXt101 with smart resize
  - ResNeXt101 with default resize
  - Training: ~100 epochs each, ~10 h per model on Colab Pro.
  - Post-processing: dictionary correction with a 9k-word dictionary. A 160k-word dictionary was dropped because of submission time limits.
  - Source: [2nd-place repo](https://github.com/vadimtimakin/2nd-place-solution-Digital-Peter)
- **Measured ensemble effect (public LB CER):** best single model 5.025 (DenseNet161). Other singles scored 5.047 and 5.286. The 3-model ensemble plus dictionary scored 4.861 (−0.164 abs, −3.3% relative vs the best single). Private LB: CER 4.814 / WER 24.72 / ACC 45.94. Their best model (5.011 CER) was left out of the ensemble for runtime reasons. — [2nd-place repo](https://github.com/vadimtimakin/2nd-place-solution-Digital-Peter)

#### Digital Peter: top-participant methods analysed in the organizers' paper
The paper states: "We analysed the best solutions provided by participants."
- **Input and augmentation:** fixed height 128 with variable width (aspect ratio kept). Albumentations: ±4° rotation, width scaling 95–105%, grid distortion.
- **Architecture changes:** BatchNorm throughout the CNN, reflection padding matched to the receptive field, subsampling ×4.
- Source: [arXiv 2103.09354](https://arxiv.org/abs/2103.09354)
- **Measured tuning (CRNN, Digital Peter test):**
  - GRU 256 -> 368 units (Adam): CER 7.1 -> 6.6
  - Adam -> SGD with momentum + cyclical LR (1e-6 -> 1e-2 -> 1e-6 over the first 4 epochs), weight decay 1e-2: CER 6.6 -> 5.7, WER 37.2 -> 33.1, ACC 29.0 -> 33.1
  - Source: [arXiv 2103.09354](https://arxiv.org/abs/2103.09354)
- **Measured LM effect:** CTC beam search with a character 6-gram LM (SRILM), objective log P(c|x) + α·log P_lm(c) + β·|c| with α = 0.8, β = 2.0. More than 100 beams brought no gain.

  | Model (optimizer) | LM | CER | WER | ACC |
  |---|---|---|---|---|
  | Adam | none | 6.6 | 37.2 | 29.0 |
  | Adam | Small (93% of train text) | 4.2 | 22.6 | 47.3 |
  | SGD | none | 5.7 | 33.1 | 33.1 |
  | SGD | Small | 3.7 | 20.7 | 50.1 |
  | SGD | Large (Small interpolated with 17th-c. texts from GramEval 2020) | 3.5 | 19.4 | 52.3 |

  A word-level, closed-vocabulary LM was judged useless: there was a >20% OOV rate on held-out text, and word LMs optimise WER rather than CER. — [arXiv 2103.09354](https://arxiv.org/abs/2103.09354)

#### ICFHR 2018 READ competition: winner (Yousef et al., gated fully-convolutional CTC network)
- **Method:** fully convolutional (no recurrence) with 16 stacked GateBlocks, trained with CTC. Augmentation: projective transforms, elastic distortion, sign flipping. Single model, no ensemble. No LM: "we do not utilize any form of language modeling." — [arXiv 1812.11894](https://ar5iv.labs.arxiv.org/html/1812.11894)
- **Measured CER by number of document-specific adaptation pages (0 / 1 / 4 / 16):**

  | Entry | 0 | 1 | 4 | 16 |
  |---|---|---|---|---|
  | Winner, 16-layer (the competition entry) | 25.35 | 12.63 | 8.28 | 5.82 |
  | 8-layer variant | 25.91 | 12.59 | 8.37 | 6.82 |
  | OSU (2nd) | 31.40 | 17.73 | 13.27 | 9.02 |
  | LITIS | 35.29 | 22.51 | 16.89 | 11.34 |
  | PRHLT | 32.79 | 22.15 | 17.90 | 13.33 |
  | ParisTech | — | — | — | 14.72 |

  The authors claim a >25% relative CER decrease over 2nd place. — [arXiv 1812.11894](https://ar5iv.labs.arxiv.org/html/1812.11894)

#### Kaggle Kuzushiji Recognition (2019): 2nd place (lopuhin)
- **Approach:** a class-agnostic Faster-RCNN detector (~0.99 validation F1), then classification on large crops with ResNet152 / ResNeXt101-WSL backbones, FPN and RoI-align.
- **Measured progression (public LB F1):**
  - best single classifier: 0.935
  - ResNet152 + ResNeXt101 blend: 0.941
  - larger blend: 0.943
  - second-level LightGBM + XGBoost stacker: 0.949
  - fine-tuning with pseudo-labels gave a single model of 0.938
- **LM:** a character language model was abandoned (log loss ~4.5 vs ~0.5 for the image model).
- Source: [lopuhin/kaggle-kuzushiji-2019](https://github.com/lopuhin/kaggle-kuzushiji-2019)
- **1st place:** Cascade R-CNN with HRNet-w32/w48, per a search snippet. The write-up itself was not fetched. — [Kaggle discussion 112788](https://www.kaggle.com/c/kuzushiji-recognition/discussion/112788)

#### Gwalther (16th-c. Latin, 3,603 training lines): TrOCR augmentation and ensemble ablation
- **Single-augmentation ablation (CER):**
  - TrOCR-base with no augmentation: 1.93
  - Improved: random rotation 1.86; elastic distortion 1.86
  - Worse: underline 2.03, Gaussian blur 2.04, erosion 2.04, re-resize 2.09, random affine 2.13, random perspective 2.27, dilation 2.31, resize 2.31
  - Source: [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)
- **Ensembles (voting over beam hypotheses):**
  - 11-model full voting: CER 1.66
  - Top-5 voting (elastic, rotation, underline, blur and baseline models, top-5 beams each, majority sentence): CER 1.60. That is −14% relative vs the best single (1.86) and −17% vs the unaugmented baseline.
  - The prior state of the art was HTR+ at 2.74 and the previous TrOCR at 3.18.
  - Source: [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)

### Inferences
- The largest measured single lever at R.O.A.D.-like scale was synthetic line data built from the competition's own character crops plus a period-appropriate text corpus (StackMix: −23% alone, −44% with blots).
  - For 17th–18th-c. English legal hands, the analogous corpus would be early-modern English legal/probate text (e.g., EEBO-TCP-style transcriptions or the R.O.A.D. training text itself).
  - This is an inference; no source tested English legal hands.
- The winner-vs-2nd gap on Digital Peter (2.53 vs 4.86 CER) was far larger than anything ensembling or dictionary post-processing bought 2nd place (−3.3%). Data synthesis and augmentation dominated.
- Augmentation choice matters in both directions. On Gwalther, 8 of 10 single augmentations hurt TrOCR fine-tuning. On HKR, blots hurt.
- The char-LM gain (−35%) was measured on a CTC CRNN with no internal LM. A Qwen-VL decoder already carries a strong LM prior, so the marginal gain from an external n-gram LM is likely much smaller. That gain is untested in the sources, and the prior can be harmful (see Q4 on normalization).

### Gaps
- The Digital Peter 1st-place team's final private-LB CER and its exact ensemble/TTA composition: the README content was not retrievable.
- The link between the winning repo and the StackMix paper is inferred from overlapping authors (Shonenkov, Karachev, Novopoltsev) and the `ctc_labeling.py` file. The paper does not state that it describes the winning entry.
- The 2nd-place dictionary post-correction was not ablated separately from the ensemble.
- Per-component ablations for the ICDAR 2024 Ciphers, BRESSAY and HWD winners were not available.

## Q3. Cross-cutting patterns: how much did augmentation/synthesis, LM, ensembling, pseudo-labelling and pretraining/adaptation add (measured)?

### Takeaway
Ranked by measured effect at the few-thousand-line scale:
1. **Synthetic lines plus targeted augmentation:** −35% to −48% relative CER.
2. **Pretraining, then fine-tuning on target lines:** very large when target data is scarce. For example, a 0 -> 16 page adaptation cut CER 25.35 -> 5.82, and 16–256 target lines give −20% to −50%.
3. **Character n-gram LM on CTC:** −35% relative.
4. **Ensembling:** −3% (3 models, Digital Peter) up to −14% (top-5 hypothesis voting, Gwalther).
5. **Pseudo-labelling:** smallest and least documented. It was the only measured-but-tiny gain found (+0.003 F1 on Kuzushiji).

Input resolution (line height) was one of the largest single hyper-parameter effects: −8 to −12 CER points.

### Cited Findings
- **Augmentation / synthesis:**
  - Digital Peter: 4.44 -> 2.50 CER (−44%). Standard photometric/geometric augs alone gave only 4.44 -> 4.15 (−6.5%). — [StackMix, arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)
  - Bentham: 2.99 -> 1.73 (−42%). IAM-B: 5.80 -> 3.77 (−35%). — [arXiv 2108.11667](https://arxiv.org/html/2108.11667v1)
  - Gwalther (3.6K lines): best single aug −3.6% (1.93 -> 1.86); several augs worsen CER by up to +20% (dilation/resize 2.31). — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)
  - Writer adaptation (CzechHWR pretrained model, 406K lines): with 64 target lines, fine-tuning gave ~25% relative CER reduction, and augmentation added ~10% more. Writer-dependent: 6% without vs 15% with augmentation. — [Kohút & Hradiš, arXiv 2302.06308](https://arxiv.org/html/2302.06308)
- **Language model:**
  - Digital Peter CRNN: char 6-gram LM cut CER 5.7 -> 3.7 (−35%), WER 33.1 -> 20.7. Adding external 17th-c. text to the LM gave 3.7 -> 3.5. — [arXiv 2103.09354](https://arxiv.org/abs/2103.09354)
  - The ICFHR 2018 READ winner used no LM. — [arXiv 1812.11894](https://ar5iv.labs.arxiv.org/html/1812.11894)
  - Kuzushiji 2nd place dropped its LM (no gain). — [lopuhin repo](https://github.com/lopuhin/kaggle-kuzushiji-2019)
- **Ensembling:**
  - Digital Peter 2nd place, 3 CNN-transformer models: 5.025 -> 4.861 public CER (−3.3%). — [repo](https://github.com/vadimtimakin/2nd-place-solution-Digital-Peter)
  - Gwalther: top-5 hypothesis voting over 5 models gave 1.86 -> 1.60 (−14% vs best single). The 11-model vote gave 1.66, so adding weaker members hurt. — [arXiv 2508.11499](https://arxiv.org/html/2508.11499v1)
  - Kuzushiji: blending plus a 2nd-level stacker raised F1 0.935 -> 0.949. — [lopuhin repo](https://github.com/lopuhin/kaggle-kuzushiji-2019)
- **Pseudo-labelling / label noise:**
  - Kuzushiji 2nd place: pseudo-label fine-tuned single model 0.938 vs 0.935 best single, public F1. — [lopuhin repo](https://github.com/lopuhin/kaggle-kuzushiji-2019)
  - Aradillas et al.: transfer learning + data augmentation + a label-noise mitigation algorithm together cut CER by "up to 6% in some cases" on ICFHR 2018 READ, Washington and Parzival. Individual contributions were not isolated in the abstract. — [arXiv 2012.02544](https://arxiv.org/abs/2012.02544)
- **Pretraining / adaptation data:**
  - ICFHR 2018 winner CER by adaptation pages: 25.35 (0) -> 12.63 (1) -> 8.28 (4) -> 5.82 (16). — [arXiv 1812.11894](https://ar5iv.labs.arxiv.org/html/1812.11894)
  - Autoregressive (transformer) HTR fine-tuning "can reliably start with just 16 lines", giving ~10% relative CER improvement at 16 lines and up to 40% at 256 lines. Confidence-based active line selection halves annotation cost. — [arXiv 2503.19546](https://arxiv.org/abs/2503.19546)
  - PP-OCRv6 pretrained on ~6.2M real + ~1M synthetic lines: fine-tuning on only 25% of CATMuS (~142K training lines) beat every model variant trained from scratch on 100% of CATMuS, across all 7 test sets. — [Kiessling, arXiv 2609.20064](https://arxiv.org/pdf/2609.20064)
- **Input resolution:** line height 48 -> 96 px cut CMMHWR Task 1 macro CER by 8.3–11.5 points (PP-OCRv6 medium 18.35 -> 10.10; tiny 24.05 -> 12.57). The optimizer change (Muon+AdamW vs Adam) was roughly neutral. — [arXiv 2609.20064](https://arxiv.org/pdf/2609.20064)
- **Optimization:** SGD + cyclical LR vs Adam: CER 6.6 -> 5.7 (−14%) on a Digital Peter CRNN. — [arXiv 2103.09354](https://arxiv.org/abs/2103.09354)

### Inferences
- For a ~4K-line competition, the evidence supports this priority order:
  1. Domain-matched synthetic lines and strike-through/blot-style augmentation, with each augmentation validated individually.
  2. Maximising effective input resolution for the line image. For Qwen-VL this likely means raising min_pixels/max_pixels so line height is not downsampled below ~64–96 px equivalent. This is inferred from the CTC result, not tested on a VLM.
  3. A small ensemble of the best 3–5 diverse models, combined by voting over hypotheses (ROVER / majority string), rather than every checkpoint.
  4. Pseudo-labelling only if unlabelled in-domain lines exist; its measured value in competitions was marginal.
- The measured ensemble gains differ by model family: −3% for CTC/transformer model averaging and −14% for sequence-level voting. Hypothesis-level voting across diverse fine-tunes is the better-evidenced recipe for autoregressive decoders like Qwen-VL.

### Gaps
- No HTR competition winner write-up was found with a clean pseudo-labelling ablation on line transcription (baseline -> with PL).
- No measured test-time augmentation (TTA) delta was found for any HTR competition winner.
- No source measured n-gram LM fusion or lexicon post-correction on top of a fine-tuned VLM decoder.

## Q4. 2025–2026 evidence: VLM-based (Qwen-VL etc.) vs dedicated HTR models, with numbers

### Takeaway
The results are mixed and depend on setup:
- **Fine-tuned VLMs win against zero-shot VLMs and commercial OCR** at page level (CHURRO, Qwen2.5-VL-3B: 70.1 vs Gemini 2.5 Pro 63.6 on handwritten; English handwritten 84.0 vs 80.0).
- **On line-level medieval HTR (ICDAR 2026 CMMHWR), a 16M-parameter CTC-style recognizer beat the Qwen3.5-9B "Medusa" entry** on Latin, Spanish and French: 8.70 vs 13.07 CER on Latin, 7.99 vs 11.19 on Spanish. Medusa won only on languages it had been specifically trained on (Occitan, Czech).
- **Zero-shot LLMs trail Transkribus on English historical (Bentham)**: 7.07 CER for Transkribus vs 8.01 for the best LLM, Qwen2-VL-7B.
- **VLMs tend to normalize historical spelling.**

### Cited Findings
- **CHURRO (EMNLP 2025).** Metric: page-level normalized Levenshtein similarity (higher is better).
  - Training: Qwen2.5-VL-3B fine-tuned for 5 epochs on CHURRO-DS (99,491 pages, 155 corpora, 46 language clusters), 32×H100, ~25 h, LR 5e-5 cosine, effective batch 128, images capped at 5,120 patches of 28×28 px.
  - Handwritten: average 70.1, which is +27.2 over zero-shot Qwen2.5-VL-3B. Zero-shot baselines: Gemini 2.5 Pro 63.6, Qwen2.5-VL-72B 54.5, Azure OCR 47.7.
  - English handwritten: CHURRO 84.0 (+16.5 over zero-shot 3B), Gemini 2.5 Pro 80.0, Qwen2.5-VL-72B 77.0, Azure OCR 73.5.
  - Printed: average 82.3 (+14.5 over zero-shot 3B) vs Gemini 2.5 Pro 80.9.
  - Zero-shot 3B beat zero-shot 72B on printed text.
  - Sources: [CHURRO, arXiv 2509.19768](https://arxiv.org/pdf/2509.19768); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1763/)
- **ICDAR 2026 CMMHWR, Medusa entry.** Medusa ("MEDUSA 0.1"), a Qwen3.5-9B line recognizer submitted to CMMHWR26, compared with CATMuS-fine-tuned dedicated recognizers.
  - CER results:

    | Test set | PP-OCRv6-medium (16M params) | kraken CRNN | Medusa (Qwen3.5-9B) |
    |---|---|---|---|
    | Task 1 Latin | 8.70 | 12.61 | 13.07 |
    | Task 1 Spanish | 7.99 | 10.47 | 11.19 |
    | Task 1 French | 3.36 | 5.01 | 3.74 |
    | Task 2 Occitan | 6.62 | — | 5.70 (Medusa better) |
    | Task 3 Czech | 18.05 | — | 15.73 (Medusa better) |
    | CoMMA Latin | 11.30 | — | 16.08 |
    | CoMMA French | 21.36 | — | 28.24 |

  - Medusa's self-reported Task 1–3 CERs (8.03 / 5.24 / 10.8) "could not be reproduced" with its released pipeline.
  - PP-OCRv6-medium is ~45× faster than Medusa.
  - In extreme few-shot adaptation (20 lines), the kraken CRNN improved faster than PP-OCRv6.
  - Source: [Kiessling, arXiv 2609.20064](https://arxiv.org/pdf/2609.20064)
- **Zero-shot LLMs vs Transkribus (Crosilla, Klic, Colavizza, 2025), CER:**

  | Dataset | Transkribus | LLMs |
  |---|---|---|
  | Bentham (English, 18th–19th c.) | Text Titan I: 7.07 | Qwen2-VL-7B 8.01; GPT-4o-mini 9.48; Claude 3.5 Sonnet 10.97; GPT-4o 16.62 |
  | IAM (modern) | Text Titan I: 9.13 | GPT-4o-mini 1.71 |
  | READ2016 (German historical) | Text Titan I: 40.63 | 71–92 |

  - Using an LLM to self-correct its own output was inconsistent. Open-source models got worse. The best case was GPT-4o on ICDAR2017 (−8% CER), and it remained unusable.
  - Source: [arXiv 2503.15195](https://arxiv.org/pdf/2503.15195)
- **TrOCR vs Qwen2.5-VL on 18th-c. English lines.** Qwen achieves lower CER/WER and is more robust to degraded input, but shows "selective linguistic regularization and orthographic normalization" that can silently alter historical forms. TrOCR preserves orthography but suffers cascading errors. — [arXiv 2602.14524](https://arxiv.org/abs/2602.14524)

### Inferences
- For R.O.A.D. (English, early-modern legal hands, ~4K lines), the evidence points to a fine-tuned Qwen-VL as a reasonable base. English is the language where VLMs are strongest, and CHURRO gained +16.5 on English handwriting from fine-tuning a 3B model.
- The CMMHWR26 comparison warns that a large VLM line recognizer can lose to a much smaller dedicated recognizer on in-distribution Latin-script lines.
- Two transferable, low-risk additions follow from this:
  1. Train a dedicated CTC/transformer line model (e.g., CRNN or PP-OCR-style at ≥96 px height, with StackMix-style synthesis) as a second system, and combine it with the Qwen-VL output by hypothesis voting or confidence selection.
  2. Explicitly guard against orthographic normalization, e.g., by checking whether the VLM "modernizes" spellings, abbreviations and long-s, since a CER metric against diplomatic ground truth penalizes it.
- These are inferences, not measured on R.O.A.D.

### Gaps
- No official CMMHWR26 final leaderboard or winner identity could be retrieved: the Codabench page did not render, and the HAL system report for Medusa returned "Access Denied". It is therefore unconfirmed whether a VLM won any CMMHWR26 task.
- No 2025–2026 competition was found with an HTR leaderboard dominated by a Qwen-VL LoRA fine-tune, with published numbers.
- CHURRO does not compare against fine-tuned dedicated HTR models (PyLaia/Transkribus/kraken) on the same pages, and its metric is page-level similarity, not line CER.
