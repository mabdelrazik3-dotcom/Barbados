# Whole-word accuracy decides the final 0.03

No one has published a measured recipe that scores above 0.93 on R.O.A.D. Getting from a ~0.905 Qwen-VL LoRA to the public #1's **0.9399** therefore means stacking bets on a thin layer of measurement. The arithmetic is unforgiving. #1 makes **37% fewer word edits and 45% fewer character edits per line** than the best-documented public single VLM. At every level above 0.90, about **three-quarters of the lost score comes from the word term**: the metric prices one word edit like 4.58 character edits, and top systems average only ~1.5 character edits per wrong word, so the contest is decided by single-letter slips that make whole words wrong. The only technique measured on this competition to add more than a few thousandths is candidate-level ensembling (5-fold + MBR: **+0.012 on the leaderboard**). Every text rule anyone posted (lexicon recasing, OOV snapping, lexicon rescoring, markup fixes, ink-band cropping) measured neutral or harmful. Measured levers plausibly carry a 0.905 system to about 0.913–0.918. The rest has to come from transductive pseudo-labelling, a more diverse backbone, GRPO and resolution choices: top teams' gated artefacts show these, but nobody has ablated them publicly. The ~1.5% mislabelled lines are worth detecting mainly to repair validation, since 12 of 820 held-out lines carry 18% of character edits, and to keep them out of RL and pseudo-label pools. Excluding them from training is allowed if documented, but comparable studies at ~1% noise measured the training gain as negligible. The six-day plan spends Monday on a noise-aware evaluation harness, Monday–Tuesday on an MBR ensemble scored with the exact competition loss, Wednesday–Friday on one pseudo-label round plus one diverse member, and the weekend on freezing and choosing two submissions by cross-validation.

*Reading convention: **measured** means someone scored it on real data and published the number. **Computed** means my own arithmetic on published numbers. **Inference** marks everything else, including every expected gain for your system.*

## A 0.905 system must remove about a third of its errors, and no one has shown how

The 28 September leaderboard puts YoussefTrabelsi at **0.939889**, 0.0043 clear of Brainiac at 0.935549, with #3 to #24 packed between 0.9339 and 0.9302. There are 24 entries at or above 0.93, 124 at or above 0.92 and 244 at or above 0.91 ([Zindi leaderboard](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)). The score is a deterministic function of mean word and character edits per line: score = 1 − 0.5·(WER_w/12 + CER_w/55). J0NNY posted the formula and another team reproduced its own submissions with it to nine decimals ([thread 33847](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847); [thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). The hidden WER_w/CER_w columns therefore show exactly where each team loses points. The table uses the best-documented public VLM, a Qwen2.5-VL-32B LoRA whose published validation predictions score 0.9018, as a proxy for a ~0.90 system ([EYEDOL val predictions](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter/blob/main/val_predictions_qwen_32b_hr.csv)).

| System | Score | WER_w | CER_w | Lost to words | Lost to characters | Word share of loss | Char edits per word edit |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-32B LoRA (val, 210 lines) | 0.9018 | 1.729 | 2.881 | 0.0720 | 0.0262 | 73% | 1.67 |
| #50 EGY_TEAM | 0.9270 | 1.3203 | 1.9773 | 0.0550 | 0.0180 | 75% | 1.50 |
| #2 Brainiac | 0.9355 | 1.1628 | 1.7602 | 0.0485 | 0.0160 | 75% | 1.51 |
| #1 YoussefTrabelsi | 0.9399 | 1.0943 | 1.5966 | 0.0456 | 0.0145 | 76% | 1.46 |

All loss splits and ratios are computed from the formula. The 32B row covers only 210 lines, so its own sampling sd is about ±0.01 (computed by scaling the 0.0048 measured at 820 lines).

To match #1 from the proxy, word edits per line must fall by **37%** and character edits by **45%** (computed). Expressed as a uniform cut in lost points, a 0.905 system needs 37% and a 0.910 system needs 33%. The step from #50 to #1 is only a 17% cut in word edits and 19% in character edits. That is why the top 50 look close in score but are far apart in engineering. Public single-model evidence clusters at your level. Qwen2.5-VL-7B LoRA r32, trained for two epochs at 256 px height, scored 0.9004 on validation and **0.90413 on the public board** ([ModarIbrahim card](https://huggingface.co/ModarIbrahim/road-qwen25vl-lora-e3)). The 32B adapter's eval loss bottomed at epoch 2 (0.416) and climbed to 0.645 by epoch 5 ([EYEDOL trainer state](https://huggingface.co/EYEDOL/barbados-qwen_32b_hr_adapter/blob/main/last-checkpoint/trainer_state.json)). A Qwen3-VL-8B with LoRA r8 trained for one epoch scores only 0.850 on its own published predictions (computed) ([mlai-dante](https://huggingface.co/mlai-dante/road-model-public)), so configuration alone spans 0.05. Size is not the lever. The 32B's 0.9018 and the 7B's 0.9004 are within one sd of each other (computed), which matches the pattern that a fine-tuned 3B VLM can beat a zero-shot 72B on historical handwriting ([CHURRO](https://arxiv.org/abs/2509.19768)).

Nobody near the top has explained how they got there. Only three of the 24 forum threads contain measured modelling numbers. The two posters who disclosed recipes now sit around 0.915: Damilare06 at 0.9162, and dantebhai at 0.9145, whose "qwen + lora + grpo and beam search" scored "slightly above 91" ([thread 34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350); [leaderboard](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)). The methods of the 0.927–0.931 tier are visible only as gated Hugging Face file lists.

The vanijonny repositories are attributed to #19 Ynnoj by username only. Their file lists show:

| Model or technique | Details in the file names |
|---|---|
| olmOCR-2 LoRAs | 112 px height, ranks 32–128, 5- and 10-fold splits, three augmentation seeds, beam-4 test decoding |
| Other backbones | PaddleOCR-VL-1.6; Qwen3-VL-8B at 128 px; Qwen3-VL-32B |
| Gemma | Greedy-versus-beam studies |
| Training extras | GRPO from an SFT initialisation; "distill_pl" distillation with pseudo-labels; a 442 MB reranker |

Sources: [vanijonny model tree](https://huggingface.co/api/models/vanijonny/barbados_htr_model/tree/main); [sweeps](https://huggingface.co/api/models/vanijonny/barbados-htr-sweeps/tree/main).

CalebE, probably #53, adds 5-fold olmOCR and Qwen3-VL runs with TTA predictions, an HTR "judge" and a ByT5 corrector ([CalebE repo](https://huggingface.co/api/models/CalebE/olmocr-barbados)). Every result file is access-restricted. These are hypotheses consistent with the artefacts, not measured gains. #1, #6 and #9 reached the top with only 25–27 submissions each. That points (inference) to a strong offline validation loop and a step-change in recipe rather than leaderboard probing.

The public board is also noisy. The public split is roughly 275–412 of the 1,374 test lines; the rules say both 20% and 30% ([competition info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)). Scaling the measured sd of 0.0048 at 820 lines gives an absolute public-score sd of about **0.007–0.008** (computed from [thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). The whole #2–#50 band spans 0.0085, about 1.2 sd, so the private board will reshuffle it, and #1's lead is not safe either. Paired comparisons of your own systems on the same lines are far less noisy, which is why the two final picks should come from cross-validation rather than public score.

## One word edit costs 4.58 character edits, so single-letter slips decide the top

In every row of the table, **73–76% of lost score comes from the word term** (computed). Cutting 0.1 word edits per line is worth **+0.0042**; cutting 0.1 character edits per line is worth only **+0.0009** (computed from the formula). Top systems make only **1.46–1.51 character edits per word edit** (computed for #1, #2 and #50). The typical remaining error is therefore a one- or two-letter slip, and the word term prices it as a whole wrong word. Removing a single-letter error from one line in ten is worth about +0.005 (computed: 0.1 × 0.5 × (1/12 + 1/55)). That is more than any post-processing rule has measured on this data.

Casing shows why text rules cannot capture that value. Damilare06 measured casing at about **13% of word errors** and a perfect casing oracle at **+0.014**, the largest error class anyone has quantified on R.O.A.D. Yet recasing from a corpus lexicon **lost at every confidence threshold from 0.60 to 1.00, by −0.016 to −0.0006**, because the model writes lowercase for uppercase about as often as the reverse ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). The same post measured two more things:

- Markup slips (^ : &) are 3% of word edits. A perfect caret would be worth at most +0.0027 and a perfect colon +0.0016.
- **43% of wrong words have both prediction and truth in the training vocabulary.** So snapping out-of-vocabulary words to the nearest training word fixed 67 words, broke 79 and cost **−0.012** ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).

The inference is that word accuracy must come from the image, through better reading or better selection among readings, not from dictionaries. A related inference that nobody has measured: under token-level Levenshtein, a merged or split word ("ofthe") costs one character edit but two word edits, the equivalent of about nine character edits. Measuring the whitespace share of your out-of-fold word edits takes about an hour and could expose a cheap, heavily weighted error class.

The weighting also decides which way of combining models is right. jdalton ran a 27-model TrOCR ensemble with word-level ROVER voting. It cut WER_w from 2.245 to 2.215 but raised CER_w from 4.684 to 4.769. His local equal-weight metric scored that as a loss (3.465 → 3.492), yet the leaderboard moved **+0.0005** ([jdalton workflow](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)). That is exactly what the Zindi formula predicts (computed: 0.5 × (0.030/12 − 0.085/55) = +0.00048). Any candidate-selection step should therefore use the exact per-line Zindi loss as its utility, and every beam width, threshold and member weight should be tuned on that loss. Rate-based local metrics mislead badly: one participant saw a leaderboard WER_w of 2.89 while local jiwer reported 0.27 ([thread 33702](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33702)).

Because the metric averages absolute edits per line, longer lines weigh more ([competition info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)), and the largest measured pocket of loss is concentrated. The high-resolution batch (crop height above 150 px) is **27% of lines but 42% of character edits, and scores 0.823 against 0.900** for the rest ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). Parity would be worth about **+0.021** overall (computed: 0.27 × 0.077), but no remedy has been measured:

- Cropping to the ink band equalised pixels per character but bought only +0.0003 locally.
- An earlier band-crop change cost that team **0.009 on the leaderboard**.
- The ~0.05 gap persists at matched pixels per character and at matched line length ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)).

The tall batch is a target for ensembling and resolution experiments, not a problem with a known fix. Decoding is equally unresolved for VLMs. TrOCR-large with beam-4 scored 0.8554 against 0.8577 greedy ([AmanDeva scorecard](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)), diverse beam search hurt at 3 and 9 beams ([jdalton workflow](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)), and the decode studies in the likely top team's (vanijonny's) repositories are gated.

## Ensembling is the only lever measured here; every text rule measured flat or negative

The ledger below puts each technique that has a number next to what it measured on this data and elsewhere.

| Technique | Measured on R.O.A.D. | Measured elsewhere | Verdict |
|---|---|---|---|
| k-fold + MBR candidate selection | +0.0121 LB, TrOCR-large ([AmanDeva](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)) | −14% CER on 3.6K lines ([Gwalther](https://arxiv.org/html/2508.11499v1)); −18% to −34% with 5-fold voting ([Wick et al.](https://arxiv.org/pdf/1802.10033)) | Do first |
| Extra seeds | +0.003 LB, 3 seeds ([jdalton](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)) | — | Small |
| Diverse models vs resampling one model | — | +8.2% vs −2.8% ([CE-OCR](https://arxiv.org/abs/2504.11101)) | Prefer diverse members |
| Word ROVER, 27 members | +0.0005 LB, CER worse ([jdalton](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)) | — | Marginal |
| GRPO after SFT | "slightly above .91" with beam, no ablation ([34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350)) | −11% ([FD-RL](https://arxiv.org/pdf/2601.08834)) and −34% ([DocPO](https://arxiv.org/html/2608.00536v2)) text edit distance | Bet |
| Pseudo-labelling | None public | Up to −55% CER with a 1.14M-line pool ([AT-ST](https://arxiv.org/abs/2104.13037)); +0.003 F1 ([Kuzushiji](https://github.com/lopuhin/kaggle-kuzushiji-2019)) | Bet (allowed) |
| Input height | Tall-batch gap not fixed by cropping ([34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)) | 48→96 px: CER 18.35→10.10 for PP-OCRv6 ([Kiessling](https://arxiv.org/pdf/2609.20064)) | Bet |
| Lexicon recasing | −0.0006 to −0.016 | — | Fails |
| OOV snapping | −0.012 | — | Fails |
| Lexicon rescoring | 0.847 → 0.842 ([jdalton](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)) | Char LM −35%, but only for a CTC net ([Digital Peter](https://arxiv.org/abs/2103.09354)) | Fails |
| Markup rules | Ceiling +0.0027 | — | Noise |
| Ink-band crop | +0.0003 local, −0.009 LB | — | Fails |
| Synthetic-font pretraining | 0.8577 → 0.8269 ([AmanDeva](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)) | — | Fails |
| 2B VLM | 0.804–0.811 LB ([AmanDeva](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)) | — | Fails |

The one ensemble measured on this competition's leaderboard was 5-fold TrOCR-large with MBR consensus. It went from 0.8577 to **0.8698**, removing about **8.5% of lost points** (computed) ([AmanDeva scorecard](https://github.com/AmanDeva/Barbados-Historic-Handwriting-Challenge/blob/HEAD/Context.md)). Three seeds removed about 2% (+0.003) ([jdalton workflow](https://github.com/jdaltonll02/road_barbados_challenge/blob/HEAD/docs/WORKFLOW.md)). Published HTR voting results are stronger but variable:

- Top-5 hypothesis voting cut CER 14% against the best single model on the 3,603-line Gwalther set ([Gwalther](https://arxiv.org/html/2508.11499v1)).
- 5-fold sequence voting cut CER 18–34% on early printed books ([Wick et al.](https://arxiv.org/pdf/1802.10033)).
- The Digital Peter runner-up's 3-model ensemble managed 3.3% ([Digital Peter 2nd place](https://github.com/vadimtimakin/2nd-place-solution-Digital-Peter)).

Diversity explains the spread. Consensus across different VLMs averaged +8.2% on OCR benchmarks, while resampling one VLM averaged −2.8% ([CE-OCR](https://arxiv.org/abs/2504.11101)). Voting helped less for deeper, less diverse networks ([Wick et al.](https://arxiv.org/pdf/1802.10033)), and adding weaker members to the Gwalther vote made it worse (1.60 → 1.66). Applied to a 0.905 system, an 8.5–14% cut lands at **0.913–0.918** (inference). That is inside the 0.91–0.92 band where 120 teams already sit (computed from the leaderboard counts). **Measured levers alone do not reach 0.93.**

The levers that could close the rest have measured analogues, but none on this data.

**GRPO.** After SFT, GRPO cut text edit distance by 11% on Qwen3-VL-4B (FD-RL). With a plain edit-distance reward it cut 34% on Qwen2.5-VL-3B (DocPO). Both followed SFT on roughly half a million document samples and used LR 1e-6 with 8 rollouts. FD-RL also showed that keeping only the uncertain half of the RL pool mattered (overall score 88.47 → 90.41) ([DocPO](https://arxiv.org/html/2608.00536v2); [FD-RL](https://arxiv.org/pdf/2601.08834)). LightOnOCR's RLVR added 1.4 benchmark points and halved repetition loops ([LightOnOCR](https://arxiv.org/html/2601.14251)). None of this is line-level handwriting on 4K lines, and the one R.O.A.D. user who reported GRPO sits at 0.9145.

**Pseudo-labelling.** Pseudo-labelling on test images is explicitly allowed if automated ([thread 34459](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459)). Its large measured gains, however, needed unlabelled pools about 100 times the labelled set: AT-ST reached up to −55% CER with 1.14M lines ([AT-ST](https://arxiv.org/abs/2104.13037)). The only competition-measured pseudo-label gain was +0.003 F1 on Kuzushiji ([lopuhin repo](https://github.com/lopuhin/kaggle-kuzushiji-2019)). Here the pool of 1,374 test and 687 unlisted lines is half the labelled set. The case for pseudo-labelling therefore rests on writer adaptation: supervised fine-tuning on 16 to 256 lines of a new hand cut CER 10–40% for autoregressive HTR ([arXiv 2503.19546](https://arxiv.org/html/2503.19546)). That was measured with true labels, not pseudo-labels.

**Resolution.** Effects are large for dedicated recognisers: raising line height from 48 to 96 px cut PP-OCRv6-medium's macro CER from 18.35 to 10.10 ([Kiessling 2026](https://arxiv.org/pdf/2609.20064)). Qwen2-VL, by contrast, degrades when small images are upscaled beyond its training distribution ([Qwen2-VL report](https://arxiv.org/abs/2409.12191)). Top-team artefacts use 112–128 px heights, where the public 0.904 model used 256.

**A dedicated recogniser as a diverse member.** One striking measurement supports this. The 16M-parameter PP-OCRv6-medium beat a Qwen3.5-9B entry on in-distribution Latin lines, 8.70 against 13.07 CER ([Kiessling 2026](https://arxiv.org/pdf/2609.20064)), and the organisers approved PP-OCRv6-medium's licence ([thread 34481](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481)). Against that, every public non-VLM R.O.A.D. pipeline (CRNN, TrOCR, Kraken) plateaued at 0.846–0.870.

**StackMix.** This was the Digital Peter winner's lever: 4.44 → 2.50 CER with blot augmentation on 6,237 lines ([StackMix](https://arxiv.org/abs/2108.11667)). It is allowed here if built from training crops ([thread 34514](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34514)). But its measured gains drew text from external corpora of up to 3M lines, which the 11 September ruling bans ([thread 34734](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734)), and came from CTC models trained for 1,000 epochs. It does not fit into six days.

**Image cleaning.** The claim that cleaning the images will "hit 0.96" comes from a user at 0.877 and has no numbers behind it ([thread 34350](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34350)). On Arabic manuscript lines, binarisation, denoising and contrast enhancement all hurt Qwen2.5-VL-7B, and only sharpening helped ([arXiv 2608.22366](https://arxiv.org/abs/2608.22366)).

## Five decisive variables remain unmeasured in public

Five things would change this plan, and none of them is public:

1. **What the top 18 do.** No one above #18 has posted a method, and the attribution of the gated artefacts rests on usernames.
2. **R.O.A.D.-measured deltas** for GRPO, pseudo-labelling, TTA, image height, VLM beam width, and the olmOCR-2, PaddleOCR-VL and Gemma backbones. The files that would show them are gated.
3. **How the gains compound.** No study combines ensembling, RL and self-training on a VLM for handwritten lines, so any stacking arithmetic, including the landing range above, is speculative.
4. **The test set's conventions.** Most multi-line crops are labelled with only the centre line, but some carry every line. Staff promised in August to investigate, and no ruling has followed ([thread 34089](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34089)). Nobody knows whether the hidden labels share the training noise rate.
5. **Logistics:**
   - whether the public share is 20% or 30%;
   - the hardware budget for top-10 code verification, which was asked about and never answered ([thread 33702](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33702));
   - whether a new backbone's licence clears the 11 September ruling (CHURRO-3B was rejected under it ([thread 34481](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481)));
   - whether re-pairing a swapped label counts as the relabelling the rules forbid.

## The 1.5% mislabelled lines distort validation more than training

### Measured effects point to small training gains and large evaluation gains

Damilare06 found about one line in 70 mislabelled: **12 of 820 held-out lines (1.5%) carry 18% of all character edits** ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). The public list of 21 "corrupted" IDs is mostly wrong. Most of those IDs are correct centre-line labels on tall multi-line crops, hence his advice "don't drop all 21 blindly" ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). MPWARE confirmed three as genuinely bad: one is empty or unreadable, and two do not match their labels ([thread 33891](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)). A well fine-tuned model has already learned the centre-line convention: none of the 820 held-out predictions exceeded twice the label length, and 96.5% were within 10% of it ([thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)). Organisers allow excluding clearly corrupted training rows if three conditions hold: the original data stays unchanged, the rows and reasons are documented, and the decision uses no test data or external labels ([thread 33891](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)). Manual labelling is forbidden ([competition info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)).

Studies at comparable noise rates say the training-side gain is small:

- **Arabic HTR:** training-set cleaning gave "negligible additional improvement" on datasets with about 1% label errors. It helped only where errors reached about 8.5% (Ajami, 10.59% → 8.78% CER with train and test both cleaned) ([Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)).
- **NMT:** adding misaligned pairs equal to 5% of the corpus cost only 0.7 BLEU ([Khayrallah & Koehn 2018](https://aclanthology.org/W18-2709/)).
- **Small historical sets react more.** On Ricordi with 88 lines, dropping 16 misaligned lines cut CER from 21.1 to 18.2, and re-pairing them with their best-matching images reached 17.4. With 295 lines the gains shrank: 9.7 → 9.4 by dropping and → 8.9 by re-pairing ([Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)).
- **Over-removal hurts.** On clean Konzil pages, dropping 27% of lines at threshold τ = 0.5 raised CER from 7.6 to 8.5, while τ = 0.7 limited the damage to 7.9 ([Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)).

The evaluation-side effect is larger and more consistent. Cleaning only the test split moved CER by 0.28–1.64 points on five Arabic datasets ([Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)). A 5–6% rise in mislabelled test examples flipped the ResNet-18/50 and VGG-11/19 rankings ([Northcutt et al. 2021](https://arxiv.org/abs/2103.14749)).

For R.O.A.D. this implies (inference) three uses, in descending order of value:

1. **Model selection.** Noisy lines inject large, heavy-tailed edits into every fold score.
2. **RL and pseudo-label pools.** A GRPO reward computed against a swapped label pushes the model toward another line's text. The "uncertain" lines that FD-RL says to keep are exactly where swaps concentrate.
3. **Training-side exclusion,** probably worth a few thousandths at most.

The noise also sets a floor. The Benchmark row (WER_w 11.53, CER_w 53.81) looks like near-empty predictions, which would make those numbers the mean reference lengths of about 11.5 words and 54 characters (inference) ([leaderboard](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/leaderboard)). Suppose the hidden test carries the same noise rate, which is unconfirmed. Then each mislabelled line costs roughly 35 character edits whatever the model reads (computed: 18% of ~2.9 edits per line, spread over 1.46% of lines). Assuming about 10 of the ~11.5 words differ on a swapped line, it also costs ~0.15 word edits per line. Together that is **about 0.01 of score for every team** (inference). The 12/820 rate alone has a 95% interval of roughly 0.8–2.5% (computed). On that assumption, about 30% of #1's character edits and 13% of its word edits cannot be removed, and the cut a 0.905 system needs on clean lines rises from 37% to roughly 42%.

### A detection recipe that uses only training data

The best-evidenced signal for line-level HTR is per-line CER between an out-of-fold prediction and the label. Aradillas's "Corrupted Label Purging" uses exactly that, with N-fold splits and a threshold of 0.5–0.7 ([Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)). A looser CER > 0.25 screen reached only 68–90% precision among the top 50 flagged lines, which is why that study required human verification ([Al-azzawi et al. 2026](https://arxiv.org/html/2601.16713)). Ranking by loss or token NLL is weaker for sequences because it also flags valid-but-hard examples ([Li et al. 2024](https://arxiv.org/abs/2310.00840)). On R.O.A.D., the hard-but-correct lines are the rare names and places that dictionaries already break.

Two further signals make the screen specific:

- **Swap test.** Check whether a line's label matches another image's prediction far better than its own. That is the fingerprint of a misaligned pair, and Aradillas used it to re-align labels ([Aradillas et al. 2020](https://arxiv.org/abs/2012.02544)).
- **Consensus test.** Check whether two independently trained model families agree with each other but not with the label. Agreement between two models was worth about 1 BLEU in NMT data filtering ([Junczys-Dowmunt 2018](https://aclanthology.org/W18-6478/)).

The recipe, step by step:

1. Take out-of-fold predictions for every training line from your existing 5-fold models, ideally for two families (Qwen2.5-VL and Qwen3-VL).
2. Normalise both strings with NFC, case-folding and whitespace collapse. Casing noise is symmetric and the test shares it, so it must not trigger a flag.
3. Treat own-CER ≥ 0.5 as a necessary condition.
4. Run the swap test with rapidfuzz against all predictions, checking page neighbours (i±1, i±2) first.
5. Exclude automatically only lines with corroborating evidence, capped at about 2% of the set.

The thresholds come from CRNN studies and NMT filtering. They are untested on R.O.A.D. and on any VLM.

```python
# Untested sketch: out-of-fold noise screen using training data only
import re, unicodedata, numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as Lev

norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).casefold()).strip()
L  = [norm(x) for x in labels]               # training labels
PA = [norm(x) for x in oof_pred_family_a]    # out-of-fold, e.g. Qwen2.5-VL folds
PB = [norm(x) for x in oof_pred_family_b]    # out-of-fold, e.g. Qwen3-VL folds
cer = lambda p, l: Lev.distance(p, l) / max(len(l), 1)
cer_a = np.array([cer(p, l) for p, l in zip(PA, L)])
cer_b = np.array([cer(p, l) for p, l in zip(PB, L)])
agree = np.array([cer(a, b) for a, b in zip(PA, PB)])

cand = np.where(cer_a >= 0.5)[0]
D = process.cdist([L[i] for i in cand], PA, scorer=Lev.normalized_distance, workers=-1)
D[np.arange(len(cand)), cand] = np.inf       # ignore the line's own image
best = D.min(1)
swap = (best <= 0.3) & (best <= 0.5 * cer_a[cand])
consensus = (agree[cand] <= 0.2) & (cer_b[cand] >= 0.5)
tier_a = cand[swap | consensus]              # auto-exclude; log ids + signal values
assert len(tier_a) <= 0.02 * len(L), "tighten thresholds rather than remove more"
```

| Tier | Rule | Action |
|---|---|---|
| A | own-CER ≥ 0.5 AND (swap: best other-image CER ≤ 0.3 and ≤ half own-CER, OR consensus: families agree within 0.2 and both are ≥ 0.5 from the label) | Exclude from SFT, GRPO pools and pseudo-label teacher training. Log the ID, the signals and the reason. |
| B | own-CER ≥ 0.5 without corroboration, or prediction/label length ratio outside [0.5, 2] | Keep, optionally at loss weight ~0.5. Never exclude on this alone. |
| Cap | Tier A above ~2% of lines (~80) | Tighten thresholds instead of removing more. |

Two R.O.A.D.-specific cautions apply (inference). First, Tier B's length signal will mostly catch the minority of multi-line crops labelled with every line, such as the examples gilgamesh listed ([thread 34089](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34089)). The test convention has not been ruled on, so keep them. Second, do not re-pair swapped labels without staff confirmation. Aradillas measured re-pairing as better than dropping, but it may count as the relabelling the rules forbid. Keep the thresholds automatic so the exclusion is reproducible and documentable; visual spot-checks can validate the detector but should not decide individual rows.

### Report every experiment on two slices

Report both numbers, each with a different job (inference built on the measured evaluation effects above):

- **Full out-of-fold set**, every line including swaps. This is the leaderboard proxy, because the test cannot be cleaned.
- **Clean slice**, the full set minus the frozen Tier A list. This drives model selection.
- **Tier B as its own slice** rather than dropped, since it contains genuinely hard lines whose improvement you want to see.

Freeze the list once, on Monday, from training-only signals. Never regenerate it per experiment, or the metric moves with the detector. Use all ~4K out-of-fold lines rather than one fold: scaling the measured 0.0048 sd shrinks it to about 0.0021 (computed). A paired bootstrap (1,000 line resamples of the Zindi-score difference between two systems) separates real gains from luck far better than comparing two absolute scores. Adopt a change only if the 95% interval of its clean-slice paired gain excludes zero and the full-set score does not fall.

| Slice | Contents | Report | Use |
|---|---|---|---|
| Full OOF | All lines | Score, WER_w, CER_w | Expected leaderboard level |
| Clean | Full minus frozen Tier A | Score, WER_w, CER_w; paired Δ with 95% CI | Model selection |
| Tier A only | Corroborated swaps | Edits per line | Noise floor; should stay nearly constant across models |
| Tier B only | Uncorroborated hard or multi-line lines | Score | Catches gains or damage on hard lines |

## A six-day plan that front-loads measured levers

The plan orders work by strength of evidence and then by cost, with a go/no-go gate on each step. The constraints are 5 submissions a day and 200 in total, two final picks, and, for the top 10, reproducible code with fixed seeds delivered within 48 hours ([competition info](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)). Backbones must carry commercial-use licences ([thread 34734](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734)).

| When | Step and time box | Evidence | Expected effect | Gate |
|---|---|---|---|---|
| Mon 28 Sep, ≤6 h | Build the harness: exact Zindi loss; OOF predictions for every training line; paired bootstrap; noise screen and frozen Tier A/B list. Audit errors by type (casing, whitespace merges/splits, one-letter slips, tall batch). | **Measured basis:** formula reproduces the LB to 1e-9; sd 0.0048 per 820 lines; 12/820 lines = 18% of char edits | None directly; makes +0.004 decisions visible | Current best's full-OOF score within ~0.008 of its public score |
| Mon–Tue 28–29 Sep, ≤10 h, mostly inference | Build the candidate pool: all folds × families × beam n-best, plus 2–3 input rescales as cheap TTA. Select per line by MBR with the exact Zindi loss; compare with word-ROVER; prune members that do not help OOF. | **Measured:** +0.0121 LB (k-fold + MBR on R.O.A.D.); −3% to −34% (HTR voting); +8.2% diverse vs −2.8% resampled | +0.004 to +0.012 (inference) | Clean-slice Δ CI above 0; submit 1–2 |
| Tue 29 Sep, overnight | Retrain the best SFT recipe on train minus Tier A, same seeds | **Measured elsewhere:** negligible at ~1% noise; 0.3–2.9 CER points on 88–295-line sets | 0 to +0.003 (inference) | Clean Δ > 0 and full not worse |
| Wed 30 Sep–Thu 1 Oct, ≤36 GPU-h | Main bet, one pseudo-label round: the ensemble labels 1,374 test + 687 unlisted lines; keep lines where families agree; retrain folds on train − Tier A + pseudo-labels; re-ensemble | **Bet:** allowed and present in top-team artefacts; measured gains need huge pools; writer adaptation −10% to −40% with true labels | Unknown | CV is optimistic because the teacher saw the validation folds; confirm with one LB submission |
| Wed 30 Sep–Thu 1 Oct, in parallel, ≤24 GPU-h | One diverse member (olmOCR-2, PaddleOCR-VL or Gemma family, licence checked). If a new family is impractical, a 112/128 px height sweep on one fold instead. | **Bet:** artefacts only; diversity gains measured elsewhere; height effects measured only for CTC | Unknown | Keep only if it improves the OOF MBR ensemble |
| Thu 1–Fri 2 Oct, stretch, ≤24 GPU-h | GRPO from the best SFT: reward = −per-line Zindi loss; pool = OOF-uncertain lines minus Tier A/B; LR 1e-6, 8 rollouts | **Bet:** −11% to −34% edit distance in document parsing at ~500K samples; R.O.A.D. report of 0.9145 without ablation | Unknown; highest variance | Abort after one run without clean gain |
| Sat 3 Oct | Freeze; build the final ensemble; reproducibility pass (seeds, weights, licences, runtime, one-command inference) | Rules | Protects rank | End-to-end rerun reproduces predictions |
| Sun 4 Oct (close) | Pick two submissions: "safe" (measured stack only) and "best CV" (including any bets that passed their gates) | Public sd ~0.007–0.008 | — | Select on clean + full OOF, not public score |

Only the first three rows rest on R.O.A.D. measurements. Together they plausibly take a 0.905 system to 0.913–0.918 (inference). To reach 0.94, lost points must fall to 63% of today's level. After a 14% ensemble cut, the bets would still have to remove a further ~27% (computed: 0.632 / 0.86 ≈ 0.73). That sits at the optimistic end of what GRPO and self-training measured in far larger settings. The tall batch is where such a gain would most likely concentrate: halving its measured gap would add about 0.010 on its own (computed).

The plan drops everything measured to fail or left unsupported:

- lexicon recasing, OOV snapping, lexicon rescoring and markup rules;
- ink-band cropping or resolution normalisation;
- synthetic-font pretraining and image "cleaning";
- 2B models;
- diverse beam search;
- any external text corpus, which the 11 September ruling bans.

## Conclusion

The metric turns R.O.A.D. into a whole-word contest. That makes the measured failures of recasing, snapping and lexicon rescoring structural rather than unlucky. Errors are symmetric and mostly in-vocabulary, so text rules break about as many words as they fix. Selecting among visual readings under the exact loss captures word-level value directly, and it is the one family of methods with a leaderboard-measured gain. Measurement quality is itself the lever. The sd is 0.0048 per 820 lines, and roughly 0.01 of every score is plausibly locked in label noise. At that level, a gain the size of anything measured here (+0.003 to +0.012) cannot be seen without a paired, noise-aware harness over all out-of-fold lines. That is the likeliest reason the leaders got there in about 25 submissions while others made more than 100.

The realistic target is a robust private score rather than the public 0.9399. The #2–#50 band spans barely more than one public-split sd, so a system that is genuinely better on unseen lines can finish above teams whose public scores are 0.005 higher. Reaching 0.94 itself would need at least two unmeasured bets, pseudo-labelling and a diverse or RL-tuned member, to both land at their optimistic end. The harness built on Monday is what will tell you by Thursday whether they did.
