# Routing test lines to class-specific HTR models, and choosing a model per class when validation sets are small

Scope: (a) routing each input at inference time to a specialist chosen by its detected class: hard routing, soft/ensemble routing, falling back to a generalist, and handling router errors. (b) Choosing a model per class without fooling ourselves when each class has only 75–311 validation lines. Applied to the Zindi R.O.A.D. Barbados HTR setup: classes A1/A2/A3/B come from a pixel rule. Fold-0 validation has A1 311, A2 75, A3 226 and B 207 lines. Test has A1 564, A2 129, A3 336 and B 345 lines, so the test shares are A1 0.410, A2 0.094, A3 0.245 and B 0.251.

Labels used below:
- **[MEASURED-LIT]**: a number reported in a cited paper.
- **[COMPUTED]**: I calculated it in this session from the stated assumptions (Python, normal approximation or Monte Carlo).
- **[INFERRED]**: my reasoning, not measured by anyone.

Core assumption for every [COMPUTED] number: per-line loss sd σ ≈ 0.0048·√820 ≈ **0.1375** (this is the unpaired, single-system sd). Paired comparisons depend on the sd of per-line *differences*, sd_d = σ·√(2(1−ρ)), where ρ is the correlation of two systems' per-line losses:
- ρ = 0.5 gives sd_d ≈ 0.14.
- ρ = 0.8 gives sd_d ≈ 0.087.
- ρ = 0.9 gives sd_d ≈ 0.061.

Fine-tuned models from the same family usually fail on the same hard lines, so ρ is probably high, but **you should measure sd_d from your own per-line loss differences**.

---

## Q1. When does hard routing to class/domain experts beat a single generalist or an ensemble? (Evidence from MT, ASR, OCR/HTR and LM mixture-of-experts)

### Takeaway
Hard routing to separately fine-tuned specialists wins when four conditions hold:
- the domains are genuinely distant from each other,
- each domain has enough data,
- the router is accurate,
- each specialist starts from a strong shared model.

Otherwise it roughly ties with the alternatives, which are a single model given the class as an extra input or a soft top-k ensemble of experts, and it often loses to them. In small or fragmented domains, per-domain fine-tuning overfits, and shared/multi-domain models match or beat it.

### Cited Findings
**Multi-domain MT (Pham, Crego & Yvon, TACL 2021, six domains, oracle domain tags)** [MEASURED-LIT]
- Fine-tuning a separate copy per domain (FT-Full) beat the single mixed model (Mixed-Nat) on every domain:
  - weighted average 42.7 vs 41.1 BLEU;
  - the gains were significant only for domains distant from the dominant MED domain: REL 90.8 vs 77.5, LAW 59.2 vs 54.6, BANK 54.5 vs 50.1, IT 46.8 vs 43.2;
  - they were small for close domains: MED 37.7 vs 37.3, TALK 34.0 vs 33.5.
  - The authors write that "all systems fail to outperform fine-tuning" when domains are well separated.
  - [Pham et al. 2021](https://aclanthology.org/2021.tacl-1.2/)
- Splitting a domain into smaller artificial sub-domains and fine-tuning on each piece hurt FT-Full [Pham et al. 2021, Table 4](https://aclanthology.org/2021.tacl-1.2.pdf) [MEASURED-LIT]:
  - LAW split 0.5/0.5: −2.3 / −5.1 BLEU;
  - MED split 0.25/0.75: −1.5 / −0.2.
  - Multi-domain systems that share parameters changed by only about ±0.4.
  - The authors attribute this to the small-data condition. **This is the direct analogue of "model per sub-class": more splits leave less data per specialist, which leads to overfitting.**
- With automatically clustered domains (k-means, k=30) [Pham et al. 2021, Table 5](https://aclanthology.org/2021.tacl-1.2.pdf) [MEASURED-LIT]:
  - For the 10 *small* clusters (about 29k lines on average), several shared multi-domain systems beat per-cluster fine-tuning: MDL-Res 71.2 and FT-Res 70.7 vs FT-Full 70.0.
  - For the large clusters, per-cluster fine-tuning stayed best: 52.9 vs ≤52.0.

**ASR dialect/accent routing** [MEASURED-LIT]
- A single LAS model given the dialect (as a one-hot input to all layers plus a dialect output token) beat LAS models trained individually per dialect by **3.1–16.5% relative WER** across 7 English dialects. The authors also report that simply pooling all dialects without dialect information underperformed per-dialect fine-tuned models. [Li et al. 2018, arXiv:1712.01541](https://arxiv.org/abs/1712.01541); [IEEE ICASSP 2018](https://ieeexplore.ieee.org/document/8461886/)
- In multi-accent ASR (UIUC/IBM), with oracle accent labels [Yang et al. 2018, arXiv:1802.02656](https://arxiv.org/html/1802.02656):
  - Accent-specific models: 10.85% average WER.
  - Multi-task shared model: 9.55%.
  - Joint accent+acoustic model: 8.9% (−18% relative).
  - **Separate accent-specific models were the weakest option.**

**Sparse mixtures of domain experts in LMs (c-BTM)** [MEASURED-LIT]
- Test contexts are routed by distance to k-means cluster centres, with a softmax temperature T and a top-k filter; the chosen experts' outputs are then ensembled. [Gururangan et al. 2023, arXiv:2303.14177](https://arxiv.org/abs/2303.14177)
  - Top-1 (hard) routing beat the dense model in perplexity.
  - Top-2/top-4 were "comparable to (and sometimes slightly better than)" using all experts.
- On downstream few-shot classification, the C4-trained models averaged [Gururangan et al. 2023, Table 1](https://arxiv.org/pdf/2303.14177):
  - dense 1-cluster 1.3B: 66.2;
  - **16-cluster top-1: 65.4 (below the dense generalist on average)**;
  - top-4: 68.8;
  - top-16: 69.9.
  - **So hard top-1 routing is not reliably better than the generalist on downstream tasks, while soft top-k ensembling is.**

**HTR/OCR, generic vs document-specific** [MEASURED-LIT]
- On German medieval manuscripts, mixed (generic) HTR models reached 6.22% average CER on unseen manuscripts. [Reul et al. 2022, arXiv:2201.07661](https://arxiv.org/abs/2201.07661)
  - Fine-tuning that mixed model on document-specific ground truth gave 3.27% CER (2 pages), 2.58% (4 pages) and 1.65% (32 pages).
  - The specialist is a *fine-tune of a strong generalist*, not a from-scratch model.
- On early printed books, cross-fold training (training fold models, i.e. a committee) plus confidence-based voting improved character accuracy by 46% when all folds started from the same mixed model. It improved by 53% when folds started from *different* pretrained models, which shows that diverse voters matter. [Reul et al. 2018, arXiv:1802.10038](https://arxiv.org/html/1802.10038)
- Choosing an OCR model per document can be done without ground truth. Mean character confidence and "lexicality" (the share of output tokens found in a historical-spelling-aware lexicon) both correlate with true accuracy, and the authors propose them for selecting the most suitable model for a given printed document. [Springmann et al. 2016, arXiv:1606.05157](https://arxiv.org/pdf/1606.05157)
- A ResearchGate item claims that adaptive per-document OCR engine selection is 23.7% more accurate than single-engine baselines on 15,000 government documents. **This is low-confidence: it is not peer-review-verified and was not checked beyond the search snippet.** [ResearchGate](https://www.researchgate.net/publication/400113988_Adaptive_OCR_Engine_Selection_and_Evaluation_for_Multi-Format_Government_Document_Digitization)

### Inferences
- **[INFERRED]** How the evidence maps onto R.O.A.D.:
  - **B vs A.** B lines are much taller (>150 px) and the router is ~100% accurate, which is the "distant domain + perfect router" case where specialists paid off in Pham et al. A B-vs-A split is the most defensible specialisation.
  - **A1/A2/A3.** These are closer to each other ("close domains") and smaller, especially A2 with only 75 validation lines. This is the regime where per-domain fine-tuning showed only small gains in Pham et al. and lost when domains were split finely.
- **[INFERRED]** The cheapest competitor to separate adapters is **one shared LoRA whose prompt carries the class**, e.g. "style: A2, short parchment line". This is the Li et al. 2018 and DC-Tag recipe. It keeps all data in one model, costs nothing extra at inference, and deserves to be a baseline candidate.
- **[INFERRED]** Specialists should be built as **continued fine-tunes of the shared LoRA on class data** (the Reul et al. 2022 pattern), not trained from base. That keeps them close to the generalist, which limits the damage when a line is misrouted.
- **[INFERRED]** The existing cross-model MBR ensemble is the analogue of c-BTM top-k soft routing. The literature suggests "MBR over {class specialist, generalist(s)}" is likely to be at least as good as hard routing to the specialist alone.

### Gaps
- I found no paper that directly compares hard class-routed LoRA specialists vs a shared VLM for *handwriting* OCR with pixel-derived style classes. The HTR evidence is about document-specific fine-tuning, which has oracle routing.
- The Li et al. 2018 abstract does not say whether the dialect was known or predicted at test time; the full text was not checked.

---

## Q2. How much do routing errors cost, and what fallbacks exist?

### Takeaway
Wrong routing can be catastrophic for models that *depend* on the class signal (−12 to −20 BLEU with random domain tags). With an accurate router the cost is small: about 3–4% relative when accent identification errs on one class, and ~0 when recall is 100%.

In practice there are three ways to protect against routing errors:
1. Define the classes with the same deterministic pixel rule at train and test time, so there is no train/test mismatch.
2. Evaluate the *whole routed pipeline* on validation using the pixel router, not gold clusters.
3. Send ambiguous or boundary lines to the generalist, or to an MBR over specialist + generalist.

### Cited Findings
**Wrong or predicted domain tags in MT (Pham et al. 2021)** [MEASURED-LIT]
- Translating in-domain test sentences with a *random* domain tag cost [Pham et al. 2021, §5.2.1 and Table 4](https://aclanthology.org/2021.tacl-1.2.pdf):
  - FT-Full −19.6, MDL-Res −18.6, DC-Tag −13.4, FT-Res −13.3, LDR −12.0 BLEU;
  - systems that ignore the tag at test time lost 0.0.
- For an unseen domain (NEWS) whose tag was predicted by per-domain LMs [same source]:
  - tag-using systems lost 1.3–3.3 BLEU and performed "significantly worse than the Mixed-Nat baseline";
  - tag-free systems lost only −0.2 to +0.2.
- Their neural domain classifier had a 16.4% average prediction error on in-domain data. [same source, footnote]
- When every test sentence was routed to one of 30 automatic clusters, "the net effect is a loss in performance for almost all systems and conditions". The best multi-domain system (MDL-Res) "is no longer able to surpass the Mix-Nat baseline". [same source, §5.3 and Table 6]

**ASR with a predicted accent (Yang et al. 2018)** [MEASURED-LIT] [Yang et al. 2018](https://arxiv.org/html/1802.02656)
- The pipeline used an independent accent-identification network with up to 97.77% accuracy (most configurations >92%).
- British English: 9.5% WER, no degradation vs oracle, because accent identification had 100% recall on British test utterances.
- American English: 8.6% vs 8.3% oracle, a ~3.6% relative degradation.
- The authors note that accent-identification errors "can cause large mismatch to acoustic models".

**Soft routing and ensembles as fallback** [MEASURED-LIT]
- c-BTM uses soft routing: ensemble weights ∝ exp(−dist²/T), filtered to the top-k experts. [Gururangan et al. 2023](https://arxiv.org/pdf/2303.14177)
  - Top-2/top-4 soft ensembles matched or beat using all experts.
  - Top-1 was weaker on downstream tasks (see Q1).
  - The authors speculate that activating unrelated experts can introduce interference.
- Diverse fold models voting together gave +46–53% character accuracy on early printed books. [Reul et al. 2018](https://arxiv.org/html/1802.10038)

### Inferences
- **[INFERRED] The "90% agreement" is not a routing error rate.** The pixel rule *is* the class definition. If specialists are trained on pixel-rule groups, there is no train/test routing mismatch. The ~10% disagreement with label-derived clusters only says that the pixel classes are ~90% "pure" in style. That lowers how specialised each adapter can be, but it does not send test lines to the wrong adapter. **Recommendation: build training groups with the same pixel rule used at test time, not with label-derived clusters.**
- **[INFERRED]** If specialists are nonetheless trained on label-derived clusters, the expected net gain of routing is:

  `Gain ≈ Σ_c π_c · [a_c·g_c + (1−a_c)·h_c]`

  - π_c is the class's test share.
  - a_c is routing accuracy.
  - g_c is the specialist's gain on correctly routed lines.
  - h_c is the gain (usually negative) of the wrong specialist on misrouted lines.
  - Worked example: with a = 0.9, g = +0.02 and h = −0.02, you keep 80% of the oracle gain (+0.016). With h = −0.05 you keep 65% (+0.013).
  - Because specialists fine-tuned from the shared LoRA stay close to it, h is likely small. **Measure it**: run each specialist on the other classes' validation lines. This is an offline confusion matrix of model × class.
- **[INFERRED]** Fallback designs, ordered from cheapest to most expensive:
  1. **Confidence-gated routing.** Lines within a margin of a threshold (e.g. height within ±5 px of 58 or 150, grey level within ±5 of 194) go to the generalist. Because this is a deterministic rule, it can be evaluated honestly on validation.
  2. **MBR / ROVER over {specialist, generalist}** for all lines, or only for boundary lines. This is the analogue of c-BTM top-k and Reul voting. Cost: 2× decoding on the lines where it is applied.
  3. **Per-line selection by model confidence** (mean token log-prob) or by agreement between candidates. This is the analogue of Springmann et al. 2016's confidence-based model selection. The selection rule must itself be tuned on held-out data.
- **[INFERRED]** A vs B routing is ~100% accurate, so like the British-English case the routing cost there is ≈0. Any specialisation risk sits inside A.

### Gaps
- I found no published measurement of misrouting cost for handwriting/OCR specialists specifically.
- The size of h_c (the cross-class penalty) for LoRA specialists derived from a shared LoRA is unknown and must be measured on your own data.

---

## Q3. How to select a model per class with 75–311 validation lines without fooling yourself

### Takeaway
At these sample sizes, per-class validation can only reliably see per-class gains of about **0.01–0.03 or more** in score. An A2 gain of 0.02 is worth only **~0.002 overall** (test share 0.094). Naively taking the per-class argmax over K candidates therefore mostly harvests noise: it inflates the validation score by 0.004–0.007 even when no class-specific advantage exists, and it can lose true score.

Use these safeguards:
- a **paired** bootstrap per class (on per-line loss differences);
- **empirical-Bayes shrinkage** of class×model effects toward the global best, or a "switch only if P(better) ≥ 0.9–0.975" rule with the global best as the default;
- an honest estimate of the chosen configuration via split-half or cross-fitting, following Cawley & Talbot's nested principle.

### Cited Findings
**Significance testing** [MEASURED-LIT]
- Paired bootstrap resampling (Koehn 2004) resamples test sentences with replacement and compares two systems on the same resampled sets. Koehn reports that "even for small test sizes of only 300 sentences" the method can show that differences are real. [Koehn 2004, EMNLP](https://www.research.ed.ac.uk/en/publications/statistical-significance-tests-for-machine-translation-evaluation/); [ResearchGate PDF](https://www.researchgate.net/publication/221013020_Statistical_Significance_Tests_for_Machine_Translation_Evaluation)
- Bisani & Ney (ICASSP 2004) give a bootstrap method for WER confidence intervals and paired significance that is directly interpretable in WER units and makes no ill-founded normal approximations. [Bisani & Ney 2004 PDF](http://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/Bisani_BootstrapEstimatesForConfidenceIntervalsInASRPerformanceEvaluation_ICASSP_2004.pdf)
- Follow-up work (titles and abstracts only; I did not read the full texts) argues that utterances sharing a speaker are dependent and proposes a *blockwise* bootstrap. [Interspeech 2020](http://www.interspeech2020.org/uploadfile/pdf/Mon-2-3-3.pdf); [arXiv:2209.05281](https://arxiv.org/pdf/2209.05281)
- Dror et al. (ACL 2018) give a protocol for choosing significance tests in NLP. They found that significance testing is often ignored or misused in ACL/TACL 2017 papers, and recommend non-parametric tests (bootstrap/permutation) when the metric's distribution is unknown. [Dror et al. 2018](https://aclanthology.org/P18-1128/)
- Dror et al. (TACL 2017) treat comparisons over several datasets as a multiple-testing problem. They count the datasets on which one algorithm wins significantly (Bonferroni/Fisher) and identify which ones with the Holm procedure. **This applies directly to "on which of A1/A2/A3/B is the specialist better?"** [Dror et al. 2017](https://arxiv.org/abs/1709.09500)

**Selection bias** [MEASURED-LIT]
- Cawley & Talbot (JMLR 2010) show that "over-fitting in model selection" is often of comparable magnitude to real differences between learning algorithms. Their remedies are low-variance selection criteria and **nested** resampling, with an inner loop for selection and an outer loop for evaluation. [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)
- Smith & Winkler (Management Science 2006), the "optimizer's curse": if you choose the alternative with the best estimate, its true value is expected to be *below* its estimate even when every estimate is unbiased. Their remedy is Bayesian shrinkage of the noisy estimates toward a prior. [Smith & Winkler 2006](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451); [PDF](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf)
- James–Stein / empirical-Bayes shrinkage (Efron & Morris 1975): estimates from small groups, e.g. 18 batting averages each based on 45 at-bats, predict future performance better when shrunk toward the grand mean, by more when the group's sampling variance is larger. [Efron & Morris overview, arXiv:2111.02829](https://arxiv.org/pdf/2111.02829); [R-bloggers worked example](https://www.r-bloggers.com/2012/04/example-9-27-baseball-and-shrinkage/)

**Minimum detectable difference per class** [COMPUTED]
Settings: paired test, two-sided α = 0.05, power 0.8, MDE = 2.80·sd_d/√n. Each cell is SE / MDE.

| sd_d (paired) | A2 (n=75) | B (n=207) | A3 (n=226) | A1 (n=311) | All (n=819) |
|---|---|---|---|---|---|
| 0.06 (ρ≈0.9) | 0.0069 / 0.019 | 0.0042 / 0.012 | 0.0040 / 0.011 | 0.0034 / 0.0095 | 0.0021 / 0.0059 |
| 0.10 (ρ≈0.74) | 0.0115 / 0.032 | 0.0070 / 0.020 | 0.0067 / 0.019 | 0.0057 / 0.016 | 0.0035 / 0.0098 |
| 0.14 (ρ≈0.5) | 0.0162 / 0.045 | 0.0097 / 0.027 | 0.0093 / 0.026 | 0.0079 / 0.022 | 0.0049 / 0.014 |
| 0.198 (ρ=0, unpaired) | 0.023 / 0.064 | 0.014 / 0.039 | 0.013 / 0.037 | 0.011 / 0.032 | 0.0069 / 0.019 |

- The unpaired row (0.198) is the sd of a difference between two independent systems, √2·σ. The user's own figure, sd 0.0048 on 820 lines, is the SE of one system's mean score (σ/√n), not of a paired difference. **Pairing is what makes small-class comparisons feasible.**

**Minimum observed class gain needed to claim "P(better) ≥ 0.9"** [COMPUTED]
One-sided, threshold = 1.28·SE:

| sd_d | A2 | B | A3 | A1 |
|---|---|---|---|---|
| 0.06 | 0.009 | 0.005 | 0.005 | 0.004 |
| 0.10 | 0.015 | 0.009 | 0.0085 | 0.007 |
| 0.14 | 0.021 | 0.0125 | 0.012 | 0.010 |

**Lines needed to detect a true per-class gain δ (power 0.8)** [COMPUTED]

| sd_d | δ = 0.005 | δ = 0.01 | δ = 0.02 | δ = 0.03 |
|---|---|---|---|---|
| 0.06 | 1,131 | 283 | 71 | 32 |
| 0.10 | 3,140 | 785 | 197 | 88 |
| 0.14 | 6,154 | 1,539 | 385 | 171 |

**What a class gain is worth overall** [COMPUTED] (test shares)
- A gain of +0.02 on a single class moves the overall score by:
  - A1: +0.0082
  - A2: +0.0019
  - A3: +0.0049
  - B: +0.0050
- **Even a large, real A2 specialist win is below the ~0.005 overall noise floor.**

**Winner's curse: apparent gain of the best of K equal systems** [COMPUTED, Monte Carlo]
Setting: true difference = 0, ρ = 0.8 (sd_d ≈ 0.087). The value shown is how far the chosen system's validation score is expected to exceed its true score.

| K | A2 (75) | B (207) | A3 (226) | A1 (311) | All (819) |
|---|---|---|---|---|---|
| 2 | 0.0041 | 0.0025 | 0.0024 | 0.0020 | 0.0012 |
| 4 | 0.0074 | 0.0045 | 0.0043 | 0.0036 | 0.0023 |
| 6 | 0.0092 | 0.0055 | 0.0053 | 0.0045 | 0.0028 |
| 10 | 0.0111 | 0.0067 | 0.0064 | 0.0055 | 0.0034 |

- At ρ = 0.5, multiply by about 1.6 (e.g. K = 6: A2 0.014, All 0.0044).
- Per-class selection across the 4 classes with K = 6 (all systems truly equal) inflates the validation-weighted score by about **0.005**, which is as large as the user's noise threshold. Global selection inflates it by about 0.003.

**Simulated selection strategies** [COMPUTED, Monte Carlo, 8,000 runs per cell]
Setup:
- K = 6 candidate systems.
- True global means are spread with sd 0.004.
- True class×system interaction sd is τ.
- Validation n per class is as above; test weights are the test shares.
- Values are the *true test-score gain vs "global best for all classes"*, followed by the validation optimism of the chosen configuration in brackets.

Strategies compared:
- **Per-class argmax**: take the best candidate in each class.
- **Switch if P ≥ 0.9 or ≥ 0.975**: keep the global best unless the best alternative in that class clears the one-sided threshold.
- **EB shrinkage**: shrink the class×system interactions toward each system's global mean by τ̂²/(τ̂²+SE_c²), with τ̂² estimated by method of moments.
- **Oracle**: the true per-class best (the ceiling).

| ρ (sd_d) | τ | Per-class argmax | Switch if P≥0.9 | Switch if P≥0.975 | EB shrinkage | Oracle |
|---|---|---|---|---|---|---|
| 0.8 (0.087) | 0 | **−0.0009** (0.0038) | −0.0001 (0.0018) | 0.0000 (0.0014) | 0.0000 (0.0015) | +0.0006 |
| 0.8 | 0.003 | 0.0000 (0.0034) | +0.0002 | +0.0001 | +0.0002 | +0.0014 |
| 0.8 | 0.01 | +0.0045 (0.0020) | +0.0038 | +0.0030 | +0.0044 | +0.0054 |
| 0.8 | 0.02 | +0.0108 | +0.0103 | +0.0097 | +0.0108 | +0.0113 |
| 0.5 (0.14) | 0 | **−0.0012** (0.0071) | −0.0001 (0.0039) | 0.0000 (0.0030) | 0.0000 (0.0032) | +0.0012 |
| 0.5 | 0.005 | +0.0007 (0.0060) | +0.0005 | +0.0002 | +0.0005 | +0.0031 |
| 0.5 | 0.01 | +0.0039 (0.0044) | +0.0028 | +0.0017 | +0.0035 | +0.0060 |
| 0.5 | 0.02 | +0.0104 | +0.0094 | +0.0081 | +0.0104 | +0.0118 |

(K = 3 gives the same pattern with smaller magnitudes.)

### Inferences
- **[INFERRED from the simulation]** What the value of per-class selection depends on:
  - It is governed by τ, the size of *genuine* class×model interactions.
  - If τ ≲ 0.005, per-class selection buys ≤ 0.0007 of true score but *reports* 0.003–0.007 of gain on validation, and naive argmax loses 0.001.
  - If τ ≈ 0.01–0.02, per-class selection is worth +0.004 to +0.011 and all sensible rules capture most of it.
  - **EB shrinkage is the robust default.** It never lost in these scenarios and captured about 80–100% of the per-class-argmax gain when that gain was real.
  - "P ≥ 0.9 over the global best" is almost as safe and easier to explain. "P ≥ 0.975" (roughly Holm-level strictness) gives up 20–40% of the real gain for little extra safety.
- **[INFERRED]** Estimate τ̂ directly from your validation table: τ̂² = mean(observed interaction²) − mean(SE_c²). The observed interaction is est[k,c] − est[k,·]. If τ̂² ≤ 0, the data give no evidence of class-specific advantages, so use one model for all.
- **[INFERRED] Recommended per-class selection protocol for the notebook:**
  1. Compute a per-line loss vector for every candidate on the same 819 validation lines. Pairing is essential.
  2. Pick G, the global best on all lines, by paired bootstrap against the runner-up.
  3. For each class c and candidate k: paired bootstrap (B ≥ 10,000) of mean(loss_G − loss_k) over that class's lines. If lines share a page or document, resample whole pages or documents (block bootstrap). Report the mean, a 90% CI and P(k better).
  4. Switch c to k only if P(better) ≥ 0.9 *and* the EB-shrunk gain > 0 *and* the gain has the same sign in both halves of a random split-half. Treat the 4 classes as a multiple-comparison family (Holm) when claiming significance.
  5. Report an **honest** score:
     - apply the whole selection procedure to half A of each class and score it on half B, then swap and average (repeat over ~50 random splits);
     - or use cross-fitted out-of-fold predictions (Q4).
     - Never report the validation score of the configuration that was selected on that same validation set.
  6. Show the expected overall gain Σ_c π_test,c · gain_c. If it is < ~0.002, prefer "one model for all" for simplicity and robustness.
- **[INFERRED]** "Per sub-class" routing (finer than A1/A2/A3/B) splits the already small n further. By the table above, a sub-class of ~40 validation lines has an MDE of about 0.04–0.06 even with good pairing. Together with Pham et al.'s losses when domains are split, **the default should be no sub-class routing** unless the gap is very large and replicates across folds.
- **[INFERRED]** Decoding variants (beam, sampling-MBR, etc.) are also "candidates" and add to K. Every extra candidate raises the winner's-curse inflation (the K = 10 row), so limit per-class choices to a short pre-registered list.

### Gaps
- The true paired sd_d and the correlation ρ between candidate systems on this dataset were not available to me. The tables above bracket plausible values, and the notebook should compute them.
- Page/document grouping of the validation lines is unknown. If lines cluster by page, the effective n is smaller and block bootstrap CIs will be wider than the tables suggest.
- I did not find literature that gives a canonical "P(better) ≥ 0.9" switching rule. It is a Bayesian-decision convention, and the numbers above are my own simulation.

---

## Q4. Should we cross-fit (train on several folds) to get more validation evidence for small classes like A2, and at what compute cost?

### Takeaway
Cross-fitting over k folds multiplies the out-of-fold evidence per class by about k. For A2 with 5 folds, that is ~375 lines instead of 75, and the MDE drops by √5 ≈ 2.2×. It also produces k fold models per class that can be ensembled at test time with MBR/voting, which Reul et al. found gives large gains when the voters are diverse.

The catch is that an honest *comparison* also needs the shared/generalist model to be cross-fitted, so that the out-of-fold lines are unseen by both sides. That costs k full trainings of the shared model. It is worth it only if class-specific decisions are plausibly worth ≥ 0.003–0.005 overall.

### Cited Findings
- Nested/outer-loop resampling is the recommended guard against selection bias when both choosing and evaluating models. [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html) [MEASURED-LIT: recommendation]
- Cross-fold training with voting on early printed books [Reul et al. 2018](https://arxiv.org/html/1802.10038) [MEASURED-LIT]:
  - 5 folds from the same mixed model: +46% character accuracy;
  - folds from different pretrained models: +53%.
- The same authors pool fold-trained models into a mixed model and report CER from two-fold cross evaluation: 8.3% on 12 Antiqua books and 5.0% on 20 Fraktur books from the RIDGES corpus. This shows cross-fold evaluation is standard practice in historical OCR. [Reul et al. 2018](https://arxiv.org/html/1802.10038) [MEASURED-LIT]

### Inferences
- **[COMPUTED]** Out-of-fold MDE (two-sided 0.05, power 0.8) at sd_d = 0.10, assuming 5 folds each about the size of fold 0 (n ≈ 5 × fold-0 counts):

  | Class | Out-of-fold lines | MDE |
  |---|---|---|
  | A2 | 375 | 0.0145 |
  | B | 1,035 | 0.0087 |
  | A3 | 1,130 | 0.0083 |
  | A1 | 1,555 | 0.0071 |

  Compare the fold-0-only MDEs of 0.032 / 0.020 / 0.019 / 0.016. At sd_d = 0.06, A2 improves from 0.019 to 0.0087.
- **[INFERRED] Compute accounting**, with F = one full fine-tune of the shared LoRA on the whole training set:
  - A class specialist built as a *continued* fine-tune from the shared LoRA on its class's training lines costs about (class share × epochs) × F. For A2 (≈9% of the data) at 1–2 epochs that is ≈ 0.1–0.2 F per fold. All four specialists together cost ≈ 1–2 F per fold.
  - Full cross-fitting (shared model plus specialists on k = 5 folds) costs ≈ 5 × (F + 1–2 F) = **10–15 F**, plus inference on all out-of-fold lines for every candidate.
  - Cheaper partial option: cross-fit only the A2 (and possibly B) specialist and its shared counterpart on 2–3 extra folds. That still requires those extra shared-model trainings.
  - Cheapest option: no retraining. Use repeated split-half *within* fold-0 validation to de-bias the selection estimate. This fixes the optimism but not the small n.
- **[INFERRED]** Given the numbers in Q3, A2's maximum plausible *overall* impact is ~0.002 (for a +0.02 class gain). **Cross-fitting purely to decide A2 is hard to justify.** Cross-fitting pays off if:
  1. the fold models are reused as a test-time ensemble (MBR over k fold models), where Reul et al. suggest real gains; or
  2. the question is B vs A specialisation, which covers about 25% and 75% of test lines, where decisions move the overall score measurably.
- **[INFERRED]** If only fold 0 exists, a pragmatic rule is:
  - allow class switches only for classes whose fold-0 evidence clears the P ≥ 0.9 bar *and* whose expected overall contribution is ≥ 0.002 (A1, A3 or B);
  - keep A2 on whatever its parent group uses (the A specialist or the global model) unless its gain is enormous (>0.03).

### Gaps
- The number of folds and total labelled lines per class in the user's setup are not stated. I assumed 5 roughly equal folds.
- I found no published study quantifying the value of cross-fitted per-class selection for HTR specialists specifically.

---

## Q5. Practical implementation: serving several PEFT/LoRA adapters per class-grouped batch on one GPU

### Takeaway
With Hugging Face PEFT you can load all class adapters onto one base model. Either run class-homogeneous batches, calling `set_adapter` per class, or use one mixed batch by passing `adapter_names=[...]` to `generate`, with `"__base__"` meaning the base model.

Grouping lines by class is the recommended fast path. Mixed-adapter batches carry "expected performance overhead", do not work with merged adapters or DoRA, and had a beam-search bug that PEFT has since fixed. The extra memory per adapter is small (~0.1 GB for a 7B model at rank 16), so a single Kaggle GPU can hold every class adapter at once.

### Cited Findings
- PEFT supports different LoRA adapters inside one inference batch via `adapter_names` (a list aligned with the samples), with `"__base__"` meaning the base model. Samples do not need to be grouped. [PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora) [MEASURED-LIT: documentation]
- Documented limitations [PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora):
  - inference only;
  - `disable_adapter()` takes precedence over `adapter_names`;
  - it cannot be used with merged adapters (unmerge first), nor after `merge_and_unload()`;
  - it does not work with DoRA;
  - `modules_to_save` is limited to Linear/Embedding/Conv layers;
  - there is expected performance overhead, especially with many different adapters per batch.
  - Recommended mitigations: bigger batches, homogeneous batches grouped by adapter, or dedicated servers (LoRAX, punica, S-LoRA) when there are many adapters.
- With beam search, transformers repeats each sample once per beam, so `adapter_names` must be expanded to match. This was addressed in PEFT PR #2287. [PEFT PR #2287](https://github.com/huggingface/peft/pull/2287); a related bug with `modules_to_save` in mixed batches is [PEFT issue #1960](https://github.com/huggingface/peft/issues/1960) [MEASURED-LIT: issue tracker]
- `PeftMixedModel` loads *different adapter types* (e.g. LoRA + LoHa) together. `set_adapter()` must activate both, otherwise only the first adapter is active. It is not needed for several LoRAs of the same type. [PEFT mixed adapter types](https://huggingface.co/docs/peft/en/developer_guides/mixed_models)

### Inferences
- **[COMPUTED] Adapter size.** Assumed Qwen2.5-7B config (from memory; verify against the model's config.json): hidden 3584, intermediate 18944, 28 layers, KV dimension 512 (GQA with 4 KV heads).
  - A LoRA on all linear projections (q, k, v, o, gate, up, down) has r × 90,112 parameters per layer, i.e. r × 2.52 M in total.
  - r = 16 gives ≈ 40 M parameters ≈ **81 MB in bf16**; r = 64 gives ≈ 320 MB.
  - Four class adapters therefore add ≲ 0.35–1.3 GB to a ~15 GB bf16 base, so they fit together on one Kaggle GPU.
- **[INFERRED] Runtime.** With class-grouped batches the total decode tokens equal single-model inference. Switching adapters is a cheap `set_adapter` call. The only extra cost is unmerged-LoRA matmuls, which you can remove per group by merge → generate → unmerge.
  - Hard routing therefore costs ≈ 1× generalist inference.
  - MBR over {specialist, generalist} costs ≈ 2× (plus MBR scoring).
  - c-BTM-style top-2 soft routing costs ≈ 2×.
- **[INFERRED] Notebook design.** Represent the policy as a dict, `{class: [candidate_1, ... candidate_m], combine: "single" | "mbr"}`, with presets:
  - `"one_for_all"`: every class maps to G.
  - `"A_vs_B"`: two adapters.
  - `"per_class"`: four adapters.
  - `"per_class_mbr"`: specialist + G per class, combined by MBR.

  Selection code should produce a per-class decision table showing Δ vs G, the 90% CI, P(better), the EB-shrunk Δ, split-half sign agreement, the overall contribution π·Δ, and a KEEP/SWITCH verdict. Only then should the chosen policy run on test.
- **[INFERRED]** The class rule (height thresholds 58/150 px, grey ≤ 194) must be computed from exactly the same preprocessing at train, validation and test time, and logged per line, so the routing is reproducible.

### Gaps
- I did not verify multi-LoRA support for the Qwen2.5-VL / Qwen3-VL *vision* modules in vLLM or other fast servers in this session. If the notebook uses vLLM instead of HF generate, check its multi-LoRA (`LoRARequest`) support for these VL architectures and whether LoRA on the vision tower is supported.
- I found no source quantifying unmerged-LoRA inference overhead for 7–32B VLMs. Measure it: time a class batch merged vs unmerged.
- Qwen3-VL-8B / 32B adapter sizes were not computed because I could not verify their configs.
