# Tag one shared model before splitting it

Do not replace the shared Qwen-VL LoRA with four class-only models, and do not route to sub-classes. In the handwriting literature, every "specialist" that wins is a strong shared model fine-tuned further on its target. When the target was already in the shared training data, which is the R.O.A.D. situation, that extra step buys only about **6% relative CER without augmentation and 15% with it**, and it can make things worse on classes the shared model already reads well ([Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308)). Class-only models trained from the base throw away 60–90% of the archive's supervision. In the closest multi-domain study, a joint model beat every single-domain model, and the gap was **9.6 BLEU on the smallest domain** ([Kobus et al. 2017](https://arxiv.org/pdf/1612.06140)). The best use of the week is three steps. First, retrain the shared adapter once with the pixel class written into the prompt and mild class-balanced sampling (T=2). Second, build one "continue on this class" specialist for B, the most distinct class, which covers 25% of test lines. Third, optionally do the same for A1, which carries 41% of the test weight. Routing by the pixel rule is safe, because the rule is the class definition; lines only go to the wrong model if preprocessing differs between training and test. The real danger is statistical. With 75–311 validation lines per class, class-level differences below about 0.01–0.03 are invisible. A +0.02 gain on A2 moves the overall score by only 0.002. Picking the best of six candidates in each class inflates the validation score by about 0.005 even when no candidate is truly better. Switch a class to a specialist only when four conditions hold: a paired bootstrap gives P(better) ≥ 0.9, the gain stays positive after empirical-Bayes shrinkage, the sign repeats across split halves, and the expected overall contribution is at least 0.002.

## Every winning HTR specialist started as a shared model

The HTR adaptation literature consistently shows one pattern: train generic, then specialize. In the ICFHR 2018 READ competition, the winning system's mean CER on five target documents fell from **25.3% with no target pages to 5.8% with 16 target pages** ([Yousef et al.](https://arxiv.org/abs/1812.11894)). Sixteen pages is roughly 300–1,000 lines, about the size of one R.O.A.D. class ([Aradillas et al.](https://arxiv.org/abs/2012.02544)). On the same documents, 12 pages of class-only training from scratch gave **21.1–45.4% CER**, fine-tuning the general model gave **3.5–11.2%**, and the general model alone gave 15.5–41.5% ([Aradillas et al., Table 4](https://arxiv.org/abs/2012.02544)). Medieval German manuscripts follow the same pattern: a mixed model at 6.22% CER dropped to 1.65% after fine-tuning on 32 document pages ([Reul et al. 2022](https://arxiv.org/abs/2201.07661)). These large gains share one condition: the target hand or document was new to the shared model.

R.O.A.D. does not meet that condition. A1, A2, A3 and B all already sit in the all-data LoRA. The closest measured setting is Kohút & Hradiš's *writer-dependent* experiment, where the target writer was already in the base training set. Fine-tuning on that writer gave **6% relative CER reduction without augmentation and 15% with augmentation**, and overfitting was a problem without augmentation ([Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308)). The same paper shows specialization making things worse when the shared model is already good. A writer at 2.3% CER rose to **2.6% after fine-tuning on 16 lines**, while a hard writer went from 12.7% to 6.6% ([Kohút & Hradiš 2023, Table 3](https://arxiv.org/html/2302.06308)). Multi-domain translation and speech point the same way. Kobus et al.'s joint model beat every per-domain model: 29.81 vs 20.25 BLEU on the 35k-line Literature domain, while the gap vanished for the 1.6M-line Parliamentary domain (37.53 vs 37.34) ([Kobus et al. 2017](https://arxiv.org/pdf/1612.06140)). A single dialect-conditioned speech model beat per-dialect models by **3.1–16.5% relative WER** ([Li et al. 2018](https://arxiv.org/abs/1712.01541)). In multi-accent ASR, separate accent-specific models were the weakest option: 10.85% WER, against 8.9% for a joint model ([Yang et al. 2018](https://arxiv.org/html/1802.02656)).

Separately fine-tuned copies win only under narrower conditions. In a six-domain translation study with oracle domain labels, per-domain fine-tuning beat the mixed model on average (42.7 vs 41.1 BLEU). The gains came almost entirely from *distant* domains: REL 90.8 vs 77.5 and LAW 59.2 vs 54.6, against MED 37.7 vs 37.3. **Splitting a domain into halves and fine-tuning each half cost 2.3–5.1 BLEU**, while shared-parameter systems moved by only about ±0.4 ([Pham et al. 2021](https://aclanthology.org/2021.tacl-1.2.pdf)). Applied to R.O.A.D., B against A is the "distant domain with a perfect router" case: a different era, taller crops, and different conventions. A1/A2/A3 are close to each other and small. Routing to sub-classes finer than these four is exactly the split that Pham et al. measured as harmful, and later sections show it is also statistically unverifiable. The answer to "per sub-class?" is no.

The direct evidence for vision-language models is thin but consistent. The strongest historical VLM, CHURRO, is a **single Qwen2.5-VL model trained jointly on 155 corpora**, and it reports no per-corpus ablations ([CHURRO](https://arxiv.org/abs/2509.19768)). Narrowly trained Qwen-VL OCR adapters fail off-distribution. One Qwen2.5-VL-7B LoRA reached 87.5% word accuracy on synthetic Manchu but **only 43.1% on real handwriting** ([Manchu VLM OCR](https://arxiv.org/abs/2507.06761)). That matches your finding that bigger models and more epochs scored worse. Specialists cut the data per model to between 383 and 1,669 lines, which aggravates this failure rather than fixing it.

There is one mechanism by which specialists could genuinely help. VLMs used on historical text silently normalize spelling toward modern or majority forms ([Vesalainen et al. 2026](https://arxiv.org/abs/2602.14524)), and those errors hide in low CER ([VLM OCR hallucinations](https://arxiv.org/abs/2607.24077)). R.O.A.D.'s class-specific conventions (heires/heyres, saide/sd, ye/yt, and B's heavier use of '^' and ':') are exactly where a shared model can bleed one class's conventions into another. That problem, though, can be fixed by telling one model which class it is reading, without splitting the data.

## Five ways to specialize, ranked by evidence per GPU-hour

In the table, F means the cost of one full shared training run.

| Option | Strongest evidence | Expected effect on R.O.A.D. | Cost | Verdict |
|---|---|---|---|---|
| 1. Separate LoRA per class, trained from the base | Joint beat single-domain on every domain ([Kobus](https://arxiv.org/pdf/1612.06140)); training from scratch on one class is 3–10× worse in CER ([Aradillas](https://arxiv.org/abs/2012.02544)) | Negative, worst for A2 (383 lines) | 4 short runs | Skip |
| 2. Shared model, then continue on one class | 6–15% relative CER for writers the model has already seen ([Kohút 2023](https://arxiv.org/html/2302.06308)) | Small gain on B; about 0 or negative on A1/A3 | 0.1–0.4 F per class | Try B first, then A1 |
| 3. Class tag in the prompt | Feature-style tags gave +0.8 BLEU on average; inline tags gave mixed results ([Kobus](https://arxiv.org/pdf/1612.06140)) | Small, targets convention bleed | 1 F, no serving cost | Do first |
| 4. Mixture of LoRA experts, or adapter merging | LLaVA-MoLE needed 64 A100s for about 16 h ([LLaVA-MoLE](https://arxiv.org/html/2401.16160v2)) | A learned router is pointless because the class is known exactly; merging is useful only as a regularizer | Merging: minutes | Merge only |
| 5. Class-balanced sampling | T=1–2 is often better than the usual T=5 ([Shaham et al. 2023](https://arxiv.org/pdf/2212.07530)) | A2 upsampled 1.68× at T=2; B and A3 barely change | Free, folded into the Option 3 run | Do together with the tag |

**Options 3 and 5 belong in the same first run.** Kobus et al. compared two ways of telling the model the domain. An inline domain token improved some domains and hurt others. A domain feature added to every token's embedding improved all six domains, by **+0.26 to +0.92 BLEU (average +0.80)**, and it still beat the untagged joint model when the domain had to be predicted with 82.7–97.8% accuracy ([Kobus et al. 2017, Table 2](https://arxiv.org/pdf/1612.06140)). In a decoder-only VLM, a prompt prefix such as `Hand class: B` is attended at every decoding step. It may therefore behave more like Kobus's per-token feature than like an inline token, but no one has measured this for VLM handwriting recognition.

Temperature sampling draws each class with probability proportional to n^(1/T). By our arithmetic at T=2, A2's share of training samples rises from 9.3% to 15.7% (1.68×), A1's falls from 40.7% to 32.8% (0.81×), and B and A3 stay within 4% of their natural shares. So the sampling mostly trades A1 repeats for A2 repeats. Because more epochs already hurt, keep T ≤ 3 and give every repeated A2 line a fresh augmentation.

**Option 2 is the only way to build a true specialist that the evidence supports**, and its settings matter more than the choice itself. Below are the settings, with where each one comes from:

- **Learning rate: 5e-5 to 1e-4, i.e. 0.25–0.5× the stage-1 rate of 2e-4.** This is our inference: the adapter already starts near a good solution. For reference, autoregressive HTR adaptation used AdamW at 5e-5 ([Kohút & Hradiš 2025](https://arxiv.org/abs/2503.19546)).
- **Pick checkpoints by CER, never by validation loss.** Fine-tuning CER curves are U-shaped, and the CER minimum comes *after* the loss minimum ([Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308)).
- **Keep both the vision and the LLM LoRA trainable.** Freezing six encoder layers of TrOCR on 1,666 historical lines raised CER from 8.0% to 9.7% ([TrOCR medieval ablation](https://arxiv.org/abs/2606.24302)). The decoder matters most when the transcription conventions are unfamiliar, and LoRA on the whole model beat LoRA on the encoder only ([Kohút & Hradiš 2025](https://arxiv.org/abs/2503.19546)).
- **Keep the augmented copy.** On this point Aradillas et al. found that target-side augmentation did not help, while Kohút & Hradiš found it "almost eliminated overfitting". Follow Kohút, whose already-seen-writer setting is closer to R.O.A.D.
- **Mix 15–20% of each batch from the other classes (replay).** LoRA degrades gracefully as replay shrinks ([replay study](https://arxiv.org/html/2505.12512v1)). Specialists can still forget: fine-tuning on a narrower corpus raised Arabic CER from 16.05% to 23.63% in one 2026 study ([PP-OCRv6](https://arxiv.org/abs/2609.20064)). At R.O.A.D.'s scale, replay's job is regularization. Its value is keeping general reading skill, not preserving performance on other classes, which the specialist never sees at test time.
- **Fallback variant for a specialist that overfits early.** Merge the shared adapter into the base, then train a fresh, smaller LoRA (r=8–16) on top. Lower rank limits forgetting: after code fine-tuning, the source-domain score was 0.414 for full fine-tuning against 0.509 for LoRA at r=64 ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673)).

**Option 1 is dominated, and Option 4 is useful only as a merge trick.** A learned router such as LoRAMoE, X-LoRA or LLaVA-MoLE learns something you can already compute exactly from the pixels, at a training scale a Kaggle GPU cannot reach. PEFT's `add_weighted_adapter` still earns a place. If a specialist overfits, combine it with the shared adapter at weights [0.5, 0.5] using `combination_type="cat"`. PEFT documents `cat` as exact, `linear` as only a rough approximation, and `svd` as requiring full precision ([PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)). This "soup" pulls the specialist halfway back toward the shared model. TIES and DARE resolve sign conflicts between adapters trained for *different* tasks, and add hyperparameters that bring no benefit here.

**On A2, the two research threads disagree, and the statistics settle it.** The adaptation evidence names A2 as the most under-served class and the cheapest specialist, at about 190 steps. The selection analysis below shows that an A2 decision cannot be checked on 75 validation lines. Even a real +0.02 gain on A2 is worth only 0.0019 overall. So A2 should get its help from T=2 upsampling inside the shared model, and at most appear as an optional ensemble member. It should not get a hard-routed specialist.

## Pixel routing costs nothing if the pixel rule defines the classes

Routing is only dangerous when the router can disagree with the class the specialist was trained on. Here the pixel rule (height thresholds of 58 and 150 px, grey level ≤ 194) *is* the class definition. The rule agrees about 90% with label-derived style clusters, but that figure measures how pure each class is in style. It is not a routing error rate. **Build every training group with the exact preprocessing function used at test time, log the class of every line, and never train specialists on label-derived clusters.** With that discipline, the only way to misroute a line is a preprocessing mismatch.

A wrong route is expensive for any model that relies on the class signal, and a tagged shared model relies on it too. Forcing a wrong domain feature cost **0.39 to 14.18 BLEU** ([Kobus et al. 2017, Table 4](https://arxiv.org/pdf/1612.06140)). Random domain tags cost tag-dependent systems **12.0–19.6 BLEU**, and systems that ignore tags lost nothing ([Pham et al. 2021](https://aclanthology.org/2021.tacl-1.2.pdf)). Routing every sentence to one of 30 automatically clustered domains produced a net loss for almost all systems in the same study. With an accurate router, the cost is close to zero. In accented ASR, British English lost nothing because accent identification had 100% recall on it, while American English went from 8.3% to 8.6% WER, a 3.6% relative loss ([Yang et al. 2018](https://arxiv.org/html/1802.02656)). The B-vs-A split resembles the British case.

If a class is ever routed imperfectly, the expected net gain is π·[a·g + (1−a)·h]: π is the class's share of test lines, a is routing accuracy, g is the specialist's gain on correctly routed lines, and h is its gain on lines from the wrong class. For example, with a=0.9, g=+0.02 and h=−0.05, you keep 65% of the gain you would get with perfect routing. Measure h once. Run each specialist on the *other* classes' validation lines, which gives an offline confusion matrix of specialist against true class. Specialists continued from the shared adapter should stay close to it, so h should be small.

Two fallbacks cap the downside. **Boundary gating:** send lines within about ±5 px of the height thresholds, or ±5 grey levels of the ink cut, to the shared model. This is a deterministic rule, so it can be scored honestly on validation. **Soft routing:** run MBR (minimum Bayes risk voting) over the specialist and the shared model(s) instead of trusting the specialist alone. The language-model evidence favours soft routing. In c-BTM, hard top-1 routing to a cluster expert averaged **65.4** on downstream tasks, below the dense generalist's 66.2, while a top-4 soft ensemble reached 68.8 ([Gururangan et al. 2023](https://arxiv.org/pdf/2303.14177)). In historical OCR, voting across fold models raised character accuracy by 46%, and by 53% when the voters started from different pretrained models ([Reul et al. 2018](https://arxiv.org/html/1802.10038)). Adding the specialist as one member of your existing cross-model MBR ensemble for its class limits the damage if it overfits, at the cost of roughly doubling decoding on that class.

Serving is straightforward in PEFT:

1. Load the base in bf16 and wrap it with `PeftModel.from_pretrained(..., adapter_name="shared")`.
2. Add each specialist with `load_adapter`.
3. Sort test lines by pixel class, call `set_adapter` once per class group, and generate.

Mixed-adapter batches through `generate(adapter_names=[...])` also work. They carry documented overhead, and they do not work with merged adapters or DoRA ([PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)). Beam search needs `adapter_names` expanded once per beam, which was fixed in [PEFT PR #2287](https://github.com/huggingface/peft/pull/2287). By our estimate an r=32 adapter on an 8B-class LLM is about 175 MB in bf16, so all four specialists fit beside the base weights on a single GPU. If inference runs through vLLM instead of HF `generate`, check separately that its multi-LoRA support covers the Qwen-VL vision tower.

## Seventy-five validation lines cannot see a 0.02 gain

The binding constraint is not routing or training; it is measurement. Our calculations use your measured standard error of 0.0048 on about 820 lines, which implies a per-line standard deviation of about 0.1375. What matters for comparing two models on the same lines is the standard deviation of per-line *differences*, which shrinks as the two models' errors become correlated (sd_d = σ·√(2(1−ρ))). Fine-tuned siblings usually fail on the same hard lines, so ρ is probably high. Measure it anyway. The table shows the smallest true gain a paired test detects (two-sided α=0.05, power 0.8), and what a +0.02 class gain is worth overall given the test shares (A1 0.410, A2 0.094, A3 0.245, B 0.251):

| Paired sd_d | A2 (n=75) | B (n=207) | A3 (n=226) | A1 (n=311) | All (n=819) |
|---|---|---|---|---|---|
| 0.06 (ρ≈0.9) | 0.019 | 0.012 | 0.011 | 0.0095 | 0.006 |
| 0.10 (ρ≈0.74) | 0.032 | 0.020 | 0.019 | 0.016 | 0.010 |
| 0.14 (ρ≈0.5) | 0.045 | 0.027 | 0.026 | 0.022 | 0.014 |
| Overall value of a +0.02 class gain | 0.0019 | 0.0050 | 0.0049 | 0.0082 | — |

**Pairing is what makes class-level comparisons possible at all.** An unpaired comparison on A2 needs a 0.064 difference. The expected specialization gain from the literature is a few percent relative CER, so fold-0 validation can see it on A1, A3 and B only if the gain is at the upper end and the models are highly correlated. On A2 it cannot see it at all.

Choosing among many candidates adds a bias on top of the noise. Cawley & Talbot show that "over-fitting in model selection" is often as large as the real differences between algorithms, and prescribe nested resampling ([Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). The optimizer's curse means the winner's true value sits below its estimate even when every estimate is unbiased; the remedy is Bayesian shrinkage ([Smith & Winkler 2006](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451)). Our Monte Carlo shows the size of this bias for R.O.A.D. We assumed K candidates that are truly equal, with ρ=0.8, and measured how far the chosen one's validation score overstates its true score:

| Candidates K | A2 | B | A3 | A1 | All |
|---|---|---|---|---|---|
| 2 | 0.004 | 0.0025 | 0.0024 | 0.002 | 0.0012 |
| 4 | 0.0074 | 0.0045 | 0.0043 | 0.0036 | 0.0023 |
| 6 | 0.009 | 0.0055 | 0.0053 | 0.0045 | 0.0028 |

Picking per class among six equal candidates inflates the validation-weighted score by about **0.005, the same size as your noise threshold**. Global selection inflates it by about 0.003.

A second simulation compared selection strategies. It used six candidates whose true class×model interactions have standard deviation τ. When τ=0, meaning no candidate is genuinely better on any class, **naive per-class argmax *lost* 0.0009–0.0012 of true test score while reporting 0.004–0.007 of gain**. Two alternatives never lost in any scenario:

- **Empirical-Bayes shrinkage.** Shrink each class×model effect toward the model's global mean by the factor τ̂²/(τ̂²+SE_c²), where SE_c is that class's standard error.
- **"Switch only if P(better) ≥ 0.9".** Keep the global best unless an alternative clears this bar on the class.

When τ was 0.01 or more, both captured 80–100% of the real gain. A stricter P ≥ 0.975 rule gave up 20–40% of the real gain for little extra safety.

You can estimate τ from your own validation table: τ̂² = mean(squared observed interactions) − mean(SE_c²). **If τ̂² ≤ 0, your data show no class-specific advantage, and one model for all is the correct answer.** The tools behind this are standard:

- Paired bootstrap resampling, which Koehn showed is informative even at 300 test sentences ([Koehn 2004](https://www.research.ed.ac.uk/en/publications/statistical-significance-tests-for-machine-translation-evaluation/)).
- Non-parametric tests when the metric's distribution is unknown ([Dror et al. 2018](https://aclanthology.org/P18-1128/)).
- Holm correction when asking "on which of the four classes does the specialist win?" ([Dror et al. 2017](https://arxiv.org/abs/1709.09500)).
- James–Stein shrinkage for small groups ([Efron & Morris](https://arxiv.org/pdf/2111.02829)).
- If validation lines cluster by page, resample whole pages (a block bootstrap).

Cross-fitting over five folds would shrink A2's detectable gain from 0.032 to 0.0145 (at sd_d=0.10). It costs roughly 10–15 F, because the shared model must be cross-fitted too, otherwise the comparison is unfair. It pays off only if the fold models are reused as a test-time voting ensemble, where Reul et al. saw large gains. With seven days, the cheaper fix is repeated split-half *within* fold 0. That removes selection optimism but not small-sample noise. Also avoid spending validation data on checkpoint picking. Either fix the step budget in advance, or use a label-free stopping rule such as Kohút & Hradiš's TRN_CER, which stops once CER on the non-augmented fine-tuning lines reaches zero ([Kohút & Hradiš 2025](https://arxiv.org/abs/2503.19546)).

## A seven-day recipe with the acceptance rule written first

Keep the 8B-class base. Your current recipe already matches documented LoRA practice: all linear layers including the MLPs, alpha = 2r, LR 2e-4, and a small batch, which LoRA tolerates better than a large one ([Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/); [Biderman et al. 2024](https://arxiv.org/abs/2405.09673)). Change the data and the prompt, not the optimizer. G* below means the best shared model at each point.

| Day | Run | Settings | Gate |
|---|---|---|---|
| 1 AM | E0: evaluation harness (no training) | Per-line error vectors for the current shared model G0 and the current MBR ensemble on all 819 fold-0 lines. Class computed by the test-time pixel function. If a second seed exists, compute sd_d and ρ. Record per-class error. | None. This sets the noise floor and shows per-class headroom. |
| 1–2 | E1: tagged + balanced shared run G1 | Prompt prefix `Hand class: {A1\|A2\|A3\|B}`; T=2 `WeightedRandomSampler` (per-line weight p_c/n_c); fresh augmentation on each repeat. Otherwise unchanged: r=32, α=64, LR 2e-4 cosine, batch 4, 1 epoch plus 1 augmented copy, vision + LLM LoRA. | G1 replaces G0 if a paired bootstrap over all 819 lines gives P(better) ≥ 0.9 and no class has P(worse) ≥ 0.9. |
| 2–3 | E2: B specialist S_B | Continue G* with `is_trainable=True` on B lines plus the augmented copy (≈520 steps per pass at batch 4); 15–20% replay from A classes; LR 1e-4 constant after 10 warm-up steps; keep the tag; save checkpoints at about 170, 340 and 520 steps. | Acceptance rule below. Pick the checkpoint on split halves, not on the full class set. |
| 3–4 | E3: A1 specialist S_A1 (optional) | Same recipe, 400–800 steps (one pass is ≈835 steps). Run only if E0 shows A1 headroom or E2 suggests τ > 0. | Acceptance rule |
| 4 | E4: regularized variant (conditional) | Only if a specialist's CER turned upward early: the `cat` merge of G* and S_c at [0.5, 0.5], or merge G* into the base and train a fresh r=8–16 LoRA. | Counts as an extra candidate (K) |
| 4–5 | E5: soft routing + cross-class check | MBR over {S_c, G*, existing ensemble} vs hard routing on class c. Run S_B on A validation lines to measure h. | Acceptance rule |
| 5–6 | Decision table + honest score | Per class: Δ vs G*, 90% CI, P(better), shrunk Δ, split-half sign, π·Δ. Estimate τ̂². Score the selected policy honestly by split-half (≈50 random splits). | Lock the policy |
| 6–7 | Test inference | Group lines by class, `set_adapter`, generate. Keep the one-model G* submission as a fallback. | — |

The acceptance rule should be written into the notebook before E2 produces any numbers. **Switch class c from G* to a candidate only if all of the following hold:**

1. A paired bootstrap over class c's validation lines (≥10,000 resamples, resampling whole pages if page IDs exist) gives **P(candidate better) ≥ 0.9**.
2. The **empirically-Bayes-shrunk gain is positive**.
3. The gain has **the same sign in both halves** of random split-halves, in a clear majority of repeats.
4. The **expected overall contribution π_c·Δ_c is at least 0.002**. That translates to minimum class gains of about **0.005 on A1, 0.008 on A3 and B, and 0.03 on A2**. A2's bar is deliberately higher than the arithmetic minimum, because its detectable gain alone is 0.019–0.045.

Also apply these limits:

- Allow at most four candidates per class (G*, the specialist, the specialist+G* MBR, and one regularized variant), because the winner's curse grows with K.
- Apply Holm correction across the four classes before claiming any class-level win is significant.
- Report only the split-half score of the whole selection procedure, never the fold-0 score of the configuration that was chosen on fold 0.

Set expectations realistically. If the already-seen-writer prior of 6–15% relative improvement holds for B alone, the overall gain is about 0.251 × (0.06–0.15) × B's current per-line error. For a B error of 0.10, that is 0.0015–0.0038, close to the noise floor. That is why the tag-and-balance run comes first and the specialists come second.

## Conclusion

Whether to use specialists or a shared model turns on one quantity: τ, the size of genuine class×model interactions. The literature says τ is small for classes that are already in training and are close in language, genre and archive, and fold-0 validation can barely measure it. So the productive investment is not building more candidates. It is getting one prompt-conditioned shared model right, and building the measurement machinery (paired per-line vectors, seed-noise estimates, shrinkage) that lets a real B-specific gain show itself if one exists. The evidence points to one design with a capped downside: a class tag that gives the model the convention signal without splitting the data, plus a B specialist that joins the ensemble as an extra voter rather than replacing the shared model.

The biggest open unknown is whether a text tag in a decoder-only VLM behaves like Kobus's consistently helpful per-token domain feature or like his inconsistent inline token. No handwriting study has measured it, so E1 is both the cheapest experiment and the most informative one. If E1 moves B's word-level spelling errors without touching the other classes, it is direct evidence that convention bleed was the class-specific problem, and that the tag has already solved most of what a specialist could.
