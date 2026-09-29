# Parameter-efficient per-class specialists from a shared LoRA-tuned VLM (Qwen2.5-VL / Qwen3-VL, handwritten line OCR)

Scope: options for turning one LoRA-tuned VLM into per-class specialists for 4 pixel-detectable classes (B 1,043 lines; A1 1,669; A2 383; A3 1,003; 4,093 total). Current recipe: LoRA r=32, alpha=64, all LLM linear layers + vision tower, LR 2e-4 cosine, 1 epoch over originals + 1 augmented copy, batch 4. Bigger models and more epochs scored worse.

Labelling: "Cited Findings" are measured results or documented facts from the linked source. "Inferences" are my own reasoning applied to this competition and have not been measured on this data.

---

## Q1. Option 1: a separate LoRA per class, trained from the base model on that class only

### Takeaway
The best multi-domain evidence says this is dominated. A joint model beat every single-domain model in Kobus et al. 2017, by up to 9.6 BLEU on the smallest domain. HTR adaptation work always starts from a general model, never from scratch. With 383 lines, A2 is the class most at risk. Use per-class training only as a continuation of the shared adapter (Option 2), not from the base model.

### Cited Findings
- Kobus et al. 2017 (English-French NMT, 6 domains, 35k to 1.6M lines per domain) compared "Single" models, each trained on its own domain only, with one "Join" model trained on all data without tags. **Join beat every Single model.** The gap was largest for the smallest domain: Literature (35k lines) scored 20.25 BLEU Single vs 29.81 Join. Medical scored 33.97 vs 41.83 and News 29.70 vs 33.83. The gap nearly vanished for the best-represented domain: Parliamentary (1.6M lines) scored 37.34 vs 37.53. — [Kobus et al., Domain Control for NMT (arXiv 1612.06140), Table 2](https://arxiv.org/pdf/1612.06140)
- The authors wrote that NMT engines "benefit from additional training data", and that differences shrink for domains with higher representation in the joint data. — [Kobus et al. 2017](https://arxiv.org/pdf/1612.06140)
- HTR writer adaptation is always done by fine-tuning a model already trained on a large general set. In one study the source was CzechHWR, with 406k lines. No source data was mixed in during target fine-tuning. — [Kohút & Hradiš 2023, Fine-tuning Is a Surprisingly Effective Domain Adaptation Baseline in HTR](https://arxiv.org/html/2302.06308)
- A multimodal-LLM study found that mixing domain datasets can hurt some capabilities. This "data conflict" is the argument for separate or specialist parameters. Adding document data to LLaVA-1.5's general mixture dropped the Tiny-LVLM-eHub score from 306.3 to 298.8 (-7.5). — [LLaVA-MoLE (arXiv 2401.16160)](https://arxiv.org/html/2401.16160v2)
- A Qwen2.5-VL-7B LoRA OCR model trained on 60k synthetic Manchu word images reached 2.64% CER on synthetic validation. On 218 real handwritten samples, CER rose to 25.4% and word accuracy fell from 87.5% to 43.1%. The authors describe this as overfitting to training patterns. Setup: r=64, alpha=64, dropout 0, LR 1e-4, 10 epochs, effective batch 8. This shows how narrowly trained VLM-LoRA OCR models fail off-distribution. — [Manchu VLM OCR (arXiv 2507.06761)](https://arxiv.org/html/2507.06761v1)

### Inferences
- All four classes are English early-modern hands, so letterforms, vocabulary and the basic reading skill are largely shared. A from-base A2 adapter sees only 383 lines and throws away roughly 3,700 lines of related supervision. Kobus's small-domain result (Literature) is the closest analogue.
- Option 1 is only reasonable for the largest class (A1, 1,669 lines). Even there, "shared, then continue on A1" (Option 2) should be at least as good, because it starts from strictly more information.
- Data needs: at least a few hundred lines per class. Compute: 4 short runs, each shorter than the shared run. Risks: overfitting (worst for A2), and losing shared abbreviation knowledge (for example, B-style ':' suspensions also appear in A classes).

### Gaps
- I found no VLM or HTR paper that directly compares "per-domain LoRA trained from base" against "joint LoRA" on handwriting. The NMT evidence is from 2017 RNN models, not VLMs.

---

## Q2. Option 2: two-stage training. A shared LoRA on all data, then a copy or stacked adapter continued on one class ("writer adaptation")

### Takeaway
This is the best-supported option for creating true specialists. HTR writer-adaptation papers report 10–50% relative CER gains from fine-tuning a general model on 16–256 target lines. For writers already present in the training set, the gain is smaller (about 15%), and that is the closest match to our case, because every class is already in the shared data. The keys are: a lower LR than stage 1; a short run; choosing the checkpoint by validation CER (not by loss); strong augmentation; and optionally 10–25% replay of other classes as a regulariser.

### Cited Findings
**Evidence of gains (HTR):**
- Fine-tuning a general HTR model on new writers gave about **25% average relative CER reduction with 16 lines, ~35% with 64 lines, and ~50% with 256 lines**. — [Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308) ([abstract](https://arxiv.org/abs/2302.06308))
- For writers **already in the training set** (writer-dependent), fine-tuning with their full augmentation set (B1C1G1M1) "almost eliminated overfitting" and gave about a **15% CER reduction on average**. — [Kohút & Hradiš 2023 (via search summary of the paper)](https://arxiv.org/abs/2302.06308)
- Their fine-tuning settings: Adam, LR 3e-4 with polynomial warm-up, then step-downs to 0.7e-4 and 0.175e-4; batch 32; 1,000–3,000 iterations for 16–256 lines. Augmentation was blur/noise/gamma, colour, geometry (slant, scale) and masking, and it consistently beat no augmentation. Fine-tuning was "surprisingly resistant to overfitting even for an extremely low number of text lines". **The CER curve was U-shaped, with its minimum later than the validation-loss minimum.** — [Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308)
- For autoregressive transformer HTR models (33M and 100M params): 16 lines gave a **10% relative CER gain**, rising to **40% at 256 lines**. Settings: AdamW LR 5e-5, batch 32 (smaller when fewer lines), up to 80 epochs, augmentation of blur, noise, geometric distortion and patch masking. — [Practical Fine-Tuning of Autoregressive Models on Limited Handwritten Texts (arXiv 2503.19546)](https://arxiv.org/html/2503.19546)
- **Which part to adapt:** when the target uses the model's native transcription style, "fine-tuning only the encoder yields the best performance". When the transcription style is unknown, "for 8 and 16 lines the decoder plays a crucial role". As the number of lines grows, encoder-only catches up with full fine-tuning. — [arXiv 2503.19546](https://arxiv.org/html/2503.19546)
- **Stopping criteria:** the same paper proposes "TRN_CER", which stops when CER on the non-augmented fine-tuning data reaches 0. It also evaluates cross-validation and fixed-epoch rules. Choosing lines to annotate by confidence halved the annotation needed for the same gain. — [arXiv 2503.19546](https://arxiv.org/html/2503.19546)

**LR, rank, forgetting (LoRA-specific):**
- LoRA's optimal LR is about **10x** that of full fine-tuning, rising to about **15x for short runs (~100 steps)**. With the standard alpha/r scaling (their experiments fix alpha=32), the optimal LR is roughly independent of rank. — [Thinking Machines, "LoRA Without Regret" (2025)](https://thinkingmachines.ai/blog/lora/)
- Apply LoRA to all layers, especially the MLPs. "Attention-only LoRA with rank 256 underperforms MLP-only with rank 128" at about the same parameter count. LoRA also tolerates large batch sizes less well than full fine-tuning, independent of rank. — [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)
- Best LoRA LRs measured in Biderman et al.: 5e-4 for code instruction fine-tuning (full fine-tuning best: 5e-5), and **2e-4 for math instruction fine-tuning** (full fine-tuning best: 1e-5). Their recommendations: target all modules; alpha = 2r; sweep LR from 1e-5 to 5e-4 and take "the highest value that enables stable training"; MLP modules drive most of the gains. — [Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (HTML)](https://arxiv.org/html/2405.09673)
- LoRA forgets less than full fine-tuning, and **rank controls how much it forgets**. On the code task, the source-domain score (average of HellaSwag, ARC-C and WinoGrande) after 4 epochs was 0.414 for full fine-tuning vs 0.509 for LoRA r=64. The paper finds LoRA limits forgetting more than weight decay or dropout do. Full fine-tuning learns perturbations with 10–100x higher rank than typical LoRA configurations. — [Biderman et al. 2024](https://arxiv.org/abs/2405.09673); [HTML](https://arxiv.org/html/2405.09673)

**Replay / mixing general data:**
- In a replay study, LoRA clearly beat full fine-tuning at every replay ratio below 1.0 in continual pre-training, and degraded "far more gracefully" as replay approached zero. In class-incremental learning, full fine-tuning stayed ahead down to a replay ratio of about 0.25, then dropped sharply at 0.1. — [Scalable Strategies for Continual Learning with Replay (arXiv 2505.12512)](https://arxiv.org/html/2505.12512v1)
- A practitioner heuristic is to mix "typically 5–20%" of the original distribution back in. This comes from a non-peer-reviewed explainer; treat it as a rule of thumb only. — [ZeroEntropy explainer](https://zeroentropy.dev/concepts/catastrophic-forgetting/)

**Analogous ASR evidence (per-domain adapters on a shared model):**
- Residual adapters in the encoder matched full-model fine-tuning gains for atypical and accented speech while updating **less than 0.5% of parameters** (over 100x fewer than full encoder fine-tuning). Results held across RNN-T and Transformer-Transducer. — [Tomanek et al. 2021, EMNLP (arXiv 2109.06952)](https://arxiv.org/abs/2109.06952)

### Inferences
**Recipe to try (my synthesis, not measured on this data):**
- **Variant 2a: "continue a copy".** Load the shared adapter with `is_trainable=True` and train it further on class-c data only. Rank stays r=32, alpha=64.
  - LR: 5e-5 to 1e-4, i.e. 0.25–0.5x of the stage-1 2e-4. The adapter is already near a good solution. The ~15x-for-short-runs finding is about the ratio to full fine-tuning; it does not argue for a higher LR here.
  - Schedule: constant with about 10 warmup steps, or short cosine.
  - Length: 1–2 passes over the class data plus 1 augmented copy. At batch 4 that is about 200–800 steps for B/A1/A3 and about 100–200 for A2.
  - Checkpointing: evaluate CER every 25–50 steps and keep the best by CER. Both HTR papers show the CER minimum can come after the loss minimum.
- **Variant 2b: "stack".** `merge_and_unload()` the shared adapter into the base, then add a fresh, smaller LoRA (r=8–16) and train it on class-c data. The low rank is itself a regulariser, per Biderman's finding that rank controls forgetting. This may suit A2 best.
- **What to freeze.** Class differences here are partly visual (hands spanning 1670s–1710s) and partly transcription conventions (B's '^' and ':' vs A's "ye/yt", "sd"). The HTR finding that the decoder matters for unfamiliar transcription styles argues for keeping the **LLM LoRA trainable**.
  - For A2 (383 lines), try freezing the vision-tower LoRA to reduce overfitting.
  - For B, whose hands differ most, keep both vision and LLM LoRA trainable.
  - This is a hypothesis to A/B test, not a measured result.
- **Replay.** Routing is by a pixel-detectable class, so a specialist never sees other classes at test time. Forgetting other classes therefore does not matter directly. Replay's value here is regularisation: it keeps generic reading skill and prevents collapse onto 383 lines. Suggested mix: 10–25% of each batch from other classes, more for A2 and less for A1.
- **Expected gain.** The ~15% relative CER gain for writers already in training is the right prior. Sizing the gain relative to the class's line count against the 16–256-line new-writer figures would overstate it.
- **Compute.** Each specialist costs about 10–40% of a shared-run epoch on the class's data. Four specialists should cost less than one shared run on an 8B model.
- **Validation.** A 20% holdout of A2 is about 77 lines, which gives a noisy CER. Use k-fold, or at least 2 seeds, before trusting a gain of under about 5% relative.

### Gaps
- I could not retrieve the WACV-W 2025 paper "Low-Rank Adaptation vs. Fine-Tuning for Handwritten Text Recognition" (HTTP 403). It likely contains the most directly relevant LoRA-vs-full-fine-tuning HTR numbers. — [PDF link](https://openaccess.thecvf.com/content/WACV2025W/VISIONDOCS/papers/Huttner_Low-Rank_Adaptation_vs._Fine-Tuning_for_Handwritten_Text_Recognition_WACVW_2025_paper.pdf)
- I found no published VLM (Qwen-VL) HTR study that does two-stage shared-then-per-collection LoRA and reports per-collection CER.
- I found no measured optimal replay ratio for LoRA specialist HTR. The 10–25% suggested above is inference.

---

## Q3. Option 3: class tokens or prompt conditioning in one shared model

### Takeaway
Domain tags are the cheapest option and have a consistent but modest positive record. In NMT, a feature-embedded tag averaged +0.8 BLEU over an untagged joint model; an inline token gave mixed results. The tag must be correct: in the same study a wrong domain tag cost 0.4 to 14 BLEU. Our classes are pixel-detectable, so misrouting risk is low. This option combines freely with Options 2 and 5.

### Cited Findings
- Kobus et al. 2017 compared two ways of adding the domain.
  - **Inline token** (for example, appending `@MED@` to the source) gave "mixed results": it improved some domains and hurt others. Examples: News 34.47 vs Join 33.83; Parliamentary 37.13 vs 37.53.
  - **Word-feature embedding** (the domain added to every token's embedding) improved every domain, by +0.26 to +0.92 BLEU, **average +0.80**. — [Kobus et al. 2017, Table 2](https://arxiv.org/pdf/1612.06140)
- With a predicted rather than oracle domain (classifier accuracy 82.7–97.8%), the Feature model still beat Join on all domains. — [Kobus et al. 2017, Table 2 "RNN" column](https://arxiv.org/pdf/1612.06140)
- **Wrong-tag penalty:** forcing a wrong domain feature lost -0.39 to -14.18 BLEU. For example, IT test sentences tagged as Tourism lost 14.18. — [Kobus et al. 2017, Table 4](https://arxiv.org/pdf/1612.06140)
- A survey summary reports that Kobus et al. and Tars & Fishel (2018) found domain features slightly better than discrete tags, and that tags improved in-domain translation over simple fine-tuning on the target domain. — [Saunders 2021, Domain Adaptation and Multi-Domain Adaptation for NMT: A Survey (arXiv 2104.06951)](https://arxiv.org/pdf/2104.06951) (seen via search snippet; not verified in the full text)
- Multi-dimensional tagging extends tags to several attributes at once (for example, domain plus other metadata). — [Stergiadis et al. 2021, Multi-Domain Adaptation in NMT Through Multidimensional Tagging (arXiv 2102.10160)](https://arxiv.org/pdf/2102.10160)
- In accented ASR, one-hot accent conditioning needs the accent known in advance, and simply concatenating accent embeddings gave only limited improvements. A pooled model without accent information underperformed accent-specific models. — [Accented Speech Recognition with Accent-specific Codebooks, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.444.pdf) (via search summary); a layer-wise adaptation approach gave 12% relative WER reduction on AESRC2020 — [Layer-wise Fast Adaptation for E2E Multi-Accent ASR (arXiv 2204.09883)](https://arxiv.org/pdf/2204.09883)
- VLMs used for historical OCR show "selective linguistic regularization and orthographic normalization" that can silently alter historical forms. Qwen had lower CER/WER than TrOCR but normalized spellings more. The same paper notes prior reports of VLM spelling modernization and hallucination in visually ambiguous regions. — [Vesalainen et al. 2026, Error Patterns in Historical OCR: TrOCR vs a VLM (arXiv 2602.14524)](https://arxiv.org/abs/2602.14524)

### Inferences
- In a decoder-only VLM, a text prompt such as `"Hand: B (1670-1710). Conventions: '^' marks superscript letters, ':' marks suspensions."` is an inline-token tag. However, it is attended at every decoding step, so it may behave more like Kobus's per-token feature than his end-of-sentence token. Two natural variants to test:
  - a bare class tag, for example `<class_B>`;
  - the tag plus a one-line description of the class's conventions.
- Prompt conditioning directly targets the VLM normalization failure mode (Vesalainen et al.). It tells the model which archaic conventions to keep ("ye" vs "the"; "heyres" vs "heires"). This should matter most for class-specific spellings.
- Implementation: add the tag to the user turn during training and inference, and retrain the shared adapter once. It adds no serving complexity, and misrouting is unlikely because classes are pixel-detectable. Optionally drop the tag on 10–20% of training samples so the model still works untagged (my suggestion, not sourced).
- Expected size: the NMT gains were modest (+0.8 BLEU average). The upside here may be larger, because the per-class conventions are systematic and label-level (for example, B's '^' appears 2x as often and its ':' 4x as often). This is unmeasured.

### Gaps
- I found no VLM or HTR paper that measures a gain from a dataset or collection tag in the prompt. The evidence is NMT/ASR only.

---

## Q4. Option 4: mixture of LoRA experts, routed adapters, and LoRA merging

### Takeaway
Learned routers (LoRAMoE, MoLE, X-LoRA, LLaVA-MoLE) solve a problem we do not have: here the domain is known from the pixels. Hard routing to one class adapter is the degenerate, optimal case, and one 2026 study finds hard routing beats learned soft composition. Merging and souping (linear, TIES, DARE, SVD via `add_weighted_adapter`) is useful mainly as a cheap regulariser: interpolate a specialist back toward the shared adapter. It is not a way to build specialists. Full MoE-LoRA training is not practical in the time left.

### Cited Findings
- **LLaVA-MoLE:** sparse top-1 routing over LoRA experts in MLP layers only, with a load-balancing loss; LoRA r=32, LR 2e-5, batch 64, **64 A100s, ~16 h** for a 3-dataset mix.
  - It recovered from the mixing drop: eHub 307.3 vs 298.8 for plain LoRA on the mix, and ChartQA 41.36 vs 36.72.
  - It "can even outperform the plain-LoRA baseline trained with twice the samples".
  - Sparse vs dense MoE GPU memory: 61% vs 83%, with dense MoE running out of memory at 3 experts on long contexts.
  - — [LLaVA-MoLE (arXiv 2401.16160)](https://arxiv.org/html/2401.16160v2)
- **MoLE** (Microsoft, 2024) treats each layer of already-trained LoRAs as an expert. It learns a gating function per layer to compose them, and outperforms direct arithmetic merging. — [Mixture of LoRA Experts (arXiv 2404.13628)](https://arxiv.org/abs/2404.13628) (via search summary)
- **X-LoRA** takes pre-trained LoRAs as experts and learns dynamic gating per token and per layer. **LoRAMoE** places multiple LoRA experts with a router in the FFN, plus a localized balancing constraint. — [Segmind overview of MoLE methods](https://blog.segmind.com/mixture-of-lora-experts-techniques/) (secondary source)
- **Hard-routed MoR-LoRA:** applies exactly one frozen LoRA expert at unit scale, and "achieves the best average performance among the learned-composition baselines" (LoRAMixer, MoLE with soft routing) on LLaMA-3B and 8B. — [Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs (arXiv 2606.31413)](https://arxiv.org/pdf/2606.31413) (via search snippet)
- **LoraHub** composes existing LoRAs with weights tuned gradient-free on about 5 examples of an unseen task. It approaches, but does not beat, in-context learning on BIG-Bench Hard, at zero-shot inference cost. — [LoraHub (arXiv 2307.13269)](https://arxiv.org/abs/2307.13269); [HTML](https://arxiv.org/html/2307.13269v2)
- **PEFT `add_weighted_adapter` merge types:** `svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`. Parameters include `weights` (may be negative), `density` (for the pruning methods), `majority_sign_method` ("total" or "frequency"), and `svd_rank`/`svd_clamp`. Documented caveats:
  - `cat` produces rank = the sum of the input ranks.
  - `cat` and `svd` are exact, while `linear` is a "rough approximation".
  - `svd` needs full precision (not fp16/bf16).
  - — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
- PEFT also provides `reduce_intruder_dimension()` (in `peft.tuners.lora.intruders`), which can be run after training when LoRA "intruder dimensions" cause forgetting. — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)

### Inferences
- With 4 classes, all known at inference time, the "router" is our class detector. A learned token-level router would have to learn from about 4k lines something we can already compute exactly. LLaVA-MoLE-scale training (64 GPUs) is impossible on 1 GPU in 7 days. **Skip MoE.**
- **Useful merging trick.** If a specialist (Option 2) overfits, build `add_weighted_adapter(["shared","spec_B"], weights=[1-λ, λ], combination_type="cat")` with λ in {0.3, 0.5, 0.7}. Choose λ by class-B validation CER.
  - `cat` is exact for a weighted sum of ΔW. `linear` interpolates A and B separately, which only approximates it, because ΔW = B·A adds cross terms.
  - The cost is a rank-64 temporary adapter, which is negligible.
  - This is a WiSE-FT/"soup"-style regulariser; I did not find it tested for HTR.
- TIES/DARE are designed to combine adapters for **different** tasks while resolving sign conflicts. For "shared + one specialist" they add hyper-parameters (`density`) with no clear benefit. `cat` or `linear` interpolation is enough.
- LoraHub would only matter for an unseen hand with no labels. That is not our case.

### Gaps
- No MoE-LoRA study on OCR/HTR was found.
- I did not verify whether current PEFT ships an `XLoraConfig` usable with Qwen-VL; check the installed version.
- The LoRALib benchmark of LoRA-MoE methods ([arXiv 2509.18137](https://www.arxiv.org/pdf/2509.18137)) turned up in search, but I did not read its results.

---

## Q5. Option 5: class-balanced sampling or loss re-weighting as a cheaper alternative

### Takeaway
Temperature-based sampling is the standard tool, from multilingual NMT. T=5 is common, but a 2023 analysis suggests T=1–2 is often better. For our proportions, T=2 moves A2 from 9.3% to 15.7% of samples (1.7x upsampling), and A1 down from 40.7% to 32.8%. This is a zero-cost change to the shared run and should be tried alongside the class tag.

### Cited Findings
- Temperature sampling draws a language pair (domain) with probability ∝ p^(1/T). T=1 is the natural distribution; T=100 is almost uniform. Sampling equally gives "a huge boost" to low-resource pairs, but high-resource pairs fall well below their bilingual baselines. T=5 is the common compromise. — [search summary drawing on multilingual NMT literature, e.g. Arivazhagan et al. 2019 "Massively Multilingual NMT"](https://www.researchgate.net/publication/334600243_Massively_Multilingual_Neural_Machine_Translation)
- "Recent research shows that T=1 and T=2 are often better ... than the commonly-used T=5". This most likely comes from Shaham et al. 2023, which attributes severe multilingual interference mainly to small models and treats temperature tuning as the key remedy. — [Causes and Cures for Interference in Multilingual Translation (arXiv 2212.07530)](https://arxiv.org/pdf/2212.07530) (attribution from search snippet; not verified in full text)

**Computed for our class sizes** (my arithmetic, p_c ∝ n_c^(1/T); "×" = effective upsampling vs natural):

| T | A1 (1,669) | B (1,043) | A3 (1,003) | A2 (383) |
|---|---|---|---|---|
| 1 | 40.7% | 25.5% | 24.5% | 9.3% |
| 2 | 32.8% (×0.81) | 26.0% (×1.02) | 25.5% (×1.04) | 15.7% (×1.68) |
| 3 | 30.2% (×0.74) | 25.8% | 25.5% | 18.5% (×1.98) |
| 5 | 28.1% (×0.69) | 25.6% | 25.4% | 20.9% (×2.24) |
| ~uniform | 25% | 25% | 25% | 25% (×2.65) |

### Inferences
- Only A2 is seriously under-represented. B and A3 are nearly unchanged at any T, so rebalancing mostly trades A1 samples for A2 samples. It will not specialize B.
- Upsampling A2 about 2x repeats each A2 line about 2x within an epoch. Given the observation that "more epochs = worse", keep T ≤ 3 and use a different augmentation for each repeat.
- Loss re-weighting (per-class weights ∝ n_c^(1/T - 1) on token-level cross-entropy) is roughly equivalent in expectation but gives noisier gradients at batch 4. Prefer sampling (my judgement, no source).
- Implementation: use a `WeightedRandomSampler` over line indices with weight p_c/n_c per line, or override the Trainer's `_get_train_sampler`.
- Balancing does not give per-class conventions. Pair it with the class tag (Option 3) and/or specialists (Option 2).

### Gaps
- No HTR/VLM study on class-balanced sampling across collections was found.

---

## Q6. Serving several LoRA adapters in one process with PEFT, and memory for 8B vs 32B

### Takeaway
PEFT supports several named adapters on one base model. You switch the active adapter with `set_adapter`, or send different adapters per sample in one `generate` call with `adapter_names=[...]`, which has overhead. For 4 classes, the simplest fast path is to **sort test lines by class, run class-homogeneous batches, and call `set_adapter(class)` between groups**. Optionally merge the active adapter for zero-latency inference. On a 96 GB GPU, adapters are tiny (about 0.1–0.5 GB each), so all four specialists fit alongside even a 32B base.

### Cited Findings
- **Named adapters:** `PeftModel.from_pretrained(base, path, adapter_name="x")`, then `model.load_adapter(path, adapter_name="y")`, `model.set_adapter("y")` and `model.delete_adapter("y")`. `unload()` and `merge_and_unload()` are **not in-place**; assign the result. — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
- **Mixed-adapter batches:** `model.generate(**inputs, adapter_names=[...])`, where `"__base__"` means no adapter. Documented limitations:
  - inference only;
  - `disable_adapter()` takes precedence;
  - **cannot be used with merged adapters** (call `unmerge_adapter()` first), and is incompatible with `merge_and_unload()`;
  - **does not work with DoRA**;
  - `modules_to_save` support is limited to Linear/Embedding/Conv layers;
  - there is "expected overhead", because the effective batch size shrinks to the samples per adapter. Mitigations: larger batches, fewer distinct adapters per batch, or dedicated servers (LoRAX, punica, S-LoRA).
  - — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
- **Merging:** `merge_adapter()` / `unmerge_adapter()` merge while keeping the original weights so you can unmerge later. `merge_and_unload()` merges permanently. — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
- LoRA's design adds "no additional inference latency" once the update is merged into the frozen weights. — [Hu et al. 2021, LoRA (arXiv 2106.09685)](https://arxiv.org/abs/2106.09685)
- Tied weights: use `ensure_weight_tying=True` in `LoraConfig` to keep `embed_tokens`/`lm_head` tied. — [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)

### Inferences
**Serving plan for the Kaggle notebook:**
1. Load the base model in bf16.
2. `PeftModel.from_pretrained(base, shared_dir, adapter_name="shared")`.
3. `load_adapter(dir_c, adapter_name=c)` for each specialist c.
4. Classify all test lines by pixel class, then group them.
5. For each class: `set_adapter(c)`, then run batched `generate`.
6. For classes where the specialist failed validation, use `set_adapter("shared")`.

Notes on this plan:
- Merged vs unmerged: an unmerged adapter adds two small matmuls per targeted linear layer. For r=32 this costs a few percent extra FLOPs, plus kernel-launch overhead that is more noticeable at small batch sizes during decoding. I found no measured number, so this is a gap. If speed matters, call `merge_adapter()` per class group and `unmerge_adapter()` before switching; that is 4 merge/unmerge pairs in total.
- Merge/unmerge in bf16 can introduce small rounding drift. Reloading the base, or keeping adapters unmerged, avoids it; this is an unverified concern.
- **Variant 2b (stacked):** `merge_and_unload()` the shared adapter into the base once. Then load the small specialists as named adapters on top of it. Mixed `adapter_names` batches still work, because the shared weights are now plain base weights.
- **Adapter size estimate** (my arithmetic, assuming a Qwen3-8B-style LLM with 36 layers, hidden 4096, MLP 12288 and 8 KV heads × 128): r=32 on all 7 linear projections is about **87M params, roughly 175 MB in bf16, LLM only**. Vision-tower LoRA adds a small amount on top. A 32B-class LLM (64 layers, hidden 5120, larger MLP) at r=32 is roughly 0.25–0.3B params (0.5–0.6 GB). The four specialists together are negligible next to the base weights.
- **Base memory (bf16, 2 bytes/param):**
  - 8B-class VLM: about 16–18 GB of weights, leaving plenty of room for KV cache, large decode batches and training.
  - 32B-class VLM: about 64–67 GB of weights. LoRA training still fits in 96 GB with gradient checkpointing, since the LoRA params, gradients and AdamW states come to about 16 bytes × 0.3B ≈ 5 GB. Headroom for activations is tight, and each specialist run costs about 4x the time of 8B.
  - Given the observation that "bigger models scored worse" and the 7-day limit, the 8B base plus specialists is the pragmatic choice.
  - These are estimates; the exact parameter counts of the Qwen3-VL-8B/32B checkpoints were not verified in this session.
- **Training API:** to continue an adapter, use `PeftModel.from_pretrained(base, shared_dir, is_trainable=True)`. For a copy-per-class, reload from disk each time rather than deep-copying in memory. This follows standard PEFT usage but was not re-verified in the docs fetched here.

### Gaps
- No measured unmerged-vs-merged LoRA decode-latency numbers for Qwen-VL were found.
- I did not verify vLLM's multi-LoRA support for Qwen2.5-VL / Qwen3-VL, including whether the vision-tower LoRA is supported there. If vLLM is used for inference, check this before relying on vision-tower LoRA.

---

## Q7. Cross-cutting failure modes and a recommended 7-day recipe

### Takeaway
Rank options by expected gain per GPU-hour:
1. Class tag in the prompt plus mild temperature sampling (T≈2), retrained into the shared adapter. This is one run.
2. Per-class continuation from that shared adapter (Option 2a/2b), with a lower LR, CER-based early stopping and some replay. Keep a specialist only where k-fold CER beats the shared model.
3. Optionally, interpolate each specialist with the shared adapter via `cat`.

Skip from-base per-class LoRAs and learned MoE routers.

### Cited Findings
- VLM OCR on historical text shows silent orthographic normalization and linguistically plausible substitutions, unlike OCR-native models. Class-specific archaic spellings are therefore exactly where errors hide. — [Vesalainen et al. 2026](https://arxiv.org/abs/2602.14524)
- A wrong domain conditioning is very costly (up to -14 BLEU with a wrong tag), even though correct conditioning helps. — [Kobus et al. 2017, Table 4](https://arxiv.org/pdf/1612.06140)
- LoRA tolerates large batches less well than full fine-tuning, and LoRA should cover MLP layers as well as attention. — [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)
- HTR fine-tuning shows a U-shaped CER curve whose minimum comes after the validation-loss minimum, so select checkpoints on CER. — [Kohút & Hradiš 2023](https://arxiv.org/html/2302.06308)
- A Qwen2.5-VL-7B LoRA OCR model with near-perfect in-distribution CER degraded sharply on real out-of-distribution handwriting (CER 2.6% to 25.4%). — [Manchu VLM OCR (arXiv 2507.06761)](https://arxiv.org/html/2507.06761v1)
- Large-scale historical HTR with Qwen2.5-VL fine-tuning works: CHURRO, fine-tuned from Qwen2.5-VL on 155 corpora (~100k pages), reports 82.3% (printed) and 70.1% (handwritten) on its benchmark. — [CHURRO (arXiv 2509.19768)](https://arxiv.org/pdf/2509.19768) (via search summary). A community Qwen2.5-VL-7B HTR project on early-modern Spanish used LoRA r=16, alpha=32. — [vlm-handwriting-recognition-es (GitHub)](https://github.com/junghare-aniket/vlm-handwriting-recognition-es)

### Inferences
**Suggested concrete plan.** These are inferences to validate with k-fold CER per class.

**Day 1–2: tag + balance, retrained into the shared adapter.**
- Add `Hand class: {B|A1|A2|A3}` (plus a one-line conventions hint) to the prompt.
- Use T=2 class sampling.
- Keep r=32, alpha=64, LR 2e-4 cosine and batch 4. The current recipe sits inside the documented LoRA LR band (Biderman's best math LR was 2e-4), so there is no reason to change it.
- Compare per-class CER with the current shared model.

**Day 2–4: specialists for B and A2 first.**
- Why these two first: B has the most distinct conventions; A2 is the smallest class and most likely under-served.
- Variant 2a: continue the shared adapter at LR 5e-5 to 1e-4, constant with warmup.
  - B: about 300–500 steps at batch 4.
  - A2: about 100–200 steps, plus 10–25% replay of other classes and fresh augmentation.
  - Evaluate CER every 25–50 steps.
- Variant 2b for A2: a merged shared model plus a new r=8–16, LLM-only LoRA.
- Keep a specialist only if its k-fold CER beats the tagged shared model by more than the seed-to-seed noise.

**Day 4–5: A1/A3 specialists (optional), and λ-interpolation of specialists with the shared adapter via `add_weighted_adapter(..., combination_type="cat")`.**

**Inference:** group test lines by class, `set_adapter` per group, and fall back to "shared" for any class without a winning specialist.

**Failure modes to watch:**
- A2 overfitting: an early CER minimum, then a rise.
- The specialist "forgets" shared abbreviations that appear rarely in its class.
- Class-detector errors routing a line to the wrong specialist. Check detector accuracy on the validation set; if it is below ~95%, prefer tag conditioning with tag dropout over hard routing.
- Normalization errors ("the" instead of "ye") increasing after training on a larger base. This may be one explanation for "bigger models scored worse", since larger LMs have stronger linguistic priors. That is a hypothesis only.
- Using `adapter_names` with DoRA or merged weights, which is unsupported.

### Gaps
- There is no direct evidence on the size of gains for VLM per-collection specialists in HTR. Every expected-gain figure above is transferred from CNN/LSTM/small-transformer HTR, NMT or ASR.
- The main questions (whether the tag helps, whether specialists beat the tagged shared model, and the best LR/steps) have to be settled empirically on this data.
