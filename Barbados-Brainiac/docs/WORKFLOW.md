# Barbados-Brainiac workflow

How the pipeline turns `Train.csv`, `Test.csv`, the line images and the two mega prompts into
`submission.csv`: 18 stages run one after another (`bash run_all.sh`), and the two approaches
meet in one candidate pool.

Score: `0.5 · (1 − word edits / 12) + 0.5 · (1 − char edits / 55)` (R.O.A.D. metric).

`docs/workflow.html` is the source of the same diagrams as a styled page.

## Layers

| Layer | What it does | Stages |
|---|---|---|
| Prepare | Builds the line table: split, family A (1639–1668 strips) or B (1669–1710 crops), near-duplicate groups kept in one fold, 5 folds. Renders the model input views. | `prepare`, `views` |
| Approach A | The 1st-place recipe: VLM label correction, dual-target LoRA (ink + transcription), pseudo labels on test and unlabelled images, final LoRA. | `label_correction` … `a_infer` |
| Generate | Fine-tuned VLMs (two soups, one rank-64 LoRA) and the approach A models write about 14 candidate lines each. The CRNN adds beam-8 strings. | `b_train_generators` … `b_pool` |
| Score | Six VLM judges (forced-decode log-probs), the CTC reader and a character n-gram LM score every candidate. | `b_judges`, `b_features` |
| Pick | A gradient-boosted ranker learns from the holdout-fold lines which evidence to trust. The judge-mix → MBR → CTC baseline is its reference. | `b_baseline`, `b_rank` |
| Stabilise | 3 CatBoost members × 5 seeds, averaged within each line, with threads and seeds pinned. | `b_rank` |
| Gates | Twin → tier 1 → Holm → K5 → seed flip ≤ 3 % decide whether the ranker replaces the baseline. | `b_rank` |
| Fuse | Writes one submission per method, fills gaps along the fallback order, and picks the primary by the gates. | `fuse`, `report` |

## End to end

```mermaid
%%{init: {"theme": "base", "flowchart": {"curve": "basis"}, "themeVariables": {"primaryColor": "#f5f0e6", "mainBkg": "#f5f0e6", "primaryBorderColor": "#6e5b3f", "nodeBorder": "#6e5b3f", "primaryTextColor": "#2a241b", "textColor": "#2a241b", "titleColor": "#2a241b", "lineColor": "#7d6b53", "clusterBkg": "#f2f2ef", "clusterBorder": "#6e5b3f", "edgeLabelBackground": "#fbfaf6", "tertiaryColor": "#f2f2ef"}}}%%
flowchart TD
  IN["Train.csv · Test.csv · images<br/>mega_prompt.md · mega_prompt_v2.md"]
  PREP["prepare · views<br/>line table · families A/B<br/>near-duplicate groups · 5 folds"]
  IN --> PREP
  subgraph SA["Approach A · 1st-place pipeline"]
    direction TB
    LC["label_correction<br/>VLM checks every label"]
    A1["a_train_first<br/>dual-target LoRA"]
    PS["a_pseudo<br/>pseudo labels"]
    A2["a_train_final<br/>labels + pseudo labels"]
    AI["a_infer<br/>candidates + own submission"]
    LC --> A1 --> PS --> A2 --> AI
  end
  subgraph SB["Approach B · generate → score → pick"]
    direction TB
    GEN["b_train_generators · b_soups<br/>b_generate · b_readers"]
    POOL[("b_pool<br/>candidate pool per line")]
    SCORE["b_judges · b_features"]
    PICK["b_baseline · b_rank<br/>stabilise · gates"]
    GEN --> POOL --> SCORE --> PICK
  end
  PREP --> LC
  PREP --> GEN
  AI -->|"fusion: A candidates join the pool"| POOL
  PS -.->|"pseudo labels for the data3 soup"| GEN
  PICK --> FUSE["fuse<br/>per-method submissions<br/>primary chosen by the gates"]
  AI --> FUSE
  FUSE --> SUBM[("submission.csv")]
  FUSE --> REP["report<br/>holdout-fold scores"]
```

## Approach A · the 1st-place pipeline, adapted to lines

```mermaid
%%{init: {"theme": "base", "flowchart": {"curve": "basis"}, "themeVariables": {"primaryColor": "#f5f0e6", "mainBkg": "#f5f0e6", "primaryBorderColor": "#6e5b3f", "nodeBorder": "#6e5b3f", "primaryTextColor": "#2a241b", "textColor": "#2a241b", "titleColor": "#2a241b", "lineColor": "#7d6b53", "clusterBkg": "#f2f2ef", "clusterBorder": "#6e5b3f", "edgeLabelBackground": "#fbfaf6", "tertiaryColor": "#f2f2ef"}}}%%
flowchart TD
  IMG["line image"] --> V["views · multi<br/>whole line + overlapping left and right parts<br/>rescaled to the vision patch grid<br/>from patchify_images.py (7 crops per plan)"]
  LBL["Train.csv Target"] --> LC
  EXT[("external corrections<br/>optional, e.g. opus_label")] -.-> LC
  V --> LC["label_correction · base Qwen3-VL-8B<br/>full mega prompt + correction header<br/>accepted within 6 char edits · rejected rows marked noisy<br/>from automated_label_correction.py"]
  LC --> A1["a_train_first · each backbone × scope<br/>answer JSON: ink (corrected) + transcription (label)<br/>LoRA r16 · dropout 0.35 · NEFTune 5 · Unsloth<br/>from train_and_create_pseudo.py"]
  A1 --> PS["a_pseudo<br/>test + unlabelled images + the scope's own prediction lines<br/>kept when mean token log-prob ≥ −0.35"]
  PS --> A2["a_train_final<br/>labels + pseudo labels<br/>from train_with_pseudos.py"]
  A2 --> AI["a_infer<br/>greedy + beam 5 (4 returned) + the ink reading<br/>from text_inference.py"]
  AI --> CL["clean<br/>collection charset · long s → s · plus sign → et-sign · superscripts → ^<br/>from clean_text_preds.py"]
  CL --> SUBA[("submissions/a_qwen3vl8b.csv<br/>submissions/a_qwen25vl7b.csv")]
  CL --> POOLA[("approach B candidate pool")]
```

The 1st place trained each field twice in one answer, the corrected value and the annotator's
raw value, and submitted the raw one. Here the model answers with an `ink` reading and a
`transcription` in the annotators' style, and the `transcription` is what gets submitted.

## Approach B · generate, score, pick, stabilise, gates

```mermaid
%%{init: {"theme": "base", "flowchart": {"curve": "basis"}, "themeVariables": {"primaryColor": "#f5f0e6", "mainBkg": "#f5f0e6", "primaryBorderColor": "#6e5b3f", "nodeBorder": "#6e5b3f", "primaryTextColor": "#2a241b", "textColor": "#2a241b", "titleColor": "#2a241b", "lineColor": "#7d6b53", "clusterBkg": "#f2f2ef", "clusterBorder": "#6e5b3f", "edgeLabelBackground": "#fbfaf6", "tertiaryColor": "#f2f2ef"}}}%%
flowchart TD
  TRAIN["Training lines<br/>folds 1–4 · noisy rows dropped"]
  TEST["TEST lines<br/>1,374"]
  subgraph GEN["1 · Generate candidates"]
    direction TB
    G1["q3_soup3<br/>Qwen3-VL-8B · 3 LoRAs averaged"]
    G2["q25_r64<br/>Qwen2.5-VL-7B · LoRA rank 64"]
    G3["q25_soup_d3<br/>Qwen2.5-VL-7B · 3 LoRAs on corrected + pseudo labels"]
    G4["a:qwen3vl8b · a:qwen25vl7b<br/>approach A models"]
    G5["ctc<br/>CRNN beam-8 strings"]
  end
  TRAIN --> GEN
  TEST --> GEN
  GEN --> POOL[("Candidate pool<br/>greedy + beam 8 + 6 samples per model<br/>deduplicated · up to 32 per line")]
  POOL --> JUD["2 · Six VLM judges<br/>forced-decode log-probs"]
  POOL --> RD["3 · Independent readers<br/>CRNN CTC · external readers<br/>char n-gram LM"]
  JUD --> BASE["b_baseline<br/>judge mix → MBR → CTC overlay"]
  JUD --> FEAT["Features<br/>F0 sources · F1 judges · F2 MBR<br/>F3 char LM · reader block"]
  RD --> FEAT
  FEAT --> RANK["4 · GBDT ranker<br/>target = exact metric loss per candidate<br/>nested CV on the holdout fold"]
  RANK --> STAB["Stabilise<br/>3 CatBoost members × 5 seeds<br/>pinned threads"]
  STAB --> GATES["Gates<br/>twin → tier 1 → Holm → K5 → seed flip ≤ 3 %"]
  BASE -->|"reference"| GATES
  GATES --> FUSE["fuse<br/>ranker if every gate passes, else baseline"]
  FUSE --> OUTB[("submissions<br/>ranker · blend · baseline · a_*")]
```

## Scopes · why the ranker's training data is honest

```mermaid
%%{init: {"theme": "base", "flowchart": {"curve": "basis"}, "themeVariables": {"primaryColor": "#f5f0e6", "mainBkg": "#f5f0e6", "primaryBorderColor": "#6e5b3f", "nodeBorder": "#6e5b3f", "primaryTextColor": "#2a241b", "textColor": "#2a241b", "titleColor": "#2a241b", "lineColor": "#7d6b53", "clusterBkg": "#f2f2ef", "clusterBorder": "#6e5b3f", "edgeLabelBackground": "#fbfaf6", "tertiaryColor": "#f2f2ef"}}}%%
flowchart LR
  subgraph OOF["scope oof"]
    direction TB
    O1["train on every fold<br/>except the holdout fold"] --> O2["LoRAs · CRNN · char LM"] --> O3["candidates + scores<br/>for the holdout lines"] --> O4["ranker training rows<br/>target = metric loss"]
  end
  subgraph FULL["scope full"]
    direction TB
    F1["train on all<br/>labelled lines"] --> F2["LoRAs · CRNN · char LM"] --> F3["candidates + scores<br/>for the test lines"] --> F4["ranker picks<br/>for the submission"]
  end
  O4 -->|"fitted ranker"| F4
```

Every trainable part is built twice. The `oof` models never see the holdout fold, so their
candidates and scores there are fair training data for the ranker. The `full` models serve the
test lines. Parallel copies and duplicate crops of one line share a group and therefore a fold.
