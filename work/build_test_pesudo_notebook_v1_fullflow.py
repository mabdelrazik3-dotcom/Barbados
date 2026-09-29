"""Builds test_pesudo.ipynb — unsloth-style flow (inference -> fine-tune -> inference) on the barbados Kaggle-offline
stack (transformers + PEFT, /kaggle/input models and wheels), with:
  * pixel-only classification of every image into class A/B and 10 sub-classes (or a refit with up to 15),
  * one DETAILED prompt per sub-class (prompt.txt style) filled with conventions measured on the training lines,
  * K nearest solved training examples of the same sub-class shown as few-shot turns (images + ground truth),
  * batch size 1 everywhere, validation every EVAL_EVERY steps with best-step restore,
  * pseudo-labelling of test (+ unlisted) with an optional GUIDE csv (best submission) and a model JUDGE,
  * optional self-training rounds with the confident pseudo-labels.
Reuses the tested infrastructure cells of build_enhanced_notebook.py (install, imports, paths, image utils +
augmentation, processor/model loaders, official metric + MBR, Fold-0 split, trial harness)."""
import importlib.util
import json
import re

ROOT = "D:/HANAFY/ROAD"
spec = importlib.util.spec_from_file_location("enh", f"{ROOT}/work/build_enhanced_notebook.py")
enh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enh)
E = enh.CELLS

_feat_src = open(f"{ROOT}/work/subclass_features.py", encoding="utf-8").read()
LINE_FEATURES_SRC = _feat_src[_feat_src.index("# <<LINE_FEATURES_BEGIN>>") + len("# <<LINE_FEATURES_BEGIN>>"):
                              _feat_src.index("# <<LINE_FEATURES_END>>")].strip("\n")
RULES_SRC = open(f"{ROOT}/work/subclass_rules.py", encoding="utf-8").read().strip("\n")


def find(marker):
    hits = [s for t, s in E if t == "code" and marker in s]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

# ============================================================================ intro
md(r'''
# R.O.A.D. Barbados — `test_pesudo`: sub-class prompts + few-shot, **inference → fine-tune → inference**, and test pseudo-labels

This notebook follows the order of the Unsloth vision notebook:
1. load a model;
2. **see what it outputs before any fine-tuning**;
3. **fine-tune**;
4. **run inference again**.

It uses the Kaggle-offline stack of the `barbados-2-*` notebooks instead of Unsloth: offline wheels, `transformers` + `peft`, and the Qwen-VL models under `/kaggle/input`.

| Step | What happens |
|---|---|
| **Classification** | Every train, val, test and unlisted image gets a class (A = 1630s–1660s, B = 1670s–1710s scans) and one of **10 sub-classes**. The rule uses pixels only, so it works on test. The sub-classes were chosen by a tree that makes each group as consistent as possible in transcription conventions (`^`, `:`, `&`, ye/yt, ff-, digits, fillers, length, capitals). |
| **Prompts** | One **detailed prompt per sub-class**, in the style of `prompt.txt`: ROLE, INPUTS, WHAT IT LOOKS LIKE, ANALYSIS STRATEGY, TRANSCRIPTION RULES with the rates measured on that sub-class's training lines, HABITS, COMMON MISTAKES, EXAMPLES, SILENT RE-CHECK and OUTPUT FORMAT. |
| **Few-shot** | Before each line, the model sees the **K most similar solved training lines of the same sub-class** (image + ground truth) as earlier conversation turns. Val and test examples always come from the training part only. |
| **Batch size 1** | Training (with gradient accumulation) and every inference call. |
| **Inference before fine-tuning** | A quick look on sampled val lines (`INFER_BEFORE="sample"`), or a full run on val + test + unlisted. |
| **Fine-tuning** (`DO_FINETUNE=True`) | LoRA on the training lines. Every `EVAL_EVERY=100` steps a validation prints loss and the official score on held-out training lines. The best step is restored at the end. |
| **Inference after fine-tuning** | All 819 Fold-0 val lines are scored with the official metric, per sub-class. Test + unlisted are decoded as 5-best lists. |
| **Pseudo-labels** | For every test line: the model's 5-best + the **GUIDE** transcription (your best submission, optional). The model **judges** the guide by scoring it with its own log-probability, then MBR under the official cost picks the pseudo-label. Each line gets a confidence and a comment. `GUIDE_VAL_CSV` measures the whole procedure on the 819 val lines. |
| **Self-training** (`SELF_TRAIN_ROUNDS`) | The confident pseudo-labels are added to TRAIN, and the model is fine-tuned and run again. |

> Rules: pseudo-labelling is allowed when it is fully automated. The guide file must come from your own models' outputs (e.g. `submission_offline_best.csv`), never from hand-corrected text.

**Switches:**
- `DO_FINETUNE=False` gives an inference-only notebook, using the base model or `INIT_ADAPTER`.
- `PROMPT_MODE="detailed"` / `"short"` / `"exp002"`.
- `FEWSHOT_K` sets the number of solved examples per line.
- `SUBCLASS_MODE="refit"` with `N_SUBCLASSES` (≤ 15) builds a new tree instead of the 10 fixed sub-classes.
''')

md("### Installation")
code(find("# Cell 1 — OFFLINE install"))
code(next(s for t, s in E if t == "code" and s.lstrip().startswith("%%capture")))
code(find("# Cell 2 — imports"))

# ============================================================================ CFG
md(r'''
### Configuration — every switch of the notebook
''')
code(r'''
# =========================================================
# Cell 3 — CFG
# =========================================================
TRIAL_ID          = "TPL_001"
TRIAL_DESCRIPTION = ("test_pesudo: 10 pixel sub-classes, detailed sub-class prompts + K nearest solved train lines as "
                     "few-shot turns, batch 1; inference -> fine-tune -> inference; guided + judged test pseudo-labels")

# the measured one-line prompt of barbados-2 / EXP_002 (PROMPT_MODE "short" and "exp002")
SHORT_PROMPT = ("This is not modern English, Transcribe the handwriting exactly. Keep the wrong spelling, abbreviations, "
                "and marks (^, ff, unusual letters).Do not modernize or correct anything.")

class CFG:
    WHEELS_DIR = WHEELS
    # ---- model ----
    BASE_MODELS = {
        "8b":  "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
        "7b":  "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2",
        "32b": "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/32b-instruct/1",
    }
    MODEL         = "8b"
    INIT_ADAPTER  = None      # optional LoRA folder to start from (its adapter_config.json inside). None = base model.
                              # An EXP_002 adapter was trained with the one-line prompt and no few-shot: use it with
                              # PROMPT_MODE="exp002" and FEWSHOT_K=0 to reproduce it.
    GRAD_CKPT_MODELS = ("32b",)
    DATA_DIR    = "/kaggle/input/datasets/haniagamal/road-barbados"
    TRAIN_CSV   = None
    TEST_CSV    = None
    IMAGES_DIR  = None
    AUTODISCOVER = True

    # ---- outputs ----
    OUTPUT_DIR     = "/kaggle/working/qwen-vl-ocr"
    FINAL_DIR      = "/kaggle/working/qwen-vl-ocr/final"
    WORK_DIR       = "/kaggle/working"
    SUBMISSION_CSV = "/kaggle/working/submission.csv"
    PREV_RESULTS_DIRS = []

    # ---- protocol ----
    RUN_MODE        = "fold0"      # "fold0": score all 819 Fold-0 val lines | "full": train on all lines, predict test
    FOLD            = 0
    N_FOLDS         = 5
    CKPT_SELECTION  = "inner_loss" # keeps an inner 8 % slice of TRAIN out of training for the step validation — keep
    INNER_CKPT_FRAC = 0.08
    TIME_BUDGET_H   = 11.3

    # ---- the three stages (Unsloth order) ----
    INFER_BEFORE    = "sample"     # "sample": BEFORE_SAMPLE_N val lines (quick look) | "full": val + test + unlisted | "off"
    BEFORE_SAMPLE_N = 60           # stratified over the sub-classes
    DO_FINETUNE     = True         # False -> inference only (INFER_BEFORE is forced to "full")
    SELF_TRAIN_ROUNDS = 1          # after fine-tuning: add the kept test/unlisted pseudo-labels to TRAIN, fine-tune again
    SELF_TRAIN_LR   = 1e-4         # continues from the current adapter
    PREDICT_UNLISTED = True        # the 687 images in neither CSV (more pseudo-labels)

    # ---- classification ----
    SUBCLASS_MODE  = "fixed"       # "fixed": the 10 sub-classes of work/10_subclass_tree.py | "refit": new tree here
    N_SUBCLASSES   = 10            # "refit" only (<= 15); ~25 % of the leaves go to class B
    SUBCLASS_MIN_LINES = 110       # "refit": smallest sub-class (training lines)

    # ---- prompts + few-shot ----
    PROMPT_MODE    = "detailed"    # "detailed": the sub-class prompt (system turn) | "short": SHORT_PROMPT as system turn
                                   # | "exp002": exact barbados-2 format (prompt + image in the user turn)
    FEWSHOT_K      = 2             # solved training lines of the same sub-class shown before each line (inference)
    FEWSHOT_K_TRAIN = 2            # during fine-tuning (keep = FEWSHOT_K so training matches inference)
    FEWSHOT_TRAIN_JITTER = 4       # training: choose K among the nearest K + jitter (varied context); 0 = fixed
    PROMPT_TEXT_EXAMPLES = 6       # text-only example transcriptions written inside each detailed prompt
    USER_INSTRUCTION = "Transcribe this line."

    # ---- training (batch size 1) ----
    PER_DEVICE_TRAIN_BATCH_SIZE  = 1
    GRADIENT_ACCUMULATION_STEPS  = 4      # effective batch 4, as in the measured EXP_002 recipe
    NUM_TRAIN_EPOCHS             = 1
    MAX_STEPS                    = -1
    LEARNING_RATE                = 2e-4
    LR_SCHEDULER                 = "cosine"
    WARMUP_RATIO                 = 0.03
    DATALOADER_NUM_WORKERS       = 2
    MAX_PIXELS                   = 2_000_000
    MIN_PIXELS                   = 256*28*28
    ATTN_IMPL                    = None
    USE_BF16                     = True
    R                            = 32
    LORA_ALPHA                   = 64
    LORA_DROPOUT                 = 0.05
    LORA_VISION                  = True
    SEED                         = 42
    MODEL_OVERRIDES = {"32b": {"LEARNING_RATE": 1e-4}}

    # ---- validation every N optimizer steps (inner slice of TRAIN, never the Fold-0 val) ----
    EVAL_EVERY          = 100
    EVAL_GEN_N          = 96       # inner lines greedy-decoded per validation (stratified over sub-classes)
    EVAL_LOSS_N         = 96
    EVAL_MAX_NEW_TOKENS = 128
    CKPT_METRIC         = "inner_official"   # "inner_official" | "inner_loss" | "last"

    # ---- decoding (batch 1) ----
    BATCH_SIZE      = 1
    MAX_NEW_TOKENS  = 160
    DECODE          = dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=5)
    DECODE_VARIANTS = {"beam5": DECODE}
    BASELINE_VARIANT = "beam5"
    LOOP_GUARD      = 3
    MBR_TEMPERATURE = 1.0
    LOG_EVERY_LINES = 100

    # ---- pseudo-labels: guide + judge ----
    GUIDE_CSV      = None          # best test submission so far (ID, Target) — or pseudo_labels_final.csv
    GUIDE_VAL_CSV  = None          # the same system's Fold-0 val predictions (ID, pred): measures guide / judge on val
    GUIDE_WEIGHT   = "auto"        # prior mass of the guide text in the MBR evidence (0 = model + judge only, 1 = copy
                                   # the guide). "auto": chosen on val with GUIDE_VAL_CSV (5-fold CV reported), else 0.5
    JUDGE_GUIDE    = True          # the model scores the guide text with its own log-probability
    PSEUDO_KEEP    = "agree_or_lowrisk"   # lines kept for self-training: "agree" | "lowrisk" | "agree_or_lowrisk"
    PSEUDO_KEEP_FRAC = 0.7         # "lowrisk": lowest-risk share kept inside every sub-class
    FIRST_PSEUDO_FROM_GUIDE = False       # also train the FIRST fine-tune on guide lines (keep_for_training if present)

    # ---- classes (pixels only) ----
    CLASS_HEIGHT_SPLIT  = 150
    GROUP_A1_MAX_HEIGHT = 58
    GROUP_A2_MAX_PAPER  = 194

    # ---- augmentation (realistic profile, query image only) ----
    USE_AUGMENT   = True
    AUG_COPIES    = 1
    AUG_PROFILE   = "realistic"
    AUG_RESAMPLE  = True
    AUG_STROKE_MODE = "both"
    AUG_ROTATE  = True;  AUG_ROTATE_P  = 0.5;  AUG_ROTATE_DEG    = 4.0
    AUG_SHEAR   = True;  AUG_SHEAR_P   = 0.4;  AUG_SHEAR_MAX     = 0.15
    AUG_ELASTIC = True;  AUG_ELASTIC_P = 0.4;  AUG_ELASTIC_ALPHA = 6.0;  AUG_ELASTIC_SIGMA = 8.0
    AUG_HJITTER = True;  AUG_HJITTER_P = 0.4;  AUG_HJITTER_FRAC  = 0.12
    AUG_PAD     = True;  AUG_PAD_P     = 0.3;  AUG_PAD_FRAC      = 0.06
    AUG_PHOTO   = True;  AUG_PHOTO_P   = 0.7;  AUG_BRIGHT = 0.20;  AUG_CONTRAST = 0.20;  AUG_GAMMA = 1.4
    AUG_TINT    = True;  AUG_TINT_P    = 0.5;  AUG_TINT_MAX = 0.12;  AUG_DESAT_MAX = 0.40
    AUG_ILLUM   = True;  AUG_ILLUM_P   = 0.4;  AUG_ILLUM_MAX = 0.35
    AUG_STROKE  = True;  AUG_STROKE_P  = 0.35
    AUG_BLEED   = True;  AUG_BLEED_P   = 0.25;  AUG_BLEED_ALPHA = 0.12
    AUG_NOISE   = True;  AUG_NOISE_P   = 0.5;  AUG_NOISE_STD = 0.1
    AUG_JPEG    = True;  AUG_JPEG_P    = 0.4;  AUG_JPEG_QMIN = 40;  AUG_JPEG_QMAX = 80

    CORRUPT_IDS = {
        "79tMUVyfIdy3GzkG",  "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO",
    }
    # ---- used by the shared helper cells ----
    OCR_PROMPT = SHORT_PROMPT; CLASS_HINT = False; HINT_LEVEL = "class"; CLASS_AUG_COPIES = None
    PSEUDO_LABEL_CSV = None; PSEUDO_PER_CLASS = True; DO_TRAIN = True; DO_INFER = True; PREDICT_TEST = True
    SPECIALISTS = {}; SPECIALIST_MODELS = (); CLASS_SAMPLING_T = None; CLASS_ROUTING = False; ROUTING_LEVEL = "class"

CFG.MODEL_PATH = CFG.BASE_MODELS[CFG.MODEL]
CFG.MODEL_PATHS = [CFG.MODEL_PATH]
if not CFG.DO_FINETUNE:
    CFG.INFER_BEFORE, CFG.SELF_TRAIN_ROUNDS = "full", 0
TRIAL_CONFIG = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)}
print("TRIAL:", TRIAL_ID, "|", TRIAL_DESCRIPTION)
print(f"MODEL {CFG.MODEL} | mode {CFG.RUN_MODE} | fine-tune {CFG.DO_FINETUNE} | before {CFG.INFER_BEFORE} | "
      f"self-train rounds {CFG.SELF_TRAIN_ROUNDS} | prompts {CFG.PROMPT_MODE} | few-shot K={CFG.FEWSHOT_K} | "
      f"guide {'yes' if CFG.GUIDE_CSV else 'no'}")
''')

code(find("# Cell 4 — resolve & sanity-check paths"))
md(r'''
### Helpers: image loading + augmentation, processor/model loaders, the official metric
These cells are shared with `barbados-2-enhanced.ipynb`. They include the pixel-limit fix for new `transformers` versions and the official leaderboard score `0.5·(1 − word_edits/12) + 0.5·(1 − char_edits/55)`.
''')
code(find("# Cell 5 — text / image utils"))
code(find("# Cell 6 — model & processor loaders"))
code(find("# Cell 7 — OFFICIAL metric"))

md(r'''
<a name="Data"></a>
### Data Prep
The Fold-0 split is identical to all earlier runs: 3,274 training lines, of which 8 % are held out for the step validation, and 819 validation lines that are **all** scored. The test CSV and the unlisted images come after them.
''')
code(find("# Cell 8 — samples, Fold-0 split"))
code(find("# Cell 11 — trial harness"))

# ============================================================================ classification
md(r'''
### Classification: class and sub-class of every train, val, test and unlisted image
Cheap pixel features (size, parchment tone, contrast, pen width, focus, ink from neighbouring lines) go through the fixed rule tree below. It was fitted offline by `work/10_subclass_tree.py` on 4,093 training lines, with a 5-fold CV choice of the number of leaves. It re-applies identically on test images (300 of 300 checked).

| Sub-class | Rule | What the training lines show |
|---|---|---|
| A01 | A, parchment grey ≤ 187 | dark parchment; protests/depositions; capitalised nouns; commas |
| A02 | A, width ≤ 941 px | short narrow crops: attestation lines, dates, digits in 22 % of lines |
| A03 | A, width 942–1048, fine pen | conveyances; ye/yt in 24 %; ~interlined~ words; easiest (0.95) |
| A04 | A, width 942–1048, heavy pen | bonds; & in 49 %, sd in 20 % |
| A05 | A, width > 1048, fine pen | long early bonds; archaic spellings (knowe, tenne, assynes, exers) |
| A06 | A, width > 1048, heavy pen, cool paper | shipping/mercantile protests; few abbreviations |
| A07 | A, width > 1048, heavy pen, warm paper | heavily abbreviated conveyances; & 41 %, : 12 % |
| B01 | B, soft focus | formal deeds; Said/Deed capitals; commas; end dashes |
| B02 | B, crisp | colon-suspension hand: ':' in 26 %, '^' in 26 % (S:^d, Exec:^rs) |
| B03 | B, width > 5032 px | very long lines: affidavits, bills; hardest (0.79) |

The data supports 10 sub-classes. More leaves did not make the groups purer under cross-validation, and class B stays flat beyond 3 leaves. `SUBCLASS_MODE="refit"` builds up to 15 anyway.
''')
code(r'''
# =========================================================
# Cell 9 — CLASSIFICATION: pixel features -> class (A/B) and sub-class for every image
# =========================================================
import random as _random
import torch.nn as _nn
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

RUN_LOG = os.path.join(CFG.WORK_DIR, "run_log.txt")

def log(msg, banner=False):
    """timestamped print (flushed) + copy in run_log.txt"""
    line = f"[{(time.time() - SESSION_T0) / 60:7.1f} min] {msg}"
    if banner:
        line = "\n" + "=" * 110 + "\n" + line + "\n" + "=" * 110
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---- pixel features: IDENTICAL code to work/subclass_features.py (the offline rule was fitted on it) ----
__LINE_FEATURES__

# ---- the fixed rule tree (work/subclass_rules.py, fitted by work/10_subclass_tree.py) ----
__RULES__

def _feat_row(p):
    try:
        return line_features(p)
    except Exception:
        return None

def compute_features(paths):
    paths = list(dict.fromkeys(p for p in paths if p and os.path.exists(p)))
    t0 = time.time()
    if os.name == "posix" and len(paths) > 200:
        from multiprocessing import Pool
        with Pool(max(1, min(8, os.cpu_count() or 2))) as pool:
            feats = pool.map(_feat_row, paths, chunksize=32)
    else:
        feats = [_feat_row(p) for p in paths]
    out = {os.path.basename(p)[:-4]: f for p, f in zip(paths, feats) if f is not None}
    log(f"[CLASSIFY] pixel features of {len(out)} images in {time.time() - t0:.0f}s")
    return out

FEATS = compute_features([s["image"] for s in core_samples + inner_samples + val_samples] + TEST_PATHS + UNL_PATHS)
FEAT_DF = pd.DataFrame.from_dict(FEATS, orient="index")
SUB_FEATS = ["height", "width", "aspect", "paper", "contrast", "ink_frac", "tint", "band_frac", "band_center",
             "edge_ratio", "xh_px", "stroke_rel", "sharp"]

def convention_targets(t):
    t = norm_text(t)
    ws = t.split()
    return [float("^" in t), float(":" in t), float("&" in t), float("+" in t),
            float(bool(re.search(r"(?i)(?:^|\s)y(?:e|t|m|r|u|or|em|eir)(?:\s|$)|(?:^|\s)y\^", t))),
            float(any(w[:2] in ("ff", "Ff", "FF") for w in ws)), float(bool(re.search(r"\d", t))),
            float(bool(re.search(r"[~_]|-$", t))), float("'" in t or '"' in t), len(t) / 100.0,
            float(np.mean([w[:1].isupper() for w in ws])) if ws else 0.0]

def refit_subclasses(n_total):
    """new pixel-only tree (<= 15 leaves) fitted on the TRAIN lines of this run; B gets ~25 % of the leaves."""
    from sklearn.tree import DecisionTreeRegressor
    n_total = int(max(2, min(15, n_total)))
    pool = [s for s in core_samples + inner_samples if s["id"] in FEATS]
    X = pd.DataFrame([FEATS[s["id"]] for s in pool])[SUB_FEATS]
    Y = np.array([convention_targets(s["text"]) for s in pool])
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    era = np.where(X.height > CFG.CLASS_HEIGHT_SPLIT, "B", "A")
    kB = max(1, int(round(n_total * 0.25)))
    kA = max(1, n_total - kB)
    models = {}
    for e, k in (("A", kA), ("B", kB)):
        m = era == e
        mdl = DecisionTreeRegressor(max_leaf_nodes=max(2, k), min_samples_leaf=CFG.SUBCLASS_MIN_LINES, random_state=0) \
              if k > 1 else None
        if mdl is not None:
            mdl.fit(X[m], Y[m])
            leaves = sorted(set(mdl.apply(X[m])))
            models[e] = (mdl, {lf: f"{e}{j + 1:02d}" for j, lf in enumerate(leaves)})
        else:
            models[e] = (None, {})
    log(f"[CLASSIFY] refit tree: A {len(models['A'][1]) or 1} leaves, B {len(models['B'][1]) or 1} leaves")
    def assign(f, height_split=CFG.CLASS_HEIGHT_SPLIT):
        e = "B" if f["height"] > height_split else "A"
        mdl, names = models[e]
        if mdl is None:
            return f"{e}01"
        lf = mdl.apply(pd.DataFrame([f])[SUB_FEATS])[0]
        return names.get(lf, f"{e}01")
    return assign

subclass_fn = refit_subclasses(CFG.N_SUBCLASSES) if CFG.SUBCLASS_MODE == "refit" else subclass_of
SUB = {i: subclass_fn(f, CFG.CLASS_HEIGHT_SPLIT) for i, f in FEATS.items()}
_largest = collections.Counter(SUB.values())
def sub_of(image_id, grp="A1"):
    """sub-class of an image; an unreadable image falls back to the largest sub-class of its class."""
    if image_id in SUB:
        return SUB[image_id]
    era = grp[0] if grp else "A"
    return max((k for k in _largest if k[0] == era), key=_largest.get, default=next(iter(_largest)))

for s in all_samples:
    s["sub"] = sub_of(s["id"], s.get("grp"))
VAL_SUB  = [s["sub"] for s in val_samples]
TEST_SUB = [sub_of(i, g) for i, g in zip(TEST_IDS, TEST_GROUP)]
UNL_SUB  = [sub_of(i, g) for i, g in zip(UNL_IDS, UNL_GROUP)]
SUB_KEYS = sorted(set(SUB.values()) | set(VAL_SUB) | set(TEST_SUB))

_rows = []
for k in SUB_KEYS:
    ids_k = [i for i, v in SUB.items() if v == k]
    med = FEAT_DF.loc[ids_k].median() if ids_k else pd.Series(dtype=float)
    _rows.append({"sub-class": k, "class": k[0], "train": sum(s["sub"] == k for s in core_samples),
                  "inner": sum(s["sub"] == k for s in inner_samples), "val": VAL_SUB.count(k), "test": TEST_SUB.count(k),
                  "unlisted": UNL_SUB.count(k), "height": med.get("height", np.nan), "width": med.get("width", np.nan),
                  "paper": med.get("paper", np.nan), "tint": med.get("tint", np.nan), "contrast": med.get("contrast", np.nan),
                  "stroke_rel": med.get("stroke_rel", np.nan), "sharp": med.get("sharp", np.nan),
                  "edge_ratio": med.get("edge_ratio", np.nan)})
SUB_TABLE = pd.DataFrame(_rows)
log(f"[CLASSIFY] mode {CFG.SUBCLASS_MODE}: {len(SUB_KEYS)} sub-classes\n" + SUB_TABLE.round(3).to_string(index=False))
_ab = pd.crosstab(pd.Series([s["sub"] for s in all_samples], name="sub-class"),
                  pd.Series([s["grp"] for s in all_samples], name="old group"))
log("[CLASSIFY] train+val lines: sub-class x old A1/A2/A3/B group\n" + _ab.to_string())
pd.DataFrame([{"ID": i, "split": sp, "class": sb[0], "subclass": sb, **FEATS.get(i, {})}
              for sp, ids_, subs_ in (("train", [s["id"] for s in core_samples + inner_samples], [s["sub"] for s in core_samples + inner_samples]),
                                      ("val", VAL_IDS, VAL_SUB), ("test", TEST_IDS, TEST_SUB), ("unlisted", UNL_IDS, UNL_SUB))
              for i, sb in zip(ids_, subs_)]).to_csv(os.path.join(CFG.WORK_DIR, "subclass_assignments.csv"), index=False)
log("[CLASSIFY] saved subclass_assignments.csv (ID, split, class, subclass, pixel features)")
''')

# ============================================================================ prompts
md(r'''
### Prompts: one detailed prompt per sub-class
Every prompt follows the structure of `prompt.txt`. It has fixed rules shared by all sub-classes (the markup conventions of the ground truth) plus sub-class material:
- **How the images look,** from their pixel medians.
- **How often each convention occurs.** The `^`, `:`, `&`, ye/yt, ff-, digits, fillers, commas and capitals percentages are **measured on this run's training lines of the sub-class**. Validation lines are never used.
- **Hand-written notes:** document types, habits and the mistakes models make on this hand.
- **Typical words and text examples,** also from the training lines.

The few-shot examples (image + ground truth) are added as conversation turns at run time.
''')
code(r'''
# =========================================================
# Cell 10 — sub-class profiles (TRAIN lines only) and the detailed prompt of every sub-class
# =========================================================
SUBCLASS_NOTES = {
    "A01": dict(title="dark parchment, 1630s-1660s: protests, depositions and declarations",
                documents="notarial protests and depositions sworn before the Secretary (Appearer, Merchant, protest, "
                          "recovered, did depose), with shipping and debt matters",
                habits=["Nouns and even 'Said' are often capitalised (Said, Land, Act, Assignes, Merchant): copy every "
                        "capital exactly as written, never normalise case",
                        "Commas are written more often than in the other A hands: copy them where the scribe wrote them",
                        "Old spellings: publique, yeare, yee, Assignes, Appearer; u written for v (seruants)"],
                mistakes=["writing 'said' for a capitalised 'Said' (or the reverse)",
                          "merging two words the scribe separated, or splitting one word",
                          "modernising names and old spellings"]),
    "A02": dict(title="short narrow crops, 1630s-1660s: attestation, witness and dating lines",
                documents="attestation clauses (Signed sealed and delivered in the presence of / prsence / pnce of us), "
                          "witness names, dates and years",
                habits=["Digits are frequent (years, days, sums): read every digit and every superscript ordinal "
                        "(19^th) carefully",
                        "Attestation formulas are abbreviated: prsence, pnce, sealed & delivered: keep the abbreviation "
                        "exactly as written",
                        "Names and signatures: copy letter by letter, including doubled letters and odd spellings"],
                mistakes=["capitalising or lower-casing Signed / Sealed differently from the image",
                          "normalising personal names", "dropping the ^ of ordinals and suspended words"]),
    "A03": dict(title="medium-width strips, fine pen, light parchment, 1630s-1660s: conveyances of plantations and land",
                documents="deeds of sale and conveyance (plantation, parish, bargained, seales, buildings, executors, "
                          "Assignes)",
                habits=["Thorn words are common: ye, yt, ym (y = th): keep them exactly, never write the/that/them",
                        "Interlined or inserted words are written between tildes: ~and~, ~assignes~: keep both tildes",
                        "A trailing ~ or - may fill the end of the line: keep it"],
                mistakes=["dropping the tildes of interlined words", "adding punctuation that is not written "
                          "(a period after M^r)", "normalising surname spellings"]),
    "A04": dict(title="medium-width strips, heavy pen, 1630s-1660s: abbreviation-dense bonds and bills of sale",
                documents="bonds, receipts and bills of sale (heires executors administrators, hundred pounds, good and "
                          "lawfull money)",
                habits=["The ampersand & is written in about half of the lines: never write 'and' for &",
                        "Abbreviations without marks: sd (said), yt, wch, pte, pcell, delivd; with marks: Adm^rs, Tho: "
                        "- copy exactly, never expand",
                        "A dash '-' often fills the end of the line: keep it"],
                mistakes=["completing abbreviations (assigns/assignes, delivd/delivered)",
                          "dropping the colon of a suspension such as Tho:", "misreading the digits of ordinals"]),
    "A05": dict(title="long lines, fine pen, 1630s-1640s: early bonds, bills and obligations",
                documents="obligations and bills (Knowe all men by these presents, bindeth, planters, servants, wooll) "
                          "with the earliest spellings",
                habits=["Archaic spellings everywhere: knowe, tenne, twentie, anie, theise, untill, withall, "
                        "Everlastinge, Peeter - never modernise",
                        "Short administrative forms: exers, admistrs, excers, assynes, admrs - copy exactly",
                        "Superscripts (w^tsoever, 16^th) and apostrophe/quote contractions (afores'd, p'forme, "
                        "plantac\"on) occur: keep the marks"],
                mistakes=["modernising spellings (theise -> this, writeing -> writing)", "dropping ^ in superscripts",
                          "capitalising 'and' / 'that' when the scribe did not"]),
    "A06": dict(title="long lines, heavy pen, cool-toned parchment, 1630s-1660s: shipping and mercantile records",
                documents="ship ownership, cargo and manifests, protests before the Secretary (ship, owners, manifest, "
                          "appearer, personally appeared)",
                habits=["Few abbreviations: superscripts appear in only about one line in twenty - do not invent ^ marks",
                        "Number words are written out (sixty, six, sixteenth): copy their capitals exactly",
                        "Doubled consonants and final e belong to the scribe: premisses, Tonne-style spellings"],
                mistakes=["changing the case of number words", "misreading rare nautical and mercantile nouns",
                          "dropping or adding doubled letters"]),
    "A07": dict(title="long lines, heavy pen, warm parchment, 1630s-1660s: heavily abbreviated conveyances",
                documents="conveyances and leases with dense legal abbreviation (aforesd, pfitts, Adm^rs, Ex^rs, Captn:, "
                          "Willm, Gent)",
                habits=["& is frequent, thorn words (ye, yt) and sd for said are common",
                        "Colon suspensions occur (Captn:, Tho:, grt:) - keep the colon",
                        "Marked and unmarked abbreviations both occur (Adm^rs and Admrs, Ex^rs and Exrs): copy the one "
                        "the scribe wrote"],
                mistakes=["changing the case of Adm^rs / Exec^rs", "dropping the colon after Captn:",
                          "dropping ^ (Adm^rs -> Admrs) or adding it where the scribe did not write it"]),
    "B01": dict(title="tall high-resolution scans, soft focus, 1670s-1710s: formal deeds of land",
                documents="formal deeds and grants (Hundred Acres, premises, This Deed, first above written, King Charles)",
                habits=["Capitalised nouns are frequent (Said, Deed, Land, Acres, Hundred): copy capitals exactly",
                        "Commas are written in about one line in five, and a dash '-' ends about one line in ten",
                        "Superscripts (y^e, M^r, ffeb^ry) and initial ff (ffeb^ry, ffrancis)"],
                mistakes=["writing 'said' for a capitalised 'Said'",
                          "dropping or adding a trailing comma, period or end-of-line dash", "normalising names"]),
    "B02": dict(title="tall high-resolution scans, crisp, dark stained parchment, 1670s-1710s: the colon-suspension hand",
                documents="deeds and wills with executors' formulas (Exec:^rs, Adm:^rs, heyres, Lands, Singular, parrish)",
                habits=["Colon suspensions in about one line in four (John:, S:^t, S:^d, Exec:^rs, Adm:^rs): the colon "
                        "comes BEFORE the ^ - write S:^d, never S^d or S^d:",
                        "Superscripts in about one line in four",
                        "Capitalised pronouns and adjectives (Said, Saide, Same, Singular, Ever): copy case exactly",
                        "Spellings: heyres, saide, hee, parrish, Lawfully"],
                mistakes=["dropping the colon of a suspension (the most frequent error in this sub-class)",
                          "dropping ^", "swapping the order of colon and caret", "lower-casing Said / Same"]),
    "B03": dict(title="very wide tall scans, 1670s-1710s: long lines of affidavits, bills and certificates",
                documents="affidavits (made oath on the Holy Evangelists), bills of exchange (Att tenn dayes Sight), "
                          "certificates and land deeds",
                habits=["Superscripts in about three lines in ten and colon + caret suspensions (S:^d, Exec:^rs)",
                        "Commas in about one line in five; '_' may fill the end of the line",
                        "Old spellings: mee, bee, wee, tenn dayes, conteyneing"],
                mistakes=["the order of colon and caret in suspensions", "stopping before the true end of the long line",
                          "added or missing commas"]),
}

def _pct(T, pat):
    return 100.0 * float(np.mean([bool(re.search(pat, t)) for t in T])) if T else 0.0

def label_profile(texts):
    T = [norm_text(t) for t in texts if norm_text(t)]
    if not T:
        return {"n": 0}
    L = np.array([len(t) for t in T]); Wd = np.array([len(t.split()) for t in T])
    return {"n": len(T), "chars": float(np.median(L)), "chars_q25": float(np.percentile(L, 25)),
            "chars_q75": float(np.percentile(L, 75)), "words": float(np.median(Wd)),
            "caret": _pct(T, r"\^"), "colon": _pct(T, ":"), "amp": _pct(T, "&"), "plus": _pct(T, r"\+"),
            "thorn": _pct(T, r"(?:^|\s)y(?:e|t|m|r|u|or|em)(?:\s|$)|(?:^|\s)y\^"), "ff": _pct(T, r"(?:^|\s)[Ff]f"),
            "digit": _pct(T, r"\d"), "tilde": _pct(T, "~"), "under": _pct(T, "_"), "enddash": _pct(T, r"-$"),
            "apos": _pct(T, "['\"]"), "comma": _pct(T, ","), "period": _pct(T, r"\."),
            "Said": _pct(T, r"(?:^|\s)Said(?:\s|$)"), "said": _pct(T, r"(?:^|\s)said(?:\s|$)"),
            "sd": _pct(T, r"(?:^|\s)s:?\^?d(?:\s|$)"),
            "cap_words": 100.0 * float(np.mean([np.mean([w[:1].isupper() for w in t.split()]) for t in T]))}

_TRAIN_BY_SUB = {k: [s for s in core_samples if s["sub"] == k] for k in SUB_KEYS}
PROFILES = {k: label_profile([s["text"] for s in v]) for k, v in _TRAIN_BY_SUB.items()}

def distinctive_words(k, top=18, min_count=5):
    wc = collections.Counter(w for s in _TRAIN_BY_SUB.get(k, []) for w in norm_text(s["text"]).split())
    allc = collections.Counter(w for v in _TRAIN_BY_SUB.values() for s in v for w in norm_text(s["text"]).split())
    n, N = max(1, sum(wc.values())), max(1, sum(allc.values()))
    lo = {w: math.log((c + .5) / n) - math.log((allc[w] + .5) / N) for w, c in wc.items()
          if c >= min_count and re.search(r"[A-Za-z]", w)}
    return [w for w, _ in sorted(lo.items(), key=lambda x: -x[1])[:top]]

def look_description(k):
    ids_k = [i for i, v in SUB.items() if v == k]
    if not ids_k:
        return "no images measured"
    m = FEAT_DF.loc[ids_k].median()
    era = FEAT_DF[FEAT_DF.height > CFG.CLASS_HEIGHT_SPLIT] if k.startswith("B") else FEAT_DF[FEAT_DF.height <= CFG.CLASS_HEIGHT_SPLIT]
    g = era.median() if len(era) else m
    parts = [f"images about {m.height:.0f} px high and {m.width:.0f} px wide (about {m.aspect:.0f}:1)",
             "dark, stained parchment" if m.paper < 175 else "light parchment" if m.paper > 198 else "medium-toned parchment",
             "warm brownish tone" if m.tint > 60 else "cool greyish tone" if m.tint < 48 else "neutral tone",
             "low ink contrast" if m.contrast < 70 else "strong ink contrast" if m.contrast > 95 else "moderate ink contrast",
             "thick, heavy pen strokes" if m.stroke_rel > 1.12 * g.stroke_rel else "thin, fine pen strokes"
             if m.stroke_rel < 0.9 * g.stroke_rel else "medium pen strokes",
             "soft focus" if m.sharp < 0.85 * g.sharp else "crisp focus" if m.sharp > 1.1 * g.sharp else "normal focus",
             "strokes from the lines above/below often reach into the image" if m.edge_ratio > 0.45
             else "little ink from neighbouring lines"]
    return "; ".join(parts)

def _share(p):
    """percentage -> words, e.g. 26 -> 'about one line in four (26 %)'"""
    if p < 0.5:
        return "almost never (<1 %)"
    if p >= 90:
        return f"in almost every line ({p:.0f} %)"
    n = max(1, int(round(100.0 / p)))
    return f"about one line in {n} ({p:.0f} %)" if n > 1 else f"in most lines ({p:.0f} %)"

def build_prompt(k, k_shots):
    P, N = PROFILES.get(k, {"n": 0}), SUBCLASS_NOTES.get(k, {})
    title = N.get("title", f"{'tall high-resolution scans, 1670s-1710s' if k.startswith('B') else '1630s-1660s'} "
                            f"records (pixel sub-class {k})")
    era = "1670s-1710s, tall high-resolution scans" if k.startswith("B") else "1630s-1660s"
    L = [f"BARBADOS HISTORIC HANDWRITING TRANSCRIBER - SUB-CLASS {k}: {title.upper()}", "",
         "ROLE",
         "- You are a palaeographic transcriber of 17th-century Barbados legal records (R.O.A.D. archive). Each image is "
         "ONE cropped line of handwriting. Produce an exact diplomatic transcription of that line: the letters, "
         "spelling, capitals, abbreviations and marks exactly as the scribe wrote them.",
         "- You transcribe; you never translate, modernise, expand, complete or correct.", "",
         "INPUTS",
         (f"- First, {k_shots} solved example(s) from this same sub-class: a line image followed by its exact "
          "ground-truth transcription. Use them to calibrate this hand's letter forms, spelling and markup. Never copy "
          "their words unless the same words are really written in the new line."
          if k_shots else "- One line image."),
         "- Then the line to transcribe. Transcribe only that last image.", "",
         "WHAT THIS SUB-CLASS LOOKS LIKE (measured on the scans)",
         f"- Scans: {look_description(k)}.",
         f"- Documents: {era}; {N.get('documents', 'Barbados deeds, bonds, protests and depositions')}."]
    if P.get("n"):
        L.append(f"- Typical line: about {P['chars']:.0f} characters and {P['words']:.0f} words (half of the lines have "
                 f"{P['chars_q25']:.0f}-{P['chars_q75']:.0f} characters). Measured on {P['n']} training lines.")
    L += ["", "TASK",
          "Read the central line of the image from left to right and write it out character-for-character in the "
          "conventions of the ground truth below. Every word of the line, and nothing else.", "",
          "ANALYSIS STRATEGY (MANDATORY ORDER)",
          "1) Find the central text line. Ignore strokes, loops and descenders that belong to the lines above or below.",
          "2) Split the line into words by the visible gaps between them, not by modern word boundaries.",
          "3) Before reading letters, spot the marks: raised letters, colons after suspended words, ampersands, tildes, "
          "dashes and fillers at the line end.",
          "4) Resolve letters with this hand's own forms (compare with the solved examples): long s, e/o, u/n/m minims, "
          "c/t, final -e, doubled consonants.",
          "5) Keep the scribe's spelling even when it looks wrong; check it against the habits of this sub-class.", "",
          "TRANSCRIPTION RULES (THE CONVENTIONS OF THE GROUND TRUTH)"]
    rules = [
        ("Superscript (raised) letters", "write ^ before the raised letters: W^m, y^e, M^r, Adm^rs, 19^th, w^ch", "caret"),
        ("Colon suspensions", "a colon marks a shortened word; keep it where written: Tho:, Captn:, John:, S:^d, "
                              "Exec:^rs (colon before the ^)", "colon"),
        ("Ampersand", "write & exactly where the scribe wrote an ampersand; never replace it by 'and' (a few hands "
                      "write a plus-like '+': keep +)", "amp"),
        ("Thorn words", "ye, yt, ym, yr (y = th) are written as they appear; never expand to the, that, them, their",
         "thorn"),
        ("Initial ff", "keep a doubled initial ff (ffrancis, ffeb^ry, ffor)", "ff"),
        ("Digits and numbers", "copy digits, years and ordinals exactly (1647, 19^th); keep roman numerals and number "
                               "words as written", "digit"),
        ("Interlined words and fillers", "words inserted above the line are written between tildes (~and~); a ~, _ "
                                         "or - that fills the end of the line is kept", "tilde"),
        ("Apostrophes and quotes", "an apostrophe or a quote mark used as an abbreviation sign stays: afores'd, "
                                   "p'forme, plantac\"on", "apos"),
        ("Commas and periods", "only where the scribe wrote them; never add sentence punctuation", "comma"),
        ("Capitals", "copy every capital exactly as written, even inside a sentence (Said, Same, Land); never "
                     "normalise case", "cap"),
        ("Spelling", "keep the original spelling: heires, assignes, publique, yeare, doe, bee, mee, tenne, heyres, "
                     "saide; never modernise", None),
        ("Unmarked abbreviations", "sd, wch, wth, pte, pish, pmisses, admrs stay exactly as written; never expand",
         "sd"),
    ]
    for j, (name, text, key) in enumerate(rules, 1):
        rate = ""
        if P.get("n") and key == "cap":
            rate = f" In this sub-class {P['cap_words']:.0f} % of the words start with a capital."
        elif P.get("n") and key == "comma":
            rate = f" In this sub-class commas occur {_share(P['comma'])}, periods {_share(P['period'])}."
        elif P.get("n") and key == "tilde":
            rate = (f" In this sub-class ~ occurs {_share(P['tilde'])}, _ {_share(P['under'])}, and a final - "
                    f"{_share(P['enddash'])}.")
        elif P.get("n") and key in P:
            rate = f" In this sub-class: {_share(P[key])}."
        L.append(f"{j}) {name}: {text}.{rate}")
    L += ["", "HABITS OF THIS SUB-CLASS (from its training lines)"]
    for h in N.get("habits", []):
        L.append(f"- {h}.")
    dw = distinctive_words(k)
    if dw:
        L.append("- Words typical of this sub-class, spelled as its scribes wrote them: " + ", ".join(dw) + ".")
    if P.get("n") and (P["Said"] + P["said"]) > 0:
        L.append(f"- 'Said' with a capital occurs {_share(P['Said'])}; lower-case 'said' {_share(P['said'])}: "
                 "follow the image, not habit.")
    L += ["", "COMMON MISTAKES TO AVOID IN THIS SUB-CLASS"]
    for m_ in N.get("mistakes", ["normalising spelling or capitals", "dropping ^ or : marks",
                                 "adding punctuation that is not written"]):
        L.append(f"- {m_[0].upper() + m_[1:]}.")
    L += ["", "WHAT TO IGNORE",
          "- Ink from the lines above and below, bleed-through from the back of the page, stains, holes, ruling, margins "
          "and page edges.",
          "- Do not describe the image. Do not add line numbers, quotes or brackets that are not written."]
    ex = [norm_text(s["text"]) for s in _TRAIN_BY_SUB.get(k, [])]
    if ex and CFG.PROMPT_TEXT_EXAMPLES:
        lo_, hi_ = P.get("chars_q25", 0), P.get("chars_q75", 1e9)
        mid = [t for t in ex if lo_ <= len(t) <= hi_] or ex
        pick = _random.Random(CFG.SEED + len(k)).sample(mid, min(CFG.PROMPT_TEXT_EXAMPLES, len(mid)))
        L += ["", "EXAMPLE TRANSCRIPTIONS FROM THIS SUB-CLASS (ground truth of other training lines)"]
        L += [f"- {t}" for t in pick]
    L += ["", "MANDATORY SILENT RE-CHECK BEFORE OUTPUT",
          "- Every visible word of the central line is present, from the first to the last; nothing from other lines.",
          "- Every ^, :, &, ~, apostrophe and comma that is written is present, in the right place, and none is invented.",
          "- Capitals and spellings are the scribe's, not modern English.",
          "", "OUTPUT FORMAT (MANDATORY)",
          "Return ONLY the transcription of the line as plain text on one line: no quotes, no labels, no explanations, "
          "no markdown.", "", "RETURN ONLY THE TRANSCRIPTION."]
    return "\n".join(L)

PROMPTS = {k: (build_prompt(k, CFG.FEWSHOT_K) if CFG.PROMPT_MODE == "detailed" else SHORT_PROMPT) for k in SUB_KEYS}
with open(os.path.join(CFG.WORK_DIR, "subclass_prompts.json"), "w", encoding="utf-8") as f:
    json.dump({"mode": CFG.PROMPT_MODE, "profiles": PROFILES, "prompts": PROMPTS}, f, indent=1, ensure_ascii=False)
_prof = pd.DataFrame(PROFILES).T
log("[PROMPTS] conventions measured on the TRAIN lines of each sub-class (% of lines):\n" +
    _prof[[c for c in ("n", "chars", "words", "caret", "colon", "amp", "thorn", "ff", "digit", "tilde", "comma",
                       "Said", "sd", "cap_words") if c in _prof]].round(1).to_string())
log("[PROMPTS] characters per prompt: " + ", ".join(f"{k}={len(v)}" for k, v in PROMPTS.items()))
''')

code(r'''
# the full prompt of the largest sub-class (all prompts are in subclass_prompts.json)
_show = max(SUB_KEYS, key=lambda k: PROFILES.get(k, {}).get("n", 0))
print(f"==================== PROMPT of sub-class {_show} ====================\n")
print(PROMPTS[_show])
''')

# ============================================================================ few-shot, conversation, collator, generation, judge
md(r'''
### Few-shot examples, the conversation, and batch-1 generation
**Few-shot examples.** For each line, the few-shot examples are the **K training lines of the same sub-class nearest in pixel features**: size, tone, contrast, pen and focus, usually the same page or hand. The nearest one comes last. During training the K are drawn from the nearest K + 4, so the context varies.

**Conversation:**
1. `system`: the sub-class prompt;
2. `user`: example image + "Transcribe this line." → `assistant`: its ground truth, repeated K times;
3. `user`: the line image + "Transcribe this line." → the model answers.

The loss covers only that last answer. It is located by the last `<|im_start|>assistant\n` header, which was checked against the processor.
''')
code(r'''
# =========================================================
# Cell 11 — few-shot retrieval, conversation, collator (batch 1), generation, forced scoring (judge)
# =========================================================
def _fs_vec(f):
    return np.array([math.log(max(f["height"], 1)), math.log(max(f["width"], 1)), f["paper"] / 50.0, f["tint"] / 20.0,
                     f["contrast"] / 30.0, f["stroke_rel"] * 20.0, f["sharp"] * 10.0, f["edge_ratio"] * 2.0,
                     f["band_frac"] * 2.0], float)

EX_POOL = {}
for k in SUB_KEYS:
    pool = [s for s in core_samples if s["sub"] == k and s["id"] in FEATS]
    M = np.stack([_fs_vec(FEATS[s["id"]]) for s in pool]) if pool else np.zeros((0, 9))
    mu, sd = (M.mean(0), M.std(0) + 1e-6) if len(M) else (np.zeros(9), np.ones(9))
    EX_POOL[k] = (pool, (M - mu) / sd, mu, sd)
log("[FEW-SHOT] example pools (Fold-0 TRAIN lines only): " + ", ".join(f"{k}={len(v[0])}" for k, v in EX_POOL.items()))

def exemplars_for(qid, sub, k, jitter=0, rng=None):
    """the k training lines of sub-class `sub` nearest to image `qid` in pixel features (never qid itself);
    nearest LAST (closest to the query in the conversation)."""
    pool, Z, mu, sd = EX_POOL.get(sub, ([], None, None, None))
    if k <= 0 or not pool or qid not in FEATS:
        return []
    d = ((Z - (_fs_vec(FEATS[qid]) - mu) / sd) ** 2).sum(1)
    order = [j for j in np.argsort(d, kind="stable") if pool[j]["id"] != qid]
    if jitter and rng is not None and len(order) > k:
        cand = order[:k + jitter]
        pick = sorted(rng.choice(len(cand), size=min(k, len(cand)), replace=False))
        order = [cand[j] for j in pick]
    return [pool[j] for j in order[:k]][::-1]

_IMG_CACHE = collections.OrderedDict()
def load_image_cached(path):
    if path in _IMG_CACHE:
        _IMG_CACHE.move_to_end(path)
        return _IMG_CACHE[path]
    img = load_image(path)
    _IMG_CACHE[path] = img
    if len(_IMG_CACHE) > 256:
        _IMG_CACHE.popitem(last=False)
    return img

def conversation(img, sub, exs):
    """-> (messages without the final answer, images in order of appearance)"""
    msgs, imgs = [], []
    if CFG.PROMPT_MODE == "exp002":                       # exact barbados-2 format: prompt + image in the user turn
        for ex in exs:
            e = load_image_cached(ex["image"])
            msgs += [{"role": "user", "content": [{"type": "text", "text": SHORT_PROMPT}, {"type": "image", "image": e}]},
                     {"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]}]
            imgs.append(e)
        msgs.append({"role": "user", "content": [{"type": "text", "text": SHORT_PROMPT}, {"type": "image", "image": img}]})
    else:
        msgs.append({"role": "system", "content": [{"type": "text", "text": PROMPTS[sub]}]})
        for ex in exs:
            e = load_image_cached(ex["image"])
            msgs += [{"role": "user", "content": [{"type": "image", "image": e}, {"type": "text", "text": CFG.USER_INSTRUCTION}]},
                     {"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]}]
            imgs.append(e)
        msgs.append({"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": CFG.USER_INSTRUCTION}]})
    imgs.append(img)
    return msgs, imgs

_HDR = {}
def answer_start(ids, tok):
    """index of the first answer token = right after the LAST '<|im_start|>assistant\n' header"""
    if "h" not in _HDR:
        _HDR["h"] = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    h = _HDR["h"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    for j in range(len(ids) - len(h), -1, -1):
        if ids[j:j + len(h)] == h:
            return j + len(h)
    raise ValueError("assistant header not found in the encoded conversation")

_AUG_VISITS = collections.Counter()
def _aug_seed(row_seed):
    if not CFG.AUG_RESAMPLE:
        return row_seed
    v = _AUG_VISITS[row_seed]; _AUG_VISITS[row_seed] += 1
    return (int(row_seed) * 1000003 + v * 7919 + int(torch.initial_seed()) % 1000000007) & 0x7fffffff

MASK_CHECKS = []
def collate_fewshot(examples, proc, train=True):
    """training / loss batches: the few-shot conversation + the answer; labels only on the answer."""
    texts, all_imgs = [], []
    for ex in examples:
        img = load_image(ex["image"])
        if train and CFG.USE_AUGMENT and ex.get("aug"):
            img = augment_image(img, _aug_seed(ex.get("seed", 0)))
        rng = np.random.RandomState((int(ex.get("seed", 0)) + _AUG_VISITS[("fs", ex["id"])]) & 0x7fffffff) if train else None
        if train:
            _AUG_VISITS[("fs", ex["id"])] += 1
        exs = exemplars_for(ex["id"], ex["sub"], CFG.FEWSHOT_K_TRAIN if train else CFG.FEWSHOT_K,
                            jitter=CFG.FEWSHOT_TRAIN_JITTER if train else 0, rng=rng)
        msgs, imgs = conversation(img, ex["sub"], exs)
        texts.append(proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]}],
                                              tokenize=False, add_generation_prompt=False))
        all_imgs += imgs
    tok = proc.tokenizer
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        batch = proc(text=texts, images=all_imgs, padding=True, return_tensors="pt")
    finally:
        tok.padding_side = old
    labels = batch["input_ids"].clone()
    for i in range(len(texts)):
        s = answer_start(batch["input_ids"][i], tok)
        labels[i, :s] = -100
        if len(MASK_CHECKS) < 3:                      # the answer tokens must decode back to the ground truth
            ans = tok.decode(batch["input_ids"][i, s:][batch["attention_mask"][i, s:].bool()], skip_special_tokens=True)
            MASK_CHECKS.append(clean_output(ans) == norm_text(examples[i]["text"]))
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch

def _gen_config(model, var):
    gc_ = copy.deepcopy(model.generation_config)
    gc_.do_sample = False
    for k_ in ("temperature", "top_p", "top_k"):
        try: setattr(gc_, k_, None)
        except Exception: pass
    nb = int(var.get("num_beams", 1)); nr = min(int(var.get("n_return", 1)), nb)
    gc_.num_beams, gc_.num_return_sequences = nb, nr
    gc_.max_new_tokens = CFG.MAX_NEW_TOKENS
    gc_.repetition_penalty = float(var.get("repetition_penalty", 1.0))
    gc_.no_repeat_ngram_size = int(var.get("no_repeat_ngram_size", 0))
    gc_.length_penalty = float(var.get("length_penalty", 1.0))
    gc_.early_stopping = nb > 1
    gc_.output_scores = True
    gc_.return_dict_in_generate = True
    return gc_, nb, nr

def loop_guard(text, max_run=None):
    max_run = max_run or CFG.LOOP_GUARD
    ws, out, run = norm_text(text).split(), [], 0
    for i, w in enumerate(ws):
        run = run + 1 if (i and w == ws[i - 1]) else 1
        if run <= max_run:
            out.append(w)
    return " ".join(out)

def generate_line(model, proc, path, sub, exs, var):
    """batch size 1: -> N-best [(text, total_logprob, n_tokens, truncated)]"""
    img = load_image(path)
    msgs, imgs = conversation(img, sub, exs)
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=imgs, return_tensors="pt")
    inputs = {k_: v.to(model.device) for k_, v in inputs.items()}
    gcfg, nb, nr = _gen_config(model, var)
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=gcfg)
    tok = proc.tokenizer
    gen = out.sequences[:, inputs["input_ids"].shape[1]:]
    stop_ids = {s_ for s_ in {tok.pad_token_id, tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
                if isinstance(s_, int) and s_ >= 0}
    is_stop = torch.zeros_like(gen, dtype=torch.bool)
    for s_ in stop_ids:
        is_stop |= gen == s_
    has_stop = is_stop.any(1)
    first = torch.where(has_stop, is_stop.float().argmax(1), torch.full_like(has_stop, gen.shape[1], dtype=torch.long))
    n_tok = (first + has_stop.long()).clamp(min=1)
    if nb > 1 and getattr(out, "sequences_scores", None) is not None:
        total = out.sequences_scores.float() * n_tok.float() ** float(gcfg.length_penalty)
    else:
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        valid = torch.arange(ts.shape[1], device=ts.device)[None, :] < n_tok[:, None].to(ts.device)
        total = torch.where(valid & torch.isfinite(ts), ts, torch.zeros_like(ts)).sum(1)
    texts = [loop_guard(clean_output(t)) for t in proc.batch_decode(gen, skip_special_tokens=True)]
    return [(texts[r], float(total[r]), int(n_tok[r]), bool(not has_stop[r])) for r in range(nr)]

def forced_logprobs(model, proc, path, sub, exs, texts):
    """JUDGE: exact log-probability the model gives to each text as the answer (same prompt + examples)."""
    img = load_image(path)
    msgs, imgs = conversation(img, sub, exs)
    tok = proc.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for t in texts:
        full = proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": t}]}],
                                        tokenize=False, add_generation_prompt=False)
        enc = proc(text=[full], images=imgs, return_tensors="pt")
        enc = {k_: v.to(model.device) for k_, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        ids = enc["input_ids"][0]
        s = answer_start(ids, tok)
        tgt = ids[s:]
        lp = torch.log_softmax(logits[0, s - 1:-1].float(), -1).gather(-1, tgt[:, None]).squeeze(-1)
        ends = (tgt == end_id).nonzero()
        stop = int(ends[0].item()) + 1 if len(ends) else len(tgt)
        out.append(float(lp[:stop].sum()))
        del logits
    return out

SPLITS = {"val": (VAL_IDS, VAL_PATHS, VAL_SUB, VAL_GT), "test": (TEST_IDS, TEST_PATHS, TEST_SUB, None),
          "unlisted": (UNL_IDS, UNL_PATHS, UNL_SUB, None)}
STORE = {}                 # stage -> split -> list of records {id, sub, path, gt, nbest, ex}

def mbr_of(nb):
    nb = [x for x in nb if x[0]]
    if not nb:
        return ""
    return mbr_select([t for t, *_ in nb], nbest_posteriors([s_ for _, s_, *_ in nb], CFG.MBR_TEMPERATURE))

def run_stage(model, proc, stage, split, idx=None):
    """decode (batch 1, beam 5, 5-best) every line of `split` (or the positions `idx`) with the current model."""
    ids, paths, subs, gts = SPLITS[split]
    idx = list(range(len(ids))) if idx is None else list(idx)
    recs, t0 = [], time.time()
    model.eval()
    for n, i in enumerate(idx):
        exs = exemplars_for(ids[i], subs[i], CFG.FEWSHOT_K)
        nb = [("", float("-inf"), 0, False)]
        if paths[i] and os.path.exists(paths[i]):
            try:
                nb = generate_line(model, proc, paths[i], subs[i], exs, CFG.DECODE)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"[{stage}] OOM on {ids[i]} — empty prediction")
        recs.append({"id": ids[i], "sub": subs[i], "path": paths[i], "gt": gts[i] if gts is not None else None,
                     "nbest": nb, "ex": [e["id"] for e in exs]})
        if (n + 1) % CFG.LOG_EVERY_LINES == 0 or n + 1 == len(idx):
            rate = (n + 1) / max(1e-9, time.time() - t0)
            msg = f"[{stage}] {split}: {n + 1}/{len(idx)} lines | {rate * 60:.0f} lines/min | ETA {(len(idx) - n - 1) / max(rate, 1e-9) / 60:.1f} min"
            if gts is not None:
                msg += f" | running official score (top-1) {official_metric([r['gt'] for r in recs], [r['nbest'][0][0] for r in recs])['score']:.4f}"
            log(msg)
    STORE.setdefault(stage, {})[split] = recs
    pd.DataFrame([{"ID": r["id"], "subclass": r["sub"], "GroundTruth": r["gt"], "pred": r["nbest"][0][0],
                   "mbr": mbr_of(r["nbest"]), "exemplars": " ".join(r["ex"]),
                   "nbest": json.dumps([[t, round(s_, 4)] for t, s_, *_ in r["nbest"]], ensure_ascii=False)} for r in recs]
                 ).to_csv(os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__{stage}__{split}.csv"), index=False)
    return recs

def stage_val_report(stage, recs, extra_hyps=None):
    """official score of top-1 and MBR on the val lines of `recs`, per sub-class; full val -> trial row."""
    gt = [r["gt"] for r in recs]
    ids = [r["id"] for r in recs]
    subs = [r["sub"] for r in recs]
    hy = {"top1": [r["nbest"][0][0] for r in recs], "mbr": [mbr_of(r["nbest"]) for r in recs]}
    hy.update(extra_hyps or {})
    full = len(recs) == len(VAL_IDS)
    guide_v = [GUIDE_VAL.get(i) for i in ids] if GUIDE_VAL else None
    for name, h in hy.items():
        m = official_metric(gt, h)
        log(f"[{stage}] VAL {name:6s}: official {m['score']:.5f} | word/line {m['word_edits_per_line']:.3f} | char/line "
            f"{m['char_edits_per_line']:.3f} | n={m['n']}" + ("" if full else " (sample)"))
        log(f"[{stage}]   per sub-class: " + json.dumps(score_by_group(gt, h, subs)))
        if guide_v and all(g is not None for g in guide_v):
            d = paired_delta(gt, guide_v, h)
            log(f"[{stage}]   vs GUIDE ({official_metric(gt, guide_v)['score']:.5f}) on the same lines: {d['delta']:+.5f} "
                f"CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}] P(better)={d['p_better']:.3f}")
        if full:
            report_trial(f"{TRIAL_ID}__{stage}__{name}", TRIAL_DESCRIPTION, CFG.MODEL, f"{stage}/{name}", h, 0.0,
                         len(core_samples), baseline_hyps=guide_v if (guide_v and all(g is not None for g in guide_v)) else None,
                         extra={"stage": stage, "prompt_mode": CFG.PROMPT_MODE, "fewshot_k": CFG.FEWSHOT_K,
                                "subclass_mode": CFG.SUBCLASS_MODE})
    return hy

def show_conversation(proc, i=0, split="val"):
    """print the exact text the model sees for one line (image tokens collapsed) + its length in tokens"""
    ids, paths, subs, _ = SPLITS[split]
    if not ids:
        return
    exs = exemplars_for(ids[i], subs[i], CFG.FEWSHOT_K)
    msgs, imgs = conversation(load_image(paths[i]), subs[i], exs)
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    n = proc(text=[txt], images=imgs, return_tensors="pt")["input_ids"].shape[1]
    shown = re.sub(r"(<\|image_pad\|>)+", "<IMAGE>", txt)
    log(f"[CONVERSATION] {split} line {ids[i]} (sub-class {subs[i]}, {len(exs)} solved examples: "
        f"{[e['id'] for e in exs]}) = {n} tokens")
    print(shown[:1500] + ("\n  [...]\n" + shown[-600:] if len(shown) > 2100 else shown[1500:]))
''')

# ============================================================================ guide + judge
md(r'''
### Pseudo-labels: guide + judge
For every test (and unlisted) line:
1. **Model candidates.** The fine-tuned model's 5-best, each with its log-probability.
2. **Guide.** The transcription in `GUIDE_CSV` (your best submission) is added to the candidates. If it's missing from the 5-best, the model **scores it** (the judge), so it competes on the model's own probability scale.
3. **Pick.** MBR under the official cost. The evidence weights are `(1 − GUIDE_WEIGHT) ×` the model posterior `+ GUIDE_WEIGHT ×` the guide.
4. **Confidence.** Risk is the expected official cost. With val predictions available, confidence is calibrated on val as the expected line score per risk decile.
5. **Output row.** `agree` means the model's top-1 equals the guide. The comment says what happened: *agreement*, *judge chose the guide*, *judge kept the model*, or *compromise*. `keep_for_training` follows `PSEUDO_KEEP`.

`GUIDE_VAL_CSV`, the same guide system's Fold-0 val predictions (e.g. `trils/EXP_002__ENS_7b8b_mbrq__D2_b5nb_ens__val.csv`), runs the identical procedure on the 819 val lines. It shows whether model + guide beats the guide alone before you trust it on test.
''')
code(r'''
# =========================================================
# Cell 12 — guide files, judge, pseudo-label writer
# =========================================================
def load_guide(path):
    """-> ({ID: text}, {ID: keep_for_training flag or None})"""
    if not path or not os.path.exists(path):
        if path:
            log(f"[GUIDE] file not found: {path}")
        return {}, {}
    d = pd.read_csv(path)
    col = next((c for c in ("Target", "pseudo_label", "pred", "mbr") if c in d.columns), None)
    assert col is not None, f"{path}: needs a Target / pseudo_label / pred column"
    txt = {str(i).strip(): norm_text(t) for i, t in zip(d.ID, d[col].fillna(""))}
    keep = {}
    if "keep_for_training" in d.columns:
        keep = {str(i).strip(): str(v).lower() in ("true", "1") for i, v in zip(d.ID, d.keep_for_training)}
    log(f"[GUIDE] {os.path.basename(path)}: {len(txt)} lines from column '{col}'"
        + (f", keep_for_training for {sum(keep.values())}" if keep else ""))
    return txt, keep

CORE_BY_ID = {s["id"]: s for s in core_samples}
GUIDE, GUIDE_KEEP = load_guide(CFG.GUIDE_CSV)
GUIDE_VAL, _ = load_guide(CFG.GUIDE_VAL_CSV)
if GUIDE:
    log(f"[GUIDE] covers {sum(i in GUIDE for i in TEST_IDS)}/{len(TEST_IDS)} test and "
        f"{sum(i in GUIDE for i in UNL_IDS)}/{len(UNL_IDS)} unlisted lines")
if GUIDE_VAL and CFG.RUN_MODE == "fold0":
    _gv = [GUIDE_VAL.get(i) for i in VAL_IDS]
    if all(g is not None for g in _gv):
        log(f"[GUIDE] guide on the 819 val lines: official {official_metric(VAL_GT, _gv)['score']:.5f} | per sub-class "
            + json.dumps(score_by_group(VAL_GT, _gv, VAL_SUB)))

def judge_evidence(model, proc, recs, guide):
    """for every record: candidate texts (model 5-best + guide) and the model posterior over them. A guide text that
    is not in the 5-best is scored by the model (forced log-probability) = the JUDGE."""
    ents, n_judged, t0 = [], 0, time.time()
    for r in recs:
        cands = [(t, s_) for t, s_, *_ in r["nbest"] if t]
        g = guide.get(r["id"]) if guide else None
        g_lp = None
        if g:
            hit = [s_ for t, s_ in cands if t == g]
            if hit:
                g_lp = hit[0]
            elif CFG.JUDGE_GUIDE and model is not None and r["path"] and os.path.exists(r["path"]):
                exs = [CORE_BY_ID[e] for e in r["ex"] if e in CORE_BY_ID]      # the same examples as the decode
                g_lp = forced_logprobs(model, proc, r["path"], r["sub"], exs, [g])[0]
                n_judged += 1
                cands.append((g, g_lp))
        texts = [t for t, _ in cands]
        post = nbest_posteriors([s_ for _, s_ in cands], CFG.MBR_TEMPERATURE) if cands else np.array([])
        p_guide = float(sum(p for t, p in zip(texts, post) if t == g)) if g else float("nan")
        if g and g not in texts:
            texts.append(g); post = np.append(post, 0.0)
        ents.append({"r": r, "texts": texts, "post": np.asarray(post, float), "g": g, "g_lp": g_lp,
                     "p_guide": p_guide, "top1": r["nbest"][0][0] if r["nbest"] else ""})
    if n_judged:
        log(f"[JUDGE] the model scored {n_judged} guide lines that were not in its 5-best "
            f"({(time.time() - t0) / 60:.1f} min)")
    return ents

def judge_table(ents, lam):
    """MBR pick with evidence (1 - lam) * model posterior + lam * guide, for every entry -> DataFrame"""
    out = []
    for e in ents:
        texts, post, g, top1, r = e["texts"], e["post"], e["g"], e["top1"], e["r"]
        l_ = lam if g else 0.0
        if texts:
            w = (1 - l_) * post + l_ * np.array([1.0 if t == g else 0.0 for t in texts])
            pick, risk = mbr_select(texts, list(w), return_risk=True)
        else:
            pick, risk = "", 1.0
        agree = bool(g) and norm_text(top1) == g
        if not g:
            comment = "no guide: MBR over the model's 5-best"
        elif agree:
            comment = "model top-1 = guide (agreement)"
        elif pick == g:
            comment = f"judge chose the guide over the model top-1 (model gives the guide p={e['p_guide']:.2f})"
        elif pick == norm_text(top1):
            comment = f"judge kept the model top-1 over the guide (model gives the guide p={e['p_guide']:.2f})"
        else:
            comment = "compromise: MBR pick differs from both the model top-1 and the guide"
        out.append({"ID": r["id"], "subclass": r["sub"], "pseudo_label": pick, "risk": float(risk), "agree": agree,
                    "model_top1": top1, "guide": g or "", "p_guide": e["p_guide"], "guide_logprob": e["g_lp"],
                    "edits_top1_vs_guide": line_edits(top1, g)[1] if g else np.nan, "n_cands": len(texts),
                    "comment": comment, "GroundTruth": r["gt"]})
    return pd.DataFrame(out)

GUIDE_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
CHOSEN_GUIDE_WEIGHT = {}
def choose_guide_weight(stage, val_ents):
    """GUIDE_WEIGHT='auto': pick the weight on the val lines; report the honest 5-fold CV score of that choice."""
    if CFG.GUIDE_WEIGHT != "auto":
        return float(CFG.GUIDE_WEIGHT)
    if not val_ents or not any(e["g"] for e in val_ents):
        log("[JUDGE] GUIDE_WEIGHT='auto' needs GUIDE_VAL_CSV on val — using 0.5")
        return 0.5
    gt = [e["r"]["gt"] for e in val_ents]
    C = {lam: np.array([line_cost(a, b) for a, b in zip(gt, judge_table(val_ents, lam).pseudo_label)]) for lam in GUIDE_GRID}
    cv = []
    for sd in (0, 1, 2):
        fold = np.random.RandomState(sd).permutation(len(gt)) % 5
        held = np.empty(len(gt))
        for f in range(5):
            tr_, te_ = fold != f, fold == f
            best = min(GUIDE_GRID, key=lambda l_: C[l_][tr_].mean())
            held[te_] = C[best][te_]
        cv.append(1 - held.mean())
    lam = min(GUIDE_GRID, key=lambda l_: C[l_].mean())
    CHOSEN_GUIDE_WEIGHT[stage] = lam
    log(f"[JUDGE] {stage}: val score by guide weight " + ", ".join(f"{l_}:{1 - C[l_].mean():.5f}" for l_ in GUIDE_GRID)
        + f" | chosen {lam} | honest 5-fold CV score of this choice {np.mean(cv):.5f} +- {np.std(cv):.5f} "
        f"(guide alone = weight 1.0, model + judge alone = weight 0.0). For RUN_MODE='full' set GUIDE_WEIGHT={lam}")
    return lam

CALIB = None
def calibrate(val_df):
    """risk decile -> mean official line score on val (honest confidence for test)"""
    global CALIB
    if val_df is None or len(val_df) < 50:
        return
    v = val_df.copy()
    v["line_score"] = [1 - line_cost(g, p) for g, p in zip(v.GroundTruth, v.pseudo_label)]
    q = np.unique(np.quantile(v.risk, np.linspace(0, 1, 11)))
    v["bin"] = np.clip(np.searchsorted(q, v.risk, side="right") - 1, 0, len(q) - 2)
    CALIB = (q, v.groupby("bin").line_score.mean().to_dict(), v.groupby("bin").line_score.apply(lambda s: (s >= 0.999).mean()).to_dict())
    log("[CALIBRATION] risk decile -> mean val line score: " + ", ".join(f"{b}:{s:.3f}" for b, s in CALIB[1].items()))
    if "agree" in v and v.agree.any():
        log(f"[CALIBRATION] val lines where model = guide: {v.agree.mean():.1%} of lines, line score "
            f"{v[v.agree].line_score.mean():.4f}, exact {((v[v.agree].line_score >= 0.999).mean()):.1%} | "
            f"disagreeing lines: {v[~v.agree].line_score.mean():.4f}")

def confidence_of(risk):
    if CALIB is None:
        return max(0.0, 1.0 - float(risk)), np.nan
    q, m, pe = CALIB
    b = int(np.clip(np.searchsorted(q, risk, side="right") - 1, 0, len(q) - 2))
    return float(m.get(b, np.nan)), float(pe.get(b, np.nan))

PSEUDO = {}                # stage -> DataFrame of the test + unlisted pseudo-labels
def pseudo_label_stage(model, proc, stage):
    """judge val (if a val guide exists) for calibration + scores, then test + unlisted -> pseudo_labels_<stage>.csv"""
    lam = 0.5 if CFG.GUIDE_WEIGHT == "auto" else float(CFG.GUIDE_WEIGHT)
    if "val" in STORE.get(stage, {}) and GUIDE_VAL:
        ve = judge_evidence(model, proc, STORE[stage]["val"], GUIDE_VAL)
        lam = choose_guide_weight(stage, ve)
        vj = judge_table(ve, lam)
        calibrate(vj)
        if len(vj) == len(VAL_IDS):
            report_trial(f"{TRIAL_ID}__{stage}__judged", TRIAL_DESCRIPTION, CFG.MODEL, f"{stage}/judged(model+guide)",
                         vj.pseudo_label.tolist(), 0.0, len(core_samples), baseline_hyps=[GUIDE_VAL[i] for i in VAL_IDS],
                         extra={"stage": stage, "guide_weight": lam})
        log(f"[{stage}] VAL judged (guide weight {lam}) per sub-class: " + json.dumps(score_by_group(
            vj.GroundTruth.tolist(), vj.pseudo_label.tolist(), vj.subclass.tolist())))
        log(f"[{stage}] VAL comments: " + json.dumps(vj.comment.str.split(" \\(").str[0].value_counts().to_dict()))
        STORE[stage]["val_judged"] = vj
    elif "val" in STORE.get(stage, {}):
        vj = judge_table(judge_evidence(model, proc, STORE[stage]["val"], {}), 0.0)
        calibrate(vj)
    frames = []
    for sp in ("test", "unlisted"):
        if sp in STORE.get(stage, {}):
            d = judge_table(judge_evidence(model, proc, STORE[stage][sp], GUIDE), lam)
            d.insert(1, "split", sp)
            frames.append(d)
    if not frames:
        return None
    P = pd.concat(frames, ignore_index=True)
    conf = [confidence_of(r) for r in P.risk]
    P["confidence"] = [c for c, _ in conf]
    P["p_exact"] = [p for _, p in conf]
    q = P.groupby("subclass").risk.transform(lambda s: s.quantile(CFG.PSEUDO_KEEP_FRAC))
    low = P.risk <= q
    ok = P.pseudo_label.str.split().str.len().between(2, 40)
    if CFG.PSEUDO_KEEP == "agree" and GUIDE:
        keep = P.agree
    elif CFG.PSEUDO_KEEP == "agree_or_lowrisk" and GUIDE:
        keep = P.agree | low
    else:
        keep = low
    P["keep_for_training"] = keep & ok
    P.loc[~ok, "comment"] = P.loc[~ok, "comment"] + " | rejected: implausible length"
    P = P[["ID", "split", "subclass", "pseudo_label", "confidence", "p_exact", "risk", "agree", "keep_for_training",
           "model_top1", "guide", "p_guide", "edits_top1_vs_guide", "n_cands", "comment"]]
    path = os.path.join(CFG.WORK_DIR, f"pseudo_labels_{stage}.csv")
    P.to_csv(path, index=False)
    PSEUDO[stage] = P
    log(f"[PSEUDO] {stage}: guide weight {lam if GUIDE else '-'} | {len(P)} lines -> {os.path.basename(path)} | kept for "
        f"training {int(P.keep_for_training.sum())} "
        f"({P.keep_for_training.mean():.0%}) | agreement with guide {P.agree.mean():.1%} | mean confidence "
        f"{np.nanmean(P.confidence):.3f}")
    log(f"[PSEUDO] {stage}: kept per sub-class " + json.dumps(P.groupby("subclass").keep_for_training.sum().astype(int).to_dict()))
    return P
''')

# ============================================================================ load model
md(r'''
### Load the model
The Qwen-VL model is loaded from `/kaggle/input` without internet, with the pixel limits enforced. If `DO_FINETUNE=True`, a LoRA adapter is attached right away, as in the Unsloth notebook. A fresh LoRA starts at zero, so the "before" outputs are the base model's.
''')
code(r'''
# =========================================================
# Cell 13 — load processor + model (+ LoRA)
# =========================================================
dtype = torch.bfloat16 if (CFG.USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP = {k: CFG.MODEL_OVERRIDES.get(CFG.MODEL, {}).get(k, getattr(CFG, k)) for k in ("LEARNING_RATE", "R", "LORA_ALPHA")}
proc = load_vlm_processor(CFG.MODEL_PATH, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
base = load_vlm_model(CFG.MODEL_PATH, dtype)
base.to(DEVICE)
base.config.use_cache = False
if CFG.DO_FINETUNE and any(t in CFG.MODEL_PATH.lower() for t in CFG.GRAD_CKPT_MODELS):
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.enable_input_require_grads()
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if CFG.LORA_VISION:
    TARGETS += [n for n, m in base.named_modules()
                if isinstance(m, _nn.Linear) and any(t in n.lower() for t in ("visual", "vision", "merger"))]
if CFG.INIT_ADAPTER:
    model = PeftModel.from_pretrained(base, CFG.INIT_ADAPTER, is_trainable=CFG.DO_FINETUNE)
    log(f"[MODEL] {CFG.MODEL} + adapter {CFG.INIT_ADAPTER} (trainable={CFG.DO_FINETUNE})")
elif CFG.DO_FINETUNE:
    model = get_peft_model(base, LoraConfig(r=HP["R"], lora_alpha=HP["LORA_ALPHA"], target_modules=TARGETS,
                                            lora_dropout=CFG.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"))
    log(f"[MODEL] {CFG.MODEL} + fresh LoRA r={HP['R']} ({len(TARGETS)} target module names, vision LoRA "
        f"{'on' if CFG.LORA_VISION else 'off'}) — B matrices start at zero, so 'before' = base model")
    model.print_trainable_parameters()
else:
    model = base
    log(f"[MODEL] {CFG.MODEL} base model, inference only")
model.eval()
model.config.use_cache = True
log(f"[MODEL] prompt lengths in tokens: " + ", ".join(
    f"{k}={len(proc.tokenizer(v)['input_ids'])}" for k, v in PROMPTS.items()))
show_conversation(proc, 0, "val" if VAL_IDS else "test")
''')
code(r'''
# @title Show current memory stats
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
else:
    start_gpu_memory, max_memory = 0.0, 0.0
    print("no GPU")
''')

# ============================================================================ before
md(r'''
Let's first see what the model outputs **before we do any fine-tuning**!
With `INFER_BEFORE="sample"`, `BEFORE_SAMPLE_N` val lines spread over all sub-classes are scored. With `"full"`, all val + test + unlisted are decoded, and pseudo-labels are written from this stage. That's the inference-only path, `DO_FINETUNE=False`.
''')
code(r'''
# =========================================================
# Cell 14 — INFERENCE BEFORE FINE-TUNING
# =========================================================
def stratified_positions(subs, n, seed=0):
    rng = np.random.RandomState(seed)
    by = collections.defaultdict(list)
    for i, s in enumerate(subs):
        by[s].append(i)
    for v in by.values():
        rng.shuffle(v)
    out, keys = [], sorted(by)
    while len(out) < min(n, len(subs)):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return sorted(out)

if CFG.INFER_BEFORE != "off":
    log(f"INFERENCE BEFORE FINE-TUNING ({CFG.INFER_BEFORE})", banner=True)
    if CFG.INFER_BEFORE == "sample":
        sp = "val" if VAL_IDS else "test"
        pos = stratified_positions(SPLITS[sp][2], CFG.BEFORE_SAMPLE_N, CFG.SEED)
        recs = run_stage(model, proc, "before", sp, pos)
        for r in recs[:3]:
            log(f"[before] {r['id']} [{r['sub']}]\n      GT  : {r['gt'] if r['gt'] is not None else '(test line: no ground truth)'}"
                f"\n      PRED: {r['nbest'][0][0]}")
        if sp == "val":
            stage_val_report("before", recs)
    else:
        if CFG.RUN_MODE == "fold0":
            stage_val_report("before", run_stage(model, proc, "before", "val"))
        run_stage(model, proc, "before", "test")
        if CFG.PREDICT_UNLISTED and UNL_IDS:
            run_stage(model, proc, "before", "unlisted")
        pseudo_label_stage(model, proc, "before")
''')

# ============================================================================ train
md(r'''
<a name="Train"></a>
### Train the model
LoRA fine-tuning with **batch size 1** and gradient accumulation 4, on the few-shot conversations of the training lines. With `FIRST_PSEUDO_FROM_GUIDE`, the guide's test lines are also used.
- Every `EVAL_EVERY` optimizer steps, and at the last step, a `[VALID]` line prints the training loss, the held-out loss, and the official score of greedy transcriptions of held-out training lines (all sub-classes).
- The best step (`CKPT_METRIC`) is restored at the end.
''')
code(r'''
# =========================================================
# Cell 15 — fine-tuning utilities: training rows, step validation, one fine-tune call
# =========================================================
GREEDY = dict(num_beams=1, repetition_penalty=1.2, no_repeat_ngram_size=0, n_return=1)
EVAL_ITEMS = [inner_samples[i] for i in stratified_positions([s["sub"] for s in inner_samples],
                                                              max(CFG.EVAL_GEN_N, CFG.EVAL_LOSS_N), CFG.SEED + 3)]
EVAL_CURVES = []

def training_rows(samples, salt=0):
    rows = []
    for j, s in enumerate(samples):
        base_seed = int((CFG.SEED + 1) * 2000003 + CFG.FOLD * 1000003 + j + salt * 7777777) & 0x7fffffff
        rows.append({"id": s["id"], "image": s["image"], "text": s["text"], "sub": s["sub"], "aug": False, "seed": base_seed})
        if CFG.USE_AUGMENT:
            for c in range(int(CFG.AUG_COPIES)):
                rows.append({"id": s["id"], "image": s["image"], "text": s["text"], "sub": s["sub"], "aug": True,
                             "seed": (base_seed + (c + 1) * 9973) & 0x7fffffff})
    return rows

def teacher_forced_loss(model, proc, items):
    tot, n = 0.0, 0
    with torch.no_grad():
        for it in items:
            b = collate_fewshot([{**it, "aug": False, "seed": 0}], proc, train=False)
            b = {k_: v.to(model.device) for k_, v in b.items()}
            nt = int((b["labels"] != -100).sum())
            tot += float(model(**b).loss) * nt
            n += nt
    return tot / max(1, n)

class StepValidation(TrainerCallback):
    def __init__(self, model, proc, tag):
        self.model, self.proc, self.tag = model, proc, tag
        self.gen, self.loss_items = EVAL_ITEMS[:CFG.EVAL_GEN_N], EVAL_ITEMS[:CFG.EVAL_LOSS_N]
        self.best, self.best_step, self.best_state, self.last_step, self.last = None, None, None, -1, None
        self.train_loss = float("nan")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.train_loss = float(logs["loss"])

    def validate(self, state):
        if state.global_step == self.last_step or not self.gen:
            return
        self.last_step = state.global_step
        m, t0 = self.model, time.time()
        was = m.training
        m.eval()
        keep = CFG.MAX_NEW_TOKENS
        CFG.MAX_NEW_TOKENS = CFG.EVAL_MAX_NEW_TOKENS
        try:
            loss = teacher_forced_loss(m, self.proc, self.loss_items)
            hyps = []
            for it in self.gen:
                exs = exemplars_for(it["id"], it["sub"], CFG.FEWSHOT_K)
                hyps.append(generate_line(m, self.proc, it["image"], it["sub"], exs, GREEDY)[0][0])
        finally:
            CFG.MAX_NEW_TOKENS = keep
            if was:
                m.train()
        met = official_metric([it["text"] for it in self.gen], hyps)
        crit = {"inner_official": met["score"], "inner_loss": -loss}.get(CFG.CKPT_METRIC)
        flag = ""
        if crit is not None and (self.best is None or crit > self.best + 1e-9):
            self.best, self.best_step = crit, state.global_step
            self.best_state = {k_: v.detach().to("cpu", copy=True) for k_, v in get_peft_model_state_dict(m).items()}
            flag = "  <- best so far"
        row = {"tag": self.tag, "step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch,
               "train_loss": self.train_loss, "inner_loss": loss, "inner_score": met["score"],
               "inner_we": met["word_edits_per_line"], "inner_ce": met["char_edits_per_line"], "n": len(self.gen),
               "seconds": time.time() - t0}
        EVAL_CURVES.append(row)
        self.last = row
        log(f"[VALID {self.tag}] step {state.global_step}/{state.max_steps} ep {state.epoch or 0:.2f} | train loss "
            f"{self.train_loss:.4f} | inner loss {loss:.4f} | inner official {met['score']:.4f} (word/line "
            f"{met['word_edits_per_line']:.3f}, char/line {met['char_edits_per_line']:.3f}, n={len(self.gen)}) | "
            f"{row['seconds']:.0f}s{flag}")

    def on_step_end(self, args, state, control, **kw):
        if CFG.EVAL_EVERY and state.global_step % int(CFG.EVAL_EVERY) == 0:
            self.validate(state)

    def on_train_end(self, args, state, control, **kw):
        self.validate(state)
        if CFG.CKPT_METRIC != "last" and self.best_state is not None and self.best_step != state.global_step:
            set_peft_model_state_dict(self.model, self.best_state)
            log(f"[CKPT {self.tag}] RESTORED step {self.best_step} ({CFG.CKPT_METRIC} {self.best:.4f}) instead of the last "
                f"step {state.global_step} (inner official {self.last['inner_score']:.4f})")
        else:
            log(f"[CKPT {self.tag}] keeping the last step {state.global_step}")
        self.best_state = None

def finetune(model, proc, samples, lr, tag):
    rows = training_rows(samples, salt=len(EVAL_CURVES) + 1)
    steps = math.ceil(len(rows) / (CFG.PER_DEVICE_TRAIN_BATCH_SIZE * CFG.GRADIENT_ACCUMULATION_STEPS)) * CFG.NUM_TRAIN_EPOCHS
    steps = CFG.MAX_STEPS if CFG.MAX_STEPS and CFG.MAX_STEPS > 0 else steps
    log(f"[TRAIN {tag}] {len(samples)} lines -> {len(rows)} rows (by sub-class "
        f"{dict(sorted(collections.Counter(s['sub'] for s in samples).items()))}) | lr {lr} | batch "
        f"{CFG.PER_DEVICE_TRAIN_BATCH_SIZE} x accumulation {CFG.GRADIENT_ACCUMULATION_STEPS} | ~{steps} steps | "
        f"validation every {CFG.EVAL_EVERY} steps on {len(EVAL_ITEMS[:CFG.EVAL_GEN_N])} inner lines", banner=True)
    model.train()
    model.config.use_cache = False
    cb = StepValidation(model, proc, tag)
    args = TrainingArguments(
        output_dir=os.path.join(CFG.OUTPUT_DIR, tag), per_device_train_batch_size=CFG.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATION_STEPS, learning_rate=lr,
        num_train_epochs=CFG.NUM_TRAIN_EPOCHS, max_steps=int(CFG.MAX_STEPS or -1), lr_scheduler_type=CFG.LR_SCHEDULER,
        warmup_ratio=CFG.WARMUP_RATIO, bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16), logging_steps=10,
        eval_strategy="no", save_strategy="no", remove_unused_columns=False,
        dataloader_num_workers=CFG.DATALOADER_NUM_WORKERS, dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none", seed=CFG.SEED)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                      data_collator=lambda x, _p=proc: collate_fewshot(x, _p, train=True), callbacks=[cb])
    t0 = time.time()
    stats = trainer.train()
    log(f"[TRAIN {tag}] done in {(time.time() - t0) / 60:.1f} min | answer-mask checks {sum(MASK_CHECKS)}/{len(MASK_CHECKS)} OK")
    if MASK_CHECKS and not all(MASK_CHECKS):
        log("[TRAIN] WARNING: an answer mask did not decode back to its ground truth — inspect collate_fewshot")
    try:
        pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(CFG.WORK_DIR, f"log_history__{tag}.csv"), index=False)
    except Exception:
        pass
    pd.DataFrame(EVAL_CURVES).to_csv(os.path.join(CFG.WORK_DIR, "eval_curves.csv"), index=False)
    ad = os.path.join(CFG.FINAL_DIR, model_short_name(CFG.MODEL_PATH), f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full", tag)
    os.makedirs(ad, exist_ok=True)
    model.save_pretrained(ad)
    proc.save_pretrained(ad)
    log(f"[TRAIN {tag}] adapter saved -> {ad}")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, tag), ignore_errors=True)
    model.eval()
    model.config.use_cache = True
    return stats
''')
code(r'''
# =========================================================
# Cell 16 — FINE-TUNE
# =========================================================
trainer_stats = None
if CFG.DO_FINETUNE:
    train_lines = list(core_samples)
    if CFG.FIRST_PSEUDO_FROM_GUIDE and GUIDE:
        extra = []
        for sp, ids_, paths_, subs_ in (("test", TEST_IDS, TEST_PATHS, TEST_SUB), ("unlisted", UNL_IDS, UNL_PATHS, UNL_SUB)):
            for i, p, s in zip(ids_, paths_, subs_):
                if i in GUIDE and (not GUIDE_KEEP or GUIDE_KEEP.get(i, False)) and os.path.exists(p) \
                        and 2 <= len(GUIDE[i].split()) <= 40:
                    extra.append({"id": i, "image": p, "text": GUIDE[i], "sub": s})
        assert not ({s["id"] for s in extra} & set(VAL_IDS)), "guide lines must not cover validation lines"
        train_lines += extra
        log(f"[TRAIN] + {len(extra)} guide pseudo-labelled lines (test/unlisted) in the first fine-tune")
    trainer_stats = finetune(model, proc, train_lines, HP["LEARNING_RATE"], "ft1")
else:
    log("DO_FINETUNE=False — inference only (the 'before' stage is the final model)")
''')
code(r'''
# @title Show final memory and time stats
if torch.cuda.is_available() and trainer_stats is not None:
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {round(used_memory / max_memory * 100, 3)} %.")
    print(f"Peak reserved memory for training % of max memory = {round(used_memory_for_lora / max_memory * 100, 3)} %.")
''')

# ============================================================================ after
md(r'''
<a name="Inference"></a>
### Inference after fine-tuning
All val lines are scored (top-1 and MBR, per sub-class, and against the guide), then test + unlisted are decoded. Pseudo-labels are written with the guide + judge.
''')
code(r'''
# =========================================================
# Cell 17 — INFERENCE AFTER FINE-TUNING + pseudo-labels
# =========================================================
LAST_STAGE = "before"
if CFG.DO_FINETUNE:
    log("INFERENCE AFTER FINE-TUNING", banner=True)
    if CFG.RUN_MODE == "fold0":
        recs = run_stage(model, proc, "after", "val")
        stage_val_report("after", recs)
        if "before" in STORE and "val" in STORE["before"]:
            bpos = {r["id"]: r for r in STORE["before"]["val"]}
            same = [r for r in recs if r["id"] in bpos]
            d = paired_delta([r["gt"] for r in same], [mbr_of(bpos[r["id"]]["nbest"]) for r in same],
                             [mbr_of(r["nbest"]) for r in same])
            log(f"[after] fine-tuning effect on the {len(same)} lines decoded before: {d['delta']:+.5f} "
                f"CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}]")
    run_stage(model, proc, "after", "test")
    if CFG.PREDICT_UNLISTED and UNL_IDS:
        run_stage(model, proc, "after", "unlisted")
    pseudo_label_stage(model, proc, "after")
    LAST_STAGE = "after"
''')

# ============================================================================ self-training
md(r'''
### Self-training with the pseudo-labels (optional)
The kept pseudo-labels of the last stage (`keep_for_training`, per sub-class) are added to the training lines, and the model is fine-tuned again from the current adapter at `SELF_TRAIN_LR`, still with validation every 100 steps. Then it is decoded and judged again. Validation lines are never pseudo-labelled.
''')
code(r'''
# =========================================================
# Cell 18 — SELF-TRAINING ROUNDS
# =========================================================
for rnd in range(1, int(CFG.SELF_TRAIN_ROUNDS) + 1):
    if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
        log(f"[TIME] budget {CFG.TIME_BUDGET_H} h reached — self-training round {rnd} skipped")
        break
    P = PSEUDO.get(LAST_STAGE)
    if P is None or not P.keep_for_training.any():
        log(f"[SELF-TRAIN] no pseudo-labels from stage '{LAST_STAGE}' — stop")
        break
    paths = dict(zip(TEST_IDS, TEST_PATHS)); paths.update(dict(zip(UNL_IDS, UNL_PATHS)))
    extra = [{"id": r.ID, "image": paths[r.ID], "text": r.pseudo_label, "sub": r.subclass}
             for r in P[P.keep_for_training].itertuples() if r.ID in paths and os.path.exists(paths[r.ID])]
    assert not ({s["id"] for s in extra} & set(VAL_IDS)), "pseudo-labels must not cover validation lines"
    stage = f"self{rnd}"
    log(f"SELF-TRAINING ROUND {rnd}: {len(core_samples)} train + {len(extra)} pseudo-labelled lines", banner=True)
    finetune(model, proc, list(core_samples) + extra, CFG.SELF_TRAIN_LR, stage)
    if CFG.RUN_MODE == "fold0":
        recs = run_stage(model, proc, stage, "val")
        stage_val_report(stage, recs)
        prev = {r["id"]: r for r in STORE[LAST_STAGE].get("val", [])}
        if prev:
            d = paired_delta([r["gt"] for r in recs], [mbr_of(prev[r["id"]]["nbest"]) for r in recs],
                             [mbr_of(r["nbest"]) for r in recs])
            log(f"[{stage}] vs '{LAST_STAGE}' on val (MBR): {d['delta']:+.5f} CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}] "
                f"P(better)={d['p_better']:.3f}")
    run_stage(model, proc, stage, "test")
    if rnd < int(CFG.SELF_TRAIN_ROUNDS) and CFG.PREDICT_UNLISTED and UNL_IDS:
        run_stage(model, proc, stage, "unlisted")
    pseudo_label_stage(model, proc, stage)
    LAST_STAGE = stage
''')

# ============================================================================ submission
md(r'''
### Submission and summary
The val table ranks every stage: model top-1, model MBR, judged model + guide, and the guide alone. In `fold0` mode, the best configuration that has test predictions writes `submission.csv`. In `full` mode, the judged picks of the last stage are used, or the model's MBR when there is no guide. Every option is also saved as its own file.
''')
code(r'''
# =========================================================
# Cell 19 — SUBMISSION + summary
# =========================================================
OPTIONS = {}                                    # name -> (val hyps or None, test hyps or None)
for stage, parts in STORE.items():
    val = parts.get("val") if isinstance(parts.get("val"), list) else None
    full_val = val is not None and len(val) == len(VAL_IDS)
    test = parts.get("test")
    t_by = {r["id"]: r for r in test} if (test and len(test) == len(TEST_IDS)) else None   # full test decodes only
    OPTIONS[f"{stage}/model_mbr"] = ([mbr_of(r["nbest"]) for r in val] if full_val else None,
                                     [mbr_of(t_by[i]["nbest"]) if i in t_by else "" for i in TEST_IDS] if t_by else None)
    OPTIONS[f"{stage}/model_top1"] = ([r["nbest"][0][0] for r in val] if full_val else None,
                                      [t_by[i]["nbest"][0][0] if i in t_by else "" for i in TEST_IDS] if t_by else None)
    if stage in PSEUDO and GUIDE:
        pj = PSEUDO[stage].set_index("ID")
        vj = parts.get("val_judged")
        OPTIONS[f"{stage}/judged"] = (vj.pseudo_label.tolist() if vj is not None and len(vj) == len(VAL_IDS) else None,
                                      [pj.at[i, "pseudo_label"] if i in pj.index else GUIDE.get(i, "") for i in TEST_IDS])
if GUIDE:
    OPTIONS["guide"] = ([GUIDE_VAL.get(i, "") for i in VAL_IDS] if (GUIDE_VAL and all(i in GUIDE_VAL for i in VAL_IDS)) else None,
                        [GUIDE.get(i, "") for i in TEST_IDS])
rows = []
for name, (v, t) in OPTIONS.items():
    r = {"option": name, "has_test": t is not None}
    if v is not None and CFG.RUN_MODE == "fold0":
        m = official_metric(VAL_GT, v)
        r.update({"val_score": m["score"], "word/line": m["word_edits_per_line"], "char/line": m["char_edits_per_line"]})
        for k in SUB_KEYS:
            ix = [i for i, s in enumerate(VAL_SUB) if s == k]
            r[k] = official_metric([VAL_GT[i] for i in ix], [v[i] for i in ix])["score"] if ix else np.nan
    rows.append(r)
SUMMARY = pd.DataFrame(rows)
if "val_score" in SUMMARY:
    SUMMARY = SUMMARY.sort_values("val_score", ascending=False)
pd.set_option("display.width", 250)
log(("SUMMARY (official score on the 819 Fold-0 val lines, per sub-class):\n" if CFG.RUN_MODE == "fold0" else
     "SUMMARY (RUN_MODE='full': no validation lines, options with test predictions):\n")
    + SUMMARY.round(4).to_string(index=False), banner=True)
SUMMARY.to_csv(os.path.join(CFG.WORK_DIR, "stage_summary.csv"), index=False)

with_test = [o for o in (SUMMARY.option if "val_score" not in SUMMARY else SUMMARY.dropna(subset=["val_score"]).option)
             if OPTIONS[o][1] is not None]
if CFG.RUN_MODE == "fold0" and with_test:
    FINAL = with_test[0]
else:
    FINAL = (f"{LAST_STAGE}/judged" if f"{LAST_STAGE}/judged" in OPTIONS else f"{LAST_STAGE}/model_mbr")
    FINAL = FINAL if FINAL in OPTIONS and OPTIONS[FINAL][1] is not None else next((o for o in OPTIONS if OPTIONS[o][1] is not None), None)
for name, (v, t) in OPTIONS.items():
    if t is not None:
        pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(x) or "the" for x in t]}).to_csv(
            os.path.join(CFG.WORK_DIR, f"submission__{name.replace('/', '__')}.csv"), index=False)
if FINAL is not None:
    sub = pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(x) or "the" for x in OPTIONS[FINAL][1]]})
    sub.to_csv(CFG.SUBMISSION_CSV, index=False)
    log(f"submission.csv <- {FINAL} | rows {len(sub)} | empty replaced {int((sub.Target == 'the').sum())}")
    if FINAL == "guide":
        log("NOTE: the guide alone is still the best option on val — the model + judge did not beat it this run")
if PSEUDO:
    last = PSEUDO[max(PSEUDO, key=lambda s: list(STORE).index(s))]
    last.to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels.csv"), index=False)
    log(f"pseudo_labels.csv <- stage '{LAST_STAGE}' ({len(last)} lines, {int(last.keep_for_training.sum())} kept for training)")
log(f"RUN FINISHED in {(time.time() - SESSION_T0) / 3600:.2f} h", banner=True)
''')

# ============================================================================ save / load
md(r'''
<a name="Save"></a>
### Saving and loading the fine-tuned model
The LoRA adapter of every fine-tune is already saved in `FINAL_DIR/<model>/<fold>/<ft1|self1|...>/` with its processor. To run **inference only** from it later, e.g. on new data or with another guide, set these and run all:
- `DO_FINETUNE=False`;
- `INIT_ADAPTER=<that folder>`, attached as a dataset;
- the same `PROMPT_MODE`, `FEWSHOT_K` and `SUBCLASS_MODE`.
''')
code(r'''
# =========================================================
# Cell 20 — save (again) the final adapter + how to reload it
# =========================================================
if CFG.DO_FINETUNE:
    final_dir = os.path.join(CFG.WORK_DIR, "final_lora")
    model.save_pretrained(final_dir)
    proc.save_pretrained(final_dir)
    log(f"final adapter (stage '{LAST_STAGE}') -> {final_dir}")

if False:   # reload for inference only (a new session)
    base = load_vlm_model(CFG.MODEL_PATH, dtype).to(DEVICE)
    model = PeftModel.from_pretrained(base, "/kaggle/input/<your-dataset>/final_lora")
    model.eval()
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        s = s.replace("__LINE_FEATURES__", LINE_FEATURES_SRC).replace("__RULES__", RULES_SRC)
        lines = s.split("\n")
        c = {"cell_type": t, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "id": f"t{i:02d}"}
        if t == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb["cells"].append(c)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/test_pesudo.ipynb"
    print("wrote", out, "cells:", len(build(out)["cells"]))
