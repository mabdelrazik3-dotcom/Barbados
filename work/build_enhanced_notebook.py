"""Assembles barbados-2-enhanced.ipynb from the cells below (kept as plain strings so they can be
unit-tested by work/dryrun_notebook.py before any GPU time is spent)."""
import json

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ============================================================================ 0. overview
md(r'''
# Barbados R.O.A.D. — Qwen-VL LoRA OCR · **Fold-0 research harness** (enhanced)

This notebook is `barbados-2.ipynb` rebuilt around a controlled **champion / challenger** protocol.
Everything is measured with the **official leaderboard metric** on the **complete Fold-0 validation set (819 lines)**.
The evidence behind each change is in `FULL_DATA_AUDIT.md` / `RESEARCH_PLAN.md`.

**Official metric** (reverse-engineered on the forum, reproduces the Benchmark row to 2e-10; higher is better):
`score = 0.5·(1 − word_edits_per_line/12) + 0.5·(1 − char_edits_per_line/55)`.
One word edit costs as much as 4.58 character edits.

### What changed vs `barbados-2.ipynb`
| # | Change | Why (full-data evidence) |
|---|---|---|
| P1 | Official metric replaces `0.5·jiwer.wer + 0.5·jiwer.cer` | The corpus-rate metric weights a word edit 5.6× a char edit, versus 4.58× officially, and it is not the leaderboard number |
| P2 | Fold-0 validation is **never** used for checkpoint selection; the whole 819-line fold is scored | The original held out 328 of the 819 val lines for `eval_loss` early stopping and scored only 491 (sd ≈ 0.006) |
| P3 | Pixel limits are passed through `size=` at load time, and the result is asserted on a probe image | In transformers ≥5.1x, `ip.max_pixels = …` is silently ignored, which cuts visual tokens from ~276 to 66–76 per line |
| P4 | Decoding challengers are scored on the **same adapter** in one session | `repetition_penalty=1.2` down-weights 14.5% of correct target tokens; 91% of lines are affected (` the`, ` and`, `,`, `^` appear in the prompt) |
| P5 | N-best output + **MBR under the official cost** + word/char ROVER across models | The ensemble code in the original was never called; there was no submission writer |
| P6 | Trial harness: `TRIAL_ID`, `trial_results.csv`, per-trial predictions/configs, paired bootstrap | Differences below about 0.005 on 819 lines are noise |
| P7 | **Realistic augmentation** by default (`AUG_PROFILE="realistic"`): tilt ±1°, noise σ≈3, JPEG q70–92, parchment-coloured fill, fresh augmentation each time a line is seen | Original: ±4° tilt, σ=25.5 noise, q40 JPEG, white fill, and one fixed copy repeated every epoch |
| P8 | Per-model hyper-parameters (`MODEL_OVERRIDES`): **32B LR 2e-4 → 1e-4** | Bigger models scored worse at the 7B/8B learning rate |
| P9 | **Class-aware pipeline:** groups A1/A2/A3/B detected from pixels; per-class scores; class-routed ensemble; per-class pseudo-label filter; optional class tag in the prompt and class-B oversampling | Two eras of documents with different spelling conventions (1630s–60s vs 1670s–1710s) |

### One model with hints, or a model per class / sub-class — both are CFG switches
| Setup | CFG |
|---|---|
| One shared model (default) | `SPECIALISTS = {}` |
| One shared model **with class hints** | `CLASS_HINT = True` (`HINT_LEVEL = "class"` or `"group"`), optionally `CLASS_SAMPLING_T = 2` |
| Shared model + **specialist per class** (continued from the shared adapter) | `SPECIALISTS = {"A": {}, "B": {}}` |
| Shared model + **specialist per sub-class** | `SPECIALISTS = {"A1": {}, "A2": {}, "A3": {}, "B": {}}`, `ROUTING_LEVEL = "group"` |
| **Fully separate** models (fresh LoRA from the base on that class only) | `SPECIALISTS = {"B": {"init": "separate", "replay": 0}}`, etc. |
| Train specialists on top of existing shared adapters | `DO_TRAIN = False`, `ADAPTERS_DIR = ...`, `TRAIN_SPECIALISTS = True` |

Each specialist decodes its own class's validation and test lines. It is scored as a trial on its own, and it is also added as an extra voter in the cross-model MBR.
**The per-class acceptance rule** then decides, class by class, whether to keep the overall champion or switch. A switch needs all of these: P(better) ≥ 0.9, a positive shrunk gain, the same sign in both split-halves, and test-share × gain ≥ 0.002.
The honest score of that whole procedure is estimated by repeated split-half. The chosen routing is written to `routing_map.json`; point `ROUTING_MAP` at that file for the final `RUN_MODE="full"` refit.

### The data classes (from pixels only, so they also work on test images)
| Group | Rule | Share | What differs |
|---|---|---|---|
| **B** | height > 150 px | 25 % | 1670s–1710s documents, ~77 px/char, 2× `^`, 4× `:`, `heyres`/`saide`, more rare words |
| **A1** | height ≤ 58 px | 41 % | tight crops, heavy `&` and `ye`/`yt` |
| **A2** | taller, parchment grey ≤ 194 | 9 % | dark parchment, large hand, almost no `&`/`ye`, short lines |
| **A3** | taller, parchment grey > 194 | 24 % | loose crops with neighbouring lines, longest lines |

The pixel rule reproduces the label-derived clusters with about 90 % agreement.

### How to run a trial
1. Set `TRIAL_ID` / `TRIAL_DESCRIPTION` in Cell 3. Leave everything else at its default for the first session (**EXP_001**).
2. *Run All.* For each model the notebook trains on Fold-0 TRAIN (using an inner 8% slice for checkpoint selection), decodes all 819 val lines once per `DECODE_VARIANTS` entry, and writes one trial row per variant. It then ensembles across models and writes `submission.csv` from the best validated configuration.
3. Download `trial_results.csv` plus `trial_predictions/` and hand them back for the next analysis round.

**Fast path (no retraining):** set `DO_TRAIN=False` and `ADAPTERS_DIR` to a previous run's `final/` adapters. Every decoding challenger then runs on inference only.
''')

# ============================================================================ 1-2. install (unchanged)
code(r'''
# =========================================================
# Cell 1 — OFFLINE install from the wheels dataset
# =========================================================
import os, sys, subprocess, glob

os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob as _glob
import os as _os
WHEELS = "/kaggle/input/datasets/haniagamal/teleco-wheels/wheels"
if not _glob.glob(_os.path.join(WHEELS, "*.whl")):
    _hits = _glob.glob("/kaggle/input/**/*.whl", recursive=True)
    if _hits:
        WHEELS = _os.path.dirname(sorted(_hits, key=len)[0])
print("wheels dir:", WHEELS, "  found:",
      len(_glob.glob(_os.path.join(WHEELS, "*.whl"))), "wheels")
''')

code(r'''
%%capture
!pip uninstall -y torchao
!pip install --no-index --find-links {WHEELS} \
    transformers datasets peft trl jiwer timm librosa soundfile \
    accelerate bitsandbytes huggingface_hub torchao==0.16
print("Dependencies installed (offline).")
''')

# ============================================================================ 3. imports
code(r'''
# =========================================================
# Cell 2 — imports
# =========================================================
import ast, glob, json, shutil, gc, time, math, re, copy, collections
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob as _glob
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import transformers
from datasets import Dataset
from transformers import AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel

try:
    from rapidfuzz.distance import Levenshtein as _RFLev      # ships with jiwer>=3
    def _lev(a, b):
        return _RFLev.distance(a, b)
except Exception:                                              # pure-python fallback
    def _lev(a, b):
        n, m = len(a), len(b)
        if n == 0: return m
        if m == 0: return n
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m; ai = a[i - 1]
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            prev = cur
        return prev[m]

def _auto_vlm_class():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, name):
            return getattr(transformers, name)
    raise ImportError("no AutoModelForImageTextToText / AutoModelForVision2Seq")
AutoVLM = _auto_vlm_class()

print("transformers", transformers.__version__, "| VLM class:", AutoVLM.__name__)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
SESSION_T0 = time.time()
''')

# ============================================================================ 4. CFG
code(r'''
# =========================================================
# Cell 3 — CFG  (+ TRIAL definition)
# =========================================================
TRIAL_ID          = "EXP_002"
TRIAL_DESCRIPTION = ("Realistic augmentation (magnitudes matched to measured archive stats, resampled each time a line "
                     "is seen) + 32B learning rate 2e-4 -> 1e-4; decoding challengers, cross-model and class-routed "
                     "ensembles, all on full Fold-0 val")

class CFG:
    WHEELS_DIR = WHEELS

    # ---- inputs ----
    MODEL_PATHS = [
        "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
        "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2",
        "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/32b-instruct/1",
    ]
    GRAD_CKPT_MODELS = ("32b",)          # substrings of model paths that need gradient checkpointing
    DATA_DIR    = "/kaggle/input/datasets/haniagamal/road-barbados"
    TRAIN_CSV   = None
    TEST_CSV    = None
    IMAGES_DIR  = None
    AUTODISCOVER = True

    # ---- outputs ----
    OUTPUT_DIR     = "/kaggle/working/qwen-vl-ocr"
    FINAL_DIR      = "/kaggle/working/qwen-vl-ocr/final"     # adapters -> FINAL_DIR/<model>/fold{k}
    WORK_DIR       = "/kaggle/working"
    SUBMISSION_CSV = "/kaggle/working/submission.csv"
    KEEP_ADAPTERS  = True
    ADAPTERS_DIR   = None     # DO_TRAIN=False: dir with <model>/fold0 ; None -> autodiscover under /kaggle/input
    PREV_RESULTS_DIRS = []    # previous sessions' /kaggle/working outputs (attached as datasets) -> champion history

    # ---- protocol ----
    RUN_MODE        = "fold0"    # "fold0": research benchmark (train F0-train, score ALL F0-val)
                                 # "full" : final fit on ALL rows, decode test with CHAMPION_DECODE, write submission
    FOLD            = 0
    N_FOLDS         = 5
    CKPT_SELECTION  = "inner_loss"   # "inner_loss": best eval_loss on an inner slice of TRAIN ; "last": no selection
    INNER_CKPT_FRAC = 0.08
    TIME_BUDGET_H   = 11.0           # stop starting new models after this many hours (outputs are persisted per model)

    # ---- training (original recipe = baseline) ----
    PER_DEVICE_TRAIN_BATCH_SIZE  = 4
    GRADIENT_ACCUMULATION_STEPS  = 1
    NUM_TRAIN_EPOCHS             = 1
    LEARNING_RATE                = 2e-4
    LR_SCHEDULER                 = "cosine"
    WARMUP_RATIO                 = 0.03
    DATALOADER_NUM_WORKERS       = 4
    MAX_PIXELS                   = 2_000_000
    MIN_PIXELS                   = 256*28*28
    ATTN_IMPL                    = None
    USE_BF16                     = True
    R                            = 32
    LORA_ALPHA                   = 64
    LORA_DROPOUT                 = 0.05
    LORA_VISION                  = True
    EVAL_STEPS                   = 100
    SAVE_TOTAL_LIMIT             = 2
    SEED                         = 42
    # per-model overrides, keyed by the size tag found in the model path ("7b", "8b", "32b", ...).
    # 32B: bigger models scored worse at 2e-4 -> halve the LR (next step if still worse: 5e-5).
    MODEL_OVERRIDES = {
        "32b": {"LEARNING_RATE": 1e-4},
    }

    # ---- data classes (FULL_DATA_AUDIT §6b) ----
    # Class A = image height <= 150 px: ~17 px per character, documents dated 1630s-1660s (75 % of lines).
    # Class B = image height  > 150 px: ~77 px per character, documents dated 1670s-1710s (25 %), 2x '^', 4x ':'.
    # The class is computed from the image pixels only (never from labels), so it is available for test images.
    # Inside A, three sub-groups (A1/A2/A3) are recovered from pixels with ~90 % agreement.
    CLASS_HEIGHT_SPLIT  = 150
    GROUP_A1_MAX_HEIGHT = 58      # A1: tight crops
    GROUP_A2_MAX_PAPER  = 194     # A2: dark parchment (mean grey above Otsu); A3 otherwise
    CLASS_AUG_COPIES   = None    # augmented copies per class/group, e.g. {"B": 2}; missing keys -> AUG_COPIES
    CLASS_HINT         = False   # True -> "Scan type: <tag>." appended to the prompt, train AND inference (EXP_003)
    HINT_LEVEL         = "class" # "class": A/B | "group": A1/A2/A3/B
    CLASS_ROUTING      = True    # post-hoc per-class routing between the champion and challengers (acceptance rule below)
    ROUTING_LEVEL      = "class" # "class" (2 decisions) | "group" (4 decisions, more selection noise)
    PSEUDO_PER_CLASS   = True    # pseudo-label risk quantile applied within each class (keeps hard class-B lines)
    CLASS_SAMPLING_T   = None    # class-balanced sampling of the SHARED model: class share ∝ n_c^(1/T), upsampling only.
                                 # T=2 -> A2 lines seen ~2.1x, A3/B ~1.3x, A1 1x (extra rows = fresh augmentations)

    # ---- SPECIALISTS: one model for all (SPECIALISTS = {}) or a model per class / sub-class ----
    # key = class "A" | "B"  or sub-class "A1" | "A2" | "A3" | "B";  value = settings (missing keys -> SPEC_DEFAULTS)
    #   init="continue": copy of the SHARED adapter, trained further on that key's lines (+ replay)  <- research recipe
    #   init="separate": a brand-new LoRA from the BASE model trained on that key's lines only (+ replay)
    # examples:  {"B": {}}                                   one B specialist (recommended first)
    #            {"A": {}, "B": {}}                          model per class
    #            {"A1": {}, "A2": {}, "A3": {}, "B": {}}     model per sub-class
    #            {"B": {"init": "separate", "replay": 0}}    a fully separate class-B model
    # Test lines of each key are decoded with its specialist; the per-class rule below decides whether it is used.
    SPECIALISTS       = {}
    SPECIALIST_MODELS = ("8b",)      # base models (size tags) that get specialists; () = every model
    SPEC_DEFAULTS     = dict(init="continue", lr=1e-4, epochs=1, max_steps=None, replay=0.15, warmup_steps=10)
    TRAIN_SPECIALISTS = True         # with DO_TRAIN=False: train specialists on top of the loaded shared adapters

    # ---- per-class selection rule (report "Tag one shared model before splitting it") ----
    # switch class c from the overall champion to a challenger only if ALL hold on c's validation lines:
    #   paired bootstrap P(better) >= ROUTE_P_BETTER, empirical-Bayes-shrunk gain > 0,
    #   same sign in both halves of most random split-halves, and test-share x gain >= ROUTE_MIN_OVERALL_GAIN
    ROUTE_P_BETTER         = 0.9
    ROUTE_MIN_OVERALL_GAIN = 0.002
    ROUTE_MAX_CANDIDATES   = 3       # challengers per class besides the champion (winner's curse grows with K)
    ROUTE_SPLITS           = 30      # repeated split-half for the honest score of the whole selection procedure

    # ---- augmentation (TRAIN only, additive copies) ----
    USE_AUGMENT   = True
    AUG_COPIES    = 1
    AUG_PROFILE   = "realistic"  # "realistic" (measured archive stats, see apply_aug_profile) | "original" (barbados-2)
    AUG_RESAMPLE  = True         # new random augmentation every time a line is seen (matters for >1 epoch)
    AUG_STROKE_MODE = "both"     # "both": erode or dilate (original) | "thicken": ink spread only
    # values below = ORIGINAL barbados-2 profile; AUG_PROFILE="realistic" overwrites them in Cell 5
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

    # corrupted rows excluded from train AND val (identical to barbados-2 so Fold-0 IDs are unchanged)
    CORRUPT_IDS = {
        "79tMUVyfIdy3GzkG",  "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO",
    }

    # ---- inference / decoding challengers (each variant = one trial row per model) ----
    MAX_NEW_TOKENS = 256
    BATCH_SIZE     = 8
    INFER_USE_BF16 = True
    DECODE_VARIANTS = {
        # D0 reproduces barbados-2 decoding exactly
        "D0_orig":   dict(num_beams=3, repetition_penalty=1.2, no_repeat_ngram_size=6, n_return=1),
        # D1: ONE change vs D0 — no repetition penalty / n-gram blocking
        "D1_norep":  dict(num_beams=3, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=1),
        # D2: D1 + 5 beams, 5-best kept -> trial rows for top-1 AND for single-model MBR
        "D2_b5nb":   dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=5),
    }
    BASELINE_VARIANT = "D0_orig"
    MBR_TEMPERATURE  = 1.0        # posterior over a system's N-best = softmax(total_logprob / T)
    ENSEMBLE_METHODS = ("mbr_nbest", "mbr_top1", "word_rover", "char_rover")
    # RUN_MODE="full": reproduce the Fold-0 champion on test — decoding variant + (optional) cross-model method
    CHAMPION_DECODE   = "D2_b5nb"    # a DECODE_VARIANTS key
    CHAMPION_ENSEMBLE = "mbr_nbest"  # one of ENSEMBLE_METHODS, "single_mbr" (MBR over one model's N-best), or None (top-1)
    ROUTING_MAP       = None         # full mode, per class "variant:target" (dict), or the path of a Fold-0 routing_map.json
                                     # targets: "mbr_nbest" | "8b" | "8b+mbr" | "8b@B" (specialist) | "8b@B+mbr" |
                                     #          "mbr_nbest+8b@B" (cross-model MBR with the specialist as an extra voter)

    # ---- transductive pseudo-labelling (allowed if fully automated; thread 34459) ----
    PREDICT_UNLISTED  = True         # also decode the 687 images that are in images/ but in neither CSV
    PSEUDO_LABEL_CSV  = None         # EXP_009: path to a previous session's pseudo_labels.csv -> added to TRAIN
    PSEUDO_KEEP_FRAC  = 0.7          # keep the lowest-MBR-risk fraction of pseudo-labelled lines

    # ---- flow control ----
    DO_TRAIN = True
    DO_INFER = True
    PREDICT_TEST = True

    OCR_PROMPT = (
        "This is not modern English, Transcribe the handwriting exactly. Keep the wrong spelling, abbreviations, and marks (^, ff, unusual letters)."
        "Do not modernize or correct anything."
    )

TRIAL_CONFIG = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)}
print("TRIAL:", TRIAL_ID, "|", TRIAL_DESCRIPTION)
print("MODE:", CFG.RUN_MODE, "| models:", len(CFG.MODEL_PATHS), "| variants:", list(CFG.DECODE_VARIANTS))
''')

# ============================================================================ 5. paths
code(r'''
# =========================================================
# Cell 4 — resolve & sanity-check paths
# =========================================================
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _find_under(root, filename):
    hits = _glob(os.path.join(root, "**", filename), recursive=True)
    return hits[0] if hits else None

def resolve_paths(cfg):
    train_csv = cfg.TRAIN_CSV or os.path.join(cfg.DATA_DIR, "Train.csv")
    test_csv  = cfg.TEST_CSV  or os.path.join(cfg.DATA_DIR, "Test.csv")
    images    = cfg.IMAGES_DIR or _first_existing([
        os.path.join(cfg.DATA_DIR, "images", "images"),
        os.path.join(cfg.DATA_DIR, "images"),
    ]) or os.path.join(cfg.DATA_DIR, "images", "images")
    if cfg.AUTODISCOVER:
        if not os.path.exists(train_csv):
            train_csv = _find_under("/kaggle/input", "Train.csv") or train_csv
        if not os.path.exists(test_csv):
            test_csv = _find_under("/kaggle/input", "Test.csv") or test_csv
        if not os.path.isdir(images) or not _glob(os.path.join(images, "*.jpg")):
            cand = _glob("/kaggle/input/**/*.jpg", recursive=True)
            if cand:
                images = os.path.dirname(cand[0])
    return train_csv, test_csv, images

TRAIN_CSV, TEST_CSV, IMAGES_DIR = resolve_paths(CFG)
TRIAL_PRED_DIR = os.path.join(CFG.WORK_DIR, "trial_predictions")
TRIAL_CFG_DIR  = os.path.join(CFG.WORK_DIR, "trial_configs")
TRIAL_RESULTS  = os.path.join(CFG.WORK_DIR, "trial_results.csv")
for _d in (CFG.OUTPUT_DIR, CFG.FINAL_DIR, TRIAL_PRED_DIR, TRIAL_CFG_DIR):
    os.makedirs(_d, exist_ok=True)

print("TRAIN_CSV :", TRAIN_CSV,  "->", os.path.exists(TRAIN_CSV))
print("TEST_CSV  :", TEST_CSV,   "->", os.path.exists(TEST_CSV))
print("IMAGES_DIR:", IMAGES_DIR, "->", os.path.isdir(IMAGES_DIR))
for _m in CFG.MODEL_PATHS:
    print("MODEL     :", _m, "->", os.path.isdir(_m))
assert os.path.exists(TRAIN_CSV), "Train.csv not found"
assert os.path.isdir(IMAGES_DIR), "images dir not found"
assert all(os.path.isdir(m) for m in CFG.MODEL_PATHS), "a model dir is missing"
''')

# ============================================================================ 6. utils + augmentation
code(r'''
# =========================================================
# Cell 5 — text / image utils + augmentation stack
# =========================================================
def clean_label(x):
    return " ".join(str(x).replace("\n", " ").split()).strip()

def clean_output(text: str) -> str:
    text = str(text)
    for tag in ["assistant", "user", "<|assistant|>", "<|user|>"]:
        if tag in text:
            text = text.split(tag)[-1]
    return " ".join(text.split()).strip()

# identical resize ladder for train AND inference (downscale only; the processor then applies MIN/MAX_PIXELS)
def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    aspect = w / h
    if   aspect > 10: img.thumbnail((2048, 384))
    elif aspect > 5:  img.thumbnail((2048, 512))
    elif aspect > 3:  img.thumbnail((1600, 640))
    else:             img.thumbnail((1536, 768))
    return img

# Augmentation magnitudes matched to the full-data forensics of all 6,159 images (FULL_DATA_AUDIT §4.2).
# measured (5th-95th pct)                        original barbados-2              realistic
REALISTIC_AUG = dict(
    AUG_ROTATE_DEG=1.0,  AUG_ROTATE_P=0.5,        # skew -1..+1 deg              | +-4 deg
    AUG_SHEAR_MAX=0.12,  AUG_SHEAR_P=0.4,         # slant spread -0.2..0.8       | 0.15
    AUG_ELASTIC_P=0.3,                            # hand wobble, kept mild       | 0.4
    AUG_HJITTER_FRAC=0.10,                        # x-height varies widely       | 0.12
    AUG_BRIGHT=0.15, AUG_CONTRAST=0.20, AUG_GAMMA=1.3, AUG_PHOTO_P=0.6,   # paper grey 145..216
    AUG_TINT_MAX=0.10, AUG_DESAT_MAX=0.30,        # paper tint R-B 40..71
    AUG_ILLUM_MAX=0.12, AUG_ILLUM_P=0.3,          # illumination CV <= 0.09      | 35 % vignette
    AUG_STROKE_P=0.15, AUG_STROKE_MODE="thicken", # strokes 1.9-3.8 px: 3x3 max-filter erases them
    AUG_BLEED_ALPHA=0.10, AUG_BLEED_P=0.25,       # mid-tone / show-through 3-20 %
    AUG_NOISE_STD=0.012, AUG_NOISE_P=0.3,         # sensor noise sigma 1.8-4.2   | 25.5 grey levels
    AUG_JPEG_QMIN=70, AUG_JPEG_QMAX=92, AUG_JPEG_P=0.3,   # all images q~85     | q40-80
    AUG_FILL="paper",                             # new pixels get the page's own parchment colour, not white
)

def apply_aug_profile(cfg):
    cfg.AUG_FILL = getattr(cfg, "AUG_FILL", "white")
    if getattr(cfg, "AUG_PROFILE", "original") == "realistic":
        for k, v in REALISTIC_AUG.items():
            setattr(cfg, k, v)
    print(f"[aug] profile={cfg.AUG_PROFILE} resample={cfg.AUG_RESAMPLE} rotate=±{cfg.AUG_ROTATE_DEG}° "
          f"noise σ={cfg.AUG_NOISE_STD*255:.1f} jpeg q{cfg.AUG_JPEG_QMIN}-{cfg.AUG_JPEG_QMAX} "
          f"stroke={cfg.AUG_STROKE_MODE}@{cfg.AUG_STROKE_P} illum={cfg.AUG_ILLUM_MAX} fill={cfg.AUG_FILL}")
apply_aug_profile(CFG)

def _paper_grey(gray):
    """mean grey of the pixels above the Otsu threshold (= the parchment), as in FULL_DATA_AUDIT §4."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    w = np.cumsum(hist); s = np.cumsum(np.arange(256) * hist); tot, sall = w[-1], s[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (s * tot - sall * w) ** 2 / (w * (tot - w))
    t = int(np.nanargmax(np.where((w > 0) & (w < tot), between, -1)))
    bg = gray[gray > t]
    return float(bg.mean()) if bg.size else float(gray.mean())

_GROUP_CACHE = {}
def image_group(path):
    """Data group from the ORIGINAL pixels only (no labels) — FULL_DATA_AUDIT §6b:
       B  : height > 150 px            -> 1670s-1710s scans, ~77 px/char, 2x '^', 4x ':'
       A1 : height <= 58 px            -> tight crops, heavy '&' and 'ye'
       A2 : taller, parchment <= 194   -> dark parchment, large hand, almost no '&' / 'ye'
       A3 : taller, parchment  > 194   -> loose crops with neighbour lines, longest lines
    The rule reproduces the label-derived clusters with ~90 % agreement."""
    if path in _GROUP_CACHE:
        return _GROUP_CACHE[path]
    with Image.open(path) as im:
        h = im.size[1]
        if h > CFG.CLASS_HEIGHT_SPLIT:
            g = "B"
        elif h <= CFG.GROUP_A1_MAX_HEIGHT:
            g = "A1"
        else:
            g = "A2" if _paper_grey(np.asarray(im.convert("L"))) <= CFG.GROUP_A2_MAX_PAPER else "A3"
    _GROUP_CACHE[path] = g
    return g

def image_class(path):
    return image_group(path)[0]          # 'A' or 'B'

def key_match(grp, key):
    """does sub-class grp ('A1', ..., 'B') belong to specialist key ('A', 'B', 'A1', ...)?"""
    return grp == key or grp[0] == key

def ocr_prompt(grp=None):
    """CLASS_HINT: append the scan type so one shared model can switch spelling conventions per era."""
    if not (CFG.CLASS_HINT and grp):
        return CFG.OCR_PROMPT
    tag = grp if CFG.HINT_LEVEL == "group" else grp[0]
    return CFG.OCR_PROMPT + f" Scan type: {tag}."

def _paper_color(img):
    """median colour of the brighter half of the page = the parchment (measured background)."""
    a = np.asarray(img.convert("RGB"), np.float32).reshape(-1, 3)
    lum = a.mean(1)
    bg = a[lum >= np.median(lum)]
    return tuple(int(v) for v in np.median(bg, axis=0))

from PIL import ImageOps, ImageFilter
import io as _io
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter, map_coordinates as _map_coordinates
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _photo(arr, rng, b, c, g):
    a = arr / 255.0
    a = np.power(np.clip(a, 0, 1), float(np.exp(rng.uniform(-np.log(g), np.log(g)))))
    a = (a - 0.5) * (1.0 + float(rng.uniform(-c, c))) + 0.5
    a = a * (1.0 + float(rng.uniform(-b, b)))
    return np.clip(a, 0, 1) * 255.0

def _tint(arr, rng, tint_max, desat_max):
    a = (arr / 255.0).copy()
    gray = a.mean(axis=2, keepdims=True)
    d = float(rng.uniform(0, desat_max)); a = a * (1 - d) + gray * d
    a[..., 0] *= 1.0 + float(rng.uniform(0, tint_max))
    a[..., 2] *= 1.0 - float(rng.uniform(0, tint_max))
    return np.clip(a, 0, 1) * 255.0

def _illumination(arr, rng, imax):
    h, w = arr.shape[:2]
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
    field = (1.0 - float(imax) * (d / (d.max() + 1e-6))).astype(np.float32)[..., None]
    return np.clip(arr * field, 0, 255)

def _bleed(arr, rng, alpha):
    a = float(rng.uniform(0, alpha))
    return np.clip(arr * (1 - a) + arr[:, ::-1, :] * a, 0, 255)

def _elastic(arr, rng, alpha, sigma):
    h, w = arr.shape[:2]
    dx = _gaussian_filter(rng.rand(h, w).astype(np.float32) * 2 - 1, sigma) * alpha
    dy = _gaussian_filter(rng.rand(h, w).astype(np.float32) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    out = np.empty_like(arr)
    for ch in range(arr.shape[2]):
        out[..., ch] = _map_coordinates(arr[..., ch], coords, order=1, mode="reflect")
    return out

def augment_image(img, seed=None):
    """Label-preserving photometric + geometric stack, seeded per row (reproducible copies)."""
    if not getattr(CFG, "USE_AUGMENT", False):
        return img
    rng = np.random.RandomState(int(seed) & 0x7fffffff) if seed is not None else np.random
    fill = _paper_color(img) if CFG.AUG_FILL == "paper" else (255, 255, 255)
    if CFG.AUG_ROTATE and rng.rand() < CFG.AUG_ROTATE_P:
        img = img.rotate(float(rng.uniform(-CFG.AUG_ROTATE_DEG, CFG.AUG_ROTATE_DEG)),
                         resample=Image.BICUBIC, expand=True, fillcolor=fill)
    if CFG.AUG_SHEAR and rng.rand() < CFG.AUG_SHEAR_P:
        sh = float(rng.uniform(-CFG.AUG_SHEAR_MAX, CFG.AUG_SHEAR_MAX)); w, h = img.size
        xshift = abs(sh) * h
        img = img.transform((w + int(round(xshift)), h), Image.AFFINE,
                            (1, sh, -xshift if sh > 0 else 0.0, 0, 1, 0),
                            resample=Image.BICUBIC, fillcolor=fill)
    if CFG.AUG_STROKE and rng.rand() < CFG.AUG_STROKE_P:
        # dark ink on light paper: MinFilter = thicker strokes (ink spread), MaxFilter = thinner (can erase)
        thin = CFG.AUG_STROKE_MODE == "both" and rng.rand() >= 0.5
        img = img.filter(ImageFilter.MaxFilter(3) if thin else ImageFilter.MinFilter(3))
    if CFG.AUG_HJITTER and rng.rand() < CFG.AUG_HJITTER_P:
        w, h = img.size
        f = float(rng.uniform(1 - CFG.AUG_HJITTER_FRAC, 1 + CFG.AUG_HJITTER_FRAC))
        img = img.resize((w, max(1, int(round(h * f)))), Image.BICUBIC)
    if CFG.AUG_PAD and rng.rand() < CFG.AUG_PAD_P:
        w, h = img.size; pf = CFG.AUG_PAD_FRAC
        img = ImageOps.expand(img, (int(rng.uniform(0, pf) * w), int(rng.uniform(0, pf) * h),
                                    int(rng.uniform(0, pf) * w), int(rng.uniform(0, pf) * h)),
                              fill=fill)
    arr = np.asarray(img.convert("RGB"), np.float32)
    if CFG.AUG_ELASTIC and _HAS_SCIPY and rng.rand() < CFG.AUG_ELASTIC_P:
        arr = _elastic(arr, rng, CFG.AUG_ELASTIC_ALPHA, CFG.AUG_ELASTIC_SIGMA)
    if CFG.AUG_PHOTO and rng.rand() < CFG.AUG_PHOTO_P:
        arr = _photo(arr, rng, CFG.AUG_BRIGHT, CFG.AUG_CONTRAST, CFG.AUG_GAMMA)
    if CFG.AUG_TINT and rng.rand() < CFG.AUG_TINT_P:
        arr = _tint(arr, rng, CFG.AUG_TINT_MAX, CFG.AUG_DESAT_MAX)
    if CFG.AUG_ILLUM and rng.rand() < CFG.AUG_ILLUM_P:
        arr = _illumination(arr, rng, CFG.AUG_ILLUM_MAX)
    if CFG.AUG_BLEED and rng.rand() < CFG.AUG_BLEED_P:
        arr = _bleed(arr, rng, CFG.AUG_BLEED_ALPHA)
    if CFG.AUG_NOISE and rng.rand() < CFG.AUG_NOISE_P:
        arr = arr + rng.randn(*arr.shape).astype(np.float32) * (CFG.AUG_NOISE_STD * 255.0)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if CFG.AUG_JPEG and rng.rand() < CFG.AUG_JPEG_P:
        buf = _io.BytesIO()
        out.save(buf, format="JPEG", quality=int(rng.randint(CFG.AUG_JPEG_QMIN, CFG.AUG_JPEG_QMAX + 1)))
        buf.seek(0); out = Image.open(buf).convert("RGB")
    return out

def resolve_image(base_dir, image_id):
    exact = os.path.join(base_dir, f"{image_id}.jpg")
    if os.path.exists(exact):
        return exact
    matches = _glob(os.path.join(base_dir, f"{image_id}*.jpg"))
    return matches[0] if matches else None

def model_short_name(msrc):
    parts = [p for p in msrc.replace("\\", "/").strip("/").split("/") if p]
    return (re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts[-4:]))[:60] or "model")

def legacy_short_name(msrc):
    """adapter dir name written by the ORIGINAL barbados-2.ipynb (last 3 path parts)."""
    parts = [p for p in msrc.replace("\\", "/").strip("/").split("/") if p]
    return (re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts[-3:]))[:48] or "model")

def model_size_tag(name):
    m = re.search(r"(?<![0-9])(\d+b)(?![0-9])", name.lower())
    return m.group(1) if m else name[-12:]

def free_cuda(*names):
    for n in names:
        if n in globals():
            del globals()[n]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
''')

# ============================================================================ 7. loaders
code(r'''
# =========================================================
# Cell 6 — model & processor loaders (pixel limits enforced + verified)
# =========================================================
def load_vlm_model(path, dtype, device_map=None):
    base = {"trust_remote_code": True}
    if device_map is not None:
        base["device_map"] = device_map
    if getattr(CFG, "ATTN_IMPL", None):
        base["attn_implementation"] = CFG.ATTN_IMPL
    for dkey in ("dtype", "torch_dtype"):
        try:
            return AutoVLM.from_pretrained(path, **{dkey: dtype}, **base)
        except TypeError:
            continue
        except Exception:
            if "attn_implementation" in base:
                base.pop("attn_implementation")
                try:
                    return AutoVLM.from_pretrained(path, **{dkey: dtype}, **base)
                except TypeError:
                    continue
            raise
    return AutoVLM.from_pretrained(path, **base)

def _repair_preprocessor(path):
    dst = os.path.join(CFG.WORK_DIR, "_patched_" + os.path.basename(path.rstrip("/\\")))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    def _ignore_weights(_d, names):
        return [n for n in names if n.endswith(
            (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack", ".onnx"))]
    shutil.copytree(path, dst, ignore=_ignore_weights)
    pcfg = os.path.join(dst, "preprocessor_config.json")
    cfg = {}
    if os.path.exists(pcfg):
        with open(pcfg, "r") as f:
            cfg = json.load(f)
    prefer = ["Qwen2VLImageProcessorFast", "Qwen2_5_VLImageProcessorFast", "Qwen3VLImageProcessorFast",
              "Qwen2VLImageProcessor", "Qwen2_5_VLImageProcessor", "Qwen3VLImageProcessor"]
    avail = [c for c in prefer if hasattr(transformers, c)]
    if avail:
        cfg["image_processor_type"] = avail[0]
    else:
        cfg.pop("image_processor_type", None)
    with open(pcfg, "w") as f:
        json.dump(cfg, f)
    return dst

def _smart_resize(h, w, factor, min_pixels, max_pixels):
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar

def _force_pixel_limits(ip, min_pixels, max_pixels):
    """transformers 5.0 honours ip.min_pixels/ip.max_pixels attributes; >=5.1x only reads ip.size.
    Set every representation so the limits hold on any version."""
    try:
        sz = dict(ip.size) if not isinstance(ip.size, dict) else dict(ip.size)
    except Exception:
        sz = {}
    sz = {k: v for k, v in sz.items() if v is not None}
    sz["shortest_edge"], sz["longest_edge"] = int(min_pixels), int(max_pixels)
    try:
        ip.size = sz
    except Exception:
        pass
    for k, v in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        try:
            setattr(ip, k, int(v))
        except Exception:
            pass

def load_vlm_processor(path, max_pixels=None, min_pixels=None):
    kw_px = {}
    if max_pixels is not None and min_pixels is not None:
        kw_px = dict(min_pixels=int(min_pixels), max_pixels=int(max_pixels))
    attempts = [dict(use_fast=True), dict(use_fast=False),
                dict(use_fast=True, trust_remote_code=True), dict(use_fast=False, trust_remote_code=True)]
    proc, last = None, None
    for src in (path, None):
        if src is None:
            print("[processor] direct load failed; repairing. cause:", repr(last)[:200])
            src = _repair_preprocessor(path)
        for kw in attempts:
            for extra in (kw_px, {}):
                try:
                    proc = AutoProcessor.from_pretrained(src, **kw, **extra); break
                except Exception as e:
                    last = e
            if proc is not None: break
        if proc is not None: break
    if proc is None:
        raise last
    ip = proc.image_processor
    if kw_px:
        _force_pixel_limits(ip, min_pixels, max_pixels)
    # ---- verify on a probe strip: the grid must match smart_resize(min/max) ----
    probe = Image.new("RGB", (1060, 57), (220, 200, 170))
    fac = int(getattr(ip, "patch_size", 14)) * int(getattr(ip, "merge_size", 2))
    out = ip(images=[probe], return_tensors="pt")
    g = out["image_grid_thw"][0].tolist()
    got = (g[1] * ip.patch_size, g[2] * ip.patch_size)
    if kw_px:
        exp = _smart_resize(57, 1060, fac, min_pixels, max_pixels)
        assert tuple(got) == tuple(exp), f"pixel limits NOT applied: processor gives {got}, expected {exp}"
    print(f"[processor] {type(proc).__name__} | {type(ip).__name__} | probe 1060x57 -> {got[1]}x{got[0]} px, "
          f"{g[1]*g[2]//(ip.merge_size**2)} visual tokens")
    return proc
''')

# ============================================================================ 8. metric + ensembles + forensics
code(r'''
# =========================================================
# Cell 7 — OFFICIAL metric, MBR / ROVER ensembling, error forensics, paired bootstrap
#   score = 0.5*(1 - word_edits_per_line/12) + 0.5*(1 - char_edits_per_line/55)   (higher = better)
# =========================================================
WORD_XMAX, CHAR_XMAX = 12.0, 55.0

def norm_text(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return " ".join(str(s).split())

def line_edits(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    return _lev(r.split(), h.split()), _lev(r, h)

def line_cost(ref, hyp):
    w, c = line_edits(ref, hyp)
    return 0.5 * w / WORD_XMAX + 0.5 * c / CHAR_XMAX

def official_metric(refs, hyps):
    ed = np.array([line_edits(r, h) for r, h in zip(refs, hyps)], float).reshape(-1, 2)
    we, ce = ed[:, 0], ed[:, 1]
    nw = max(1, sum(len(norm_text(r).split()) for r in refs)); nc = max(1, sum(len(norm_text(r)) for r in refs))
    return {"score": float(0.5 * (1 - we.mean() / WORD_XMAX) + 0.5 * (1 - ce.mean() / CHAR_XMAX)),
            "word_edits_per_line": float(we.mean()), "char_edits_per_line": float(ce.mean()),
            "wer_corpus": float(we.sum() / nw), "cer_corpus": float(ce.sum() / nc),
            "n": len(refs), "empty": int(sum(not norm_text(h) for h in hyps))}

def mbr_select(cands, weights=None, return_risk=False):
    """Minimum-Bayes-risk pick under the official per-line cost. The minimum expected cost ('risk')
    doubles as a confidence: low risk = the candidate pool agrees."""
    cands = [norm_text(c) for c in cands]
    weights = [1.0] * len(cands) if weights is None else weights
    agg = collections.OrderedDict()
    for c, w in zip(cands, weights):
        agg[c] = agg.get(c, 0.0) + float(w)
    uniq = list(agg)
    if len(uniq) == 1:
        return (uniq[0], 0.0) if return_risk else uniq[0]
    wv = np.array([agg[u] for u in uniq]); wv = wv / wv.sum()
    risks = [sum(w * line_cost(c, h) for c, w in zip(uniq, wv)) for h in uniq]
    k = int(np.argmin(risks))
    return (uniq[k], float(risks[k])) if return_risk else uniq[k]

def nbest_posteriors(total_logprobs, temperature=1.0):
    s = np.asarray(total_logprobs, float) / max(temperature, 1e-6)
    s = s - s.max(); p = np.exp(s)
    return p / p.sum()

def _align(cons, hyp):
    n, m = len(cons), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i-1, j-1] + (cons[i-1] != hyp[j-1]), D[i-1, j] + 1, D[i, j-1] + 1)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i-1, j-1] + (cons[i-1] != hyp[j-1]):
            out.append((i - 1, hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i-1, j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]

def rover(hyps, weights=None, level="word"):
    """ROVER: align every system's top-1 into a confusion network, weighted vote per slot
    (an absent system votes for deletion). level='word' matches the metric's 4.58x word premium."""
    split = (lambda s: norm_text(s).split()) if level == "word" else (lambda s: list(norm_text(s)))
    join = " ".join if level == "word" else "".join
    seqs = [split(h) for h in hyps]
    weights = [1.0] * len(seqs) if weights is None else list(weights)
    order = list(np.argsort(-np.asarray(weights), kind="stable"))
    seqs = [seqs[k] for k in order]; weights = [weights[k] for k in order]
    slots = [{0: t} for t in seqs[0]]
    for s in range(1, len(seqs)):
        cons = [collections.Counter(sl.values()).most_common(1)[0][0] if sl else "" for sl in slots]
        new = []
        for ci, t in _align(cons, seqs[s]):
            if ci is None:
                new.append({s: t})
            else:
                sl = dict(slots[ci])
                if t is not None:
                    sl[s] = t
                new.append(sl)
        slots = new
    out = []
    for sl in slots:
        agg = collections.Counter()
        for s, t in sl.items():
            agg[t] += weights[s]
        agg[None] += sum(weights[s] for s in range(len(seqs)) if s not in sl)
        best = max(agg.items(), key=lambda kv: kv[1])[0]
        if best is not None:
            out.append(best)
    return norm_text(join(out))

def paired_delta(refs, hyp_a, hyp_b, n_boot=10000, seed=0):
    """challenger B vs champion A — paired bootstrap over whole lines."""
    la = np.array([line_cost(r, h) for r, h in zip(refs, hyp_a)])
    lb = np.array([line_cost(r, h) for r, h in zip(refs, hyp_b)])
    d = la - lb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    ea = np.array([line_edits(r, h) for r, h in zip(refs, hyp_a)]); eb = np.array([line_edits(r, h) for r, h in zip(refs, hyp_b)])
    return {"delta": float(d.mean()), "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5)),
            "p_better": float((boots > 0).mean()), "improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum()),
            "word_edits_saved": int(ea[:, 0].sum() - eb[:, 0].sum()), "char_edits_saved": int(ea[:, 1].sum() - eb[:, 1].sum())}

def _word_err_class(r, h):
    if r.lower() == h.lower(): return "casing"
    strip = lambda x: "".join(ch for ch in x if ch.isalnum())
    if strip(r) == strip(h): return "punct/markup"
    if strip(r).lower() == strip(h).lower(): return "casing+punct"
    d = _lev(r, h)
    return "1-char" if d == 1 else "2-char" if d == 2 else "3+char"

def error_forensics(refs, hyps, top=20):
    try:
        from rapidfuzz.distance import Levenshtein as L
    except Exception:
        return {"note": "rapidfuzz unavailable"}
    ops, csub, wsub, wcls = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    cdel, cins, wdel, wins = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for r, h in zip(refs, hyps):
        r, h = norm_text(r), norm_text(h); rw, hw = r.split(), h.split()
        for op in L.editops(rw, hw):
            ops["word_" + op.tag] += 1
            if op.tag == "replace":
                wsub[f"{rw[op.src_pos]} -> {hw[op.dest_pos]}"] += 1; wcls[_word_err_class(rw[op.src_pos], hw[op.dest_pos])] += 1
            elif op.tag == "delete": wdel[rw[op.src_pos]] += 1
            else: wins[hw[op.dest_pos]] += 1
        for op in L.editops(r, h):
            ops["char_" + op.tag] += 1
            if op.tag == "replace": csub[f"{r[op.src_pos]!r}->{h[op.dest_pos]!r}"] += 1
            elif op.tag == "delete": cdel[repr(r[op.src_pos])] += 1
            else: cins[repr(h[op.dest_pos])] += 1
    return {"ops": dict(ops), "word_sub_classes": dict(wcls), "word_sub": wsub.most_common(top),
            "word_del": wdel.most_common(top), "word_ins": wins.most_common(top),
            "char_sub": csub.most_common(top), "char_del": cdel.most_common(top), "char_ins": cins.most_common(top)}

def score_by_group(refs, hyps, groups):
    g = np.asarray(groups); out = {}
    for k in sorted(set(g.tolist()), key=str):
        idx = np.where(g == k)[0]
        m = official_metric([refs[i] for i in idx], [hyps[i] for i in idx])
        out[str(k)] = f"n={len(idx):4d} score={m['score']:.4f} w/line={m['word_edits_per_line']:.3f} c/line={m['char_edits_per_line']:.3f}"
    return out

# sanity checks
assert line_edits("a b c", "") == (3, 5)
assert abs(official_metric(["a b c"], [""])["score"] - (0.5 * (1 - 3/12) + 0.5 * (1 - 5/55))) < 1e-12
assert rover(["the said John", "the sd John", "the said Jon"]) == "the said John"
assert mbr_select(["a b c", "a b c", "a x c"]) == "a b c"
print("official metric + ensemble helpers OK")
''')

# ============================================================================ 9. samples + folds
code(r'''
# =========================================================
# Cell 8 — samples, Fold-0 split (identical IDs to barbados-2), repaired protocol
# =========================================================
def build_samples(df, base_dir):
    corrupt = set(CFG.CORRUPT_IDS)
    samples, missing, excluded = [], 0, 0
    for _, row in df.iterrows():
        image_id, label = str(row["ID"]).strip(), clean_label(row["Target"])
        if not image_id or not label:
            continue
        if image_id in corrupt:
            excluded += 1; continue
        path = resolve_image(base_dir, image_id)
        if path is None:
            missing += 1; continue
        samples.append({"id": image_id, "image": path, "text": label})
    print(f"[DATASET] found={len(samples)}, missing={missing}, excluded_corrupt={excluded}")
    return samples

df = pd.read_csv(TRAIN_CSV).sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
all_samples = build_samples(df, IMAGES_DIR)
FOLDS = list(KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED).split(all_samples))

if CFG.RUN_MODE == "fold0":
    tr_idx, va_idx = FOLDS[CFG.FOLD]
else:                                   # "full": everything is training data, no validation
    tr_idx, va_idx = np.arange(len(all_samples)), np.array([], int)
_rng = np.random.RandomState(CFG.SEED + 1000 + CFG.FOLD)
_perm = _rng.permutation(tr_idx)
n_inner = int(round(CFG.INNER_CKPT_FRAC * len(_perm))) if CFG.CKPT_SELECTION == "inner_loss" else 0
inner_idx, core_idx = _perm[:n_inner], _perm[n_inner:]
core_samples  = [all_samples[i] for i in core_idx]
inner_samples = [all_samples[i] for i in inner_idx]
val_samples   = [all_samples[i] for i in va_idx]
VAL_IDS  = [s["id"] for s in val_samples]
VAL_GT   = [s["text"] for s in val_samples]
VAL_PATHS = [s["image"] for s in val_samples]
assert not (set(VAL_IDS) & {s["id"] for s in core_samples + inner_samples}), "validation leak!"
print(f"mode={CFG.RUN_MODE} fold={CFG.FOLD}: train={len(core_samples)} inner_ckpt={len(inner_samples)} "
      f"val={len(val_samples)} (ALL scored)")
# data group of every image (A1/A2/A3 = 1630s-60s scans, B = 1670s-1710s high-res scans) — from pixels only
for s in all_samples:
    s["grp"] = image_group(s["image"])
VAL_GROUP = [s["grp"] for s in val_samples]
VAL_CLASS = [g[0] for g in VAL_GROUP]
VAL_LEN_BUCKET = [("<50" if len(t) < 50 else "50-69" if len(t) < 70 else "70+") for t in VAL_GT]
_cc = lambda ss: dict(sorted(collections.Counter(s["grp"] for s in ss).items()))
print(f"group balance  train={_cc(core_samples)}  inner={_cc(inner_samples)}  val={_cc(val_samples)}")

_test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
_test_df.columns = [c.lstrip("\ufeff") for c in _test_df.columns]
TEST_IDS   = [str(r["ID"]).strip() for r in _test_df.to_dict("records")]
TEST_PATHS = [os.path.join(IMAGES_DIR, f"{rid}.jpg") for rid in TEST_IDS]
print("test rows:", len(TEST_IDS), "| missing test images:", sum(not os.path.exists(p) for p in TEST_PATHS))
_known = set(TEST_IDS) | {str(i).strip() for i in pd.read_csv(TRAIN_CSV).ID}
UNL_IDS = sorted(os.path.basename(p)[:-4] for p in _glob(os.path.join(IMAGES_DIR, "*.jpg"))
                 if os.path.basename(p)[:-4] not in _known)
UNL_PATHS = [os.path.join(IMAGES_DIR, f"{i}.jpg") for i in UNL_IDS]
print("unlisted images (in images/, in neither CSV):", len(UNL_IDS))
TEST_GROUP = [image_group(p) if os.path.exists(p) else "A1" for p in TEST_PATHS]
UNL_GROUP  = [image_group(p) for p in UNL_PATHS]
TEST_CLASS = [g[0] for g in TEST_GROUP]
print("group balance  test=", dict(sorted(collections.Counter(TEST_GROUP).items())),
      " unlisted=", dict(sorted(collections.Counter(UNL_GROUP).items())))

# ---- optional pseudo-labels (test + unlisted) from a previous session's ensemble ----
pseudo_samples = []
if CFG.PSEUDO_LABEL_CSV and os.path.exists(CFG.PSEUDO_LABEL_CSV):
    _pl = pd.read_csv(CFG.PSEUDO_LABEL_CSV)
    _pl = _pl[_pl.Target.fillna("").str.split().str.len().between(3, 30)]        # drop empty / runaway lines
    _pl = _pl[_pl.ID.map(lambda i: os.path.exists(os.path.join(IMAGES_DIR, f"{i}.jpg")))]
    _pl["grp"] = [image_group(os.path.join(IMAGES_DIR, f"{i}.jpg")) for i in _pl.ID]
    if "keep_for_training" in _pl:  # file from work/08_build_pseudo_labels.py: calibrated per-class selection already made
        _pl = _pl[_pl.keep_for_training.astype(str).str.lower().isin(["true", "1"])]
    elif CFG.PSEUDO_PER_CLASS:      # class B is harder -> higher risk; a global cut would drop most of it
        _q = _pl.groupby(_pl.grp.str[0]).risk.transform(lambda r: r.quantile(CFG.PSEUDO_KEEP_FRAC))
        _pl = _pl[_pl.risk <= _q]
    else:
        _pl = _pl[_pl.risk <= _pl.risk.quantile(CFG.PSEUDO_KEEP_FRAC)]
    pseudo_samples = [{"id": r.ID, "image": os.path.join(IMAGES_DIR, f"{r.ID}.jpg"), "text": clean_label(r.Target),
                       "grp": r.grp} for r in _pl.itertuples()]
    assert not ({s["id"] for s in pseudo_samples} & set(VAL_IDS)), "pseudo-labels must not cover validation lines"
    print(f"pseudo-labelled lines added to TRAIN: {len(pseudo_samples)} (risk <= q{CFG.PSEUDO_KEEP_FRAC:.2f}"
          f"{' per class' if CFG.PSEUDO_PER_CLASS else ''}) by group:", dict(collections.Counter(_pl.grp)))
pd.DataFrame({"ID": VAL_IDS}).to_csv(os.path.join(CFG.WORK_DIR, f"fold{CFG.FOLD}_val_ids.csv"), index=False)
''')

# ============================================================================ 10. collator (unchanged logic)
code(r'''
# =========================================================
# Cell 9 — collator with assistant-boundary label masking (masking unchanged from barbados-2)
# =========================================================
MASK_FAIL_COUNT = 0
_AUG_VISITS = collections.Counter()

def _aug_seed(row_seed):
    """AUG_RESAMPLE: a new, still reproducible, augmentation every time a line is seen.
    Mixes the per-process visit count (num_workers=0) with the DataLoader worker seed, which
    PyTorch re-draws every epoch from the Trainer-seeded generator (num_workers>0)."""
    if not CFG.AUG_RESAMPLE:
        return row_seed
    v = _AUG_VISITS[row_seed]; _AUG_VISITS[row_seed] += 1
    return (int(row_seed) * 1000003 + v * 7919 + int(torch.initial_seed()) % 1000000007) & 0x7fffffff

def collate_fn(examples, processor):
    global MASK_FAIL_COUNT
    images, texts, classes = [], [], []
    for ex in examples:
        if os.path.exists(ex["image"]):
            img = load_image(ex["image"])
            if CFG.USE_AUGMENT and ex.get("aug", False):
                img = augment_image(img, _aug_seed(ex.get("seed")))
            images.append(img); texts.append(ex["text"]); classes.append(ex.get("grp"))
    if len(images) == 0:
        return None
    tokenizer = processor.tokenizer
    full_msgs, prompt_msgs = [], []
    for img, label, cls in zip(images, texts, classes):
        user_turn = {"role": "user", "content": [{"type": "text", "text": ocr_prompt(cls)},
                                                 {"type": "image", "image": img}]}
        full_msgs.append([user_turn, {"role": "assistant", "content": [{"type": "text", "text": label}]}])
        prompt_msgs.append([user_turn])
    texts_out = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_msgs]
    batch = processor(text=texts_out, images=images, padding=True, truncation=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    for i, pm in enumerate(prompt_msgs):
        try:
            prompt_txt = processor.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
            prompt_ids = processor(text=[prompt_txt], images=[images[i]], return_tensors="pt", truncation=True)["input_ids"][0]
            n_prompt = int((prompt_ids != tokenizer.pad_token_id).sum().item())
            labels[i, :n_prompt] = -100
        except Exception:
            MASK_FAIL_COUNT += 1
    labels[labels == tokenizer.pad_token_id] = -100
    image_token = getattr(processor, "image_token", None) or "<|image_pad|>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if isinstance(image_token_id, int) and image_token_id >= 0:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch
''')

# ============================================================================ 11. generation
code(r'''
# =========================================================
# Cell 10 — batched N-best generation with sequence scores
#   returns, per image, a list of (text, total_logprob, n_tokens, truncated) sorted best-first
# =========================================================
def _gen_config(model, var):
    gc_ = copy.deepcopy(model.generation_config)
    gc_.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        try: setattr(gc_, k, None)
        except Exception: pass
    nb = int(var.get("num_beams", 1)); nr = min(int(var.get("n_return", 1)), nb)
    gc_.num_beams = nb
    gc_.num_return_sequences = nr
    gc_.max_new_tokens = CFG.MAX_NEW_TOKENS
    gc_.repetition_penalty = float(var.get("repetition_penalty", 1.0))
    gc_.no_repeat_ngram_size = int(var.get("no_repeat_ngram_size", 0))
    gc_.length_penalty = float(var.get("length_penalty", 1.0))
    gc_.early_stopping = True if nb > 1 else False
    gc_.output_scores = True
    gc_.return_dict_in_generate = True
    return gc_, nb, nr

def generate_nbest(model, processor, images, var, classes=None):
    classes = classes or [None] * len(images)
    prompts = [processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": ocr_prompt(c)}, {"type": "image", "image": im}]}],
        tokenize=False, add_generation_prompt=True) for im, c in zip(images, classes)]
    tok = processor.tokenizer
    old_side = tok.padding_side; tok.padding_side = "left"
    try:
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = old_side
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gcfg, nb, nr = _gen_config(model, var)
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=gcfg)
    P = inputs["input_ids"].shape[1]
    gen = out.sequences[:, P:]
    stop_ids = {tok.pad_token_id, tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    stop_ids = {s for s in stop_ids if isinstance(s, int) and s >= 0}
    is_stop = torch.zeros_like(gen, dtype=torch.bool)
    for s in stop_ids:
        is_stop |= gen == s
    has_stop = is_stop.any(1)
    first_stop = torch.where(has_stop, is_stop.float().argmax(1), torch.full_like(has_stop, gen.shape[1], dtype=torch.long))
    n_tok = (first_stop + has_stop.long()).clamp(min=1)            # generated tokens incl. EOS
    if nb > 1 and getattr(out, "sequences_scores", None) is not None:
        lp = float(gcfg.length_penalty)
        total = out.sequences_scores.float() * n_tok.float() ** lp  # undo length normalisation
    else:
        ts = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        valid = torch.arange(ts.shape[1], device=ts.device)[None, :] < n_tok[:, None].to(ts.device)
        total = torch.where(valid & torch.isfinite(ts), ts, torch.zeros_like(ts)).sum(1)
    texts = [clean_output(t) for t in processor.batch_decode(gen, skip_special_tokens=True)]
    res = []
    for b in range(len(images)):
        rows = []
        for r in range(nr):
            k = b * nr + r
            rows.append((texts[k], float(total[k]), int(n_tok[k]), bool(not has_stop[k])))
        res.append(rows)
    return res

def transcribe(model, processor, paths, var, desc=None):
    """-> list (aligned with paths) of N-best lists. Missing images -> [("", -inf, 0, False)]."""
    bs = max(1, int(CFG.BATCH_SIZE) // max(1, int(var.get("num_beams", 1)) // 3))
    out = [[("", float("-inf"), 0, False)] for _ in paths]
    idxs = [i for i, p in enumerate(paths) if p and os.path.exists(p)]
    for b in tqdm(range(0, len(idxs), bs), desc=desc, leave=False):
        chunk = idxs[b:b + bs]
        imgs = [load_image(paths[i]) for i in chunk]
        cls = [image_group(paths[i]) for i in chunk] if CFG.CLASS_HINT else None
        try:
            res = generate_nbest(model, processor, imgs, var, cls)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res = [generate_nbest(model, processor, [im], var, [c] if cls else None)[0]
                   for im, c in zip(imgs, cls or [None] * len(imgs))]
        for i, r in zip(chunk, res):
            out[i] = r
    return out
''')

# ============================================================================ 12. trial harness
code(r'''
# =========================================================
# Cell 11 — trial harness: persistence, champion tracking, standard report block
# =========================================================
def _prev_results():
    frames = []
    for d in [CFG.WORK_DIR] + list(CFG.PREV_RESULTS_DIRS):
        p = os.path.join(d, "trial_results.csv")
        if os.path.exists(p):
            try: frames.append(pd.read_csv(p))
            except Exception: pass
    return pd.concat(frames, ignore_index=True).drop_duplicates("trial") if frames else pd.DataFrame()

def champion_row():
    r = _prev_results()
    if len(r) == 0 or "score" not in r: return None
    r = r[r.status.isin(["PROMOTED", "BASELINE", "EVALUATED"]) & r.score.notna()]
    return None if len(r) == 0 else r.sort_values("score", ascending=False).iloc[0]

def save_predictions(trial, ids, gts, nbests, chosen=None, split="val"):
    rows = []
    for k, i in enumerate(ids):
        nb = nbests[k]
        top = chosen[k] if chosen is not None else nb[0][0]
        rows.append({"ID": i, "GroundTruth": gts[k] if gts is not None else None, "pred": top,
                     "logprob": nb[0][1] if nb else None, "n_tokens": nb[0][2] if nb else None,
                     "truncated": nb[0][3] if nb else None,
                     "nbest": json.dumps([[t, round(s, 4)] for t, s, _, _ in nb], ensure_ascii=False)})
    p = os.path.join(TRIAL_PRED_DIR, f"{trial}__{split}.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p

def report_trial(trial, desc, model, variant, hyps, runtime_s, n_train, trunc=0, baseline_hyps=None, extra=None):
    m = official_metric(VAL_GT, hyps)
    base = paired_delta(VAL_GT, baseline_hyps, hyps) if baseline_hyps is not None else None
    ch = champion_row()
    ch_score = float(ch.score) if ch is not None else float("nan")
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print("=" * 66)
    print(f"TRIAL: {trial}\nDESCRIPTION: {desc}\nMODEL: {model} | DECODE: {variant}")
    print("=" * 66)
    print(f"Official validation score : {m['score']:.5f}")
    print(f"Word edits / line (LB WER) : {m['word_edits_per_line']:.4f}   corpus WER {m['wer_corpus']:.4f}")
    print(f"Char edits / line (LB CER) : {m['char_edits_per_line']:.4f}   corpus CER {m['cer_corpus']:.4f}")
    if base is not None:
        print(f"Delta vs baseline          : {base['delta']:+.5f}  CI95 [{base['ci95_lo']:+.5f}, {base['ci95_hi']:+.5f}]"
              f"  P(better)={base['p_better']:.3f}  improved={base['improved']} worsened={base['worsened']}"
              f"  word/char edits saved={base['word_edits_saved']}/{base['char_edits_saved']}")
    print(f"Delta vs champion          : {m['score'] - ch_score:+.5f}  (champion={ch_score:.5f})")
    print(f"Runtime                    : {runtime_s/60:.1f} min   Peak GPU RAM : {peak:.1f} GB")
    print(f"Train examples             : {n_train}   Validation examples : {m['n']}")
    print(f"Empty predictions          : {m['empty']}   Truncated predictions : {trunc}")
    print("By class (A<=150px, B>150px):", score_by_group(VAL_GT, hyps, VAL_CLASS))
    print("By group (A1/A2/A3/B)      :", score_by_group(VAL_GT, hyps, VAL_GROUP))
    print("By label length          :", score_by_group(VAL_GT, hyps, VAL_LEN_BUCKET))
    print("=" * 66)
    row = {"trial": trial, "session": TRIAL_ID, "description": desc, "model": model, "variant": variant,
           "score": m["score"], "word_edits_per_line": m["word_edits_per_line"], "char_edits_per_line": m["char_edits_per_line"],
           "wer_corpus": m["wer_corpus"], "cer_corpus": m["cer_corpus"],
           "delta_baseline": base["delta"] if base else np.nan, "delta_ci95_lo": base["ci95_lo"] if base else np.nan,
           "delta_ci95_hi": base["ci95_hi"] if base else np.nan, "p_better_baseline": base["p_better"] if base else np.nan,
           "delta_champion": m["score"] - ch_score, "runtime_min": runtime_s / 60, "peak_gpu_gb": peak,
           "n_train": n_train, "n_val": m["n"], "empty": m["empty"], "truncated": trunc,
           "status": "EVALUATED", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for g in ("A", "B"):                                   # per-class score columns for the trial table
        idx = [i for i, c in enumerate(VAL_CLASS) if c == g]
        row[f"score_{g}"] = official_metric([VAL_GT[i] for i in idx], [hyps[i] for i in idx])["score"] if idx else np.nan
    row.update({"lr": CFG.LEARNING_RATE, "aug_profile": CFG.AUG_PROFILE, "class_hint": CFG.CLASS_HINT,
                "class_aug_copies": json.dumps(CFG.CLASS_AUG_COPIES)})
    if extra: row.update(extra)
    prev = pd.read_csv(TRIAL_RESULTS) if os.path.exists(TRIAL_RESULTS) else pd.DataFrame()
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(TRIAL_RESULTS, index=False)
    with open(os.path.join(TRIAL_CFG_DIR, f"{trial}.json"), "w") as f:
        json.dump({"trial": trial, "session": TRIAL_ID, "description": desc, "model": model, "variant": variant,
                   "decode": CFG.DECODE_VARIANTS.get(variant), "cfg": {k: (list(v) if isinstance(v, (set, tuple)) else v)
                   for k, v in TRIAL_CONFIG.items()}}, f, indent=1, default=str)
    return m
''')

# ============================================================================ 13. main loop
code(r'''
# =========================================================
# Cell 12 — train each model on Fold-0 TRAIN, decode ALL Fold-0 val with every decoding variant,
#           decode test with the variants the ensemble / submission need. One model resident at a time.
# =========================================================
import torch.nn as _nn

def _vision_linear_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, _nn.Linear) and any(t in n.lower() for t in ("visual", "vision", "merger"))]

def build_lora_model(dtype, msrc, r=None, alpha=None):
    m = load_vlm_model(msrc, dtype)
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.config.use_cache = False
    if any(s in msrc.lower() for s in CFG.GRAD_CKPT_MODELS):
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if CFG.LORA_VISION:
        vt = _vision_linear_names(m); targets = targets + vt
        print(f"[lora] LORA_VISION=ON -> +{len(vt)} vision/merger modules")
    lora = LoraConfig(r=r or CFG.R, lora_alpha=alpha or CFG.LORA_ALPHA, target_modules=targets,
                      lora_dropout=CFG.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(m, lora); m.print_trainable_parameters()
    return m

def _resolve_adapter(msrc, fold):
    """Exact-name lookup only (new name, then the original notebook's name) — never a fuzzy match,
    because e.g. 8b-instruct/1 and 32b-instruct/1 share every suffix token except the size."""
    names = [model_short_name(msrc), legacy_short_name(msrc)]
    roots = [CFG.ADAPTERS_DIR] if CFG.ADAPTERS_DIR else (sorted(_glob("/kaggle/input/*")) + sorted(_glob("/kaggle/input/*/*")))
    for r in roots:
        for nm in names:
            for c in (os.path.join(r, nm, f"fold{fold}"), os.path.join(r, f"fold{fold}", nm)):
                if os.path.isfile(os.path.join(c, "adapter_config.json")):
                    return c
    for r in roots:
        for cfgp in _glob(os.path.join(r, "**", "adapter_config.json"), recursive=True):
            parts = os.path.dirname(cfgp).replace("\\", "/").split("/")
            if parts[-1] == f"fold{fold}" and len(parts) >= 2 and parts[-2] in names:
                return os.path.dirname(cfgp)
    return None

def load_infer_model(dtype, msrc, mname, fold, merge=True):
    """merge=False keeps the shared adapter as PEFT adapter 'default' so specialists can be added / switched."""
    adir = _resolve_adapter(msrc, fold)
    if adir is None:
        raise FileNotFoundError(f"no adapter for {mname} fold{fold} (ADAPTERS_DIR={CFG.ADAPTERS_DIR!r})")
    print(f"[infer] {mname} fold{fold}: adapter <- {adir}" + ("" if merge else "  (kept unmerged for specialists)"))
    m = PeftModel.from_pretrained(load_vlm_model(msrc, dtype), adir)
    if merge:
        try: m = m.merge_and_unload()
        except Exception as e: print("[infer] merge skipped:", e)
    m.to("cuda" if torch.cuda.is_available() else "cpu"); m.config.use_cache = True; m.eval()
    return m, adir

def wants_specialists(mname):
    return bool(CFG.SPECIALISTS) and (not CFG.SPECIALIST_MODELS or model_size_tag(mname) in CFG.SPECIALIST_MODELS)

def _rows(samples, model_num, n_aug, salt=0):
    rows = [{"image": s["image"], "text": s["text"], "grp": s["grp"], "aug": False, "seed": -1} for s in samples]
    if CFG.USE_AUGMENT:
        for _j, s in enumerate(samples):
            for _c in range(n_aug(s)):
                rows.append({"image": s["image"], "text": s["text"], "grp": s["grp"], "aug": True,
                             "seed": int((CFG.SEED + 1) * 2000003 + CFG.FOLD * 1000003 + model_num + _c * 9973 + _j
                                         + salt * 7777777) & 0x7fffffff})
    return rows

def train_specialist(model, proc, mname, key, spec, HP, model_num, shared_dir, save_dir):
    """Adds PEFT adapter 'spec_<key>' to the (unmerged) PeftModel and trains it on key's lines + replay."""
    sp = {**CFG.SPEC_DEFAULTS, **(spec or {})}
    name = f"spec_{key}"
    pool = core_samples + pseudo_samples
    own = [s for s in pool if key_match(s["grp"], key)]
    others = [s for s in pool if not key_match(s["grp"], key)]
    rows = _rows(own, model_num, lambda s: CFG.AUG_COPIES, salt=11)
    rng = np.random.RandomState(CFG.SEED + 97 + model_num)
    n_rep = int(round(sp["replay"] / max(1e-9, 1 - sp["replay"]) * len(rows))) if sp["replay"] > 0 else 0
    if n_rep and others:
        pick = rng.choice(len(others), size=min(n_rep, len(others)), replace=False)
        rows += [{"image": others[i]["image"], "text": others[i]["text"], "grp": others[i]["grp"], "aug": True,
                  "seed": int(rng.randint(0, 2**31 - 1))} for i in pick]
    if sp["init"] == "continue":
        model.load_adapter(shared_dir, adapter_name=name, is_trainable=True)
    elif sp["init"] == "separate":
        model.add_adapter(name, copy.deepcopy(model.peft_config["default"]))
    else:
        raise ValueError(f"SPECIALISTS[{key!r}]['init'] must be 'continue' or 'separate'")
    model.set_adapter(name)
    model.config.use_cache = False
    if any(t in CFG.MODEL_PATHS[model_num].lower() for t in CFG.GRAD_CKPT_MODELS):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try: model.enable_input_require_grads()
    except Exception: pass
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{mname}] specialist {name}: init={sp['init']} lines={len(own)} rows={len(rows)} "
          f"(replay {len(rows) - len(own) * (1 + CFG.AUG_COPIES)}) lr={sp['lr']} trainable={n_tr/1e6:.1f}M")
    args = TrainingArguments(
        output_dir=os.path.join(CFG.OUTPUT_DIR, f"{mname}_{name}"),
        per_device_train_batch_size=HP["PER_DEVICE_TRAIN_BATCH_SIZE"],
        gradient_accumulation_steps=HP["GRADIENT_ACCUMULATION_STEPS"],
        learning_rate=sp["lr"], num_train_epochs=sp["epochs"], max_steps=int(sp["max_steps"] or -1),
        lr_scheduler_type="constant_with_warmup", warmup_steps=int(sp["warmup_steps"]),
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16), logging_steps=10,
        eval_strategy="no", save_strategy="no", remove_unused_columns=False,
        dataloader_num_workers=CFG.DATALOADER_NUM_WORKERS, dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none", seed=CFG.SEED + 5)
    tr = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                 data_collator=lambda x, _p=proc: collate_fn(x, _p))
    tr.train()
    try:
        pd.DataFrame(tr.state.log_history).to_csv(os.path.join(CFG.WORK_DIR, f"log_history__{mname}__{name}.csv"), index=False)
    except Exception: pass
    if CFG.KEEP_ADAPTERS:                      # PEFT writes non-default adapters to <save_dir>/<adapter_name>/
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, selected_adapters=[name])
        print(f"[{mname}] specialist saved -> {os.path.join(save_dir, name)}")
    del tr; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, f"{mname}_{name}"), ignore_errors=True)
    try: model.gradient_checkpointing_disable()
    except Exception: pass
    model.config.use_cache = True
    model.set_adapter("default"); model.eval()
    return name

dtype = torch.bfloat16 if (CFG.USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
VAL_NBEST, TEST_NBEST, UNL_NBEST, MODEL_ORDER = {}, {}, {}, []   # (mname, variant) -> list of N-best lists
SPEC_VAL_NBEST, SPEC_TEST_NBEST = {}, {}                          # (mname, key, variant) -> full-length N-best lists
                                                                  #   (specialist on key's lines, shared elsewhere)
TRAIN_SIZE = len(core_samples)

for model_num, msrc in enumerate(CFG.MODEL_PATHS):
    if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
        print(f"[time] budget {CFG.TIME_BUDGET_H}h reached — skipping {msrc}"); continue
    mname = model_short_name(msrc); MODEL_ORDER.append(mname)
    print(f"\n=========== MODEL {mname} ({CFG.RUN_MODE}, fold {CFG.FOLD}) ===========")
    t_model = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    proc = load_vlm_processor(msrc, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
    _ov = CFG.MODEL_OVERRIDES.get(model_size_tag(mname), {})
    HP = {k: _ov.get(k, getattr(CFG, k)) for k in ("LEARNING_RATE", "NUM_TRAIN_EPOCHS", "R", "LORA_ALPHA",
                                                    "PER_DEVICE_TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS")}
    print(f"[{mname}] hyper-parameters: {HP}" + (f"  (overrides: {_ov})" if _ov else ""))
    if CFG.DO_TRAIN:
        _train_src = core_samples + pseudo_samples
        _copies = CFG.CLASS_AUG_COPIES or {}
        _extra = {}
        if CFG.CLASS_SAMPLING_T:                     # share of group g ∝ n_g^(1/T)  <=>  repeat factor (n_max/n_g)^(1-1/T)
            _n = collections.Counter(s["grp"] for s in _train_src); _nmax = max(_n.values())
            _f = {g: (_nmax / n) ** (1 - 1 / float(CFG.CLASS_SAMPLING_T)) for g, n in _n.items()}
            _rs = np.random.RandomState(CFG.SEED + 31 + model_num)
            _extra = {id(s): int(np.floor(_f[s["grp"]] - 1) + (_rs.rand() < (_f[s["grp"]] - 1) % 1)) for s in _train_src}
            print(f"[{mname}] class-balanced sampling T={CFG.CLASS_SAMPLING_T}: repeat factors",
                  {g: round(v, 2) for g, v in sorted(_f.items())})
        rows = _rows(_train_src, model_num,
                     lambda s: _copies.get(s["grp"], _copies.get(s["grp"][0], CFG.AUG_COPIES)) + _extra.get(id(s), 0))
        print(f"[{mname}] training rows by group (orig+aug):", dict(sorted(collections.Counter(r["grp"] for r in rows).items())))
        train_ds = Dataset.from_list(rows)
        eval_ds = Dataset.from_list([{"image": s["image"], "text": s["text"], "grp": s["grp"], "aug": False, "seed": -1}
                                     for s in inner_samples]) if inner_samples else None
        print(f"[{mname}] train rows={len(train_ds)} inner_ckpt={len(inner_samples)} val(scored)={len(VAL_IDS)}")
        model = build_lora_model(dtype, msrc, r=HP["R"], alpha=HP["LORA_ALPHA"])
        sel = eval_ds is not None
        args = TrainingArguments(
            output_dir=os.path.join(CFG.OUTPUT_DIR, mname),
            per_device_train_batch_size=HP["PER_DEVICE_TRAIN_BATCH_SIZE"],
            gradient_accumulation_steps=HP["GRADIENT_ACCUMULATION_STEPS"],
            learning_rate=HP["LEARNING_RATE"], num_train_epochs=HP["NUM_TRAIN_EPOCHS"],
            lr_scheduler_type=CFG.LR_SCHEDULER, warmup_ratio=CFG.WARMUP_RATIO,
            bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16), logging_steps=10,
            eval_strategy="steps" if sel else "no", eval_steps=CFG.EVAL_STEPS,
            save_strategy="steps" if sel else "no", save_steps=CFG.EVAL_STEPS, save_total_limit=CFG.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=sel, metric_for_best_model="eval_loss", greater_is_better=False,
            remove_unused_columns=False, dataloader_num_workers=CFG.DATALOADER_NUM_WORKERS,
            dataloader_pin_memory=torch.cuda.is_available(), report_to="none", seed=CFG.SEED,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                          data_collator=lambda x, _p=proc: collate_fn(x, _p))
        print(f"[{mname}] training..."); trainer.train()
        print(f"[{mname}] label-mask failures: {MASK_FAIL_COUNT}")
        _ad = os.path.join(CFG.FINAL_DIR if CFG.KEEP_ADAPTERS else os.path.join(CFG.OUTPUT_DIR, "_tmp_shared"), mname,
                           f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full")
        if CFG.KEEP_ADAPTERS or wants_specialists(mname):
            os.makedirs(_ad, exist_ok=True); trainer.save_model(_ad); proc.save_pretrained(_ad)
        SHARED_DIR = _ad
        try:
            pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(CFG.WORK_DIR, f"log_history__{mname}.csv"), index=False)
        except Exception: pass
        del trainer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, mname), ignore_errors=True)
        model.config.use_cache = True
        try: model.gradient_checkpointing_disable()
        except Exception: pass
        model.eval()
    else:
        model, SHARED_DIR = load_infer_model(dtype, msrc, mname, CFG.FOLD if CFG.RUN_MODE == "fold0" else "full",
                                             merge=not wants_specialists(mname))
    t_train = time.time() - t_model

    if CFG.DO_INFER and CFG.RUN_MODE == "fold0":
        base_hyps = None
        scores = {}
        for vname, var in CFG.DECODE_VARIANTS.items():
            t0 = time.time()
            nb = transcribe(model, proc, VAL_PATHS, var, desc=f"val/{mname}/{vname}")
            VAL_NBEST[(mname, vname)] = nb
            top1 = [x[0][0] for x in nb]
            trunc = sum(x[0][3] for x in nb)
            if vname == CFG.BASELINE_VARIANT: base_hyps = top1
            trial = f"{TRIAL_ID}__{mname}__{vname}"
            save_predictions(trial, VAL_IDS, VAL_GT, nb)
            m = report_trial(trial, TRIAL_DESCRIPTION, mname, vname, top1, t_train + time.time() - t0, TRAIN_SIZE,
                             trunc, baseline_hyps=base_hyps if vname != CFG.BASELINE_VARIANT else None,
                             extra={"lr": HP["LEARNING_RATE"]})
            scores[vname] = m["score"]
            if int(var.get("n_return", 1)) > 1:           # single-model MBR over this variant's N-best
                mbr = [mbr_select([t for t, *_ in x], nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)) for x in nb]
                report_trial(f"{trial}_mbr", TRIAL_DESCRIPTION, mname, f"{vname}+mbr", mbr, time.time() - t0, TRAIN_SIZE,
                             trunc, baseline_hyps=base_hyps, extra={"lr": HP["LEARNING_RATE"]})
                scores[f"{vname}+mbr"] = official_metric(VAL_GT, mbr)["score"]
                save_predictions(f"{trial}_mbr", VAL_IDS, VAL_GT, nb, chosen=mbr)
        best_var = max((v for v in scores if "+mbr" not in v), key=scores.get)
        print(f"[{mname}] best decoding on val: {best_var} ({scores[best_var]:.5f}) | all: "
              + ", ".join(f"{k}={v:.5f}" for k, v in scores.items()))
        test_vars = {best_var} | {v for v, c in CFG.DECODE_VARIANTS.items() if int(c.get("n_return", 1)) > 1}
    else:
        test_vars = {CFG.CHAMPION_DECODE} if CFG.CHAMPION_DECODE in CFG.DECODE_VARIANTS else set(CFG.DECODE_VARIANTS)
        test_vars |= {s.rpartition(":")[0] for s in (CFG.ROUTING_MAP or {}).values() if s.rpartition(":")[0] in CFG.DECODE_VARIANTS}

    if CFG.DO_INFER and CFG.PREDICT_TEST:
        for vname in sorted(test_vars):
            nb = transcribe(model, proc, TEST_PATHS, CFG.DECODE_VARIANTS[vname], desc=f"test/{mname}/{vname}")
            TEST_NBEST[(mname, vname)] = nb
            save_predictions(f"{TRIAL_ID}__{mname}__{vname}", TEST_IDS, None, nb, split="test")
    if CFG.DO_INFER and CFG.PREDICT_UNLISTED and UNL_IDS:
        for vname in sorted(v for v in test_vars if int(CFG.DECODE_VARIANTS[v].get("n_return", 1)) > 1):
            nb = transcribe(model, proc, UNL_PATHS, CFG.DECODE_VARIANTS[vname], desc=f"unlisted/{mname}/{vname}")
            UNL_NBEST[(mname, vname)] = nb
            save_predictions(f"{TRIAL_ID}__{mname}__{vname}", UNL_IDS, None, nb, split="unlisted")

    # ---------------- per-class / per-sub-class specialists (shared decoding above is untouched) ----------------
    if wants_specialists(mname) and CFG.DO_INFER:
        for key, spec in CFG.SPECIALISTS.items():
            name = f"spec_{key}"
            save_dir = os.path.join(CFG.FINAL_DIR, mname, f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full")
            found = [d for d in (os.path.join(SHARED_DIR, name), os.path.join(save_dir, name))
                     if os.path.isfile(os.path.join(d, "adapter_config.json"))]
            if CFG.DO_TRAIN or CFG.TRAIN_SPECIALISTS:
                t_sp = time.time()
                train_specialist(model, proc, mname, key, spec, HP, model_num, SHARED_DIR, save_dir)
                t_sp = time.time() - t_sp
            elif found:
                model.load_adapter(found[0], adapter_name=name); t_sp = 0.0
                print(f"[{mname}] specialist {name} <- {found[0]}")
            else:
                print(f"[{mname}] specialist {name}: no saved adapter and TRAIN_SPECIALISTS=False -> skipped"); continue
            model.set_adapter(name)
            vi = [i for i, g in enumerate(VAL_GROUP) if key_match(g, key)]
            ti = [i for i, g in enumerate(TEST_GROUP) if key_match(g, key)]
            for vname in sorted(test_vars):
                var = CFG.DECODE_VARIANTS[vname]
                t0 = time.time()
                if vi and (mname, vname) in VAL_NBEST:
                    sub = transcribe(model, proc, [VAL_PATHS[i] for i in vi], var, desc=f"val/{mname}/{name}/{vname}")
                    comb = list(VAL_NBEST[(mname, vname)])
                    for j, i in enumerate(vi):
                        comb[i] = sub[j]
                    SPEC_VAL_NBEST[(mname, key, vname)] = comb
                    shared_top = [x[0][0] for x in VAL_NBEST[(mname, vname)]]
                    d = paired_delta([VAL_GT[i] for i in vi], [shared_top[i] for i in vi], [sub[j][0][0] for j in range(len(vi))])
                    print(f"[{mname}] {name}/{vname} on its {len(vi)} val lines vs shared: delta={d['delta']:+.5f} "
                          f"CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}] P(better)={d['p_better']:.3f}")
                    trial = f"{TRIAL_ID}__{mname}@{key}__{vname}"
                    save_predictions(trial, VAL_IDS, VAL_GT, comb)
                    report_trial(trial, TRIAL_DESCRIPTION, f"{mname}@{key}", vname, [x[0][0] for x in comb],
                                 t_sp + time.time() - t0, TRAIN_SIZE, sum(x[0][3] for x in comb), baseline_hyps=shared_top,
                                 extra={"lr": {**CFG.SPEC_DEFAULTS, **(spec or {})}["lr"], "specialist": key,
                                        "spec_init": {**CFG.SPEC_DEFAULTS, **(spec or {})}["init"],
                                        "spec_delta_on_key": d["delta"], "spec_p_better_on_key": d["p_better"]})
                if CFG.PREDICT_TEST and ti and (mname, vname) in TEST_NBEST:
                    sub = transcribe(model, proc, [TEST_PATHS[i] for i in ti], var, desc=f"test/{mname}/{name}/{vname}")
                    comb = list(TEST_NBEST[(mname, vname)])
                    for j, i in enumerate(ti):
                        comb[i] = sub[j]
                    SPEC_TEST_NBEST[(mname, key, vname)] = comb
                    save_predictions(f"{TRIAL_ID}__{mname}@{key}__{vname}", TEST_IDS, None, comb, split="test")
            model.set_adapter("default")
    free_cuda("model"); del proc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[{mname}] done in {(time.time()-t_model)/60:.1f} min")
''')

# ============================================================================ 14. ensembles + submission
code(r'''
# =========================================================
# Cell 13 — cross-model ensembles on val (each = one trial), champion selection, submission
# =========================================================
def _pool(sys_nb):
    cands, w = [], []
    for x in sys_nb:
        p = nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)
        cands += [t for t, *_ in x]; w += list(p / len(sys_nb))
    return cands, w

def _ensemble(nb_by_model, method):
    """nb_by_model: list over systems of per-line N-best lists -> per-line chosen text."""
    n = len(nb_by_model[0]); out = []
    for i in range(n):
        sys_nb = [nb[i] for nb in nb_by_model]
        if method in ("mbr_nbest", "single_mbr"):
            out.append(mbr_select(*_pool(sys_nb)))
        elif method == "mbr_top1":
            out.append(mbr_select([x[0][0] for x in sys_nb]))
        elif method == "word_rover":
            out.append(rover([x[0][0] for x in sys_nb], level="word"))
        elif method == "char_rover":
            out.append(rover([x[0][0] for x in sys_nb], level="char"))
    return out

CANDIDATES = {}          # name -> (val_hyps or None, test_hyps or None)
for (mn, vn), nb in VAL_NBEST.items():
    CANDIDATES[f"{mn}__{vn}"] = ([x[0][0] for x in nb], [x[0][0] for x in TEST_NBEST[(mn, vn)]] if (mn, vn) in TEST_NBEST else None)
    if int(CFG.DECODE_VARIANTS[vn].get("n_return", 1)) > 1:
        f = lambda xs: [mbr_select([t for t, *_ in x], nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)) for x in xs]
        CANDIDATES[f"{mn}__{vn}+mbr"] = (f(nb), f(TEST_NBEST[(mn, vn)]) if (mn, vn) in TEST_NBEST else None)

# ---- specialist candidates: the specialist alone, its N-best MBR, and cross-model MBR with it as an extra voter ----
_top1 = lambda xs: [x[0][0] for x in xs]
_smbr = lambda xs: [mbr_select([t for t, *_ in x], nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)) for x in xs]
for (mn, key, vn) in sorted(set(list(SPEC_VAL_NBEST) + list(SPEC_TEST_NBEST))):
    vnb, tnb = SPEC_VAL_NBEST.get((mn, key, vn)), SPEC_TEST_NBEST.get((mn, key, vn))
    nm = f"{mn}@{key}__{vn}"
    CANDIDATES[nm] = (_top1(vnb) if vnb else None, _top1(tnb) if tnb else None)
    if int(CFG.DECODE_VARIANTS[vn].get("n_return", 1)) > 1:
        CANDIDATES[nm + "+mbr"] = (_smbr(vnb) if vnb else None, _smbr(tnb) if tnb else None)
        systems = [m for m in MODEL_ORDER if (m, vn) in (VAL_NBEST if vnb else TEST_NBEST)]
        if systems:
            en = f"ENS[{'+'.join(model_size_tag(s_) for s_ in systems)}+{model_size_tag(mn)}@{key}]__{vn}__mbr_nbest"
            vh = _ensemble([VAL_NBEST[(m, vn)] for m in systems] + [vnb], "mbr_nbest") if vnb else None
            th = (_ensemble([TEST_NBEST[(m, vn)] for m in systems] + [tnb], "mbr_nbest")
                  if tnb and all((m, vn) in TEST_NBEST for m in systems) else None)
            CANDIDATES[en] = (vh, th)
            if vh is not None and CFG.RUN_MODE == "fold0":
                report_trial(f"{TRIAL_ID}__{en}", TRIAL_DESCRIPTION, en, f"{vn}/mbr_nbest+specialist", vh, 0.0, TRAIN_SIZE,
                             baseline_hyps=CANDIDATES.get(f"{MODEL_ORDER[0]}__{CFG.BASELINE_VARIANT}", (None,))[0],
                             extra={"specialist": key})

_base_name = f"{MODEL_ORDER[0]}__{CFG.BASELINE_VARIANT}" if MODEL_ORDER else None
_base_hyps = CANDIDATES[_base_name][0] if _base_name in CANDIDATES else None
if CFG.RUN_MODE == "fold0" and len({m for m, _ in VAL_NBEST}) >= 2:
    base_hyps = _base_hyps
    for vname in CFG.DECODE_VARIANTS:
        systems = [m for m in MODEL_ORDER if (m, vname) in VAL_NBEST]
        if len(systems) < 2: continue
        for method in CFG.ENSEMBLE_METHODS:
            if method == "mbr_nbest" and int(CFG.DECODE_VARIANTS[vname].get("n_return", 1)) < 2: continue
            t0 = time.time()
            vh = _ensemble([VAL_NBEST[(m, vname)] for m in systems], method)
            th = _ensemble([TEST_NBEST[(m, vname)] for m in systems], method) if all((m, vname) in TEST_NBEST for m in systems) else None
            name = f"ENS[{'+'.join(model_size_tag(s_) for s_ in systems)}]__{vname}__{method}"
            CANDIDATES[name] = (vh, th)
            report_trial(f"{TRIAL_ID}__{name}", TRIAL_DESCRIPTION, "+".join(systems), f"{vname}/{method}", vh,
                         time.time() - t0, TRAIN_SIZE, baseline_hyps=base_hyps)
            pd.DataFrame({"ID": VAL_IDS, "GroundTruth": VAL_GT, "pred": vh}).to_csv(
                os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__{name}__val.csv"), index=False)

# ---- per-class routing with the ACCEPTANCE RULE (never plain per-class argmax: that loses when nothing
#      is truly better on a class, while reporting a gain) ----
def _spec_of(name):
    """candidate name -> (variant, ROUTING_MAP target) usable in RUN_MODE='full'."""
    if name.startswith("ENS["):
        inside, rest = name[4:name.index("]")], name[name.index("]") + 3:]
        vn, method = rest.split("__")
        spec = [t for t in inside.split("+") if "@" in t]
        return vn, method + ("+" + spec[0] if spec else "")
    mn, vn = name.split("__")[:2]
    mbr = vn.endswith("+mbr"); vn = vn.replace("+mbr", "")
    if "@" in mn:
        base, key = mn.split("@")
        tag = f"{model_size_tag(base)}@{key}"
    else:
        tag = model_size_tag(mn)
    return vn, tag + ("+mbr" if mbr else "")

def _decide(idx, pool, loss, kv, champ, cands, share, rng, verbose=False):
    """per-class policy {class: candidate} chosen with the 4-condition rule using only lines idx."""
    classes = sorted(set(kv[idx].tolist()))
    st, glob = {}, {}
    for c in classes:
        ic = idx[kv[idx] == c]
        for k in cands[c]:
            d = loss[champ][ic] - loss[k][ic]
            st[(c, k)] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf, d)
    for k in {k for c in classes for k in cands[c]}:
        glob[k] = float((loss[champ][idx] - loss[k][idx]).mean())
    I = np.array([st[ck][0] - glob[ck[1]] for ck in st]); SE2 = np.array([st[ck][1] ** 2 for ck in st])
    tau2 = max(0.0, float(np.mean(I ** 2) - np.mean(SE2[np.isfinite(SE2)]))) if len(I) else 0.0
    policy, table = {}, []
    for c in classes:
        best, best_gain = champ, 0.0
        for k in cands[c]:
            dm, se, d = st[(c, k)]
            n = len(d)
            boots = d[rng.integers(0, n, (2000, n))].mean(1) if n > 1 else np.array([dm])
            p = float((boots > 0).mean())
            w = tau2 / (tau2 + se ** 2) if (tau2 > 0 and np.isfinite(se)) else 0.0
            shrunk = glob[k] + w * (dm - glob[k])
            sign = 0.0
            if n >= 4:
                ok_ = 0
                for _ in range(20):
                    pm = rng.permutation(n); h1, h2 = d[pm[: n // 2]], d[pm[n // 2:]]
                    ok_ += (h1.mean() > 0) and (h2.mean() > 0)
                sign = ok_ / 20
            accept = (p >= CFG.ROUTE_P_BETTER and shrunk > 0 and sign >= 0.5
                      and share.get(c, 0) * dm >= CFG.ROUTE_MIN_OVERALL_GAIN)
            table.append((c, k, dm, p, shrunk, sign, share.get(c, 0) * dm, accept))
            if accept and shrunk > best_gain:
                best, best_gain = k, shrunk
        policy[c] = best
    if verbose:
        print(f"   tau^2 (class x candidate interaction variance) = {tau2:.2e}"
              + ("  -> no class-specific advantage in the data: one model for all" if tau2 == 0 else ""))
        print(f"   {'class':5s} {'challenger':60s} {'delta':>8s} {'P(>)':>6s} {'shrunk':>8s} {'sign':>5s} {'pi*d':>7s}  accept")
        for c, k, dm, p, sh, sg, pd_, a in table:
            print(f"   {c:5s} {k[:60]:60s} {dm:+8.4f} {p:6.3f} {sh:+8.4f} {sg:5.2f} {pd_:+7.4f}  {'YES' if a else '-'}")
    return policy

if CFG.RUN_MODE == "fold0" and CFG.CLASS_ROUTING:
    pool = {k: v for k, v in CANDIDATES.items() if v[0] is not None and v[1] is not None}
    if len(pool) >= 2:
        kv = np.array(VAL_CLASS if CFG.ROUTING_LEVEL == "class" else VAL_GROUP)
        kt = TEST_CLASS if CFG.ROUTING_LEVEL == "class" else TEST_GROUP
        share = {c: n / len(kt) for c, n in collections.Counter(kt).items()}
        loss = {k: np.array([line_cost(r, h) for r, h in zip(VAL_GT, v[0])]) for k, v in pool.items()}
        champ = min(pool, key=lambda k: loss[k].mean())
        overall_rank = sorted((k for k in pool if k != champ), key=lambda k: loss[k].mean())
        def _spec_key(name):
            head = name[4:name.index("]")].split("+") if name.startswith("ENS[") else [name.split("__")[0]]
            keys = [t.split("@")[1] for t in head if "@" in t]
            return keys[0] if keys else None
        def _relevant(c, key):         # specialist `key` covers (part of) routing unit `c`
            return key is not None and (key_match(c, key) or key_match(key, c) if len(key) > 1 or len(c) > 1 else c == key)
        cands = {}
        for c in sorted(set(kv.tolist())):
            spec_rel = [k for k in overall_rank if _relevant(c, _spec_key(k))]
            others = [k for k in overall_rank if _spec_key(k) is None]
            cands[c] = (spec_rel[:max(1, CFG.ROUTE_MAX_CANDIDATES - 1)] + others)[:CFG.ROUTE_MAX_CANDIDATES]
        rng = np.random.default_rng(CFG.SEED)
        print(f"\nPER-{CFG.ROUTING_LEVEL.upper()} ROUTING (acceptance rule) — champion: {champ} = {1 - loss[champ].mean():.5f}")
        full_policy = _decide(np.arange(len(VAL_IDS)), pool, loss, kv, champ, cands, share, rng, verbose=True)
        # honest score of the WHOLE procedure: decide on one half, apply to the other, repeated
        hs, first = [], None
        for r_ in range(CFG.ROUTE_SPLITS):
            pm = rng.permutation(len(VAL_IDS)); halves = (pm[0::2], pm[1::2])
            hyp = [None] * len(VAL_IDS)
            for a, b in ((0, 1), (1, 0)):
                pol = _decide(halves[a], pool, loss, kv, champ, cands, share, rng)
                for i in halves[b]:
                    hyp[i] = pool[pol.get(kv[i], champ)][0][i]
            hs.append(official_metric(VAL_GT, hyp)["score"])
            first = first or hyp
        test_h = [pool[full_policy.get(c, champ)][1][i] for i, c in enumerate(kt)]
        rname = f"ROUTED_by_{CFG.ROUTING_LEVEL}"
        CANDIDATES[rname] = (first, test_h)
        switched = {c: k for c, k in full_policy.items() if k != champ}
        print(f"   policy on all val: " + (", ".join(f"{c} -> {k}" for c, k in switched.items()) if switched
                                          else "keep the champion for every class"))
        print(f"   honest split-half score over {CFG.ROUTE_SPLITS} splits: {np.mean(hs):.5f} ± {np.std(hs):.5f} "
              f"(champion alone {1 - loss[champ].mean():.5f})")
        report_trial(f"{TRIAL_ID}__{rname}", TRIAL_DESCRIPTION, json.dumps(full_policy)[:200], "routed (acceptance rule)",
                     first, 0.0, TRAIN_SIZE, baseline_hyps=_base_hyps,
                     extra={"honest_score_mean": float(np.mean(hs)), "honest_score_sd": float(np.std(hs))})
        pd.DataFrame({"ID": VAL_IDS, "GroundTruth": VAL_GT, "pred": first}).to_csv(
            os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__{rname}__val.csv"), index=False)
        specs = {c: "{}:{}".format(*_spec_of(full_policy.get(c, champ))) for c in sorted(set(kt) | set(kv.tolist()))}
        print(f"   -> for RUN_MODE='full' set ROUTING_MAP={specs}  (ROUTING_LEVEL={CFG.ROUTING_LEVEL!r}) "
              f"and SPECIALISTS with the same keys")
        with open(os.path.join(CFG.WORK_DIR, "routing_map.json"), "w") as f:
            json.dump({"ROUTING_LEVEL": CFG.ROUTING_LEVEL, "ROUTING_MAP": specs, "champion": champ,
                       "SPECIALISTS": CFG.SPECIALISTS}, f, indent=1)

# ---- choose the best validated configuration that has test predictions ----
if CFG.RUN_MODE == "fold0":
    ranked = sorted(((official_metric(VAL_GT, v)["score"], k) for k, (v, t) in CANDIDATES.items() if v is not None), reverse=True)
    print("\nVALIDATION RANKING (official score):")
    for s, k in ranked[:15]:
        print(f"  {s:.5f}  {k}{'' if CANDIDATES[k][1] is not None else '   (no test preds)'}")
    best = next((k for s, k in ranked if CANDIDATES[k][1] is not None), None)
else:                                      # "full": reproduce the Fold-0 champion recipe on test
    vn = CFG.CHAMPION_DECODE
    systems = [m for m in MODEL_ORDER if (m, vn) in TEST_NBEST]
    method = CFG.CHAMPION_ENSEMBLE
    if isinstance(CFG.ROUTING_MAP, str) and os.path.exists(CFG.ROUTING_MAP):
        _rm = json.load(open(CFG.ROUTING_MAP)); CFG.ROUTING_MAP = _rm["ROUTING_MAP"]; CFG.ROUTING_LEVEL = _rm["ROUTING_LEVEL"]
        print("ROUTING_MAP loaded:", CFG.ROUTING_MAP, "| level:", CFG.ROUTING_LEVEL)
    _outs = {}
    _METHODS = ("mbr_nbest", "mbr_top1", "word_rover", "char_rover")
    def _nb_of(tag, v):
        """'8b' -> shared N-best of that model; '8b@B' -> that model's specialist-B N-best (shared elsewhere)."""
        base, _, key = tag.partition("@")
        ms = [m for m in MODEL_ORDER if model_size_tag(m) == base and (m, v) in TEST_NBEST] or \
             [m for m in MODEL_ORDER if (m, v) in TEST_NBEST][:1]
        if key:
            assert (ms[0], key, v) in SPEC_TEST_NBEST, f"specialist {tag} ({v}) missing — add {key!r} to SPECIALISTS"
            return SPEC_TEST_NBEST[(ms[0], key, v)]
        return TEST_NBEST[(ms[0], v)]
    def _spec_out(spec):
        """ROUTING_MAP value 'variant:target' -> test outputs (grammar documented at CFG.ROUTING_MAP)."""
        v, _, target = spec.rpartition(":")
        v = v or vn
        if spec not in _outs:
            sy = [m for m in MODEL_ORDER if (m, v) in TEST_NBEST]
            head, _, extra = target.partition("+")
            if head in _METHODS:
                nbs = [TEST_NBEST[(m, v)] for m in sy] + ([_nb_of(extra, v)] if extra else [])
                _outs[spec] = _ensemble(nbs, head) if len(nbs) >= 2 else _top1(nbs[0])
            else:
                nb0 = _nb_of(target.replace("+mbr", ""), v)
                _outs[spec] = _ensemble([nb0], "single_mbr") if target.endswith("+mbr") else _top1(nb0)
        return _outs[spec]
    if systems and CFG.ROUTING_MAP:
        keys = TEST_CLASS if CFG.ROUTING_LEVEL == "class" else TEST_GROUP
        default = f"{vn}:{method or model_size_tag(systems[0])}"
        best = f"ROUTED{CFG.ROUTING_MAP}"
        CANDIDATES[best] = (None, [_spec_out(CFG.ROUTING_MAP.get(k, default))[i] for i, k in enumerate(keys)])
    elif systems and method in ("mbr_nbest", "mbr_top1", "word_rover", "char_rover") and len(systems) >= 2:
        best = f"ENS[{'+'.join(model_size_tag(s_) for s_ in systems)}]__{vn}__{method}"
        CANDIDATES[best] = (None, _ensemble([TEST_NBEST[(m, vn)] for m in systems], method))
    elif systems:
        best = f"{systems[0]}__{vn}" + ("+mbr" if method else "")
        nb0 = TEST_NBEST[(systems[0], vn)]
        CANDIDATES[best] = (None, _ensemble([nb0], "single_mbr") if method else [x[0][0] for x in nb0])
    else:
        best = None

# ---- pseudo-labels for test + unlisted: MBR over every model's N-best, risk = expected official cost ----
_pl_rows = []
for split_ids, store in ((TEST_IDS, TEST_NBEST), (UNL_IDS, UNL_NBEST)):
    keys = [k for k in store if int(CFG.DECODE_VARIANTS[k[1]].get("n_return", 1)) > 1]
    if not keys:
        continue
    for i, rid in enumerate(split_ids):
        txt, risk = mbr_select(*_pool([store[k][i] for k in keys]), return_risk=True)
        _pl_rows.append({"ID": rid, "Target": txt, "risk": risk, "n_systems": len(keys)})
if _pl_rows:
    _pl = pd.DataFrame(_pl_rows)
    _pl.to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels.csv"), index=False)
    print(f"pseudo_labels.csv: {len(_pl)} lines from {_pl.n_systems.max()} system(s); "
          f"risk quantiles 50/70/90% = {np.round(_pl.risk.quantile([.5, .7, .9]).values, 4).tolist()}")

if best is not None and CANDIDATES[best][1] is not None:
    sub = pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(t) or "the" for t in CANDIDATES[best][1]]})
    sub.to_csv(CFG.SUBMISSION_CSV, index=False)
    print(f"\nsubmission.csv <- {best}  | rows={len(sub)} | empty-replaced={sum(not norm_text(t) for t in CANDIDATES[best][1])}")
    with open(os.path.join(CFG.WORK_DIR, "submission_source.json"), "w") as f:
        json.dump({"trial_session": TRIAL_ID, "source": best}, f)

# ---- champion / challenger bookkeeping over ALL sessions ----
if os.path.exists(TRIAL_RESULTS):
    res = pd.read_csv(TRIAL_RESULTS)
    hist = _prev_results()
    base_rows = res[(res.session == TRIAL_ID) & (res.variant == CFG.BASELINE_VARIANT)]
    prev_champ = hist[(hist.session != TRIAL_ID) & hist.status.isin(["PROMOTED", "BASELINE"])] if len(hist) and "session" in hist else pd.DataFrame()
    bar = prev_champ.score.max() if len(prev_champ) else (base_rows.score.max() if len(base_rows) else -1)
    sess = res.session == TRIAL_ID
    res.loc[sess & (res.variant == CFG.BASELINE_VARIANT), "status"] = "BASELINE"
    best_i = res[sess].score.idxmax()
    res.loc[sess & (res.index != best_i) & (res.status == "EVALUATED"), "status"] = "REJECTED"
    res.loc[best_i, "status"] = "PROMOTED" if res.loc[best_i, "score"] > bar else res.loc[best_i, "status"]
    res.to_csv(TRIAL_RESULTS, index=False)
    print("\nTRIAL TABLE (this session):")
    print(res[sess][["trial", "variant", "score", "word_edits_per_line", "char_edits_per_line", "delta_baseline",
                     "p_better_baseline", "runtime_min", "status"]].to_string(index=False))
''')

# ============================================================================ 15. forensics
code(r'''
# =========================================================
# Cell 14 — error forensics: champion vs baseline (where did the change help / hurt?)
# =========================================================
if CFG.RUN_MODE == "fold0" and CANDIDATES:
    vals = {k: v for k, (v, t) in CANDIDATES.items() if v is not None}
    champ = max(vals, key=lambda k: official_metric(VAL_GT, vals[k])["score"])
    base_k = f"{MODEL_ORDER[0]}__{CFG.BASELINE_VARIANT}" if MODEL_ORDER else None
    print("CHAMPION:", champ)
    fx = error_forensics(VAL_GT, vals[champ])
    for k, v in fx.items():
        print(f"  {k}: {v}")
    if base_k in vals and base_k != champ:
        d = paired_delta(VAL_GT, vals[base_k], vals[champ])
        print("\nchampion vs baseline:", d)
        lb = np.array([line_cost(r, h) for r, h in zip(VAL_GT, vals[base_k])])
        lc = np.array([line_cost(r, h) for r, h in zip(VAL_GT, vals[champ])])
        for gname, g in (("class", VAL_CLASS), ("group", VAL_GROUP), ("len", VAL_LEN_BUCKET)):
            g = np.asarray(g)
            print(f"  delta by {gname}:", {str(k): round(float((lb - lc)[g == k].mean()), 5) for k in sorted(set(g.tolist()), key=str)})
        worst = np.argsort(lc - lb)[::-1][:10]
        print("\n  lines most WORSENED by the champion:")
        for i in worst:
            if lc[i] > lb[i]:
                print(f"   {VAL_IDS[i]} | GT: {VAL_GT[i]}\n      base: {vals[base_k][i]}\n      new : {vals[champ][i]}")
    # the 1.5% suspected label-noise lines: very high loss AND fluent output
    lc = np.array([line_cost(r, h) for r, h in zip(VAL_GT, vals[champ])])
    print(f"\n  loss share of worst 2% lines: {np.sort(lc)[::-1][:max(1, len(lc)//50)].sum() / lc.sum():.3f}")
''')

md(r'''
### Next trials (from `RESEARCH_PLAN.md`)
This session (**EXP_002**, the defaults above) = realistic augmentation for every model + 32B at LR 1e-4, with decoding challengers, cross-model ensembles and class routing.
The class-aware *training* options are **off** by default so that each one is tested alone against the EXP_002 champion (usually 8B only, about 1 h each):

| Trial | Change (CFG) | Hypothesis |
|---|---|---|
| EXP_003 | `CLASS_HINT=True` (`HINT_LEVEL="class"`) | One shared model, told the era, switches spelling conventions (`ye`/`the`, `heyres`/`heires`) instead of guessing |
| EXP_004 | `CLASS_AUG_COPIES={"B": 2}` | Class B is 25 % of lines but the harder regime; one extra augmented copy per B line |
| EXP_005 | `PSEUDO_LABEL_CSV=<pseudo_labels.csv from EXP_002>` (per-class filtered) | Transductive self-training is allowed; +2,061 images, class balance preserved |
| FINAL | `RUN_MODE="full"`, `CHAMPION_DECODE` / `CHAMPION_ENSEMBLE` / `ROUTING_MAP` as printed by the best Fold-0 run | Refit on all 4,093 lines and reproduce the validated recipe on test |
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        lines = s.split("\n")
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        c = {"cell_type": t, "metadata": {}, "source": src, "id": f"c{i:02d}"}
        if t == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb["cells"].append(c)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "D:/HANAFY/ROAD/barbados-2-enhanced.ipynb"
    nb = build(out)
    print("wrote", out, "cells:", len(nb["cells"]))
