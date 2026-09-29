"""Builds barbados-2-mega.ipynb: shared + class-wise + sub-class-wise LoRA systems (any base model), validation every
EVAL_EVERY steps on the inner slice with best-step restore, beam-5 N-best of every system on val / test / unlisted,
cross-system RESCORING (every system scores every pooled candidate), MBR ensembles, submission and pseudo-labels.
Reuses the tested infrastructure cells of build_enhanced_notebook.py (install, imports, paths, utils/augmentation,
loaders, metric, data split, collator, generation, trial harness)."""
import importlib.util
import json

spec = importlib.util.spec_from_file_location("enh", "D:/HANAFY/ROAD/work/build_enhanced_notebook.py")
enh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enh)
E = enh.CELLS


def find(marker):
    hits = [s for t, s in E if t == "code" and marker in s]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

# ============================================================================ intro
md(r'''
# Barbados R.O.A.D. — **MEGA notebook**: shared + class-wise + sub-class-wise models, validated every 100 steps, pooled by MBR

One run trains every system in `CFG.SYSTEMS`. Each system's N-best lists are pooled, and the pool is selected by **minimum Bayes risk under the official metric**.

| System (default) | Scheme | LoRA adapters | Prompt |
|---|---|---|---|
| `8b_shared` | one model for all lines | 1 | original prompt (measured 0.9095 / MBR 0.9112) |
| `8b_class` | one fresh model per class | A, B | one prompt per era |
| `8b_subclass` | one fresh model per sub-class | A1, A2, A3, B | one prompt per sub-class |
| `7b_shared` | one model for all lines | 1 | original prompt (measured 0.9055 / MBR 0.9073) |

Optional systems are in the CFG comments: a B specialist continued from `8b_shared`, and a 32B shared model.
A system can also cover only some classes (`keys=[...]`). It then votes only on the lines of those classes.

### Validation every `EVAL_EVERY = 100` steps (inner slice of TRAIN, never the Fold-0 val)
Each adapter is checked at every 100 steps and at its last step. Each check prints:
- its training loss;
- the teacher-forced loss on held-out inner lines of its class;
- the **official score** of greedy transcriptions of up to 160 of them.

The best step by `CKPT_METRIC` is kept in memory and restored at the end, which answers "more epochs got worse". The curves are saved to `eval_curves.csv`.
`MONITOR_VAL_N > 0` also decodes some Fold-0 val lines at each check. This is **printed only**: selecting on it would leak the benchmark.

### Why pool, and why rescore (measured on the EXP_002 outputs, `work/selection_experiments.py`)
**The pooled 7B+8B MBR scores 0.9175.** The best candidate already in the pool would score **0.9497** (the oracle). Can a better selection rule close that gap, using only what the pool already contains?

| Selection rule, honest 5-fold CV over lines | Δ vs pooled MBR |
|---|---|
| Posterior temperature | −0.0005 |
| Per-class system weights | 0.0000 |
| Character 6-gram LM prior (train labels) | 0.0000 (the LM only hurts) |
| Out-of-vocabulary penalty | −0.0002 |
| Median-string search (no knob) | +0.0003, P = 0.78 |
| Gradient-boosted reranker (earlier) | −0.0028 |

Selection on the existing evidence is saturated. The candidate the oracle wants is **in only one model's 5-best list in about 85% of the lines**, so the other model gives it zero weight.
Two things add real evidence:
1. **More, diverse systems.** Class and sub-class models make different errors. Going from one model to two added +0.0064.
2. **Cross-rescoring (`RESCORE=True`).** Every system computes the exact log-probability of every pooled candidate, so every candidate gets a vote from every model. MBR (`rescore_mbr`) and product-of-experts MAP (`rescore_map`) then run over the rescored pool.

Every method is one trial row on the 819 val lines, with a paired bootstrap against `pool_mbr_quality` (the measured champion recipe). The best one writes `submission.csv`, and its settings go to `mega_champion.json` for the `RUN_MODE="full"` refit.

### Runtime (RTX PRO 6000, from EXP_002 timings)
- About 20 min of training per 8B system and 40 min per 7B system.
- About 1 min per validation check.
- About 20 min of decoding per system (val + test + unlisted).
- About 10 min of rescoring per system.

Default 4 systems ≈ 4–5 h. `TIME_BUDGET_H` skips whole base models, and the rescoring pass, when time runs short. Everything already produced stays on disk.
''')

# ============================================================================ reused infrastructure
code(find("# Cell 1 — OFFLINE install"))
code(next(s for t, s in E if t == "code" and s.lstrip().startswith("%%capture")))
code(find("# Cell 2 — imports"))

# ============================================================================ CFG
code(r'''
# =========================================================
# Cell 3 — CFG for the MEGA run
# =========================================================
TRIAL_ID          = "MEGA_001"
TRIAL_DESCRIPTION = ("Shared + class-wise (A/B) + sub-class-wise (A1/A2/A3/B) Qwen3-VL-8B systems and a shared Qwen2.5-VL-7B; "
                     "inner validation every 100 steps with best-step restore; beam-5 N-best pooled; cross-system rescoring; "
                     "MBR ensembles on all 819 Fold-0 val lines")

SHARED_PROMPT = ("This is not modern English, Transcribe the handwriting exactly. Keep the wrong spelling, abbreviations, "
                 "and marks (^, ff, unusual letters).Do not modernize or correct anything.")      # measured prompt (EXP_002)
COMMON_PROMPT = ("Transcribe the handwritten line exactly as written. This is not modern English: keep the original "
                 "spelling, capital letters, abbreviations and marks (^ for superscript letters, : for suspensions, "
                 "&, ff). Do not modernize, expand or correct anything. If parts of neighbouring lines are visible, "
                 "transcribe only the central line.")
_A = (" This line comes from a Barbados legal record of the 1630s-1660s. Common conventions in these documents: "
      "& for 'and', ye and yt for 'the' and 'that', sd for 'said', and spellings such as heires, assignes, publique.")
_B = (" This line comes from a Barbados legal record of the 1670s-1710s, scanned at high resolution. Common "
      "conventions in these documents: superscript abbreviations written with ^ (W^m, y^e, M^r, Adm^rs, Ex^rs), "
      "colon suspensions (Tho:, Exec:^rs, S:^d), ff at the start of words, and spellings such as heyres and saide.")

class CFG:
    WHEELS_DIR = WHEELS
    # ---- base models (tag -> path) ----
    BASE_MODELS = {
        "8b":  "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
        "7b":  "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2",
        "32b": "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/32b-instruct/1",
    }
    # ---- SYSTEMS: each one is trained (validated every EVAL_EVERY steps), decoded (N-best) and pooled ----
    #   scheme "shared"  : ONE LoRA on all lines, prompt SHARED_PROMPT
    #   scheme "class"   : one fresh LoRA per class A / B, each trained on its class only, prompt CLASS_PROMPTS[class]
    #   scheme "subclass": one fresh LoRA per sub-class A1 / A2 / A3 / B
    #   optional: keys=[...]           only these classes (the system votes only on their lines)
    #             init="from:<name>"   start every adapter from a SHARED system of the same base model (trained earlier)
    #             replay=0.15          share of other-class lines mixed into each class adapter
    #             hp={...}             LEARNING_RATE, NUM_TRAIN_EPOCHS, MAX_STEPS, R, LORA_ALPHA, AUG_COPIES, ...
    #             class_hp={"A2": {"NUM_TRAIN_EPOCHS": 2}}
    # Systems of the same base model share one loaded base; the order below is the training order.
    SYSTEMS = [
        dict(name="8b_shared",   model="8b", scheme="shared"),
        dict(name="8b_class",    model="8b", scheme="class"),
        dict(name="8b_subclass", model="8b", scheme="subclass"),
        dict(name="7b_shared",   model="7b", scheme="shared"),
        # dict(name="8b_B_cont", model="8b", scheme="class", keys=["B"], init="from:8b_shared", replay=0.15,
        #      hp={"LEARNING_RATE": 1e-4, "LR_SCHEDULER": "constant_with_warmup"}),
        # dict(name="32b_shared", model="32b", scheme="shared"),
    ]
    BASELINE_SYSTEM = "8b_shared"     # every other system / ensemble is compared with this system's top-1
    GRAD_CKPT_MODELS = ("32b",)
    DATA_DIR    = "/kaggle/input/datasets/haniagamal/road-barbados"
    TRAIN_CSV   = None
    TEST_CSV    = None
    IMAGES_DIR  = None
    AUTODISCOVER = True

    # ---- outputs ----
    OUTPUT_DIR     = "/kaggle/working/qwen-vl-ocr"
    FINAL_DIR      = "/kaggle/working/qwen-vl-ocr/final"     # adapters -> FINAL_DIR/<model>/fold0/<adapter>/
    WORK_DIR       = "/kaggle/working"
    SUBMISSION_CSV = "/kaggle/working/submission.csv"
    KEEP_ADAPTERS  = True     # always True in practice: the rescoring pass reloads them
    ADAPTERS_DIR   = None     # DO_TRAIN=False: folder holding <model>/fold0/<adapter>/ ; None -> FINAL_DIR, then /kaggle/input
    PREV_RESULTS_DIRS = []

    # ---- protocol ----
    RUN_MODE        = "fold0"        # "fold0": score ALL 819 val lines | "full": train on all lines, predict test
    FOLD            = 0
    N_FOLDS         = 5
    CKPT_SELECTION  = "inner_loss"   # keeps the inner slice OUT of training (needed by the step validation) — keep
    INNER_CKPT_FRAC = 0.08
    TIME_BUDGET_H   = 11.3

    # ---- validation during training (inner slice of TRAIN) ----
    EVAL_EVERY          = 100        # optimizer steps between validations (the last step is always validated)
    EVAL_GEN_N          = 160        # inner lines greedy-decoded per validation (the adapter's own class)
    EVAL_LOSS_N         = 400        # inner lines for the teacher-forced loss
    EVAL_BATCH          = 16
    EVAL_MAX_NEW_TOKENS = 128
    CKPT_METRIC         = "inner_official"   # "inner_official" | "inner_loss" | "last" — best step restored at the end
    MONITOR_VAL_N       = 0          # >0: also decode this many Fold-0 VAL lines per validation — PRINTED ONLY, never used

    # ---- training (defaults; per model: MODEL_OVERRIDES, per system: hp, per class: class_hp) ----
    PER_DEVICE_TRAIN_BATCH_SIZE  = 4
    GRADIENT_ACCUMULATION_STEPS  = 1
    NUM_TRAIN_EPOCHS             = 1
    MAX_STEPS                    = -1
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
    SEED                         = 42
    MODEL_OVERRIDES = {"32b": {"LEARNING_RATE": 1e-4}}

    # ---- classes (pixels only) and prompts ----
    CLASS_HEIGHT_SPLIT  = 150
    GROUP_A1_MAX_HEIGHT = 58
    GROUP_A2_MAX_PAPER  = 194
    CLASS_PROMPTS = {
        "A":  COMMON_PROMPT + _A,
        "B":  COMMON_PROMPT + _B,
        "A1": COMMON_PROMPT + _A + " The crop is tight around the line; '&' and ye/yt are very common.",
        "A2": COMMON_PROMPT + _A + " The parchment is dark and the hand is large; in these documents '&' and ye are rare and lines are short.",
        "A3": COMMON_PROMPT + _A + " The crop is loose and often shows parts of the lines above and below; the central line is usually long.",
    }
    SHARED_PROMPT     = SHARED_PROMPT
    SHARED_CLASS_HINT = False        # True -> shared systems get " Scan type: A." / " B." appended (train + inference)
    CLASS_HINT        = True         # internal: every line's class is passed to the prompt builder — keep True
    HINT_LEVEL        = "group"
    CLASS_AUG_COPIES  = None

    # ---- augmentation (realistic profile) ----
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

    # ---- decoding: beam 5, 5-best kept (EXP_002: best single-model setting with MBR) + loop guard ----
    MAX_NEW_TOKENS  = 256
    BATCH_SIZE      = 8
    DECODE          = dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=5)
    DECODE_VARIANTS = {"D2_b5nb": DECODE}
    BASELINE_VARIANT = "D2_b5nb"
    LOOP_GUARD      = 3              # collapse a token repeated more than 3x in a row (7B D1 'x x x ...' loop)
    MBR_TEMPERATURE = 1.0            # offline CV: tuning T gave -0.0005 -> keep 1.0

    # ---- cross-system rescoring ----
    RESCORE        = True
    RESCORE_SPLITS = ("val", "test")           # add "unlisted" for rescored pseudo-labels (+ ~30 % time)
    RESCORE_BATCH  = 8
    RESCORE_T      = 1.0

    # ---- ensembles / final choice ----
    ENSEMBLE_METHODS = ("pool_mbr_equal", "pool_mbr_quality", "pool_mbr_quality_median",
                        "rescore_mbr", "rescore_mbr_median", "rescore_map")
    FINAL_METHOD  = None     # RUN_MODE="full": a method name; None -> read mega_champion.json (FINAL_CHAMPION_JSON)
    FINAL_WEIGHTS = None     # RUN_MODE="full": {system: weight}; None -> from mega_champion.json, else equal
    FINAL_CHAMPION_JSON = None   # path of the Fold-0 run's mega_champion.json (attach its output as a dataset)

    # ---- pseudo-labels ----
    PREDICT_UNLISTED = True
    PSEUDO_LABEL_CSV = None
    PSEUDO_KEEP_FRAC = 0.7
    PSEUDO_PER_CLASS = True

    # ---- flow ----
    DO_TRAIN = True
    DO_INFER = True
    PREDICT_TEST = True
    OCR_PROMPT = SHARED_PROMPT
    # used only by the shared helper cells
    SPECIALISTS = {}; SPECIALIST_MODELS = (); CLASS_SAMPLING_T = None; CLASS_ROUTING = False; ROUTING_LEVEL = "class"

CFG.MODEL_PATHS = list(dict.fromkeys(CFG.BASE_MODELS[s["model"]] for s in CFG.SYSTEMS))
TRIAL_CONFIG = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)}
print("TRIAL:", TRIAL_ID, "|", TRIAL_DESCRIPTION)
print("MODE:", CFG.RUN_MODE, "| systems:", [s["name"] for s in CFG.SYSTEMS], "| base models:", len(CFG.MODEL_PATHS))
''')

code(find("# Cell 4 — resolve & sanity-check paths"))
code(find("# Cell 5 — text / image utils"))
code(find("# Cell 6 — model & processor loaders"))
code(find("# Cell 7 — OFFICIAL metric"))
code(find("# Cell 8 — samples, Fold-0 split"))
code(find("# Cell 9 — collator"))
code(find("# Cell 10 — batched N-best generation"))
code(find("# Cell 11 — trial harness"))

# ============================================================================ mega setup
code(r'''
# =========================================================
# Cell 12 — MEGA setup: logging, prompts per scheme, systems, training rows, step validation callback
# =========================================================
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

def gpu_mem():
    if not torch.cuda.is_available():
        return "cpu"
    return f"GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB used, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"

SCHEME_KEYS = {"shared": ["ALL"], "class": ["A", "B"], "subclass": ["A1", "A2", "A3", "B"]}

def scheme_key(scheme, grp):
    return "ALL" if scheme == "shared" else (grp if scheme == "subclass" else grp[0])

ACTIVE_SCHEME = "shared"

def ocr_prompt(grp=None):
    """overrides Cell 5 — shared systems: SHARED_PROMPT (+ optional scan-type hint); class / sub-class systems: the
    prompt of the line's class. The same function builds the prompt in training (collator) and decoding."""
    if ACTIVE_SCHEME == "shared" or not grp:
        return CFG.SHARED_PROMPT + (f" Scan type: {grp[0]}." if (CFG.SHARED_CLASS_HINT and grp) else "")
    return CFG.CLASS_PROMPTS[scheme_key(ACTIVE_SCHEME, grp)]

def adapter_name(sysname, key):
    return sysname if key == "ALL" else f"{sysname}__{key}"

def loop_guard(text, max_run=None):
    max_run = max_run or CFG.LOOP_GUARD
    ws, out, run = norm_text(text).split(), [], 0
    for i, w in enumerate(ws):
        run = run + 1 if (i and w == ws[i - 1]) else 1
        if run <= max_run:
            out.append(w)
    return " ".join(out)

# ---- parse / validate SYSTEMS ----
SYS = collections.OrderedDict()
for _s in CFG.SYSTEMS:
    s = {"keys": None, "init": "fresh", "replay": 0.0, "hp": {}, "class_hp": {}, **_s}
    assert s["model"] in CFG.BASE_MODELS, f"{s['name']}: unknown base model {s['model']!r}"
    assert s["scheme"] in SCHEME_KEYS, f"{s['name']}: scheme must be one of {list(SCHEME_KEYS)}"
    s["keys"] = list(s["keys"] or SCHEME_KEYS[s["scheme"]])
    for k in s["keys"]:
        assert k in SCHEME_KEYS[s["scheme"]], f"{s['name']}: key {k!r} is not a {s['scheme']} key"
        assert s["scheme"] == "shared" or k in CFG.CLASS_PROMPTS, f"CLASS_PROMPTS has no prompt for {k!r}"
    if s["init"].startswith("from:"):
        src = s["init"][5:]
        assert src in SYS and SYS[src]["model"] == s["model"] and SYS[src]["scheme"] == "shared", \
            f"{s['name']}: init source {src!r} must be an EARLIER shared system on the same base model"
    SYS[s["name"]] = s
assert CFG.BASELINE_SYSTEM in SYS, "BASELINE_SYSTEM must be one of SYSTEMS"
BASE_ORDER = list(dict.fromkeys(s["model"] for s in SYS.values()))
FOLD_TAG = f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full"

def sys_hp(s, key):
    hp = {k: getattr(CFG, k) for k in ("LEARNING_RATE", "NUM_TRAIN_EPOCHS", "MAX_STEPS", "R", "LORA_ALPHA", "AUG_COPIES",
                                        "PER_DEVICE_TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "WARMUP_RATIO",
                                        "LR_SCHEDULER")}
    hp.update(CFG.MODEL_OVERRIDES.get(s["model"], {}))
    hp.update(s["hp"])
    hp.update(s["class_hp"].get(key, {}))
    return hp

def _in_key(s, key, grp):
    return key == "ALL" or scheme_key(s["scheme"], grp) == key

def covers(s, grp):
    return scheme_key(s["scheme"], grp) in s["keys"]

def rows_for(samples, n_aug, salt):
    rows = [{"image": x["image"], "text": x["text"], "grp": x["grp"], "aug": False, "seed": -1} for x in samples]
    if CFG.USE_AUGMENT:
        for j, x in enumerate(samples):
            for c in range(int(n_aug)):
                rows.append({"image": x["image"], "text": x["text"], "grp": x["grp"], "aug": True,
                             "seed": int((CFG.SEED + 1) * 2000003 + CFG.FOLD * 1000003 + c * 9973 + j + salt * 7777777)
                                     & 0x7fffffff})
    return rows

def adapter_rows(s, key, hp, salt):
    pool = core_samples + pseudo_samples
    own = [x for x in pool if _in_key(s, key, x["grp"])]
    rows = rows_for(own, hp["AUG_COPIES"], salt)
    if s["replay"] > 0 and key != "ALL":
        others = [x for x in pool if not _in_key(s, key, x["grp"])]
        n_rep = min(len(others), int(round(s["replay"] / (1 - s["replay"]) * len(rows))))
        pick = np.random.RandomState(CFG.SEED + 13 + salt).choice(len(others), n_rep, replace=False) if n_rep else []
        rows += [{"image": others[i]["image"], "text": others[i]["text"], "grp": others[i]["grp"], "aug": True,
                  "seed": int(CFG.SEED * 31 + i + salt) & 0x7fffffff} for i in pick]
    return own, rows

# fixed order of the inner slice / val lines so every adapter is validated on the same lines of its class
_INNER_ORDER = [inner_samples[i] for i in np.random.RandomState(CFG.SEED + 7).permutation(len(inner_samples))]
_VAL_ORDER = [val_samples[i] for i in np.random.RandomState(CFG.SEED + 8).permutation(len(val_samples))]

def eval_sets(s, key):
    inner = [x for x in _INNER_ORDER if _in_key(s, key, x["grp"])]
    mon = [x for x in _VAL_ORDER if _in_key(s, key, x["grp"])][:CFG.MONITOR_VAL_N] if CFG.MONITOR_VAL_N else []
    return inner[:CFG.EVAL_GEN_N], inner[:CFG.EVAL_LOSS_N], mon

GREEDY = dict(num_beams=1, repetition_penalty=1.2, no_repeat_ngram_size=0, n_return=1)
EVAL_CURVES, ADAPTER_SUMMARY = [], []

def teacher_forced_loss(model, proc, items):
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(items), CFG.EVAL_BATCH):
            batch = collate_fn([{**x, "aug": False, "seed": -1} for x in items[b:b + CFG.EVAL_BATCH]], proc)
            if batch is None:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            nt = int((batch["labels"] != -100).sum())
            tot += float(model(**batch).loss) * nt
            n += nt
    return tot / max(1, n)

class StepValidation(TrainerCallback):
    """validates the adapter being trained every CFG.EVAL_EVERY optimizer steps and at the last step; keeps the best
    step's LoRA weights in CPU RAM and restores them at the end (CKPT_METRIC)."""
    def __init__(self, s, key, model, proc):
        self.s, self.key, self.model, self.proc = s, key, model, proc
        self.tag, self.adapter = f"{s['name']}/{key}", adapter_name(s["name"], key)
        self.gen, self.loss_items, self.mon = eval_sets(s, key)
        self.best, self.best_step, self.best_state, self.last_step = None, None, None, -1
        self.train_loss, self.first, self.last = float("nan"), None, None

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.train_loss = float(logs["loss"])

    def validate(self, state):
        if state.global_step == self.last_step:
            return
        self.last_step = state.global_step
        m, t0 = self.model, time.time()
        was_training = m.training
        m.eval()
        keep = (CFG.MAX_NEW_TOKENS, CFG.BATCH_SIZE)
        CFG.MAX_NEW_TOKENS, CFG.BATCH_SIZE = CFG.EVAL_MAX_NEW_TOKENS, CFG.EVAL_BATCH
        res = {}
        try:
            loss = teacher_forced_loss(m, self.proc, self.loss_items) if self.loss_items else float("nan")
            for nm, items in (("inner", self.gen), ("monitor", self.mon)):
                if items:
                    nb = transcribe(m, self.proc, [x["image"] for x in items], GREEDY)
                    res[nm] = official_metric([x["text"] for x in items], [loop_guard(x[0][0]) for x in nb])
        finally:
            CFG.MAX_NEW_TOKENS, CFG.BATCH_SIZE = keep
            if was_training:
                m.train()
        sc = res["inner"]["score"] if "inner" in res else float("nan")
        crit = {"inner_official": sc, "inner_loss": -loss}.get(CFG.CKPT_METRIC)
        flag = ""
        if crit is not None and np.isfinite(crit) and (self.best is None or crit > self.best + 1e-9):
            self.best, self.best_step = crit, state.global_step
            self.best_state = {k: v.detach().to("cpu", copy=True)
                               for k, v in get_peft_model_state_dict(m, adapter_name=self.adapter).items()}
            flag = "  <- best so far"
        row = {"system": self.s["name"], "adapter": self.key, "step": state.global_step, "max_steps": state.max_steps,
               "epoch": state.epoch, "train_loss": self.train_loss, "inner_loss": loss, "inner_score": sc,
               "inner_we": res.get("inner", {}).get("word_edits_per_line", np.nan),
               "inner_ce": res.get("inner", {}).get("char_edits_per_line", np.nan), "n_inner": len(self.gen),
               "monitor_val_score": res.get("monitor", {}).get("score", np.nan), "seconds": time.time() - t0}
        EVAL_CURVES.append(row)
        self.first = self.first or row
        self.last = row
        mon = f" | monitor-val {row['monitor_val_score']:.4f} (n={len(self.mon)}, NOT used)" if self.mon else ""
        log(f"[VALID {self.tag}] step {state.global_step}/{state.max_steps} ep {state.epoch or 0:.2f} | train loss "
            f"{self.train_loss:.4f} | inner loss {loss:.4f} | inner official {sc:.4f} (word/line {row['inner_we']:.3f}, "
            f"char/line {row['inner_ce']:.3f}, n={len(self.gen)}){mon} | {row['seconds']:.0f}s{flag}")

    def on_step_end(self, args, state, control, **kw):
        if CFG.EVAL_EVERY and state.global_step % int(CFG.EVAL_EVERY) == 0:
            self.validate(state)

    def on_train_end(self, args, state, control, **kw):
        self.validate(state)                               # the last step is always validated
        restored = False
        if CFG.CKPT_METRIC != "last" and self.best_state is not None and self.best_step != state.global_step:
            set_peft_model_state_dict(self.model, self.best_state, adapter_name=self.adapter)
            restored = True
            log(f"[CKPT {self.tag}] RESTORED step {self.best_step} ({CFG.CKPT_METRIC} = {self.best:.4f}) instead of the "
                f"last step {state.global_step} (inner official {self.last['inner_score']:.4f})")
        else:
            log(f"[CKPT {self.tag}] keeping the last step {state.global_step}")
        best_row = next((r for r in EVAL_CURVES if r["system"] == self.s["name"] and r["adapter"] == self.key
                         and r["step"] == self.best_step), self.last)
        ADAPTER_SUMMARY.append({"system": self.s["name"], "adapter": self.key, "steps": state.global_step,
                                "best_step": self.best_step, "best_inner_score": best_row["inner_score"],
                                "last_inner_score": self.last["inner_score"], "restored_best": restored})
        self.best_state = None

log(f"TRIAL {TRIAL_ID} | mode {CFG.RUN_MODE} ({FOLD_TAG}) | validation every {CFG.EVAL_EVERY} steps on the inner slice "
    f"({len(inner_samples)} lines) | checkpoint rule {CFG.CKPT_METRIC}", banner=True)
_pool_lines = core_samples + pseudo_samples
for s in SYS.values():
    parts = []
    for k in s["keys"]:
        n_tr = sum(_in_key(s, k, x["grp"]) for x in _pool_lines)
        n_in = sum(_in_key(s, k, x["grp"]) for x in inner_samples)
        n_va = sum(_in_key(s, k, g) for g in VAL_GROUP)
        n_te = sum(_in_key(s, k, g) for g in TEST_GROUP)
        parts.append(f"{k}: train {n_tr} / inner {n_in} / val {n_va} / test {n_te}")
    log(f"SYSTEM {s['name']:14s} base={s['model']:4s} scheme={s['scheme']:8s} init={s['init']:14s} | " + " | ".join(parts))
    if any(sum(_in_key(s, k, x["grp"]) for x in _pool_lines) < 150 for k in s["keys"]):
        log(f"  WARNING {s['name']}: an adapter has fewer than 150 training lines — expect a weak model there")
for k in ("A", "B", "A1", "A2", "A3"):
    if any(s["scheme"] != "shared" and k in s["keys"] for s in SYS.values()):
        log(f"PROMPT[{k}] {CFG.CLASS_PROMPTS[k]}")
log(f"PROMPT[shared] {CFG.SHARED_PROMPT}" + ("  + ' Scan type: A/B.'" if CFG.SHARED_CLASS_HINT else ""))
''')

# ============================================================================ train + decode
code(r'''
# =========================================================
# Cell 13 — for every base model: train its systems' adapters (validated every EVAL_EVERY steps), then decode
#           val / test / unlisted with every system (beam 5, 5-best). One base model resident at a time.
# =========================================================
dtype = torch.bfloat16 if (CFG.USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NB = {"val": {}, "test": {}, "unlisted": {}}     # split -> system -> per-line N-best ([] = system does not cover the line)
SPLITS = {"val": (VAL_IDS, VAL_PATHS, VAL_GROUP, VAL_GT), "test": (TEST_IDS, TEST_PATHS, TEST_GROUP, None),
          "unlisted": (UNL_IDS, UNL_PATHS, UNL_GROUP, None)}
DONE_KEYS, SYS_MIN = {}, {}
CURRENT_BASE, model, base = None, None, None
TRAIN_SIZE = len(core_samples) + len(pseudo_samples)

def _vision_linear_names(m):
    return [n for n, mod in m.named_modules()
            if isinstance(mod, _nn.Linear) and any(t in n.lower() for t in ("visual", "vision", "merger"))]

def _find_adapter(mname, name):
    roots = ([CFG.ADAPTERS_DIR] if CFG.ADAPTERS_DIR else []) + [CFG.FINAL_DIR] + \
            ([] if CFG.ADAPTERS_DIR else sorted(_glob("/kaggle/input/*")) + sorted(_glob("/kaggle/input/*/*")))
    for r in roots:
        p = os.path.join(r, mname, FOLD_TAG, name)
        if os.path.isfile(os.path.join(p, "adapter_config.json")):
            return p
    for r in roots:                        # exact <model>/<fold>/<adapter> match anywhere below a root (never fuzzy)
        for c in _glob(os.path.join(r, "**", FOLD_TAG, name, "adapter_config.json"), recursive=True):
            parts = os.path.dirname(c).replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[-3] == mname:
                return os.path.dirname(c)
    return None

def attach_adapter(model, base, path, name):
    if model is None:
        return PeftModel.from_pretrained(base, path, adapter_name=name)
    model.load_adapter(path, adapter_name=name)
    return model

def save_nbest(sysname, split, nb):
    ids, _, _, gts = SPLITS[split]
    rows = [{"ID": i, "GroundTruth": gts[k] if gts is not None else None, "pred": nb[k][0][0] if nb[k] else "",
             "logprob": nb[k][0][1] if nb[k] else None, "n_tokens": nb[k][0][2] if nb[k] else None,
             "truncated": nb[k][0][3] if nb[k] else None,
             "nbest": json.dumps([[t, round(sc, 4)] for t, sc, _, _ in nb[k]], ensure_ascii=False)}
            for k, i in enumerate(ids)]
    pd.DataFrame(rows).to_csv(os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__{sysname}__D2_b5nb__{split}.csv"), index=False)

def decode_system(model, proc, s, split):
    global ACTIVE_SCHEME
    ACTIVE_SCHEME = s["scheme"]
    ids, paths, groups, _ = SPLITS[split]
    out = [[] for _ in paths]
    for key in DONE_KEYS.get(s["name"], []):
        idx = [i for i, g in enumerate(groups) if _in_key(s, key, g)]
        if not idx:
            continue
        model.set_adapter(adapter_name(s["name"], key))
        t0 = time.time()
        sub = transcribe(model, proc, [paths[i] for i in idx], CFG.DECODE, desc=f"{split}/{s['name']}/{key}")
        for j, i in enumerate(idx):
            out[i] = [(loop_guard(t), sc, n, tr) for t, sc, n, tr in sub[j]]
        log(f"[DECODE {s['name']}/{key}] {split}: {len(idx)} lines in {(time.time() - t0) / 60:.1f} min "
            f"({sum(sub[j][0][3] for j in range(len(idx)))} truncated)")
    return out

def single_mbr(nb_line):
    return mbr_select([t for t, *_ in nb_line], nbest_posteriors([sc for _, sc, *_ in nb_line], CFG.MBR_TEMPERATURE)) \
        if nb_line else ""

def report_system(sname):
    """val report of one system: top-1 and single-system MBR, per class / group, vs BASELINE_SYSTEM."""
    nb = NB["val"][sname]
    idx = [i for i, x in enumerate(nb) if x]
    top1, mbr = [x[0][0] if x else "" for x in nb], [single_mbr(x) for x in nb]
    base_top = [x[0][0] if x else "" for x in NB["val"].get(CFG.BASELINE_SYSTEM, nb)]
    if len(idx) == len(nb):
        for tag, hyps in (("top1", top1), ("mbr", mbr)):
            report_trial(f"{TRIAL_ID}__{sname}__{tag}", TRIAL_DESCRIPTION, sname, f"D2_b5nb/{tag}", hyps,
                         SYS_MIN.get(sname, 0) * 60, TRAIN_SIZE, sum(x[0][3] for x in nb if x),
                         baseline_hyps=None if (sname == CFG.BASELINE_SYSTEM and tag == "top1") else base_top,
                         extra={"system": sname, "scheme": SYS[sname]["scheme"], "base_model": SYS[sname]["model"]})
    else:                                    # partial system: score only the lines it covers
        g = [VAL_GT[i] for i in idx]
        for tag, hyps in (("top1", top1), ("mbr", mbr)):
            m = official_metric(g, [hyps[i] for i in idx])
            d = paired_delta(g, [base_top[i] for i in idx], [hyps[i] for i in idx])
            log(f"[VAL {sname}/{tag}] covers {len(idx)} val lines: {m['score']:.5f} | vs {CFG.BASELINE_SYSTEM} top-1 on "
                f"those lines {d['delta']:+.5f} CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}] P(better)={d['p_better']:.3f}")

for btag in BASE_ORDER:
    systems = [s for s in SYS.values() if s["model"] == btag]
    if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
        log(f"[TIME] budget {CFG.TIME_BUDGET_H} h reached — skipping base model {btag} ({[s['name'] for s in systems]})")
        continue
    msrc = CFG.BASE_MODELS[btag]
    mname = model_short_name(msrc)
    log(f"BASE MODEL {btag}: {msrc} | systems {[s['name'] for s in systems]} | {gpu_mem()}", banner=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    proc = load_vlm_processor(msrc, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
    base = load_vlm_model(msrc, dtype)
    base.to(DEVICE)
    base.config.use_cache = False
    if CFG.DO_TRAIN and any(t in msrc.lower() for t in CFG.GRAD_CKPT_MODELS):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] + \
              (_vision_linear_names(base) if CFG.LORA_VISION else [])
    ad_root = os.path.join(CFG.FINAL_DIR, mname, FOLD_TAG)
    model, CURRENT_BASE = None, btag
    for s in systems:
        ACTIVE_SCHEME = s["scheme"]
        t_sys = time.time()
        DONE_KEYS[s["name"]] = []
        for kn, key in enumerate(s["keys"]):
            name, hp = adapter_name(s["name"], key), sys_hp(s, key)
            if CFG.DO_TRAIN:
                own, rows = adapter_rows(s, key, hp, salt=list(SYS).index(s["name"]) * 10 + kn + 1)
                if not own:
                    log(f"[TRAIN {s['name']}/{key}] no training lines — adapter skipped (its lines get no vote)")
                    continue
                lcfg = LoraConfig(r=hp["R"], lora_alpha=hp["LORA_ALPHA"], target_modules=TARGETS,
                                  lora_dropout=CFG.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
                if model is None:
                    model = get_peft_model(base, lcfg, adapter_name=name)
                else:
                    model.add_adapter(name, lcfg)
                if s["init"].startswith("from:"):
                    src = adapter_name(s["init"][5:], "ALL")
                    set_peft_model_state_dict(model, get_peft_model_state_dict(model, adapter_name=src), adapter_name=name)
                    log(f"[TRAIN {s['name']}/{key}] initialised from adapter {src}")
                model.set_adapter(name)
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                steps = math.ceil(len(rows) / (hp["PER_DEVICE_TRAIN_BATCH_SIZE"] * hp["GRADIENT_ACCUMULATION_STEPS"])) \
                        * hp["NUM_TRAIN_EPOCHS"]
                steps = hp["MAX_STEPS"] if hp["MAX_STEPS"] and hp["MAX_STEPS"] > 0 else steps
                log(f"[TRAIN {s['name']}/{key}] {len(own)} lines -> {len(rows)} rows | lr {hp['LEARNING_RATE']} "
                    f"({hp['LR_SCHEDULER']}) epochs {hp['NUM_TRAIN_EPOCHS']} r {hp['R']} | ~{steps} steps, validation "
                    f"every {CFG.EVAL_EVERY} | trainable {n_trainable / 1e6:.1f}M | {gpu_mem()}")
                cb = StepValidation(s, key, model, proc)
                args = TrainingArguments(
                    output_dir=os.path.join(CFG.OUTPUT_DIR, name),
                    per_device_train_batch_size=hp["PER_DEVICE_TRAIN_BATCH_SIZE"],
                    gradient_accumulation_steps=hp["GRADIENT_ACCUMULATION_STEPS"],
                    learning_rate=hp["LEARNING_RATE"], num_train_epochs=hp["NUM_TRAIN_EPOCHS"],
                    max_steps=int(hp["MAX_STEPS"] or -1), lr_scheduler_type=hp["LR_SCHEDULER"],
                    warmup_ratio=hp["WARMUP_RATIO"], bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
                    logging_steps=10, eval_strategy="no", save_strategy="no", remove_unused_columns=False,
                    dataloader_num_workers=CFG.DATALOADER_NUM_WORKERS, dataloader_pin_memory=torch.cuda.is_available(),
                    report_to="none", seed=CFG.SEED)
                trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                                  data_collator=lambda x, _p=proc: collate_fn(x, _p), callbacks=[cb])
                t0 = time.time()
                trainer.train()
                log(f"[TRAIN {s['name']}/{key}] done in {(time.time() - t0) / 60:.1f} min | label-mask failures so far "
                    f"{MASK_FAIL_COUNT} | {gpu_mem()}")
                try:
                    pd.DataFrame(trainer.state.log_history).to_csv(
                        os.path.join(CFG.WORK_DIR, f"log_history__{name}.csv"), index=False)
                except Exception:
                    pass
                os.makedirs(ad_root, exist_ok=True)
                model.save_pretrained(ad_root, selected_adapters=[name])   # -> ad_root/<name>/
                proc.save_pretrained(ad_root)
                del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, name), ignore_errors=True)
            else:
                p = _find_adapter(mname, name)
                if p is None:
                    log(f"[LOAD {s['name']}/{key}] no saved adapter {name!r} for {mname}/{FOLD_TAG} — skipped")
                    continue
                model = attach_adapter(model, base, p, name)
                log(f"[LOAD {s['name']}/{key}] adapter <- {p}")
            DONE_KEYS[s["name"]].append(key)
        SYS_MIN[s["name"]] = (time.time() - t_sys) / 60
    if model is None:
        log(f"[{btag}] nothing trained or loaded — base model skipped")
        free_cuda("base")
        continue
    model.eval()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    if ADAPTER_SUMMARY:
        log("ADAPTER SUMMARY (inner validation):\n" + pd.DataFrame(
            [r for r in ADAPTER_SUMMARY if SYS[r["system"]]["model"] == btag]).round(4).to_string(index=False))
    if CFG.DO_INFER:
        for s in systems:
            if not DONE_KEYS.get(s["name"]):
                continue
            t0 = time.time()
            for split in ("val", "test", "unlisted"):
                if split == "val" and CFG.RUN_MODE != "fold0":
                    continue
                if split == "test" and not CFG.PREDICT_TEST:
                    continue
                if split == "unlisted" and not (CFG.PREDICT_UNLISTED and UNL_IDS):
                    continue
                NB[split][s["name"]] = decode_system(model, proc, s, split)
                save_nbest(s["name"], split, NB[split][s["name"]])
            SYS_MIN[s["name"]] = SYS_MIN.get(s["name"], 0) + (time.time() - t0) / 60
            if CFG.RUN_MODE == "fold0":
                report_system(s["name"])
    pd.DataFrame(EVAL_CURVES).to_csv(os.path.join(CFG.WORK_DIR, "eval_curves.csv"), index=False)
    if btag != BASE_ORDER[-1] or not CFG.RESCORE:        # keep the last base model loaded for the rescoring pass
        free_cuda("model", "base")
        model = base = None
        del proc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"[{btag}] finished | {gpu_mem()}")

if ADAPTER_SUMMARY:
    _as = pd.DataFrame(ADAPTER_SUMMARY)
    _as.to_csv(os.path.join(CFG.WORK_DIR, "adapter_summary.csv"), index=False)
    log("ALL ADAPTERS — best vs last inner validation step:\n" + _as.round(4).to_string(index=False))
    _late = _as[_as.best_step < _as.steps]
    if len(_late):
        log(f"{len(_late)} of {len(_as)} adapters peaked BEFORE their last step -> training longer would not help them; "
            f"their best step was restored" if CFG.CKPT_METRIC != "last" else "their last step was kept (CKPT_METRIC='last')")
''')

# ============================================================================ rescoring
code(r'''
# =========================================================
# Cell 14 — CROSS-SYSTEM RESCORING: every system computes the exact log-probability of every pooled candidate
#           (teacher forcing, its own prompt + adapter for the line's class). Measured motivation: the oracle
#           candidate is in only ONE model's 5-best in ~85 % of the lines the pooled MBR gets wrong.
# =========================================================
RESC = {}                                             # split -> list of {"cands": [...], "lp": {system: np.array}}

def pooled_candidates(split):
    n = len(SPLITS[split][0])
    return [list(dict.fromkeys(t for sn in NB[split] for t, *_ in NB[split][sn][i] if t)) or [""] for i in range(n)]

def forced_logprobs(model, proc, path, grp, cands):
    img = load_image(path)
    user = {"role": "user", "content": [{"type": "text", "text": ocr_prompt(grp)}, {"type": "image", "image": img}]}
    tok = proc.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    prompt_txt = proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    n_prompt = int(proc(text=[prompt_txt], images=[img], return_tensors="pt")["input_ids"].shape[1])
    out = np.full(len(cands), np.nan)
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        for b in range(0, len(cands), CFG.RESCORE_BATCH):
            cs = cands[b:b + CFG.RESCORE_BATCH]
            full = [proc.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": c}]}],
                                             tokenize=False, add_generation_prompt=False) for c in cs]
            batch = proc(text=full, images=[img] * len(cs), padding=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.inference_mode():
                logits = model(**batch).logits
            ids = batch["input_ids"][:, n_prompt:]
            lp = torch.log_softmax(logits[:, n_prompt - 1:-1].float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            is_end = ids == end_id
            first_end = torch.where(is_end.any(1), is_end.float().argmax(1), torch.full_like(is_end[:, 0], ids.shape[1] - 1,
                                                                                            dtype=torch.long))
            mask = (torch.arange(ids.shape[1], device=ids.device)[None, :] <= first_end[:, None]) & \
                   batch["attention_mask"][:, n_prompt:].bool()
            out[b:b + len(cs)] = (lp * mask).sum(1).cpu().numpy()
            del logits, lp
    finally:
        tok.padding_side = old
    return out

def rescore_system(model, proc, s, split):
    global ACTIVE_SCHEME
    ACTIVE_SCHEME = s["scheme"]
    _, paths, groups, _ = SPLITS[split]
    for key in DONE_KEYS.get(s["name"], []):
        idx = [i for i, g in enumerate(groups) if _in_key(s, key, g) and os.path.exists(paths[i])]
        if not idx:
            continue
        model.set_adapter(adapter_name(s["name"], key))
        t0 = time.time()
        for i in tqdm(idx, desc=f"rescore {split}/{s['name']}/{key}", leave=False):
            try:
                lp = forced_logprobs(model, proc, paths[i], groups[i], RESC[split][i]["cands"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                keep, CFG.RESCORE_BATCH = CFG.RESCORE_BATCH, 1
                lp = forced_logprobs(model, proc, paths[i], groups[i], RESC[split][i]["cands"])
                CFG.RESCORE_BATCH = keep
            RESC[split][i]["lp"][s["name"]] = lp
        n_c = np.mean([len(RESC[split][i]["cands"]) for i in idx])
        log(f"[RESCORE {s['name']}/{key}] {split}: {len(idx)} lines x {n_c:.1f} candidates in {(time.time() - t0) / 60:.1f} min")
    # sanity check: the forced log-prob of the system's own top-1 must equal its beam-search log-prob
    diffs = []                       # complete (non-truncated) top-1 only: truncated beams carry no end-token score
    for i, x in enumerate(NB[split][s["name"]]):
        if x and not x[0][3] and s["name"] in RESC[split][i]["lp"] and np.isfinite(x[0][1]):
            j = RESC[split][i]["cands"].index(x[0][0]) if x[0][0] in RESC[split][i]["cands"] else None
            if j is not None:
                diffs.append(abs(RESC[split][i]["lp"][s["name"]][j] - x[0][1]))
        if len(diffs) >= 50:
            break
    if diffs:
        log(f"[RESCORE check {s['name']}] |forced - beam| log-prob of own top-1: median {np.median(diffs):.3f} over "
            f"{len(diffs)} lines" + ("  (OK)" if np.median(diffs) < 0.5 else "  (LARGE: loop-guarded or re-tokenised text; "
                                                                             "rescoring still ranks consistently)"))

if CFG.RESCORE and CFG.DO_INFER and len(NB["test"] or NB["val"]) >= 1:
    splits = [sp for sp in CFG.RESCORE_SPLITS if NB.get(sp)]
    for sp in splits:
        RESC[sp] = [{"cands": c, "lp": {}} for c in pooled_candidates(sp)]
        log(f"[RESCORE] {sp}: {len(RESC[sp])} lines, {np.mean([len(r['cands']) for r in RESC[sp]]):.1f} pooled candidates "
            f"per line from {len(NB[sp])} systems")
    for btag in ([CURRENT_BASE] if model is not None else []) + [b for b in BASE_ORDER if b != CURRENT_BASE or model is None]:
        systems = [s for s in SYS.values() if s["model"] == btag and DONE_KEYS.get(s["name"])]
        if not systems:
            continue
        if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
            log(f"[TIME] budget reached — rescoring of {btag} skipped (rescore methods then use the systems scored so far)")
            continue
        msrc = CFG.BASE_MODELS[btag]
        mname = model_short_name(msrc)
        if not (model is not None and btag == CURRENT_BASE):
            free_cuda("model", "base")
            proc = load_vlm_processor(msrc, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
            base = load_vlm_model(msrc, dtype)
            base.to(DEVICE)
            model = None
            for s in systems:
                for key in DONE_KEYS[s["name"]]:
                    p = _find_adapter(mname, adapter_name(s["name"], key))
                    assert p is not None, f"adapter {adapter_name(s['name'], key)} not found for rescoring"
                    model = attach_adapter(model, base, p, adapter_name(s["name"], key))
            model.eval()
            model.config.use_cache = False
            CURRENT_BASE = btag
        log(f"RESCORING with base {btag}: {[s['name'] for s in systems]} | {gpu_mem()}", banner=True)
        for s in systems:
            for sp in splits:
                rescore_system(model, proc, s, sp)
    free_cuda("model", "base")
    model = base = None
    for sp in splits:
        with open(os.path.join(CFG.WORK_DIR, f"rescore__{sp}.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(RESC[sp]):
                f.write(json.dumps({"ID": SPLITS[sp][0][i], "cands": r["cands"],
                                    "lp": {k: [None if not np.isfinite(v) else round(float(v), 4) for v in a]
                                           for k, a in r["lp"].items()}}, ensure_ascii=False) + "\n")
    log(f"[RESCORE] saved rescore__<split>.jsonl for {splits}")
else:
    free_cuda("model", "base")
    model = base = None
    log("[RESCORE] off (CFG.RESCORE=False) — rescore_* methods are skipped")
''')

# ============================================================================ ensembles
code(r'''
# =========================================================
# Cell 15 — ensembles over ALL systems (each one = a trial on the 819 val lines), champion, submission, pseudo-labels
# =========================================================
try:
    from rapidfuzz.distance import Levenshtein as _RFL
except Exception:
    _RFL = None

def _softmax(x, T=1.0):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    out = np.zeros(len(x))
    if m.any():
        z = x[m] / max(T, 1e-6)
        z = np.exp(z - z.max())
        out[m] = z / z.sum()
    return out

def median_refine(start, cands, w, max_iter=6):
    """extended-hypothesis MBR: greedy single-word edits (taken from other candidates) while the expected official cost
    under the pooled posterior falls. Offline on EXP_002: +0.0003 (P=0.78) over pooled MBR — kept as a challenger."""
    if _RFL is None:
        return start
    ev = [(c, float(x)) for c, x in zip(cands, w) if x > 1e-9]
    risk = lambda h: sum(x * line_cost(c, h) for c, x in ev)
    h, best = start.split(), risk(start)
    for _ in range(max_iter):
        props = set()
        for c, _x in ev:
            cw = c.split()
            for op in _RFL.editops(h, cw):
                nh = (h[:op.src_pos] + [cw[op.dest_pos]] + h[op.src_pos + 1:] if op.tag == "replace" else
                      h[:op.src_pos] + h[op.src_pos + 1:] if op.tag == "delete" else
                      h[:op.src_pos] + [cw[op.dest_pos]] + h[op.src_pos:])
                props.add(" ".join(nh))
        props.discard(" ".join(h))
        if not props:
            break
        cand = min(props, key=risk)
        r = risk(cand)
        if r < best - 1e-12:
            h, best = cand.split(), r
        else:
            break
    return " ".join(h)

def pool_evidence(split, i, weights, systems):
    """own N-best posteriors of every system that covers line i, mixed with the system weights."""
    agg, tot = collections.OrderedDict(), 0.0
    for sn in systems:
        x = NB[split].get(sn, [[]] * (i + 1))[i]
        if not x:
            continue
        p = nbest_posteriors([sc for _, sc, *_ in x], CFG.MBR_TEMPERATURE)
        for (t, *_), q in zip(x, p):
            agg[t] = agg.get(t, 0.0) + weights[sn] * q
        tot += weights[sn]
    if not agg:
        return [""], np.array([1.0])
    return list(agg), np.array(list(agg.values())) / max(tot, 1e-12)

def rescore_evidence(split, i, weights, systems):
    """posterior over the WHOLE pooled candidate list from every system's forced log-probs."""
    r = RESC[split][i]
    w, tot = np.zeros(len(r["cands"])), 0.0
    for sn in systems:
        if sn in r["lp"] and np.isfinite(r["lp"][sn]).any():
            w += weights[sn] * _softmax(r["lp"][sn], CFG.RESCORE_T)
            tot += weights[sn]
    return r["cands"], (w / tot if tot > 0 else np.full(len(r["cands"]), 1.0 / len(r["cands"])))

def run_method(method, split, weights, systems):
    """-> (hyps, risks) for every line of split."""
    n = len(SPLITS[split][0])
    if method.startswith("rescore") and split not in RESC:
        return None, None
    hyps, risks = [], []
    for i in range(n):
        if method.startswith("pool_mbr"):
            cands, w = pool_evidence(split, i, weights, systems)
        else:
            cands, w = rescore_evidence(split, i, weights, systems)
        if method == "rescore_map":
            tot = np.zeros(len(cands)); ok = np.ones(len(cands), bool)
            r = RESC[split][i]
            for sn in systems:
                if sn in r["lp"] and np.isfinite(r["lp"][sn]).any():
                    tot += weights[sn] * np.nan_to_num(r["lp"][sn], nan=-1e4)
            h = cands[int(np.argmax(tot))]
            risk = float(sum(x * line_cost(c, h) for c, x in zip(cands, w)))
        else:
            h, risk = mbr_select(cands, list(w), return_risk=True)
            if method.endswith("_median"):
                h = median_refine(h, cands, w)
                risk = float(sum(x * line_cost(c, h) for c, x in zip(cands, w)))
        hyps.append(h)
        risks.append(risk)
    return hyps, risks

SYSTEMS_DONE = [sn for sn in SYS if sn in NB["test"] or sn in NB["val"]]
CANDIDATES, RISKS, QW, W_FINAL, best, base_top = {}, {}, {}, {}, None, None
if CFG.RUN_MODE == "fold0" and NB["val"]:
    # quality weights fixed by a rule (1 / lost points of each system's single MBR on the val lines it covers)
    QW, EW = {}, {sn: 1.0 for sn in SYSTEMS_DONE}
    for sn in SYSTEMS_DONE:
        idx = [i for i, x in enumerate(NB["val"][sn]) if x]
        sc = official_metric([VAL_GT[i] for i in idx], [single_mbr(NB["val"][sn][i]) for i in idx])["score"]
        QW[sn] = 1.0 / max(1e-4, 1 - sc)
    log("system weights (1/lost points of single-system MBR): " + ", ".join(f"{k}={v:.1f}" for k, v in QW.items()))
    for sn in SYSTEMS_DONE:
        tv = [single_mbr(x) for x in NB["val"][sn]]
        tt = [single_mbr(x) for x in NB["test"][sn]] if sn in NB["test"] else None
        if all(NB["val"][sn]):
            CANDIDATES[f"{sn}__mbr"] = (tv, tt)
            CANDIDATES[f"{sn}__top1"] = ([x[0][0] for x in NB["val"][sn]],
                                         [x[0][0] for x in NB["test"][sn]] if sn in NB["test"] else None)
    base_top = [x[0][0] for x in NB["val"][CFG.BASELINE_SYSTEM]] if CFG.BASELINE_SYSTEM in NB["val"] else None
    RISKS = {}
    for method in CFG.ENSEMBLE_METHODS:
        if method.startswith("rescore") and "val" not in RESC:
            continue
        if method.endswith("_median") and _RFL is None:
            log(f"[ENS] {method} skipped (rapidfuzz missing)"); continue
        wts = EW if method == "pool_mbr_equal" else QW
        t0 = time.time()
        vh, vr = run_method(method, "val", wts, SYSTEMS_DONE)
        th, tr_ = run_method(method, "test", wts, SYSTEMS_DONE) if NB["test"] else (None, None)
        name = f"ENS[{'+'.join(SYSTEMS_DONE)}]__{method}"
        CANDIDATES[name] = (vh, th)
        RISKS[name] = (vr, tr_)
        report_trial(f"{TRIAL_ID}__{method}", TRIAL_DESCRIPTION, "+".join(SYSTEMS_DONE), method, vh, time.time() - t0,
                     TRAIN_SIZE, baseline_hyps=base_top, extra={"system": "ENSEMBLE", "weights": json.dumps(
                         {k: round(v, 2) for k, v in wts.items()})})
        pd.DataFrame({"ID": VAL_IDS, "GroundTruth": VAL_GT, "pred": vh, "risk": vr}).to_csv(
            os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__ENS_{method}__val.csv"), index=False)

    # ---- ranking with per-class / per-group columns and paired bootstrap vs the measured champion recipe ----
    ref_name = next((k for k in CANDIDATES if k.endswith("__pool_mbr_quality")), None)
    rows = []
    for k, (vh, th) in CANDIDATES.items():
        m = official_metric(VAL_GT, vh)
        r = {"candidate": k, "score": m["score"], "word/line": m["word_edits_per_line"], "char/line": m["char_edits_per_line"],
             "has_test": th is not None}
        for c in ("A", "B"):
            ix = [i for i, x in enumerate(VAL_CLASS) if x == c]
            r[c] = official_metric([VAL_GT[i] for i in ix], [vh[i] for i in ix])["score"] if ix else np.nan
        for g in ("A1", "A2", "A3"):
            ix = [i for i, x in enumerate(VAL_GROUP) if x == g]
            r[g] = official_metric([VAL_GT[i] for i in ix], [vh[i] for i in ix])["score"] if ix else np.nan
        if ref_name and k != ref_name:
            d = paired_delta(VAL_GT, CANDIDATES[ref_name][0], vh, n_boot=4000)
            r["vs_pool_mbr_quality"], r["P(better)"] = d["delta"], d["p_better"]
        rows.append(r)
    RANK = pd.DataFrame(rows).sort_values("score", ascending=False)
    RANK.to_csv(os.path.join(CFG.WORK_DIR, "mega_ranking_val.csv"), index=False)
    pd.set_option("display.width", 250)
    log("VALIDATION RANKING (official score, all 819 lines):\n" + RANK.round(5).to_string(index=False), banner=True)

    # ---- how much of the pool's headroom is captured ----
    orc = [min((line_cost(g, t) for sn in SYSTEMS_DONE for t, *_ in NB["val"][sn][i]), default=1.0)
           for i, g in enumerate(VAL_GT)]
    best_single = RANK[~RANK.candidate.str.startswith("ENS[")].score.max()
    best_ens = RANK[RANK.candidate.str.startswith("ENS[")].score.max()
    log(f"POOL ORACLE (best candidate of all systems per line) = {1 - np.mean(orc):.5f} | best single system "
        f"{best_single:.5f} | best ensemble {best_ens:.5f} -> the ensemble captures "
        f"{(best_ens - best_single) / max(1e-9, (1 - np.mean(orc)) - best_single):.0%} of the headroom above the best system")

    # ---- leave-one-system-out: what does each system add? ----
    for method in ("pool_mbr_quality", "rescore_mbr"):
        if method.startswith("rescore") and "val" not in RESC:
            continue
        full = official_metric(VAL_GT, run_method(method, "val", QW, SYSTEMS_DONE)[0])["score"]
        parts = []
        for sn in SYSTEMS_DONE:
            rest = [x for x in SYSTEMS_DONE if x != sn]
            if rest:
                parts.append(f"{sn}: {official_metric(VAL_GT, run_method(method, 'val', QW, rest)[0])['score'] - full:+.5f}")
        log(f"LEAVE-ONE-SYSTEM-OUT for {method} (score change when the system is removed; negative = it helps): "
            + " | ".join(parts))

    best = next((k for k in RANK.candidate if CANDIDATES[k][1] is not None), None)
    if best is not None:
        # "single_mbr:<system>" / "single_top1:<system>" when one system alone wins
        champ_method = best.split("__")[-1] if best.startswith("ENS[") else \
            ("single_" + best.split("__")[-1] + ":" + best.split("__")[0])
        W_FINAL = EW if champ_method == "pool_mbr_equal" else QW
        json.dump({"source": best, "method": champ_method, "weights": W_FINAL,
                   "systems": SYSTEMS_DONE, "val_score": float(RANK.set_index("candidate").at[best, "score"]),
                   "SYSTEMS": CFG.SYSTEMS}, open(os.path.join(CFG.WORK_DIR, "mega_champion.json"), "w"), indent=1, default=str)
        log(f"CHAMPION: {best} (val {RANK.set_index('candidate').at[best, 'score']:.5f}) -> mega_champion.json "
            f"(use it with RUN_MODE='full': FINAL_CHAMPION_JSON=<path>)")
elif CFG.RUN_MODE == "full" and NB["test"]:        # reproduce the Fold-0 champion recipe on test
    _ch = json.load(open(CFG.FINAL_CHAMPION_JSON)) if CFG.FINAL_CHAMPION_JSON and os.path.exists(CFG.FINAL_CHAMPION_JSON) else {}
    method = CFG.FINAL_METHOD or _ch.get("method") or ("rescore_mbr" if RESC.get("test") else "pool_mbr_quality")
    W_FINAL = dict(CFG.FINAL_WEIGHTS or {k: v for k, v in (_ch.get("weights") or {}).items() if k in SYSTEMS_DONE}
                   or {sn: 1.0 for sn in SYSTEMS_DONE})
    for sn in SYSTEMS_DONE:
        W_FINAL.setdefault(sn, float(np.mean(list(W_FINAL.values()))))
    if method.startswith("rescore") and "test" not in RESC:
        log(f"FULL MODE: {method} needs the rescoring pass (RESCORE=True) — falling back to pool_mbr_quality")
        method = "pool_mbr_quality"
    log(f"FULL MODE: method {method} | weights {W_FINAL}")
    if method.startswith("single_"):
        kind, sn = method[len("single_"):].split(":")
        th = [single_mbr(x) if kind == "mbr" else (x[0][0] if x else "") for x in NB["test"][sn]]
        tr_ = [0.0] * len(th)
    else:
        th, tr_ = run_method(method, "test", W_FINAL, SYSTEMS_DONE)
    best = f"ENS[{'+'.join(SYSTEMS_DONE)}]__{method}"
    CANDIDATES[best] = (None, th)
    RISKS = {best: (None, tr_)}

if best is not None and CANDIDATES[best][1] is not None:
    sub = pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(t) or "the" for t in CANDIDATES[best][1]]})
    sub.to_csv(CFG.SUBMISSION_CSV, index=False)
    log(f"submission.csv <- {best} | rows {len(sub)} | empty replaced {sum(not norm_text(t) for t in CANDIDATES[best][1])}")
    with open(os.path.join(CFG.WORK_DIR, "submission_source.json"), "w") as f:
        json.dump({"trial_session": TRIAL_ID, "source": best}, f)

# ---- pseudo-labels (test + unlisted) with the champion's posterior; risk = expected official cost ----
_pl = []
_meth = best.split("__")[-1] if (best and best.startswith("ENS[")) else "pool_mbr_quality"
_meth = "pool_mbr_quality" if _meth.startswith("single_") else _meth
_w = W_FINAL or {sn: 1.0 for sn in SYSTEMS_DONE}
for sp in ("test", "unlisted"):
    if not NB[sp]:
        continue
    m_ = _meth if (not _meth.startswith("rescore") or sp in RESC) else "pool_mbr_quality"
    hy, rk = run_method(m_, sp, _w, [sn for sn in SYSTEMS_DONE if sn in NB[sp]])
    for i, rid in enumerate(SPLITS[sp][0]):
        _pl.append({"ID": rid, "split": sp, "group": SPLITS[sp][2][i], "Target": hy[i], "risk": rk[i], "method": m_,
                    "n_systems": sum(bool(NB[sp][sn][i]) for sn in NB[sp])})
if _pl:
    _pl = pd.DataFrame(_pl)
    _pl.to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels.csv"), index=False)
    log(f"pseudo_labels.csv: {len(_pl)} lines | risk quantiles 50/70/90% = "
        f"{np.round(_pl.risk.quantile([.5, .7, .9]).values, 4).tolist()} (calibrate with work/08_build_pseudo_labels.py)")

if os.path.exists(TRIAL_RESULTS) and CFG.RUN_MODE == "fold0":
    res = pd.read_csv(TRIAL_RESULTS)
    sess = res.session == TRIAL_ID
    if sess.any():
        bi = res[sess].score.idxmax()
        res.loc[sess & (res.index != bi) & (res.status == "EVALUATED"), "status"] = "REJECTED"
        res.loc[bi, "status"] = "PROMOTED"
        res.to_csv(TRIAL_RESULTS, index=False)
        log("TRIAL TABLE (this session):\n" + res[sess][["trial", "variant", "score", "score_A", "score_B",
                                                          "word_edits_per_line", "char_edits_per_line", "delta_baseline",
                                                          "p_better_baseline", "status"]].round(5).to_string(index=False))
''')

# ============================================================================ forensics + summary
code(r'''
# =========================================================
# Cell 16 — forensics of the champion + run summary
# =========================================================
if CFG.RUN_MODE == "fold0" and CANDIDATES and best is not None:
    vh = CANDIDATES[best][0]
    log(f"CHAMPION {best}", banner=True)
    log("per class : " + json.dumps(score_by_group(VAL_GT, vh, VAL_CLASS)))
    log("per group : " + json.dumps(score_by_group(VAL_GT, vh, VAL_GROUP)))
    fx = error_forensics(VAL_GT, vh, top=12)
    for k in ("ops", "word_sub_classes", "word_sub", "char_sub", "char_del", "char_ins"):
        log(f"  {k}: {fx.get(k)}")
    if base_top is not None:
        lb = np.array([line_cost(r, h) for r, h in zip(VAL_GT, base_top)])
        lc = np.array([line_cost(r, h) for r, h in zip(VAL_GT, vh)])
        worst = [i for i in np.argsort(lc - lb)[::-1][:8] if lc[i] > lb[i]]
        if worst:
            log(f"  lines the champion made WORSE than {CFG.BASELINE_SYSTEM} top-1:")
            for i in worst:
                log(f"   {VAL_IDS[i]} [{VAL_GROUP[i]}] GT : {VAL_GT[i]}\n{'':14s}base: {base_top[i]}\n{'':14s}new : {vh[i]}")
if EVAL_CURVES:
    ec = pd.DataFrame(EVAL_CURVES)
    ec.to_csv(os.path.join(CFG.WORK_DIR, "eval_curves.csv"), index=False)
    piv = ec.pivot_table(index="step", columns=["system", "adapter"], values="inner_score")
    log("INNER-VALIDATION CURVES (official score of greedy decoding, by step):\n" + piv.round(4).to_string())
log(f"RUN FINISHED in {(time.time() - SESSION_T0) / 3600:.2f} h | outputs in {CFG.WORK_DIR}: submission.csv, "
    f"trial_results.csv, mega_ranking_val.csv, mega_champion.json, eval_curves.csv, adapter_summary.csv, "
    f"trial_predictions/*__D2_b5nb__<split>.csv, rescore__<split>.jsonl, pseudo_labels.csv, run_log.txt", banner=True)
''')

md(r'''
### After the run
1. **Hand back** `trial_results.csv`, `mega_ranking_val.csv`, `eval_curves.csv`, `adapter_summary.csv`, `run_log.txt`, `trial_predictions/` and `rescore__val.jsonl`. `python work/offline_ensembles.py <dir> --session MEGA_001` re-runs any ensemble offline.
2. **Read the leave-one-system-out lines.** A system whose removal *raises* the score is dead weight: drop it from `SYSTEMS` in the final run.
3. **Final submission:** set `RUN_MODE="full"` and `FINAL_CHAMPION_JSON=<attached Fold-0 output>/mega_champion.json`. Keep the same `SYSTEMS`, minus the dead weight. All systems are retrained on all lines, and the champion method and weights are reproduced on test.
4. **Pseudo-label round:** first run `python work/08_build_pseudo_labels.py` on the `trial_predictions/` folder, then set `PSEUDO_LABEL_CSV=pseudo_labels_final.csv`.

| Knob | When to change it |
|---|---|
| `NUM_TRAIN_EPOCHS=2` + `CKPT_METRIC="inner_official"` | The curves show the score still rising at the last step; the best step is restored anyway |
| `SYSTEMS += 8b_B_cont` | A class-B specialist continued from `8b_shared`. B holds 39 % of the loss |
| `SHARED_CLASS_HINT=True` | One shared model told the era (EXP_003 idea) |
| `RESCORE_SPLITS += ("unlisted",)` | Rescored pseudo-labels for the 687 unlisted images |
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        lines = s.split("\n")
        c = {"cell_type": t, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "id": f"m{i:02d}"}
        if t == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb["cells"].append(c)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "D:/HANAFY/ROAD/barbados-2-mega.ipynb"
    print("wrote", out, "cells:", len(build(out)["cells"]))
