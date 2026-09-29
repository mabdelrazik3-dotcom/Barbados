"""Builds test_pesudo.ipynb — PSEUDO-LABELLING ONLY (no training).

Already-trained adapters ("LABELERS": EXP_002 shared adapters, barbados-2-classwise / mega per-class adapters, or the
base model with the detailed sub-class prompts + few-shot) read every test and unlisted line (batch 1, beam 5, 5-best)
and JUDGE the guide transcription (your best submission) with their own log-probability. Their votes and the guide are
pooled by MBR under the official cost. Confidence is calibrated on the Fold-0 val lines, and every line gets
keep_for_training + a comment. Output: pseudo_labels.csv for the next (full-training) notebook.

The full inference -> fine-tune -> inference version of this notebook is kept in build_test_pesudo_notebook_v1_fullflow.py
(its fine-tuning cells are the starting point of the full-training notebook)."""
import ast
import importlib.util
import json

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


# the image / text helpers of barbados-2-enhanced Cell 5, WITHOUT the augmentation stack (no training here)
_c5 = find("# Cell 5 — text / image utils")
_seg = {n.name: ast.get_source_segment(_c5, n) for n in ast.parse(_c5).body if isinstance(n, ast.FunctionDef)}
UTILS_SRC = "\n\n".join(_seg[n] for n in ("clean_label", "clean_output", "load_image", "_paper_grey", "image_group",
                                          "image_class", "resolve_image", "model_short_name", "legacy_short_name",
                                          "model_size_tag", "free_cuda"))

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

# ============================================================================ intro
md(r'''
# R.O.A.D. Barbados — `test_pesudo`: **pseudo-labels for the test (and unlisted) images**

This notebook does **one thing**: write the best possible transcriptions for the unlabelled images, each with a calibrated confidence. It trains nothing. The next notebook trains on the training lines + these pseudo-labels.

| Step | What happens |
|---|---|
| **Classification** | Every image gets a class (A = 1630s–1660s, B = 1670s–1710s) and one of **10 pixel sub-classes**. It uses pixels only, so it works on test. |
| **Prompts** | A detailed prompt per sub-class, used by labelers with `prompt="detailed"`. Adapters that were trained with another prompt keep their own format. |
| **Labelers** | Each entry of `CFG.LABELERS` is an already-trained model: an EXP_002 shared adapter, `barbados-2-classwise` / mega per-class adapters, or the base model with the detailed prompts + few-shot training lines. Each reads every line with **batch size 1**, beam 5, and keeps its 5-best. |
| **Judge** | The **guide** (your best submission, `GUIDE_CSV`) is scored by every labeler with its own log-probability, so the guide competes on the models' own scale. `RESCORE_POOL=True` also makes every labeler score every other labeler's candidates. |
| **Pooling** | For each line: labeler posteriors (weighted by their val quality) + guide weight → MBR under the official cost picks the pseudo-label. |
| **Calibration on val** | The same procedure runs on the Fold-0 val lines (`GUIDE_VAL_CSV` for the guide). This picks the guide weight by 5-fold CV, reports honest scores of every labeler / the pool / the guide, and turns each line's risk into an **expected line score** and **P(exact)**. |
| **Output** | `pseudo_labels.csv` with ID, split, class, sub-class, pseudo-label, confidence, P(exact), risk, keep_for_training, each labeler's reading, the guide and a comment. It also writes `pseudo_labels_val_check.csv` and `submission_pseudo.csv` (for a leaderboard check). |

> Rules: pseudo-labelling is allowed when it is fully automated. The guide must be your own models' output (e.g. `submission_offline_best.csv`), never hand-corrected text. Calibration on val is honest only when the labelers' adapters were trained on Fold-0 **train** (EXP_002, CLS_001 and MEGA_001 fold-0 adapters are).
''')

md("### Installation")
code(find("# Cell 1 — OFFLINE install"))
code(next(s for t, s in E if t == "code" and s.lstrip().startswith("%%capture")))
code(find("# Cell 2 — imports"))

# ============================================================================ CFG
md("### Configuration")
code(r'''
# =========================================================
# Cell 3 — CFG
# =========================================================
TRIAL_ID          = "PSL_001"
TRIAL_DESCRIPTION = ("test_pesudo: pseudo-labels for test + unlisted from LABELERS + guide judged by every labeler, "
                     "pooled by MBR, confidence calibrated on Fold-0 val")

# the measured one-line prompt of barbados-2 / EXP_002 / enhanced / mega shared systems (user turn: prompt + image)
SHORT_PROMPT = ("This is not modern English, Transcribe the handwriting exactly. Keep the wrong spelling, abbreviations, "
                "and marks (^, ff, unusual letters).Do not modernize or correct anything.")
# the prompts of barbados-2-classwise / mega class systems (user turn: prompt + image), keyed by class / old group
_CW = ("Transcribe the handwritten line exactly as written. This is not modern English: keep the original spelling, "
       "capital letters, abbreviations and marks (^ for superscript letters, : for suspensions, &, ff). Do not modernize, "
       "expand or correct anything. If parts of neighbouring lines are visible, transcribe only the central line.")
_CW_A = (" This line comes from a Barbados legal record of the 1630s-1660s. Common conventions in these documents: "
         "& for 'and', ye and yt for 'the' and 'that', sd for 'said', and spellings such as heires, assignes, publique.")
_CW_B = (" This line comes from a Barbados legal record of the 1670s-1710s, scanned at high resolution. Common "
         "conventions in these documents: superscript abbreviations written with ^ (W^m, y^e, M^r, Adm^rs, Ex^rs), "
         "colon suspensions (Tho:, Exec:^rs, S:^d), ff at the start of words, and spellings such as heyres and saide.")
CLASSWISE_PROMPTS = {
    "A": _CW + _CW_A, "B": _CW + _CW_B,
    "A1": _CW + _CW_A + " The crop is tight around the line; '&' and ye/yt are very common.",
    "A2": _CW + _CW_A + " The parchment is dark and the hand is large; in these documents '&' and ye are rare and lines are short.",
    "A3": _CW + _CW_A + " The crop is loose and often shows parts of the lines above and below; the central line is usually long.",
}

class CFG:
    WHEELS_DIR = WHEELS
    BASE_MODELS = {
        "8b":  "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
        "7b":  "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2",
        "32b": "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/32b-instruct/1",
    }
    # ---- LABELERS: already-trained models that read the lines and judge the guide; their votes are pooled ----
    #  adapter  : None = base model | "auto" = the barbados-2 / enhanced adapter <model_short_name>/fold<FOLD> found
    #             under ADAPTER_SEARCH_ROOTS | a folder with adapter_config.json | scheme class/subclass: the folder
    #             that holds one sub-folder per class (adapter_pattern, e.g. class_A1 = barbados-2-classwise,
    #             8b_subclass__A1 = mega)
    #  prompt   : the format the adapter was TRAINED with: "exp002" (barbados-2, enhanced, mega shared) |
    #             "classwise" (barbados-2-classwise, mega class/subclass) | "detailed" (sub-class prompts below) | "short"
    #  fewshot_k: solved training lines of the same sub-class shown first (0 for adapters trained without examples)
    #  scheme   : "shared" | "class" (A/B adapters) | "subclass" (A1/A2/A3/B adapters)
    LABELERS = [
        dict(name="8b_exp002", model="8b", adapter="auto", prompt="exp002", fewshot_k=0),
        dict(name="7b_exp002", model="7b", adapter="auto", prompt="exp002", fewshot_k=0),
        # dict(name="8b_subclass", model="8b", scheme="subclass", prompt="classwise", adapter_pattern="class_{key}",
        #      adapter="/kaggle/input/<cls001-output>/qwen-vl-ocr/final/qwen_3_vl_transformers_8b_instruct_1/fold0"),
        # dict(name="8b_base_fewshot", model="8b", adapter=None, prompt="detailed", fewshot_k=3),
    ]
    ADAPTER_SEARCH_ROOTS = ["/kaggle/input"]
    ADAPTER_FOLD = None       # folder name "auto" looks for: None -> "fold<FOLD>" in RUN_MODE="fold0", "full" otherwise
    REUSE_DIRS = []           # earlier outputs of this notebook (attached): labeler_outputs/<name>__<split>.csv are
                              # re-used instead of decoded again (resume after a timeout, or add one labeler later)
    DATA_DIR    = "/kaggle/input/datasets/haniagamal/road-barbados"
    TRAIN_CSV   = None
    TEST_CSV    = None
    IMAGES_DIR  = None
    AUTODISCOVER = True
    OUTPUT_DIR  = "/kaggle/working/pseudo"
    FINAL_DIR   = "/kaggle/working/pseudo"
    WORK_DIR    = "/kaggle/working"
    SUBMISSION_CSV = "/kaggle/working/submission_pseudo.csv"
    PREV_RESULTS_DIRS = []

    # ---- protocol ----
    RUN_MODE        = "fold0"     # "fold0": the Fold-0 val lines calibrate confidence and choose the guide weight
                                  #   (honest only if the labelers' adapters were trained on Fold-0 TRAIN, e.g. EXP_002)
                                  # "full" : adapters trained on all 4,093 lines -> no val lines, no calibration
    FOLD            = 0
    N_FOLDS         = 5
    CKPT_SELECTION  = "last"      # no inner slice: every training line is available as a few-shot example
    INNER_CKPT_FRAC = 0.0
    VAL_LIMIT       = None        # decode only this many val lines (stratified over sub-classes); None = all 819
    PSEUDO_SPLITS   = ("test", "unlisted")
    TIME_BUDGET_H   = 11.3

    # ---- classification ----
    SUBCLASS_MODE  = "fixed"      # "fixed": the 10 sub-classes of work/10_subclass_tree.py | "refit": new tree here
    N_SUBCLASSES   = 10           # "refit" only (<= 15)
    SUBCLASS_MIN_LINES = 110
    CLASS_HEIGHT_SPLIT  = 150
    GROUP_A1_MAX_HEIGHT = 58
    GROUP_A2_MAX_PAPER  = 194

    # ---- detailed prompts (labelers with prompt="detailed") ----
    PROMPT_TEXT_EXAMPLES = 6
    USER_INSTRUCTION = "Transcribe this line."

    # ---- decoding (batch size 1) ----
    MAX_PIXELS      = 2_000_000
    MIN_PIXELS      = 256*28*28
    ATTN_IMPL       = None
    USE_BF16        = True
    MAX_NEW_TOKENS  = 160
    DECODE          = dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=5)
    LOOP_GUARD      = 3
    MBR_TEMPERATURE = 1.0
    LOG_EVERY_LINES = 200

    # ---- guide + judge + pooling ----
    GUIDE_CSV      = "submission_offline_best.csv"   # best test submission so far (ID, Target), or
                                  #   pseudo_labels_final.csv (test + unlisted). A bare file name is searched under
                                  #   GUIDE_SEARCH_ROOTS; None = no guide
    GUIDE_VAL_CSV  = "EXP_002__ENS_7b8b_mbrq__D2_b5nb_ens__val.csv"  # the same system's Fold-0 val predictions
                                  #   (ID, pred): 7B+8B pooled MBR, measured 0.9175
    GUIDE_SEARCH_ROOTS = ["/kaggle/input"]
    GUIDE_WEIGHT   = "auto"       # prior mass of the guide in the evidence (0 = labelers + judge only, 1 = copy the guide)
                                  #   "auto": chosen on val by 5-fold CV (needs GUIDE_VAL_CSV), else 0.5
    JUDGE_GUIDE    = True         # every labeler scores the guide text with its own log-probability
    RESCORE_POOL   = False        # every labeler also scores every OTHER labeler's candidates (second pass, ~1 h/labeler)
    LABELER_WEIGHTS = "quality"   # "quality": 1 / lost points of each labeler's MBR on val | "equal"

    # ---- which pseudo-labels the next notebook should train on ----
    PSEUDO_KEEP      = "agree_or_lowrisk"   # "agree" (pick = guide = a labeler's top-1) | "lowrisk" | "agree_or_lowrisk"
    PSEUDO_KEEP_FRAC = 0.7        # "lowrisk": the lowest-risk share kept inside every sub-class
    MIN_CONFIDENCE   = None       # optional extra cut on the calibrated expected line score (e.g. 0.95)
    MIN_WORDS, MAX_WORDS = 2, 40

    SEED = 42
    CORRUPT_IDS = {
        "79tMUVyfIdy3GzkG",  "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO",
    }
    # ---- used by the shared helper cells ----
    PSEUDO_LABEL_CSV = None; PSEUDO_PER_CLASS = True; LEARNING_RATE = None; AUG_PROFILE = "-"; CLASS_HINT = False
    CLASS_AUG_COPIES = None; DECODE_VARIANTS = {"beam5": DECODE}; BASELINE_VARIANT = "beam5"

CFG.MODEL_PATHS = list(dict.fromkeys(CFG.BASE_MODELS[L["model"]] for L in CFG.LABELERS))
TRIAL_CONFIG = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)}
print("TRIAL:", TRIAL_ID, "|", TRIAL_DESCRIPTION)
print(f"mode {CFG.RUN_MODE} | labelers {[L['name'] for L in CFG.LABELERS]} | guide {'yes' if CFG.GUIDE_CSV else 'no'} | "
      f"val guide {'yes' if CFG.GUIDE_VAL_CSV else 'no'} | guide weight {CFG.GUIDE_WEIGHT} | rescore pool {CFG.RESCORE_POOL}")
''')

code(find("# Cell 4 — resolve & sanity-check paths"))
md(r'''
### Helpers: image loading, processor/model loaders, the official metric
These are shared with `barbados-2-enhanced.ipynb`, minus the augmentation, since nothing is trained here. They include the pixel-limit fix for new `transformers` versions and the official score `0.5·(1 − word_edits/12) + 0.5·(1 − char_edits/55)`.
''')
code(r'''
# =========================================================
# Cell 5 — text / image helpers (from barbados-2-enhanced Cell 5, no augmentation)
# =========================================================
_GROUP_CACHE = {}

__UTILS__
''')
code(find("# Cell 6 — model & processor loaders"))
code(find("# Cell 7 — OFFICIAL metric"))

md(r'''
### Data: Fold-0 split, test and unlisted images
The split is the same as every earlier run. The 819 Fold-0 val lines are used **only to measure and calibrate** the labelers. All training lines are available as few-shot examples.
''')
code(find("# Cell 8 — samples, Fold-0 split"))
code(find("# Cell 11 — trial harness"))

# ============================================================================ classification
md(r'''
### Classification: class and sub-class of every image
Cheap pixel features (size, parchment tone, contrast, pen width, focus, ink from neighbouring lines) go through the fixed rule tree. It was fitted offline by `work/10_subclass_tree.py`, and it re-applies identically here (300 of 300 checked).

The sub-class drives three things:
- the detailed prompts;
- the few-shot examples;
- the per-sub-class keep rule and report.

| Sub-class | Rule | Character (training lines) |
|---|---|---|
| A01 | A, parchment grey ≤ 187 | dark parchment; protests/depositions; capitals, commas |
| A02 | A, width ≤ 941 px | narrow crops: attestation and dating lines |
| A03 | A, width 942–1048, fine pen | conveyances; ye/yt; ~interlined~ words |
| A04 | A, width 942–1048, heavy pen | bonds; & in half of the lines, sd |
| A05 | A, width > 1048, fine pen | long early bonds; archaic spellings |
| A06 | A, width > 1048, heavy pen, cool paper | shipping and mercantile records |
| A07 | A, width > 1048, heavy pen, warm paper | heavily abbreviated conveyances |
| B01 | B, soft focus | formal deeds; capitals; commas; end dashes |
| B02 | B, crisp | colon-suspension hand (S:^d, Exec:^rs) |
| B03 | B, width > 5032 px | very long lines: affidavits, bills |
''')
code(r'''
# =========================================================
# Cell 9 — CLASSIFICATION: pixel features -> class (A/B) and sub-class for every image
# =========================================================
import random as _random

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
    """new pixel-only tree (<= 15 leaves) fitted on the TRAIN lines; B gets ~25 % of the leaves."""
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
        if k > 1 and m.sum() > 2 * CFG.SUBCLASS_MIN_LINES:
            mdl = DecisionTreeRegressor(max_leaf_nodes=k, min_samples_leaf=CFG.SUBCLASS_MIN_LINES, random_state=0)
            mdl.fit(X[m], Y[m])
            models[e] = (mdl, {lf: f"{e}{j + 1:02d}" for j, lf in enumerate(sorted(set(mdl.apply(X[m]))))})
        else:
            models[e] = (None, {})
    log(f"[CLASSIFY] refit tree: A {len(models['A'][1]) or 1} leaves, B {len(models['B'][1]) or 1} leaves")
    def assign(f, height_split=CFG.CLASS_HEIGHT_SPLIT):
        e = "B" if f["height"] > height_split else "A"
        mdl, names = models[e]
        if mdl is None:
            return f"{e}01"
        return names.get(mdl.apply(pd.DataFrame([f])[SUB_FEATS])[0], f"{e}01")
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
    _rows.append({"sub-class": k, "class": k[0], "train": sum(s["sub"] == k for s in core_samples + inner_samples),
                  "val": VAL_SUB.count(k), "test": TEST_SUB.count(k), "unlisted": UNL_SUB.count(k),
                  "height": med.get("height", np.nan), "width": med.get("width", np.nan), "paper": med.get("paper", np.nan),
                  "tint": med.get("tint", np.nan), "contrast": med.get("contrast", np.nan),
                  "stroke_rel": med.get("stroke_rel", np.nan), "sharp": med.get("sharp", np.nan)})
SUB_TABLE = pd.DataFrame(_rows)
log(f"[CLASSIFY] mode {CFG.SUBCLASS_MODE}: {len(SUB_KEYS)} sub-classes\n" + SUB_TABLE.round(3).to_string(index=False))
pd.DataFrame([{"ID": i, "split": sp, "class": sb[0], "group": gp, "subclass": sb, **FEATS.get(i, {})}
              for sp, ids_, subs_, grps_ in (
                  ("train", [s["id"] for s in core_samples + inner_samples], [s["sub"] for s in core_samples + inner_samples],
                   [s["grp"] for s in core_samples + inner_samples]),
                  ("val", VAL_IDS, VAL_SUB, VAL_GROUP), ("test", TEST_IDS, TEST_SUB, TEST_GROUP),
                  ("unlisted", UNL_IDS, UNL_SUB, UNL_GROUP))
              for i, sb, gp in zip(ids_, subs_, grps_)]).to_csv(os.path.join(CFG.WORK_DIR, "subclass_assignments.csv"), index=False)
log("[CLASSIFY] saved subclass_assignments.csv (ID, split, class, old group, subclass, pixel features)")
''')

# ============================================================================ prompts
md(r'''
### Detailed prompts: one per sub-class
These are only used by labelers with `prompt="detailed"`, e.g. the base model with few-shot examples. Adapters trained with the one-line prompt keep that prompt, because a trained adapter reads best in its own training format.

Each prompt follows `prompt.txt`: ROLE, INPUTS, WHAT THIS SUB-CLASS LOOKS LIKE, TASK, ANALYSIS STRATEGY, 12 TRANSCRIPTION RULES with the rates measured on this sub-class's training lines, HABITS, COMMON MISTAKES, WHAT TO IGNORE, EXAMPLES, SILENT RE-CHECK and OUTPUT FORMAT.
''')
code(r'''
# =========================================================
# Cell 10 — sub-class profiles (training lines only) and the detailed prompt of every sub-class
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

_TRAIN_BY_SUB = {k: [s for s in core_samples + inner_samples if s["sub"] == k] for k in SUB_KEYS}
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
    """percentage -> words, e.g. 26 -> 'about one line in 4 (26 %)'"""
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

_K_DETAILED = max([L_.get("fewshot_k", 0) for L_ in CFG.LABELERS if L_.get("prompt") == "detailed"] or [0])
PROMPTS = {k: build_prompt(k, _K_DETAILED) for k in SUB_KEYS}
with open(os.path.join(CFG.WORK_DIR, "subclass_prompts.json"), "w", encoding="utf-8") as f:
    json.dump({"profiles": PROFILES, "prompts": PROMPTS}, f, indent=1, ensure_ascii=False)
_prof = pd.DataFrame(PROFILES).T
log("[PROMPTS] conventions measured on the training lines of each sub-class (% of lines):\n" +
    _prof[[c for c in ("n", "chars", "words", "caret", "colon", "amp", "thorn", "ff", "digit", "tilde", "comma",
                       "Said", "sd", "cap_words") if c in _prof]].round(1).to_string())
if any(L_.get("prompt") == "detailed" for L_ in CFG.LABELERS):
    _show = max(SUB_KEYS, key=lambda k: PROFILES.get(k, {}).get("n", 0))
    print(f"\n==================== detailed PROMPT of sub-class {_show} (all in subclass_prompts.json) ====================\n")
    print(PROMPTS[_show])
''')

# ============================================================================ conversation, generation, judge
md(r'''
### Conversation formats, batch-1 generation and the judge
Each labeler talks to the model in the format its adapter was trained with:

| `prompt` | Conversation |
|---|---|
| `exp002` | user: [one-line prompt, line image] — exactly `barbados-2` / EXP_002 / mega shared |
| `classwise` | user: [class prompt (A/B or A1/A2/A3/B), line image] — exactly `barbados-2-classwise` / mega class systems |
| `detailed` | system: the sub-class prompt; then K solved examples (user: image, assistant: ground truth); then user: line image |
| `short` | as `detailed`, with the one-line prompt as the system turn |

**Few-shot examples** are the K training lines of the same sub-class nearest in pixel features, with the nearest one last.

**The judge** computes the exact log-probability a labeler gives to a text: the guide, or another labeler's candidate. It uses the same conversation. The text's score is the sum over its tokens up to `<|im_end|>`, which matches the generation scores.
''')
code(r'''
# =========================================================
# Cell 11 — few-shot retrieval, conversation per labeler, batch-1 generation, forced scoring (judge)
# =========================================================
def _fs_vec(f):
    return np.array([math.log(max(f["height"], 1)), math.log(max(f["width"], 1)), f["paper"] / 50.0, f["tint"] / 20.0,
                     f["contrast"] / 30.0, f["stroke_rel"] * 20.0, f["sharp"] * 10.0, f["edge_ratio"] * 2.0,
                     f["band_frac"] * 2.0], float)

EX_POOL = {}
for k in SUB_KEYS:
    pool = [s for s in core_samples + inner_samples if s["sub"] == k and s["id"] in FEATS]
    M = np.stack([_fs_vec(FEATS[s["id"]]) for s in pool]) if pool else np.zeros((0, 9))
    mu, sd = (M.mean(0), M.std(0) + 1e-6) if len(M) else (np.zeros(9), np.ones(9))
    EX_POOL[k] = (pool, (M - mu) / sd, mu, sd)
TRAIN_BY_ID = {s["id"]: s for s in core_samples + inner_samples}
if any(L_.get("fewshot_k", 0) for L_ in CFG.LABELERS):
    log("[FEW-SHOT] example pools (training lines only): " + ", ".join(f"{k}={len(v[0])}" for k, v in EX_POOL.items()))

def exemplars_for(qid, sub, k):
    """the k training lines of sub-class `sub` nearest to image `qid` in pixel features (never qid itself), nearest LAST"""
    pool, Z, mu, sd = EX_POOL.get(sub, ([], None, None, None))
    if k <= 0 or not pool or qid not in FEATS:
        return []
    d = ((Z - (_fs_vec(FEATS[qid]) - mu) / sd) ** 2).sum(1)
    order = [j for j in np.argsort(d, kind="stable") if pool[j]["id"] != qid]
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

def labeler_key(lab, grp):
    """adapter / prompt key of a line for a labeler: 'ALL' (shared) | 'A'/'B' (class) | 'A1'..'B' (subclass, old groups)"""
    return "ALL" if lab["scheme"] == "shared" else (grp if lab["scheme"] == "subclass" else grp[0])

def conversation(lab, img, sub, grp, exs):
    """-> (messages without the answer, images in order of appearance), in the labeler's training format"""
    msgs, imgs = [], []
    if lab["prompt"] in ("exp002", "classwise"):
        txt = SHORT_PROMPT if lab["prompt"] == "exp002" else CLASSWISE_PROMPTS[labeler_key(lab, grp)]
        for ex in exs:
            e = load_image_cached(ex["image"])
            msgs += [{"role": "user", "content": [{"type": "text", "text": txt}, {"type": "image", "image": e}]},
                     {"role": "assistant", "content": [{"type": "text", "text": ex["text"]}]}]
            imgs.append(e)
        msgs.append({"role": "user", "content": [{"type": "text", "text": txt}, {"type": "image", "image": img}]})
    else:
        system = PROMPTS[sub] if lab["prompt"] == "detailed" else SHORT_PROMPT
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
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

def generate(model, proc, msgs, imgs, var):
    """batch size 1 -> N-best [(text, total_logprob, n_tokens, truncated)]"""
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

def forced_logprobs(model, proc, msgs, imgs, texts):
    """JUDGE: exact log-probability the model gives to each text as the answer (same conversation), batch 1"""
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

SPLIT_DATA = {"val": (VAL_IDS, VAL_PATHS, VAL_SUB, VAL_GROUP, VAL_GT),
              "test": (TEST_IDS, TEST_PATHS, TEST_SUB, TEST_GROUP, None),
              "unlisted": (UNL_IDS, UNL_PATHS, UNL_SUB, UNL_GROUP, None)}
''')

# ============================================================================ guide
md(r'''
### The guide files
`GUIDE_CSV` is the transcription you currently trust most for every test line, usually your best submission. `pseudo_labels_final.csv` (test + unlisted) also works.

`GUIDE_VAL_CSV` is the same system's Fold-0 val predictions. With it, the notebook measures:
- the guide alone;
- the labelers alone;
- the pooled combination.

It then chooses the guide weight by cross-validation before anything is applied to test.
''')
code(r'''
# =========================================================
# Cell 12 — guide files
# =========================================================
def find_file(path):
    """a full path, or a bare file name searched under GUIDE_SEARCH_ROOTS (attached Kaggle datasets)"""
    if not path or os.path.exists(path):
        return path
    for r in CFG.GUIDE_SEARCH_ROOTS:
        hits = sorted(_glob(os.path.join(r, "**", os.path.basename(path)), recursive=True), key=len)
        if hits:
            log(f"[GUIDE] {os.path.basename(path)} found at {hits[0]}")
            return hits[0]
    return path

def load_guide(path):
    """-> {ID: text} from a submission (Target), a pseudo-label file (pseudo_label / Target) or val predictions (pred)"""
    path = find_file(path)
    if not path or not os.path.exists(path):
        if path:
            log(f"[GUIDE] file not found: {path} — attach it as a dataset or give the full path (running without it)")
        return {}
    d = pd.read_csv(path)
    col = next((c for c in ("Target", "pseudo_label", "pred", "mbr") if c in d.columns), None)
    assert col is not None, f"{path}: needs a Target / pseudo_label / pred column"
    txt = {str(i).strip(): norm_text(t) for i, t in zip(d.ID, d[col].fillna("")) if norm_text(t)}
    log(f"[GUIDE] {os.path.basename(path)}: {len(txt)} lines from column '{col}'")
    return txt

GUIDE = load_guide(CFG.GUIDE_CSV)
GUIDE_VAL = load_guide(CFG.GUIDE_VAL_CSV) if CFG.RUN_MODE == "fold0" else {}
def guide_for(split):
    return GUIDE_VAL if split == "val" else GUIDE
if GUIDE:
    log(f"[GUIDE] covers {sum(i in GUIDE for i in TEST_IDS)}/{len(TEST_IDS)} test and "
        f"{sum(i in GUIDE for i in UNL_IDS)}/{len(UNL_IDS)} unlisted lines")
if GUIDE_VAL:
    _gv = [GUIDE_VAL.get(i, "") for i in VAL_IDS]
    log(f"[GUIDE] guide on the {len(VAL_IDS)} val lines: official {official_metric(VAL_GT, _gv)['score']:.5f} | "
        "per sub-class " + json.dumps(score_by_group(VAL_GT, _gv, VAL_SUB)))
''')

# ============================================================================ labelers
md(r'''
### Run the labelers
One labeler at a time is loaded, with its adapter(s). Each labeler reads:
- the val lines, for measurement and calibration;
- the test lines;
- the unlisted lines.

Batch size is 1, with beam 5 and the 5-best kept, and the labeler judges the guide on every line where the guide is not already in its 5-best. The outputs go to `labeler_outputs/<name>__<split>.csv`. With `REUSE_DIRS`, a finished labeler or split is read back instead of decoded again, so an interrupted Kaggle session can resume.
''')
code(r'''
# =========================================================
# Cell 13 — LABELERS: decode + judge the guide
# =========================================================
dtype = torch.bfloat16 if (CFG.USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAB_DIR = os.path.join(CFG.WORK_DIR, "labeler_outputs")
os.makedirs(LAB_DIR, exist_ok=True)
SCHEME_KEYS = {"shared": ["ALL"], "class": ["A", "B"], "subclass": ["A1", "A2", "A3", "B"]}
ADAPTER_FOLD_NAME = CFG.ADAPTER_FOLD or (f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full")

def resolve_adapter_auto(msrc):
    """exact-name lookup of a barbados-2 / enhanced adapter: <root>/<model_short_name>/fold<FOLD> (never fuzzy)"""
    names = [model_short_name(msrc), legacy_short_name(msrc)]
    fold = ADAPTER_FOLD_NAME
    for r in CFG.ADAPTER_SEARCH_ROOTS:
        for nm in names:
            for c in (os.path.join(r, nm, fold), os.path.join(r, fold, nm)):
                if os.path.isfile(os.path.join(c, "adapter_config.json")):
                    return c
    for r in CFG.ADAPTER_SEARCH_ROOTS:
        for cfgp in _glob(os.path.join(r, "**", "adapter_config.json"), recursive=True):
            parts = os.path.dirname(cfgp).replace("\\", "/").split("/")
            if parts[-1] == fold and len(parts) >= 2 and parts[-2] in names:
                return os.path.dirname(cfgp)
    return None

LABS = []
for _L in CFG.LABELERS:
    L = {"adapter": None, "prompt": "exp002", "fewshot_k": 0, "scheme": "shared", "adapter_pattern": "class_{key}", **_L}
    assert L["model"] in CFG.BASE_MODELS, f"labeler {L['name']}: unknown model {L['model']!r}"
    assert L["prompt"] in ("exp002", "classwise", "detailed", "short"), f"labeler {L['name']}: bad prompt {L['prompt']!r}"
    assert L["scheme"] in SCHEME_KEYS, f"labeler {L['name']}: scheme must be shared / class / subclass"
    assert not (L["prompt"] == "classwise" and L["scheme"] == "shared"), f"{L['name']}: 'classwise' prompts need class/subclass"
    msrc = CFG.BASE_MODELS[L["model"]]
    if L["scheme"] == "shared":
        ad = resolve_adapter_auto(msrc) if L["adapter"] == "auto" else L["adapter"]
        if L["adapter"] and (not ad or not os.path.isfile(os.path.join(ad, "adapter_config.json"))):
            log(f"[LABELER] {L['name']}: adapter {L['adapter']!r} not found (looked for "
                f"<{model_short_name(msrc)}>/{ADAPTER_FOLD_NAME}/adapter_config.json under {CFG.ADAPTER_SEARCH_ROOTS}) — "
                f"SKIPPED. Set ADAPTER_FOLD or give the folder path.")
            continue
        L["paths"] = {"ALL": ad}
    else:
        L["paths"] = {k: os.path.join(L["adapter"], L["adapter_pattern"].format(key=k)) for k in SCHEME_KEYS[L["scheme"]]}
        missing = [p for p in L["paths"].values() if not os.path.isfile(os.path.join(p, "adapter_config.json"))]
        if missing:
            log(f"[LABELER] {L['name']}: missing class adapters {missing} — SKIPPED")
            continue
    LABS.append(L)
def _adapter_desc(L):
    if L["scheme"] == "shared":
        return L["paths"]["ALL"] or "none (base model)"
    return f"{L['adapter']} / {L['adapter_pattern']} for {sorted(L['paths'])}"
log("LABELERS:\n" + "\n".join(f"  {L['name']:18s} model {L['model']:4s} scheme {L['scheme']:8s} prompt {L['prompt']:9s} "
                              f"few-shot K={L['fewshot_k']} | adapter {_adapter_desc(L)}" for L in LABS), banner=True)
assert LABS, "no usable labeler — check CFG.LABELERS / ADAPTER_SEARCH_ROOTS"

LAB_OUT = {}      # name -> split -> {ID: {"nbest": [(text, lp, n_tok, truncated)], "judged": {text: lp}, "ex": [ids]}}

def load_labeler(L):
    msrc = CFG.BASE_MODELS[L["model"]]
    proc = load_vlm_processor(msrc, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
    base = load_vlm_model(msrc, dtype)
    base.to(DEVICE)
    if L["scheme"] == "shared":
        model = PeftModel.from_pretrained(base, L["paths"]["ALL"]) if L["paths"]["ALL"] else base
    else:
        model = None
        for k, p in L["paths"].items():
            if model is None:
                model = PeftModel.from_pretrained(base, p, adapter_name=k)
            else:
                model.load_adapter(p, adapter_name=k)
    model.eval()
    model.config.use_cache = True
    return model, proc

def _use_adapter(model, L, grp):
    if L["scheme"] != "shared":
        model.set_adapter(labeler_key(L, grp))

def _lines(split):
    ids, paths, subs, grps, gts = SPLIT_DATA[split]
    idx = list(range(len(ids)))
    if split == "val" and CFG.VAL_LIMIT and CFG.VAL_LIMIT < len(ids):
        rng = np.random.RandomState(CFG.SEED)
        by = collections.defaultdict(list)
        for i, s in enumerate(subs):
            by[s].append(i)
        for v in by.values():
            rng.shuffle(v)
        idx, keys = [], sorted(by)
        while len(idx) < CFG.VAL_LIMIT:
            for k in keys:
                if by[k] and len(idx) < CFG.VAL_LIMIT:
                    idx.append(by[k].pop())
        idx = sorted(idx)
    return idx

def _save_split(L, split, out):
    ids, paths, subs, grps, gts = SPLIT_DATA[split]
    gpos = {i: n for n, i in enumerate(ids)}
    pd.DataFrame([{"ID": i, "subclass": subs[gpos[i]], "group": grps[gpos[i]],
                   "GroundTruth": gts[gpos[i]] if gts is not None else None,
                   "pred": e["nbest"][0][0] if e["nbest"] else "",
                   "nbest": json.dumps([[t, round(s_, 4), n_, tr_] for t, s_, n_, tr_ in e["nbest"]], ensure_ascii=False),
                   "judged": json.dumps({t: round(s_, 4) for t, s_ in e["judged"].items()}, ensure_ascii=False),
                   "exemplars": " ".join(e["ex"])} for i, e in out.items()]
                 ).to_csv(os.path.join(LAB_DIR, f"{L['name']}__{split}.csv"), index=False)

def _load_reused(L, split):
    for d in CFG.REUSE_DIRS:
        for p in (os.path.join(d, f"{L['name']}__{split}.csv"), os.path.join(d, "labeler_outputs", f"{L['name']}__{split}.csv")):
            if os.path.exists(p):
                df = pd.read_csv(p)
                out = {str(r.ID): {"nbest": [(t, float(s_), int(n_), bool(tr_)) for t, s_, n_, tr_ in json.loads(r.nbest)],
                                   "judged": {t: float(s_) for t, s_ in json.loads(r.judged).items()},
                                   "ex": str(r.exemplars).split() if isinstance(r.exemplars, str) else []}
                       for r in df.itertuples()}
                log(f"[{L['name']}] {split}: {len(out)} lines RE-USED from {p}")
                return out
    return None

def val_report(tag, ids, hyps):
    gpos = {i: n for n, i in enumerate(VAL_IDS)}
    gt = [VAL_GT[gpos[i]] for i in ids]
    subs = [VAL_SUB[gpos[i]] for i in ids]
    m = official_metric(gt, hyps)
    msg = (f"[VAL] {tag:34s} official {m['score']:.5f} | word/line {m['word_edits_per_line']:.3f} | char/line "
           f"{m['char_edits_per_line']:.3f} | n={m['n']}")
    if GUIDE_VAL and all(i in GUIDE_VAL for i in ids):
        d = paired_delta(gt, [GUIDE_VAL[i] for i in ids], hyps)
        msg += f" | vs guide {d['delta']:+.5f} CI95 [{d['ci95_lo']:+.5f},{d['ci95_hi']:+.5f}] P(better)={d['p_better']:.3f}"
    log(msg)
    log(f"[VAL]   per sub-class: " + json.dumps(score_by_group(gt, hyps, subs)))
    if len(ids) == len(VAL_IDS) and list(ids) == list(VAL_IDS):
        report_trial(f"{TRIAL_ID}__{tag.replace(' ', '_')}", TRIAL_DESCRIPTION, tag, "pseudo-label", hyps, 0.0,
                     len(core_samples) + len(inner_samples),
                     baseline_hyps=[GUIDE_VAL[i] for i in ids] if (GUIDE_VAL and all(i in GUIDE_VAL for i in ids)) else None)
    return m["score"]

def mbr_own(e):
    nb = [x for x in e["nbest"] if x[0]]
    return mbr_select([t for t, *_ in nb], nbest_posteriors([s_ for _, s_, *_ in nb], CFG.MBR_TEMPERATURE)) if nb else ""

LAB_VAL_SCORE = {}
splits = (["val"] if CFG.RUN_MODE == "fold0" and VAL_IDS else []) + [s for s in CFG.PSEUDO_SPLITS if SPLIT_DATA[s][0]]
for L in LABS:
    if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
        log(f"[TIME] budget {CFG.TIME_BUDGET_H} h reached — labeler {L['name']} skipped")
        continue
    name = L["name"]
    LAB_OUT[name] = {}
    todo = []
    for sp in splits:
        r = _load_reused(L, sp)
        if r is not None:
            LAB_OUT[name][sp] = r
        else:
            todo.append(sp)
    if todo:
        t_l = time.time()
        log(f"LABELER {name}: loading {L['model']} ({L['scheme']}, prompt={L['prompt']}, K={L['fewshot_k']}) for {todo}", banner=True)
        model, proc = load_labeler(L)
        ids0, paths0, subs0, grps0, _ = SPLIT_DATA[todo[0]]
        j0 = next((j for j, p in enumerate(paths0) if p and os.path.exists(p)), None)
        if j0 is not None:                       # show the exact conversation of one line
            _use_adapter(model, L, grps0[j0])
            _ex = exemplars_for(ids0[j0], subs0[j0], L["fewshot_k"])
            _m, _im = conversation(L, load_image(paths0[j0]), subs0[j0], grps0[j0], _ex)
            _txt = proc.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
            log(f"[{name}] conversation of line {ids0[j0]} ({subs0[j0]}/{grps0[j0]}, {len(_ex)} examples): "
                f"{proc(text=[_txt], images=_im, return_tensors='pt')['input_ids'].shape[1]} tokens\n"
                + re.sub(r"(<\|image_pad\|>)+", "<IMAGE>", _txt)[-700:])
        for sp in todo:
            ids, paths, subs, grps, gts = SPLIT_DATA[sp]
            guide = guide_for(sp)
            out, t0, n_judged = {}, time.time(), 0
            idx = _lines(sp)
            for n, i in enumerate(idx):
                if not (paths[i] and os.path.exists(paths[i])):
                    out[ids[i]] = {"nbest": [("", float("-inf"), 0, False)], "judged": {}, "ex": []}
                    continue
                _use_adapter(model, L, grps[i])
                exs = exemplars_for(ids[i], subs[i], L["fewshot_k"])
                msgs, imgs = conversation(L, load_image(paths[i]), subs[i], grps[i], exs)
                try:
                    nb = generate(model, proc, msgs, imgs, CFG.DECODE)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    nb = [("", float("-inf"), 0, False)]
                    log(f"[{name}] OOM on {ids[i]} — empty")
                judged = {}
                g = guide.get(ids[i]) if guide else None
                if g and CFG.JUDGE_GUIDE and g not in {t for t, *_ in nb}:
                    judged[g] = forced_logprobs(model, proc, msgs, imgs, [g])[0]
                    n_judged += 1
                out[ids[i]] = {"nbest": nb, "judged": judged, "ex": [e["id"] for e in exs]}
                if (n + 1) % CFG.LOG_EVERY_LINES == 0 or n + 1 == len(idx):
                    rate = (n + 1) / max(1e-9, time.time() - t0)
                    msg = (f"[{name}] {sp}: {n + 1}/{len(idx)} lines | {rate * 60:.0f} lines/min | ETA "
                           f"{(len(idx) - n - 1) / max(rate, 1e-9) / 60:.1f} min | guide judged on {n_judged} lines")
                    if gts is not None:
                        done = [j for j in idx[:n + 1] if ids[j] in out]
                        msg += (f" | running top-1 official "
                                f"{official_metric([gts[j] for j in done], [out[ids[j]]['nbest'][0][0] for j in done])['score']:.4f}")
                    log(msg)
            LAB_OUT[name][sp] = out
            _save_split(L, sp, out)
        free_cuda("model")
        del proc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[{name}] done in {(time.time() - t_l) / 60:.1f} min")
    if "val" in LAB_OUT[name]:
        vo = LAB_OUT[name]["val"]
        vids = [i for i in VAL_IDS if i in vo]
        val_report(f"{name} top-1", vids, [vo[i]["nbest"][0][0] for i in vids])
        LAB_VAL_SCORE[name] = val_report(f"{name} MBR (own 5-best)", vids, [mbr_own(vo[i]) for i in vids])
''')

# ============================================================================ rescoring
md(r'''
### Optional: every labeler scores every other labeler's candidates (`RESCORE_POOL=True`)
By default each labeler scores only its own 5-best plus the guide. With `RESCORE_POOL`, a second pass reloads each labeler and scores the whole pooled candidate list of every line. A candidate proposed by only one model then also gets the others' opinion.

In the 7B+8B pool, the best candidate sat in only one model's list in about 85 % of the lines MBR got wrong. The cost is about one extra hour per labeler.
''')
code(r'''
# =========================================================
# Cell 14 — optional pool rescoring
# =========================================================
if CFG.RESCORE_POOL and len(LAB_OUT) >= 2:
    for L in LABS:
        name = L["name"]
        if name not in LAB_OUT:
            continue
        if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
            log(f"[TIME] budget reached — rescoring by {name} skipped")
            continue
        log(f"RESCORING THE POOL with {name}", banner=True)
        model, proc = load_labeler(L)
        for sp, out in LAB_OUT[name].items():
            ids, paths, subs, grps, gts = SPLIT_DATA[sp]
            gpos = {i: n for n, i in enumerate(ids)}
            t0, n_sc = time.time(), 0
            for iid, e in out.items():
                own = {t for t, *_ in e["nbest"]} | set(e["judged"])
                pool = {t for nm in LAB_OUT if sp in LAB_OUT[nm] and iid in LAB_OUT[nm][sp]
                        for t, *_ in LAB_OUT[nm][sp][iid]["nbest"] if t}
                missing = sorted(pool - own)
                if not missing:
                    continue
                i = gpos[iid]
                _use_adapter(model, L, grps[i])
                exs = [TRAIN_BY_ID[x] for x in e["ex"] if x in TRAIN_BY_ID]
                msgs, imgs = conversation(L, load_image(paths[i]), subs[i], grps[i], exs)
                for t, s_ in zip(missing, forced_logprobs(model, proc, msgs, imgs, missing)):
                    e["judged"][t] = s_
                n_sc += len(missing)
            log(f"[{name}] {sp}: scored {n_sc} candidates of other labelers in {(time.time() - t0) / 60:.1f} min")
            _save_split(L, sp, out)
        free_cuda("model")
        del proc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    log("[RESCORE] off — each labeler scored its own 5-best + the guide")
''')

# ============================================================================ combine
md(r'''
### Pool the labelers + the guide → pseudo-labels
For every line:
1. **Evidence.** Each labeler's posterior over everything it scored (its 5-best, the judged guide, and with `RESCORE_POOL` the others' candidates). The labelers are weighted by val quality (`LABELER_WEIGHTS`).
2. **Guide prior.** The guide gets a prior weight: `GUIDE_WEIGHT`, or `"auto"` = chosen on val by 5-fold CV.
3. **Pick.** MBR under the official cost gives the pseudo-label and its **risk** (expected cost).
4. **Calibration on val.** Val lines are grouped into risk deciles, and each decile gives its **mean line score** (`confidence`) and **share of exact lines** (`p_exact`).
5. **`keep_for_training`.** Follows `PSEUDO_KEEP`, per sub-class. The val lines show what the rule would keep, and how good those kept lines are.
''')
code(r'''
# =========================================================
# Cell 15 — pooling, guide weight, calibration, pseudo_labels.csv
# =========================================================
NAMES = [L["name"] for L in LABS if L["name"] in LAB_OUT]
ALPHA = {n: 1.0 for n in NAMES}
if CFG.LABELER_WEIGHTS == "quality" and all(n in LAB_VAL_SCORE for n in NAMES):
    ALPHA = {n: 1.0 / max(1e-4, 1 - LAB_VAL_SCORE[n]) for n in NAMES}
log("[POOL] labeler weights: " + ", ".join(f"{n}={a:.1f}" for n, a in ALPHA.items()))

def _posterior(e, use_judged=True):
    items = collections.OrderedDict()
    for t, s_, *_ in e["nbest"]:
        if t and t not in items:
            items[t] = s_
    if use_judged:
        for t, s_ in e["judged"].items():
            items.setdefault(t, s_)
    if not items:
        return [], np.array([])
    return list(items), nbest_posteriors(list(items.values()), CFG.MBR_TEMPERATURE)

def pool_line(split, iid, lam, names=None, use_judged=True):
    names = NAMES if names is None else names
    agg, tot = collections.OrderedDict(), 0.0
    for nm in names:
        e = LAB_OUT.get(nm, {}).get(split, {}).get(iid)
        if not e:
            continue
        texts, p = _posterior(e, use_judged)
        for t, q in zip(texts, p):
            agg[t] = agg.get(t, 0.0) + ALPHA[nm] * float(q)
        tot += ALPHA[nm] if texts else 0.0
    g = guide_for(split).get(iid)
    texts = list(agg)
    w = np.array([v / tot for v in agg.values()]) if tot > 0 else np.zeros(len(texts))
    if g and (lam > 0 or not texts):
        if g not in agg:
            texts.append(g); w = np.append(w, 0.0)
        l_ = lam if tot > 0 else 1.0
        w = (1 - l_) * w + l_ * np.array([1.0 if t == g else 0.0 for t in texts])
    if not texts:
        return "", 1.0, None
    pick, risk = mbr_select(texts, list(w), return_risk=True)
    return pick, float(risk), g

GUIDE_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
LAM = 0.5 if CFG.GUIDE_WEIGHT == "auto" else float(CFG.GUIDE_WEIGHT)
VAL_IDS_DONE = [i for i in VAL_IDS if all(i in LAB_OUT[n].get("val", {}) for n in NAMES)] if NAMES else []
VAL_CHECK = None
if VAL_IDS_DONE:
    log("VALIDATION OF THE PSEUDO-LABEL PROCEDURE", banner=True)
    gpos = {i: n for n, i in enumerate(VAL_IDS)}
    vgt = [VAL_GT[gpos[i]] for i in VAL_IDS_DONE]
    if len(NAMES) >= 2:
        val_report("pooled labelers (own 5-best)", VAL_IDS_DONE, [pool_line("val", i, 0.0, use_judged=False)[0] for i in VAL_IDS_DONE])
    if GUIDE_VAL:
        val_report("pooled labelers + judged guide", VAL_IDS_DONE, [pool_line("val", i, 0.0)[0] for i in VAL_IDS_DONE])
        val_report("guide alone", VAL_IDS_DONE, [GUIDE_VAL.get(i, "") for i in VAL_IDS_DONE])
        if CFG.GUIDE_WEIGHT == "auto":
            C = {l_: np.array([line_cost(a, pool_line("val", i, l_)[0]) for a, i in zip(vgt, VAL_IDS_DONE)]) for l_ in GUIDE_GRID}
            cv = []
            for sd in (0, 1, 2):
                fold = np.random.RandomState(sd).permutation(len(vgt)) % 5
                held = np.empty(len(vgt))
                for f in range(5):
                    tr_, te_ = fold != f, fold == f
                    best = min(GUIDE_GRID, key=lambda l_: C[l_][tr_].mean())
                    held[te_] = C[best][te_]
                cv.append(1 - held.mean())
            LAM = min(GUIDE_GRID, key=lambda l_: C[l_].mean())
            log(f"[POOL] val score by guide weight: " + ", ".join(f"{l_}:{1 - C[l_].mean():.5f}" for l_ in GUIDE_GRID)
                + f" | chosen {LAM} | honest 5-fold CV score of this choice {np.mean(cv):.5f} +- {np.std(cv):.5f}"
                + f"  (weight 1.0 = the guide alone). In RUN_MODE='full' set GUIDE_WEIGHT={LAM}")
    rows = []
    for a, i in zip(vgt, VAL_IDS_DONE):
        pick, risk, g = pool_line("val", i, LAM)
        tops = [LAB_OUT[n]["val"][i]["nbest"][0][0] for n in NAMES]
        rows.append({"ID": i, "subclass": VAL_SUB[gpos[i]], "GroundTruth": a, "pseudo_label": pick, "risk": risk,
                     "guide": g or "", "n_top1_agree": sum(t == pick for t in tops),
                     "line_score": 1 - line_cost(a, pick)})
    VAL_CHECK = pd.DataFrame(rows)
    val_report(f"PSEUDO-LABEL RULE (guide weight {LAM})", VAL_IDS_DONE, VAL_CHECK.pseudo_label.tolist())
    if len(NAMES) >= 2:
        parts = []
        full = official_metric(vgt, VAL_CHECK.pseudo_label.tolist())["score"]
        for n in NAMES:
            rest = [x for x in NAMES if x != n]
            s_ = official_metric(vgt, [pool_line("val", i, LAM, names=rest)[0] for i in VAL_IDS_DONE])["score"]
            parts.append(f"{n}: {s_ - full:+.5f}")
        log("[POOL] leave-one-labeler-out (score change when the labeler is removed; negative = it helps): " + " | ".join(parts))

CALIB = None
if VAL_CHECK is not None and len(VAL_CHECK) >= 50:
    q = np.unique(np.quantile(VAL_CHECK.risk, np.linspace(0, 1, 11)))
    VAL_CHECK["bin"] = np.clip(np.searchsorted(q, VAL_CHECK.risk, side="right") - 1, 0, max(0, len(q) - 2))
    CALIB = (q, VAL_CHECK.groupby("bin").line_score.mean().to_dict(),
             VAL_CHECK.groupby("bin").line_score.apply(lambda s: float((s >= 0.999).mean())).to_dict())
    log("[CALIBRATION] risk decile -> mean val line score (P exact): " +
        ", ".join(f"{b}:{CALIB[1][b]:.3f} ({CALIB[2][b]:.0%})" for b in sorted(CALIB[1])))

def confidence_of(risk):
    if CALIB is None:
        return max(0.0, 1.0 - float(risk)), np.nan
    q, m, pe = CALIB
    b = int(np.clip(np.searchsorted(q, risk, side="right") - 1, 0, max(0, len(q) - 2)))
    return float(m.get(b, np.nan)), float(pe.get(b, np.nan))

def keep_mask(df):
    low = df.risk <= df.groupby("subclass").risk.transform(lambda s: s.quantile(CFG.PSEUDO_KEEP_FRAC))
    agree = (df.pseudo_label == df.guide) & (df.n_top1_agree >= 1) if (df.guide != "").any() else pd.Series(False, index=df.index)
    keep = agree if (CFG.PSEUDO_KEEP == "agree" and agree.any()) else (agree | low) if CFG.PSEUDO_KEEP == "agree_or_lowrisk" else low
    nw = df.pseudo_label.str.split().str.len()
    keep = keep & nw.between(CFG.MIN_WORDS, CFG.MAX_WORDS)
    if CFG.MIN_CONFIDENCE is not None and "confidence" in df:
        keep = keep & (df.confidence >= CFG.MIN_CONFIDENCE)
    return keep

if VAL_CHECK is not None:
    VAL_CHECK["confidence"] = [confidence_of(r)[0] for r in VAL_CHECK.risk]
    VAL_CHECK["keep_for_training"] = keep_mask(VAL_CHECK)
    k_ = VAL_CHECK.keep_for_training
    log(f"[KEEP RULE on val] '{CFG.PSEUDO_KEEP}' keeps {k_.mean():.0%} of the lines: mean line score kept "
        f"{VAL_CHECK[k_].line_score.mean():.4f} (exact {(VAL_CHECK[k_].line_score >= 0.999).mean():.0%}) vs rejected "
        f"{VAL_CHECK[~k_].line_score.mean():.4f}" if k_.any() and (~k_).any() else "[KEEP RULE on val] keeps all or nothing")
    VAL_CHECK.to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels_val_check.csv"), index=False)

# ---- the pseudo-labels ----
def comment_of(pick, g, tops, conf):
    n_, k_ = len(tops), sum(t == pick for t in tops)
    if not g:
        c = f"no guide: pooled MBR of the labelers ({k_}/{n_} labelers' top-1 = pick)"
    elif pick == g and k_ == n_:
        c = f"all {n_} labelers' top-1 and the guide agree"
    elif pick == g:
        c = f"guide chosen by the judged pool ({k_}/{n_} labelers' top-1 = guide)"
    elif k_:
        c = f"labelers' reading chosen over the guide ({k_}/{n_} labelers' top-1 = pick)"
    else:
        c = "compromise: pick differs from the guide and from every labeler's top-1"
    return c + (" | low confidence" if conf is not None and conf < 0.85 else "")

rows = []
for sp in CFG.PSEUDO_SPLITS:
    ids, paths, subs, grps, _ = SPLIT_DATA[sp]
    for i, iid in enumerate(ids):
        if not any(iid in LAB_OUT[n].get(sp, {}) for n in NAMES) and iid not in guide_for(sp):
            continue
        pick, risk, g = pool_line(sp, iid, LAM)
        tops = [LAB_OUT[n][sp][iid]["nbest"][0][0] for n in NAMES if iid in LAB_OUT[n].get(sp, {})]
        conf, pex = confidence_of(risk)
        r = {"ID": iid, "split": sp, "class": subs[i][0], "group": grps[i], "subclass": subs[i], "pseudo_label": pick,
             "Target": pick, "confidence": conf, "p_exact": pex, "risk": risk, "n_labelers": len(tops),
             "n_top1_agree": sum(t == pick for t in tops), "guide": g or "", "guide_agrees": bool(g) and pick == g}
        for n in NAMES:
            e = LAB_OUT[n].get(sp, {}).get(iid)
            r[f"{n}_top1"] = e["nbest"][0][0] if e else ""
        r["comment"] = comment_of(pick, g, tops, conf if CALIB is not None else None)
        rows.append(r)
PL = pd.DataFrame(rows)
if len(PL):
    PL["keep_for_training"] = keep_mask(PL)
    PL.loc[~PL.pseudo_label.str.split().str.len().between(CFG.MIN_WORDS, CFG.MAX_WORDS), "comment"] += " | rejected: implausible length"
    assert not (set(PL.ID) & set(VAL_IDS)), "pseudo-labels must never cover validation lines"
    PL.to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels.csv"), index=False)
    test_pl = PL[PL.split == "test"].set_index("ID")
    pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(test_pl.at[i, "pseudo_label"]) if i in test_pl.index else
                                             GUIDE.get(i, "the") or "the" for i in TEST_IDS]}).to_csv(CFG.SUBMISSION_CSV, index=False)
    summ = PL.groupby(["split", "subclass"]).agg(lines=("ID", "size"), kept=("keep_for_training", "sum"),
                                                 mean_conf=("confidence", "mean"), guide_agrees=("guide_agrees", "mean"),
                                                 all_top1_agree=("n_top1_agree", lambda s: float((s == len(NAMES)).mean())))
    log("PSEUDO-LABELS per split and sub-class:\n" + summ.round(3).to_string(), banner=True)
    kept = PL[PL.keep_for_training]
    log(f"pseudo_labels.csv: {len(PL)} lines ({dict(PL.split.value_counts())}) | kept for training {len(kept)} "
        f"({len(kept) / max(1, len(PL)):.0%}) | expected line score of the kept lines {kept.confidence.mean():.4f}"
        + ("" if CALIB is not None else " (uncalibrated: 1 - risk)")
        + f" | comments: {json.dumps(PL.comment.str.split(' [(|]').str[0].value_counts().head(6).to_dict())}")
    log(f"{os.path.basename(CFG.SUBMISSION_CSV)}: the test pseudo-labels in submission format (for a leaderboard check)")
log(f"RUN FINISHED in {(time.time() - SESSION_T0) / 3600:.2f} h | outputs in {CFG.WORK_DIR}: pseudo_labels.csv, "
    f"pseudo_labels_val_check.csv, {os.path.basename(CFG.SUBMISSION_CSV)}, labeler_outputs/, subclass_assignments.csv, "
    f"subclass_prompts.json, trial_results.csv, run_log.txt", banner=True)
''')

md(r'''
### Next notebook: full training on train + pseudo-labels
Attach this notebook's output, then set:

| Setting | Value |
|---|---|
| `PSEUDO_LABEL_CSV` | `<output>/pseudo_labels.csv` |
| lines used | rows with `keep_for_training == True` (ID, `Target`, `subclass`); `barbados-2-enhanced` already reads this format |
| weighting (optional) | `confidence` (expected line score) or `p_exact` |
| validation | never pseudo-labelled: `pseudo_labels.csv` holds only test and unlisted lines |

`pseudo_labels_val_check.csv` shows how the same rule performs on val lines, so you know how clean the kept pseudo-labels are.
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        s = (s.replace("__LINE_FEATURES__", LINE_FEATURES_SRC).replace("__RULES__", RULES_SRC)
              .replace("__UTILS__", UTILS_SRC))
        lines = s.split("\n")
        c = {"cell_type": t, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "id": f"p{i:02d}"}
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
