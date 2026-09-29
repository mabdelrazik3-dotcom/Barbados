"""Builds barbados-2-classwise.ipynb: ONE SEPARATE LoRA MODEL PER CLASS (fresh from the base model, trained on that
class's lines only) with a CLASS-SPECIFIC PROMPT, and class-routed inference. Reuses the tested infrastructure cells
of build_enhanced_notebook.py (install, imports, paths, utils/augmentation, loaders, metric, data split, collator,
generation, trial harness) and replaces CFG, the main loop, the ensemble/submission cell and the forensics cell."""
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
# Barbados R.O.A.D. — **class-wise models**: one separate LoRA + one specialised prompt per class

This notebook trains **a separate model for every data class**. Each is a brand-new LoRA adapter on the base Qwen-VL model, trained **only on that class's lines**, with a **prompt written for that class**. Every validation and test line is decoded by the model of its own class.

### Classes (from the image pixels only, so they work identically on test)
| Key | Rule | Train lines | Documents / conventions (measured on Train.csv) |
|---|---|---:|---|
| **B** | image height > 150 px | 1,043 | 1670s–1710s, high-res scans; `^` in 25 % of lines, `:` suspensions in 17 %, ff- words, spellings `heyres`, `saide` |
| **A** | height ≤ 150 px | 3,055 | 1630s–1660s; `&` in 14 %, ye/yt in 11 %, `sd` for said, `heires`, `assignes` |
| A1 | A and height ≤ 58 px | 1,669 | tight crops; heavy `&` and ye/yt |
| A2 | A, taller, parchment grey ≤ 194 | 383 | dark parchment, larger hand; `&` and ye rare; short lines |
| A3 | A, taller, parchment grey > 194 | 1,003 | loose crops showing neighbouring lines; longest lines |

`CLASS_SCHEME = "class"` trains 2 models (A, B). `CLASS_SCHEME = "subclass"` trains 4 models (A1, A2, A3, B).

### What happens
1. **Train.** For each base model in `MODEL_PATHS` and each class, a fresh LoRA is trained on that class's training lines with that class's prompt, using realistic augmentation. Optional `CLASS_REPLAY` mixes in a share of other-class lines; the default 0 means fully separate. The adapters are saved as `final/<model>/fold0/class_<key>/`.
2. **Decode.** Every validation, test and unlisted line is decoded by its class's adapter, with every decoding variant. Scores are printed overall, per class and per group.
3. **Compare (optional).** `BASELINE_PRED_CSV` can point to a *shared-model* validation prediction file from `barbados-2-enhanced.ipynb`. Every trial then shows the paired difference, and the forensics cell shows it per class.
4. **Combine and write.** With several base models, their class-routed candidates are combined by MBR. The best validated configuration writes `submission.csv`; `pseudo_labels.csv` is also written.

> **Evidence note** (`reports/Per class specialist HTR models.md`): in every published comparison, models trained on one domain only lost to one shared model, most on the smallest domain. This notebook lets you measure that on R.O.A.D. directly. Compare it with a shared-model run before submitting.
''')

# ============================================================================ reused infrastructure
code(find("# Cell 1 — OFFLINE install"))
code(next(s for t, s in E if t == "code" and s.lstrip().startswith("%%capture")))
code(find("# Cell 2 — imports"))

# ============================================================================ CFG (class-wise)
code(r'''
# =========================================================
# Cell 3 — CFG for CLASS-WISE models
# =========================================================
TRIAL_ID          = "CLS_001"
TRIAL_DESCRIPTION = ("Separate LoRA per class, fresh from the base model, trained on that class only, "
                     "with a class-specific prompt; class-routed decoding; full Fold-0 val")

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
    # ---- inputs ----
    MODEL_PATHS = [
        "/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/8b-instruct/1",
    ]
    GRAD_CKPT_MODELS = ("32b",)
    DATA_DIR    = "/kaggle/input/datasets/haniagamal/road-barbados"
    TRAIN_CSV   = None
    TEST_CSV    = None
    IMAGES_DIR  = None
    AUTODISCOVER = True
    # ---- outputs ----
    OUTPUT_DIR     = "/kaggle/working/qwen-vl-ocr"
    FINAL_DIR      = "/kaggle/working/qwen-vl-ocr/final"     # -> FINAL_DIR/<model>/fold0/class_<key>/
    WORK_DIR       = "/kaggle/working"
    SUBMISSION_CSV = "/kaggle/working/submission.csv"
    KEEP_ADAPTERS  = True
    ADAPTERS_DIR   = None     # DO_TRAIN=False: folder holding <model>/fold0/class_<key>/ ; None -> search /kaggle/input
    PREV_RESULTS_DIRS = []

    # ---- protocol ----
    RUN_MODE        = "fold0"     # "fold0": train class models on Fold-0 train, score ALL 819 val lines | "full": all lines
    FOLD            = 0
    N_FOLDS         = 5
    CKPT_SELECTION  = "last"      # class models train on ALL their Fold-0 train lines (small classes -> no inner slice)
    INNER_CKPT_FRAC = 0.08
    TIME_BUDGET_H   = 11.0

    # ---- CLASS-WISE setup ----
    CLASS_SCHEME = "class"        # "class": models A, B | "subclass": models A1, A2, A3, B
    CLASS_PROMPTS = {             # one prompt per class model, used in training AND inference
        "A":  COMMON_PROMPT + _A,
        "B":  COMMON_PROMPT + _B,
        "A1": COMMON_PROMPT + _A + " The crop is tight around the line; '&' and ye/yt are very common.",
        "A2": COMMON_PROMPT + _A + " The parchment is dark and the hand is large; in these documents '&' and ye are rare and lines are short.",
        "A3": COMMON_PROMPT + _A + " The crop is loose and often shows parts of the lines above and below; the central line is usually long.",
    }
    CLASS_REPLAY = 0.0            # share of each class model's rows drawn from OTHER classes (0 = fully separate)
    CLASS_HP = {}                 # per-class overrides, e.g. {"A2": {"NUM_TRAIN_EPOCHS": 2}, "B": {"LEARNING_RATE": 1e-4}}
    BASELINE_PRED_CSV = None      # optional: shared-model val predictions (ID, pred), e.g. an enhanced-run
                                  #   trial_predictions/EXP_002__qwen_3_vl_transformers_8b_instruct_1__D0_orig__val.csv (measured 0.90947)

    # ---- training (per class model; overridable per model and per class) ----
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
    SEED                         = 42
    MODEL_OVERRIDES = {"32b": {"LEARNING_RATE": 1e-4}}

    # ---- class rule (pixels only) ----
    CLASS_HEIGHT_SPLIT  = 150
    GROUP_A1_MAX_HEIGHT = 58
    GROUP_A2_MAX_PAPER  = 194
    CLASS_HINT = True             # internal: pass each line's class to the prompt builder (required here)
    HINT_LEVEL = "group"
    CLASS_AUG_COPIES = None

    # ---- augmentation (realistic profile, see barbados-2-enhanced) ----
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

    # ---- decoding / ensembles / final run ----
    MAX_NEW_TOKENS = 256
    BATCH_SIZE     = 8
    DECODE_VARIANTS = {   # EXP_002 (GPU, 8B): D0 0.90947 vs D1 0.90902 (P(D1 better)=0.07) -> keep the penalty
        "D0_orig":  dict(num_beams=3, repetition_penalty=1.2, no_repeat_ngram_size=6, n_return=1),
        "D2_b5nb":  dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0, n_return=5),
    }
    BASELINE_VARIANT = "D0_orig"
    MBR_TEMPERATURE  = 1.0
    ENSEMBLE_METHODS = ("mbr_nbest", "mbr_top1")
    CHAMPION_DECODE   = "D2_b5nb"
    CHAMPION_ENSEMBLE = "mbr_nbest"   # RUN_MODE="full": across base models; with one model -> MBR over its 5-best
    # ---- pseudo-labels ----
    PREDICT_UNLISTED = True
    PSEUDO_LABEL_CSV = None
    PSEUDO_KEEP_FRAC = 0.7
    PSEUDO_PER_CLASS = True
    # ---- flow ----
    DO_TRAIN = True
    DO_INFER = True
    PREDICT_TEST = True
    OCR_PROMPT = COMMON_PROMPT        # fallback only (lines always carry their class)
    # unused by this notebook, kept so the shared helper cells run unchanged
    SPECIALISTS = {}; SPECIALIST_MODELS = (); CLASS_SAMPLING_T = None; CLASS_ROUTING = False; ROUTING_LEVEL = "class"
    LR_SCHEDULER_SPEC = None

TRIAL_CONFIG = {k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)}
print("TRIAL:", TRIAL_ID, "|", TRIAL_DESCRIPTION)
print("CLASS_SCHEME:", CFG.CLASS_SCHEME, "| models:", len(CFG.MODEL_PATHS), "| replay:", CFG.CLASS_REPLAY)
''')

code(find("# Cell 4 — resolve & sanity-check paths"))
code(find("# Cell 5 — text / image utils"))
code(find("# Cell 6 — model & processor loaders"))
code(find("# Cell 7 — OFFICIAL metric"))
code(find("# Cell 8 — samples, Fold-0 split"))
code(find("# Cell 9 — collator"))
code(find("# Cell 10 — batched N-best generation"))
code(find("# Cell 11 — trial harness"))

# ============================================================================ class-wise main loop
code(r'''
# =========================================================
# Cell 12 — CLASS-WISE: for each base model, train ONE fresh LoRA per class on that class only (with its own
#           prompt), then decode every line with the adapter of its class. One base model resident at a time.
# =========================================================
import torch.nn as _nn

CLASS_KEYS = ["A1", "A2", "A3", "B"] if CFG.CLASS_SCHEME == "subclass" else ["A", "B"]

def class_key(grp):
    return grp if CFG.CLASS_SCHEME == "subclass" else grp[0]

def ocr_prompt(grp=None):
    """overrides Cell 5: each class model has its own prompt, identical in training and inference."""
    return CFG.CLASS_PROMPTS[class_key(grp)] if grp else CFG.OCR_PROMPT

for _k in CLASS_KEYS:
    assert _k in CFG.CLASS_PROMPTS, f"CLASS_PROMPTS has no prompt for class {_k}"
    print(f"[prompt {_k}] {CFG.CLASS_PROMPTS[_k]}")
_cnt = collections.Counter(class_key(s["grp"]) for s in core_samples + pseudo_samples)
print("training lines per class model:", {k: _cnt.get(k, 0) for k in CLASS_KEYS},
      "| val lines per class:", dict(collections.Counter(class_key(g) for g in VAL_GROUP)))

def _vision_linear_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, _nn.Linear) and any(t in n.lower() for t in ("visual", "vision", "merger"))]

def _hp(mname, key):
    hp = {k: getattr(CFG, k) for k in ("LEARNING_RATE", "NUM_TRAIN_EPOCHS", "R", "LORA_ALPHA", "AUG_COPIES",
                                        "PER_DEVICE_TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS")}
    hp.update(CFG.MODEL_OVERRIDES.get(model_size_tag(mname), {}))
    hp.update(CFG.CLASS_HP.get(key, {}))
    return hp

def _rows_for(samples, n_aug, model_num, salt):
    rows = [{"image": s["image"], "text": s["text"], "grp": s["grp"], "aug": False, "seed": -1} for s in samples]
    if CFG.USE_AUGMENT:
        for _j, s in enumerate(samples):
            for _c in range(int(n_aug)):
                rows.append({"image": s["image"], "text": s["text"], "grp": s["grp"], "aug": True,
                             "seed": int((CFG.SEED + 1) * 2000003 + CFG.FOLD * 1000003 + model_num + _c * 9973 + _j
                                         + salt * 7777777) & 0x7fffffff})
    return rows

FOLD_TAG = f"fold{CFG.FOLD}" if CFG.RUN_MODE == "fold0" else "full"

def _find_class_adapter(mname, key):
    names = [model_short_name(m) for m in [mname]] + [mname]
    roots = ([CFG.ADAPTERS_DIR] if CFG.ADAPTERS_DIR else []) + [CFG.FINAL_DIR] + \
            (sorted(_glob("/kaggle/input/*")) + sorted(_glob("/kaggle/input/*/*")) if not CFG.ADAPTERS_DIR else [])
    for r in roots:
        for nm in names:
            p = os.path.join(r, nm, FOLD_TAG, f"class_{key}")
            if os.path.isfile(os.path.join(p, "adapter_config.json")):
                return p
    for r in roots:
        for cfgp in _glob(os.path.join(r, "**", f"class_{key}", "adapter_config.json"), recursive=True):
            parts = os.path.dirname(cfgp).replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[-2] == FOLD_TAG and parts[-3] == mname:
                return os.path.dirname(cfgp)
    return None

BASE_HYPS = None
if CFG.RUN_MODE == "fold0" and CFG.BASELINE_PRED_CSV and os.path.exists(CFG.BASELINE_PRED_CSV):
    _b = pd.read_csv(CFG.BASELINE_PRED_CSV).set_index("ID")
    BASE_HYPS = [norm_text(_b.at[i, "pred"]) if i in _b.index else "" for i in VAL_IDS]
    print(f"baseline predictions loaded for {sum(i in _b.index for i in VAL_IDS)}/{len(VAL_IDS)} val lines "
          f"-> score {official_metric(VAL_GT, BASE_HYPS)['score']:.5f}")

dtype = torch.bfloat16 if (CFG.USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_NBEST, TEST_NBEST, UNL_NBEST, MODEL_ORDER = {}, {}, {}, []
TRAIN_SIZE = len(core_samples)
_EMPTY = [("", float("-inf"), 0, False)]

def decode_split(model, proc, paths, groups, var, desc):
    """every line decoded by the adapter of its own class (batched per class)."""
    out = [None] * len(paths)
    for key in CLASS_KEYS:
        idx = [i for i, g in enumerate(groups) if class_key(g) == key]
        if not idx:
            continue
        model.set_adapter(f"class_{key}")
        sub = transcribe(model, proc, [paths[i] for i in idx], var, desc=f"{desc}/{key}")
        for j, i in enumerate(idx):
            out[i] = sub[j]
    return [o if o is not None else list(_EMPTY) for o in out]

for model_num, msrc in enumerate(CFG.MODEL_PATHS):
    if (time.time() - SESSION_T0) / 3600 > CFG.TIME_BUDGET_H:
        print(f"[time] budget {CFG.TIME_BUDGET_H}h reached — skipping {msrc}"); continue
    mname = model_short_name(msrc); MODEL_ORDER.append(mname)
    print(f"\n=========== BASE MODEL {mname} ({CFG.RUN_MODE}, {FOLD_TAG}) — class models {CLASS_KEYS} ===========")
    t_model = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    proc = load_vlm_processor(msrc, max_pixels=CFG.MAX_PIXELS, min_pixels=CFG.MIN_PIXELS)
    base = load_vlm_model(msrc, dtype); base.to(DEVICE); base.config.use_cache = False
    grad_ckpt = any(s in msrc.lower() for s in CFG.GRAD_CKPT_MODELS)
    if CFG.DO_TRAIN and grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] + \
              (_vision_linear_names(base) if CFG.LORA_VISION else [])     # computed on the BASE, before wrapping
    ad_dir = os.path.join(CFG.FINAL_DIR, mname, FOLD_TAG)
    model = None
    for key in CLASS_KEYS:
        name = f"class_{key}"
        hp = _hp(mname, key)
        own = [s for s in core_samples + pseudo_samples if class_key(s["grp"]) == key]
        if CFG.DO_TRAIN:
            rows = _rows_for(own, hp["AUG_COPIES"], model_num, salt=CLASS_KEYS.index(key) + 1)
            if CFG.CLASS_REPLAY > 0:
                others = [s for s in core_samples + pseudo_samples if class_key(s["grp"]) != key]
                n_rep = int(round(CFG.CLASS_REPLAY / (1 - CFG.CLASS_REPLAY) * len(rows)))
                pick = np.random.RandomState(CFG.SEED + 13 + model_num).choice(len(others), min(n_rep, len(others)), replace=False)
                rows += [{"image": others[i]["image"], "text": others[i]["text"], "grp": others[i]["grp"], "aug": True,
                          "seed": int(CFG.SEED * 31 + i) & 0x7fffffff} for i in pick]
            lcfg = LoraConfig(r=hp["R"], lora_alpha=hp["LORA_ALPHA"], target_modules=TARGETS,
                              lora_dropout=CFG.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
            if model is None:
                model = get_peft_model(base, lcfg, adapter_name=name)
            else:
                model.add_adapter(name, lcfg)
            model.set_adapter(name)
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[{mname}] class model {name}: {len(own)} lines -> {len(rows)} rows "
                  f"(replay {CFG.CLASS_REPLAY:.0%}) | hp {hp} | trainable {n_tr/1e6:.1f}M")
            args = TrainingArguments(
                output_dir=os.path.join(CFG.OUTPUT_DIR, f"{mname}_{name}"),
                per_device_train_batch_size=hp["PER_DEVICE_TRAIN_BATCH_SIZE"],
                gradient_accumulation_steps=hp["GRADIENT_ACCUMULATION_STEPS"],
                learning_rate=hp["LEARNING_RATE"], num_train_epochs=hp["NUM_TRAIN_EPOCHS"],
                lr_scheduler_type=CFG.LR_SCHEDULER, warmup_ratio=CFG.WARMUP_RATIO,
                bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16), logging_steps=10,
                eval_strategy="no", save_strategy="no", remove_unused_columns=False,
                dataloader_num_workers=CFG.DATALOADER_NUM_WORKERS, dataloader_pin_memory=torch.cuda.is_available(),
                report_to="none", seed=CFG.SEED)
            trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                              data_collator=lambda x, _p=proc: collate_fn(x, _p))
            t0 = time.time(); trainer.train()
            print(f"[{mname}] {name} trained in {(time.time()-t0)/60:.1f} min | label-mask failures so far: {MASK_FAIL_COUNT}")
            try:
                pd.DataFrame(trainer.state.log_history).to_csv(
                    os.path.join(CFG.WORK_DIR, f"log_history__{mname}__{name}.csv"), index=False)
            except Exception: pass
            if CFG.KEEP_ADAPTERS:                  # PEFT writes non-default adapters to <ad_dir>/<name>/
                os.makedirs(ad_dir, exist_ok=True)
                model.save_pretrained(ad_dir, selected_adapters=[name]); proc.save_pretrained(ad_dir)
            del trainer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            shutil.rmtree(os.path.join(CFG.OUTPUT_DIR, f"{mname}_{name}"), ignore_errors=True)
        else:
            p = _find_class_adapter(mname, key)
            assert p is not None, f"no saved adapter class_{key} for {mname} ({FOLD_TAG}) under ADAPTERS_DIR={CFG.ADAPTERS_DIR!r}"
            if model is None:
                model = PeftModel.from_pretrained(base, p, adapter_name=name)
            else:
                model.load_adapter(p, adapter_name=name)
            print(f"[{mname}] class model {name} <- {p}")
    model.eval(); model.config.use_cache = True
    try: model.gradient_checkpointing_disable()
    except Exception: pass
    t_train = time.time() - t_model

    if CFG.DO_INFER and CFG.RUN_MODE == "fold0":
        scores = {}
        for vname, var in CFG.DECODE_VARIANTS.items():
            t0 = time.time()
            nb = decode_split(model, proc, VAL_PATHS, VAL_GROUP, var, f"val/{mname}/{vname}")
            VAL_NBEST[(mname, vname)] = nb
            top1 = [x[0][0] for x in nb]
            trial = f"{TRIAL_ID}__{mname}__{vname}"
            save_predictions(trial, VAL_IDS, VAL_GT, nb)
            m = report_trial(trial, TRIAL_DESCRIPTION, f"{mname} (class models {'/'.join(CLASS_KEYS)})", vname, top1,
                             t_train + time.time() - t0, TRAIN_SIZE, sum(x[0][3] for x in nb), baseline_hyps=BASE_HYPS,
                             extra={"class_scheme": CFG.CLASS_SCHEME, "class_replay": CFG.CLASS_REPLAY})
            scores[vname] = m["score"]
            if int(var.get("n_return", 1)) > 1:
                mbr = [mbr_select([t for t, *_ in x], nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)) for x in nb]
                report_trial(f"{trial}_mbr", TRIAL_DESCRIPTION, f"{mname} (class models)", f"{vname}+mbr", mbr, 0.0,
                             TRAIN_SIZE, baseline_hyps=BASE_HYPS, extra={"class_scheme": CFG.CLASS_SCHEME})
                scores[f"{vname}+mbr"] = official_metric(VAL_GT, mbr)["score"]
                save_predictions(f"{trial}_mbr", VAL_IDS, VAL_GT, nb, chosen=mbr)
        best_var = max((v for v in scores if "+mbr" not in v), key=scores.get)
        print(f"[{mname}] best decoding on val: {best_var} | " + ", ".join(f"{k}={v:.5f}" for k, v in scores.items()))
        test_vars = {best_var} | {v for v, c in CFG.DECODE_VARIANTS.items() if int(c.get("n_return", 1)) > 1}
    else:
        test_vars = {CFG.CHAMPION_DECODE}

    if CFG.DO_INFER and CFG.PREDICT_TEST:
        for vname in sorted(test_vars):
            nb = decode_split(model, proc, TEST_PATHS, TEST_GROUP, CFG.DECODE_VARIANTS[vname], f"test/{mname}/{vname}")
            TEST_NBEST[(mname, vname)] = nb
            save_predictions(f"{TRIAL_ID}__{mname}__{vname}", TEST_IDS, None, nb, split="test")
    if CFG.DO_INFER and CFG.PREDICT_UNLISTED and UNL_IDS:
        for vname in sorted(v for v in test_vars if int(CFG.DECODE_VARIANTS[v].get("n_return", 1)) > 1):
            nb = decode_split(model, proc, UNL_PATHS, UNL_GROUP, CFG.DECODE_VARIANTS[vname], f"unlisted/{mname}/{vname}")
            UNL_NBEST[(mname, vname)] = nb
            save_predictions(f"{TRIAL_ID}__{mname}__{vname}", UNL_IDS, None, nb, split="unlisted")
    free_cuda("model", "base"); del proc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[{mname}] done in {(time.time()-t_model)/60:.1f} min")
''')

# ============================================================================ ensembles + submission
code(r'''
# =========================================================
# Cell 13 — candidates (per base model and across base models), champion, submission, pseudo-labels
# =========================================================
def _pool(sys_nb):
    cands, w = [], []
    for x in sys_nb:
        p = nbest_posteriors([s for _, s, *_ in x], CFG.MBR_TEMPERATURE)
        cands += [t for t, *_ in x]; w += list(p / len(sys_nb))
    return cands, w

def _ensemble(nb_by_model, method):
    out = []
    for i in range(len(nb_by_model[0])):
        sys_nb = [nb[i] for nb in nb_by_model]
        out.append(mbr_select(*_pool(sys_nb)) if method in ("mbr_nbest", "single_mbr")
                   else mbr_select([x[0][0] for x in sys_nb]))
    return out

_top1 = lambda xs: [x[0][0] for x in xs]
CANDIDATES = {}
for (mn, vn), nb in VAL_NBEST.items():
    CANDIDATES[f"{mn}__{vn}"] = (_top1(nb), _top1(TEST_NBEST[(mn, vn)]) if (mn, vn) in TEST_NBEST else None)
    if int(CFG.DECODE_VARIANTS[vn].get("n_return", 1)) > 1:
        CANDIDATES[f"{mn}__{vn}+mbr"] = (_ensemble([nb], "single_mbr"),
                                         _ensemble([TEST_NBEST[(mn, vn)]], "single_mbr") if (mn, vn) in TEST_NBEST else None)
if CFG.RUN_MODE == "fold0" and len(MODEL_ORDER) >= 2:
    for vname in CFG.DECODE_VARIANTS:
        systems = [m for m in MODEL_ORDER if (m, vname) in VAL_NBEST]
        if len(systems) < 2: continue
        for method in CFG.ENSEMBLE_METHODS:
            if method == "mbr_nbest" and int(CFG.DECODE_VARIANTS[vname].get("n_return", 1)) < 2: continue
            vh = _ensemble([VAL_NBEST[(m, vname)] for m in systems], method)
            th = _ensemble([TEST_NBEST[(m, vname)] for m in systems], method) if all((m, vname) in TEST_NBEST for m in systems) else None
            name = f"ENS[{'+'.join(model_size_tag(s_) for s_ in systems)}]__{vname}__{method}"
            CANDIDATES[name] = (vh, th)
            report_trial(f"{TRIAL_ID}__{name}", TRIAL_DESCRIPTION, name, f"{vname}/{method}", vh, 0.0, TRAIN_SIZE,
                         baseline_hyps=BASE_HYPS, extra={"class_scheme": CFG.CLASS_SCHEME})
            pd.DataFrame({"ID": VAL_IDS, "GroundTruth": VAL_GT, "pred": vh}).to_csv(
                os.path.join(TRIAL_PRED_DIR, f"{TRIAL_ID}__{name}__val.csv"), index=False)

if CFG.RUN_MODE == "fold0":
    ranked = sorted(((official_metric(VAL_GT, v)["score"], k) for k, (v, t) in CANDIDATES.items() if v is not None), reverse=True)
    print("\nVALIDATION RANKING (official score):")
    for s, k in ranked[:12]:
        print(f"  {s:.5f}  {k}{'' if CANDIDATES[k][1] is not None else '   (no test preds)'}")
    best = next((k for s, k in ranked if CANDIDATES[k][1] is not None), None)
else:
    vn = CFG.CHAMPION_DECODE
    systems = [m for m in MODEL_ORDER if (m, vn) in TEST_NBEST]
    nbr = int(CFG.DECODE_VARIANTS[vn].get("n_return", 1))
    if len(systems) >= 2 and CFG.CHAMPION_ENSEMBLE:
        best = f"ENS[{'+'.join(model_size_tag(s_) for s_ in systems)}]__{vn}__{CFG.CHAMPION_ENSEMBLE}"
        CANDIDATES[best] = (None, _ensemble([TEST_NBEST[(m, vn)] for m in systems], CFG.CHAMPION_ENSEMBLE))
    elif systems:
        best = f"{systems[0]}__{vn}" + ("+mbr" if (CFG.CHAMPION_ENSEMBLE and nbr > 1) else "")
        nb0 = TEST_NBEST[(systems[0], vn)]
        CANDIDATES[best] = (None, _ensemble([nb0], "single_mbr") if (CFG.CHAMPION_ENSEMBLE and nbr > 1) else _top1(nb0))
    else:
        best = None

if best is not None and CANDIDATES[best][1] is not None:
    sub = pd.DataFrame({"ID": TEST_IDS, "Target": [norm_text(t) or "the" for t in CANDIDATES[best][1]]})
    sub.to_csv(CFG.SUBMISSION_CSV, index=False)
    print(f"\nsubmission.csv <- {best} | rows={len(sub)} | test lines per class model:",
          dict(collections.Counter(class_key(g) for g in TEST_GROUP)))
    with open(os.path.join(CFG.WORK_DIR, "submission_source.json"), "w") as f:
        json.dump({"trial_session": TRIAL_ID, "source": best, "class_scheme": CFG.CLASS_SCHEME}, f)

# pseudo-labels (test + unlisted): MBR over every base model's class-routed N-best; risk = expected official cost
_pl_rows = []
for split_ids, store in ((TEST_IDS, TEST_NBEST), (UNL_IDS, UNL_NBEST)):
    keys = [k for k in store if int(CFG.DECODE_VARIANTS[k[1]].get("n_return", 1)) > 1]
    for i, rid in enumerate(split_ids if keys else []):
        txt, risk = mbr_select(*_pool([store[k][i] for k in keys]), return_risk=True)
        _pl_rows.append({"ID": rid, "Target": txt, "risk": risk, "n_systems": len(keys)})
if _pl_rows:
    pd.DataFrame(_pl_rows).to_csv(os.path.join(CFG.WORK_DIR, "pseudo_labels.csv"), index=False)
    print(f"pseudo_labels.csv: {len(_pl_rows)} lines (refine with work/08_build_pseudo_labels.py)")

if os.path.exists(TRIAL_RESULTS):
    res = pd.read_csv(TRIAL_RESULTS)
    sess = res.session == TRIAL_ID
    if sess.any():
        bi = res[sess].score.idxmax()
        res.loc[sess & (res.index != bi) & (res.status == "EVALUATED"), "status"] = "REJECTED"
        res.loc[bi, "status"] = "PROMOTED"
        res.to_csv(TRIAL_RESULTS, index=False)
        print("\nTRIAL TABLE (this session):")
        print(res[sess][["trial", "variant", "score", "score_A", "score_B", "word_edits_per_line", "char_edits_per_line",
                         "delta_baseline", "p_better_baseline", "status"]].to_string(index=False))
''')

# ============================================================================ forensics per class
code(r'''
# =========================================================
# Cell 14 — per-class comparison and error forensics of the best class-wise system
# =========================================================
if CFG.RUN_MODE == "fold0" and CANDIDATES:
    vals = {k: v for k, (v, t) in CANDIDATES.items() if v is not None}
    champ = max(vals, key=lambda k: official_metric(VAL_GT, vals[k])["score"])
    print("CHAMPION:", champ)
    print("per class :", score_by_group(VAL_GT, vals[champ], VAL_CLASS))
    print("per group :", score_by_group(VAL_GT, vals[champ], VAL_GROUP))
    if BASE_HYPS is not None:
        print("\nCLASS-WISE vs BASELINE (shared model), paired bootstrap per class:")
        for c in sorted(set(VAL_CLASS)) + sorted(set(VAL_GROUP) - set(VAL_CLASS)):
            idx = [i for i, (a, g) in enumerate(zip(VAL_CLASS, VAL_GROUP)) if c in (a, g)]
            d = paired_delta([VAL_GT[i] for i in idx], [BASE_HYPS[i] for i in idx], [vals[champ][i] for i in idx])
            print(f"  {c:3s} n={len(idx):4d}  delta={d['delta']:+.5f}  CI95 [{d['ci95_lo']:+.5f}, {d['ci95_hi']:+.5f}]  "
                  f"P(class-wise better)={d['p_better']:.3f}  word/char edits saved={d['word_edits_saved']}/{d['char_edits_saved']}")
        d = paired_delta(VAL_GT, BASE_HYPS, vals[champ])
        print(f"  ALL n={len(VAL_GT)}  delta={d['delta']:+.5f}  CI95 [{d['ci95_lo']:+.5f}, {d['ci95_hi']:+.5f}]  P={d['p_better']:.3f}")
    fx = error_forensics(VAL_GT, vals[champ])
    for k, v in fx.items():
        print(f"  {k}: {v}")
''')

md(r'''
### Suggested runs
| Run | CFG | Why |
|---|---|---|
| CLS_001 | defaults: `CLASS_SCHEME="class"` (A, B), 8B, no replay | Two separate era models with era-specific prompts |
| CLS_002 | `CLASS_SCHEME="subclass"` | Four separate models (A1, A2, A3, B). A2 has only ~300 training lines |
| CLS_003 | `CLASS_REPLAY=0.15` | Mostly separate models that also see 15 % other-class lines |
| CLS_004 | `CLASS_HP={"A2": {"NUM_TRAIN_EPOCHS": 2}}` | More passes for the smallest class |
| FINAL | `RUN_MODE="full"`, same scheme and prompts, `CHAMPION_DECODE` / `CHAMPION_ENSEMBLE` | Retrain on all 4,093 lines |

Set `BASELINE_PRED_CSV` to a shared-model validation file from `barbados-2-enhanced.ipynb` (e.g. `trial_predictions/EXP_002__qwen_3_vl_transformers_8b_instruct_1__D0_orig__val.csv`, measured 0.90947) to get the per-class verdict. Keep the class-wise models only where the difference is positive with P ≥ 0.9.
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        lines = s.split("\n")
        c = {"cell_type": t, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "id": f"k{i:02d}"}
        if t == "code":
            c["execution_count"] = None; c["outputs"] = []
        nb["cells"].append(c)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "D:/HANAFY/ROAD/barbados-2-classwise.ipynb"
    print("wrote", out, "cells:", len(build(out)["cells"]))
