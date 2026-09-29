"""CPU dry run of barbados-2-enhanced.ipynb with tiny random-weight Qwen2.5-VL / Qwen3-VL models.
Executes every code cell in order (skipping the pip magic), patching only paths and data size.
Scores are meaningless (random weights); the point is that every code path runs end to end."""
import json
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = "D:/HANAFY/ROAD"
OUT = f"{ROOT}/work/dryrun"
OUT = os.environ.get("DRY_OUT", OUT)
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
nb = json.load(open(os.environ.get("DRY_NB", f"{ROOT}/barbados-2-enhanced.ipynb"), encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
DO_TRAIN = os.environ.get("DRY_DO_TRAIN", "1") == "1"
t0 = time.time()
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if src.lstrip().startswith("%%capture"):
        continue
    print(f"\n######## executing code cell {i}: {src.splitlines()[1] if len(src.splitlines()) > 1 else ''}", flush=True)
    exec(compile(src, f"<cell{i}>", "exec"), ns)
    if "class CFG" in src:
        C = ns["CFG"]
        C.MODEL_PATHS = [f"{ROOT}/work/tiny/q3", f"{ROOT}/work/tiny/q25"]
        C.GRAD_CKPT_MODELS = ("q25",)
        C.DATA_DIR = ROOT
        C.IMAGES_DIR = f"{ROOT}/images"
        C.WORK_DIR = OUT
        C.OUTPUT_DIR = f"{OUT}/qwen-vl-ocr"
        C.FINAL_DIR = f"{OUT}/qwen-vl-ocr/final"
        C.SUBMISSION_CSV = f"{OUT}/submission.csv"
        C.PER_DEVICE_TRAIN_BATCH_SIZE = 2
        C.EVAL_STEPS = 4
        C.DATALOADER_NUM_WORKERS = 0
        C.BATCH_SIZE = 4
        C.MAX_NEW_TOKENS = 10
        C.USE_BF16 = False
        C.DO_TRAIN = DO_TRAIN
        C.RUN_MODE = os.environ.get("DRY_MODE", "fold0")
        if os.environ.get("DRY_PSEUDO"):
            C.PSEUDO_LABEL_CSV = os.environ["DRY_PSEUDO"]
        if os.environ.get("DRY_HINT") == "1":
            C.CLASS_HINT = True
        if os.environ.get("DRY_COPIES"):
            C.CLASS_AUG_COPIES = json.loads(os.environ["DRY_COPIES"])
        if os.environ.get("DRY_ROUTING_MAP"):
            C.ROUTING_MAP = json.loads(os.environ["DRY_ROUTING_MAP"])
        if os.environ.get("DRY_OVERRIDES"):
            C.MODEL_OVERRIDES = json.loads(os.environ["DRY_OVERRIDES"])
        if os.environ.get("DRY_SPECIALISTS"):
            C.SPECIALISTS = json.loads(os.environ["DRY_SPECIALISTS"])
            C.SPECIALIST_MODELS = ()          # tiny test models have no size tag -> give every model specialists
            C.SPEC_DEFAULTS = {**C.SPEC_DEFAULTS, "max_steps": 3}
        if os.environ.get("DRY_ROUTING_LEVEL"):
            C.ROUTING_LEVEL = os.environ["DRY_ROUTING_LEVEL"]
        if os.environ.get("DRY_SAMPLING_T"):
            C.CLASS_SAMPLING_T = float(os.environ["DRY_SAMPLING_T"])
        if os.environ.get("DRY_SCHEME"):
            C.CLASS_SCHEME = os.environ["DRY_SCHEME"]
        if os.environ.get("DRY_REPLAY"):
            C.CLASS_REPLAY = float(os.environ["DRY_REPLAY"])
        if os.environ.get("DRY_BASELINE"):
            C.BASELINE_PRED_CSV = os.environ["DRY_BASELINE"]
        if os.environ.get("DRY_TRAIN_SPECIALISTS") == "0":
            C.TRAIN_SPECIALISTS = False
        if not DO_TRAIN:
            C.ADAPTERS_DIR = os.environ["DRY_ADAPTERS"]
        if hasattr(C, "LABELERS"):                     # test_pesudo.ipynb (pseudo-label only)
            C.BASE_MODELS = {"q3": f"{ROOT}/work/tiny/q3", "q25": f"{ROOT}/work/tiny/q25"}
            C.LABELERS = json.loads(os.environ["DRY_LABELERS"]) if os.environ.get("DRY_LABELERS") else [
                dict(name="q3_auto", model="q3", adapter="auto", prompt="exp002", fewshot_k=0),
                dict(name="q3_subclass", model="q3", scheme="subclass", prompt="classwise",
                     adapter=f"{ROOT}/work/dryrun_mega_adapters/ROAD_work_tiny_q3/fold0", adapter_pattern="q3_subclass__{key}"),
                dict(name="q25_base_fewshot", model="q25", adapter=None, prompt="detailed", fewshot_k=2),
            ]
            C.ADAPTER_SEARCH_ROOTS = [f"{ROOT}/work/dryrun/qwen-vl-ocr/final"]
            C.MODEL_PATHS = list(dict.fromkeys(C.BASE_MODELS[L["model"]] for L in C.LABELERS))
            C.MAX_NEW_TOKENS, C.LOG_EVERY_LINES = 8, 5
            C.GUIDE_CSV = os.environ.get("DRY_GUIDE", f"{ROOT}/submission_offline_best.csv") or None
            C.GUIDE_VAL_CSV = os.environ.get("DRY_GUIDE_VAL", f"{ROOT}/trils/EXP_002__ENS_7b8b_mbrq__D2_b5nb_ens__val.csv") or None
            C.RESCORE_POOL = os.environ.get("DRY_RESCORE") == "1"
            C.SUBCLASS_MODE = os.environ.get("DRY_SUBMODE", "fixed")
            if os.environ.get("DRY_REUSE"):
                C.REUSE_DIRS = [os.environ["DRY_REUSE"]]
            if os.environ.get("DRY_ADAPTER_FOLD"):
                C.ADAPTER_FOLD = os.environ["DRY_ADAPTER_FOLD"]
            if os.environ.get("DRY_WEIGHT"):
                C.GUIDE_WEIGHT = os.environ["DRY_WEIGHT"] if os.environ["DRY_WEIGHT"] == "auto" else float(os.environ["DRY_WEIGHT"])
        if hasattr(C, "FEWSHOT_K"):                    # test_pesudo v1 (full flow)
            C.BASE_MODELS = {"q3": f"{ROOT}/work/tiny/q3", "q25": f"{ROOT}/work/tiny/q25"}
            C.MODEL = os.environ.get("DRY_MODEL", "q3")
            C.MODEL_PATH = C.BASE_MODELS[C.MODEL]
            C.MODEL_PATHS = [C.MODEL_PATH]
            C.GRAD_CKPT_MODELS = (C.MODEL,)
            C.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            C.EVAL_EVERY, C.EVAL_GEN_N, C.EVAL_LOSS_N, C.EVAL_MAX_NEW_TOKENS = 2, 3, 3, 6
            C.MAX_STEPS, C.BEFORE_SAMPLE_N, C.MAX_NEW_TOKENS, C.LOG_EVERY_LINES = 4, 4, 8, 5
            C.SELF_TRAIN_ROUNDS = int(os.environ.get("DRY_ROUNDS", "1"))
            C.GUIDE_CSV = os.environ.get("DRY_GUIDE", f"{ROOT}/submission_offline_best.csv") or None
            C.GUIDE_VAL_CSV = os.environ.get("DRY_GUIDE_VAL", f"{ROOT}/trils/EXP_002__ENS_7b8b_mbrq__D2_b5nb_ens__val.csv") or None
            C.PROMPT_MODE = os.environ.get("DRY_PROMPT", "detailed")
            C.SUBCLASS_MODE = os.environ.get("DRY_SUBMODE", "fixed")
            C.FIRST_PSEUDO_FROM_GUIDE = os.environ.get("DRY_FIRST_PSEUDO") == "1"
            if os.environ.get("DRY_INIT_ADAPTER"):
                C.INIT_ADAPTER = os.environ["DRY_INIT_ADAPTER"]
            if os.environ.get("DRY_FINETUNE") == "0":
                C.DO_FINETUNE, C.INFER_BEFORE, C.SELF_TRAIN_ROUNDS = False, "full", 0
            if os.environ.get("DRY_BEFORE"):
                C.INFER_BEFORE = os.environ["DRY_BEFORE"]
        if hasattr(C, "SYSTEMS"):                      # barbados-2-mega.ipynb
            C.BASE_MODELS = {"q3": f"{ROOT}/work/tiny/q3", "q25": f"{ROOT}/work/tiny/q25"}
            C.SYSTEMS = json.loads(os.environ.get("DRY_SYSTEMS", "null")) or [
                dict(name="q3_shared", model="q3", scheme="shared"),
                dict(name="q3_class", model="q3", scheme="class"),
                dict(name="q3_subclass", model="q3", scheme="subclass", class_hp={"A2": {"MAX_STEPS": 2}}),
                dict(name="q3_B_cont", model="q3", scheme="class", keys=["B"], init="from:q3_shared", replay=0.2,
                     hp={"LR_SCHEDULER": "constant_with_warmup"}),
                dict(name="q25_shared", model="q25", scheme="shared"),
            ]
            C.BASELINE_SYSTEM = C.SYSTEMS[0]["name"]
            C.MODEL_PATHS = list(dict.fromkeys(C.BASE_MODELS[s["model"]] for s in C.SYSTEMS))
            C.EVAL_EVERY, C.EVAL_GEN_N, C.EVAL_LOSS_N, C.EVAL_BATCH = 2, 3, 3, 2
            C.MONITOR_VAL_N = int(os.environ.get("DRY_MONITOR", "2"))
            C.MAX_STEPS, C.RESCORE_BATCH, C.EVAL_MAX_NEW_TOKENS = 5, 3, 8
            C.RESCORE_SPLITS = ("val", "test", "unlisted")
            if os.environ.get("DRY_CHAMPION"):
                C.FINAL_CHAMPION_JSON = os.environ["DRY_CHAMPION"]
        ns["TRIAL_CONFIG"] = {k: v for k, v in vars(C).items() if not k.startswith("_") and not callable(v)}
    if "Cell 8 " in src:
        if hasattr(ns["CFG"], "SYSTEMS"):              # mega: every class / sub-class needs train + inner lines
            strat = lambda ss, k: [x for g in ("A1", "A2", "A3", "B") for x in [y for y in ss if y["grp"] == g][:k]]
            ns["core_samples"] = strat(ns["core_samples"], 3)
            ns["inner_samples"] = strat(ns["inner_samples"], 1)
        if hasattr(ns["CFG"], "FEWSHOT_K") or hasattr(ns["CFG"], "LABELERS"):   # test_pesudo: few-shot pools
            n_core = int(os.environ.get("DRY_CORE", "60"))
            ns["core_samples"] = ns["core_samples"][:n_core]
            ns["inner_samples"] = ns["inner_samples"][:6]
        else:
            ns["core_samples"] = ns["core_samples"][:12]
            ns["inner_samples"] = ns["inner_samples"][:4]
        # keep up to 3 validation lines per group so class routing has every class to work with
        keep = []
        for g in ("A1", "A2", "A3", "B"):
            keep += [i for i, s in enumerate(ns["val_samples"]) if s["grp"] == g][:3]
        keep = sorted(keep)
        vs = [ns["val_samples"][i] for i in keep]
        ns["val_samples"] = vs
        ns["VAL_IDS"] = [s["id"] for s in vs]
        ns["VAL_GT"] = [s["text"] for s in vs]
        ns["VAL_PATHS"] = [s["image"] for s in vs]
        ns["VAL_GROUP"] = [s["grp"] for s in vs]
        ns["VAL_CLASS"] = [g[0] for g in ns["VAL_GROUP"]]
        ns["VAL_LEN_BUCKET"] = [ns["VAL_LEN_BUCKET"][i] for i in keep]
        tkeep = sorted([i for i, g in enumerate(ns["TEST_GROUP"]) if g == "B"][:3] +
                       [i for i, g in enumerate(ns["TEST_GROUP"]) if g != "B"][:4])
        for k in ("TEST_IDS", "TEST_PATHS", "TEST_GROUP", "TEST_CLASS"):
            ns[k] = [ns[k][i] for i in tkeep]
        for k in ("UNL_IDS", "UNL_PATHS", "UNL_GROUP"):
            ns[k] = ns[k][:5]
        print("dry-run groups  val:", ns["VAL_GROUP"], " test:", ns["TEST_GROUP"])
        ns["pseudo_samples"] = ns["pseudo_samples"][:6]
        print("dry-run pseudo samples:", len(ns["pseudo_samples"]))
print(f"\nDRY RUN COMPLETE in {time.time() - t0:.0f}s")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
print(open(f"{OUT}/submission.csv", encoding="utf-8").read()[:300])
