"""Reconstruct barbados-2.ipynb's exact Fold-0 split and define the repaired protocol.

Notebook protocol (Cell 6 + Cell 9): shuffle Train.csv (random_state=42), drop 5 CORRUPT_IDS, KFold(5,
shuffle, 42). Fold-0 VALIDATION (819 rows) is then split 40/60 into an early-stop slice (328, used for
checkpoint selection on eval_loss) and an 'OOF' slice (491) that is actually scored.

Repaired protocol: Fold-0 VALIDATION (all 819 rows) is never touched by training or selection.
Checkpoint selection uses an inner split carved from Fold-0 TRAIN (5%, seed 42).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = "D:/HANAFY/ROAD"
SEED = 42
CORRUPT_IDS = {"79tMUVyfIdy3GzkG", "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO"}


def clean_label(x):
    return " ".join(str(x).replace("\n", " ").split()).strip()


df = pd.read_csv(f"{ROOT}/Train.csv").sample(frac=1, random_state=SEED).reset_index(drop=True)
rows = []
for _, r in df.iterrows():
    lab = clean_label(r["Target"])
    if not r["ID"] or not lab or r["ID"] in CORRUPT_IDS:
        continue
    rows.append((r["ID"], lab))
ids = np.array([a for a, _ in rows])
labs = [b for _, b in rows]
print("samples", len(ids))
lens = [len(t) for t in labs]
print("label len p50=%d p95=%d p99=%d max=%d" % (np.percentile(lens, 50), np.percentile(lens, 95),
                                                 np.percentile(lens, 99), max(lens)))

folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(ids))
out = pd.DataFrame({"ID": ids, "Target": labs, "fold": -1})
for k, (_, va) in enumerate(folds):
    out.loc[va, "fold"] = k
tr_idx, va_idx = folds[0]
rng = np.random.RandomState(SEED + 0)
va_perm = rng.permutation(va_idx)
n_es = int(round(0.4 * len(va_perm)))
out["nb_role"] = "train"
out.loc[va_perm[:n_es], "nb_role"] = "earlystop"
out.loc[va_perm[n_es:], "nb_role"] = "oof"
out.loc[~out.index.isin(va_idx), "nb_role"] = "train"
# repaired protocol: inner checkpoint split from fold-0 TRAIN only
rng2 = np.random.RandomState(SEED + 1000)
tr_perm = rng2.permutation(tr_idx)
n_in = int(round(0.05 * len(tr_perm)))
out["role"] = np.where(out.fold == 0, "val", "train")
out.loc[tr_perm[:n_in], "role"] = "inner_ckpt"
print(out.groupby("fold").size().to_dict())
print("notebook roles (fold0):", out[out.fold == 0].nb_role.value_counts().to_dict())
print("repaired roles:", out.role.value_counts().to_dict())
out.to_csv(f"{ROOT}/work/fold_assignments.csv", index=False)
full = pd.read_csv(f"{ROOT}/Train.csv")
excl = full[full.ID.isin(CORRUPT_IDS)]
print("excluded corrupt rows:\n", excl.to_string())
