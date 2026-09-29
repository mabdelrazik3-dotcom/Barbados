"""Sub-class discovery (<= 15 leaves) from PIXEL features only, supervised by LABEL conventions.

Splits use only line_features() (available for test images); the targets they are chosen to purify are the
transcription conventions measured on Train.csv (^, :, &, +, ye/yt thorn words, ff-, digits, line-end fillers,
apostrophe/quote abbreviations, line length, capitalised words). Era split (height > 150) is forced first.
Leaf counts per era are chosen by 5-fold CV of the explained convention variance.
Outputs: work/subclass_rules.py (plain if/else rules, rounded thresholds), work/subclass_assign.csv,
         work/subclass_profiles.json (label + pixel + model-error profile per sub-class, for the prompts)
"""
import collections
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

ROOT = "D:/HANAFY/ROAD"
MAX_TOTAL = 15
MIN_LEAF = 110
FEATS = ["height", "width", "aspect", "paper", "contrast", "ink_frac", "tint", "band_frac", "band_center",
         "edge_ratio", "xh_px", "stroke_rel", "sharp"]
CORRUPT = {"79tMUVyfIdy3GzkG", "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO"}

F = pd.read_csv(f"{ROOT}/work/subclass_features.csv").set_index("ID")
tr = pd.read_csv(f"{ROOT}/Train.csv")
tr = tr[~tr.ID.isin(CORRUPT)].copy()
te = pd.read_csv(f"{ROOT}/Test.csv", encoding="utf-8-sig")
te.columns = [c.lstrip("\ufeff") for c in te.columns]
known = set(tr.ID) | set(te.ID) | CORRUPT
unl = [i for i in F.index if i not in known]
THORN = re.compile(r"(?i)\by(e|t|m|r|u|or|em|eir|^e|^t|^m|^r)\b|\by\^[a-z]+")


def conv(t):
    t = " ".join(str(t).split())
    ws = t.split()
    return {
        "caret": float("^" in t), "colon": float(":" in t), "amp": float("&" in t), "plus": float("+" in t),
        "thorn": float(bool(re.search(r"(?i)(^|\s)y(e|t|m|r|u|or|em|eir)(\s|$)|(^|\s)y\^", t))),
        "ff": float(any(w[:2] in ("ff", "Ff", "FF") for w in ws)), "digit": float(bool(re.search(r"\d", t))),
        "filler": float(bool(re.search(r"[~_]|\s-$|-$", t))), "apost": float("'" in t or '"' in t),
        "n_chars": len(t) / 100.0, "cap_words": np.mean([w[:1].isupper() for w in ws]) if ws else 0.0,
    }


Y = pd.DataFrame([conv(t) for t in tr.Target], index=tr.ID)
X = F.loc[tr.ID, FEATS]
Z = (Y - Y.mean()) / Y.std(ddof=0)
era = np.where(X.height > 150, "B", "A")


def cv_r2(Xe, Ze, leaves, seed=0):
    if leaves == 1:
        return 0.0
    kf = KFold(5, shuffle=True, random_state=seed)
    num = den = 0.0
    for a, b in kf.split(Xe):
        m = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=MIN_LEAF, random_state=0).fit(Xe.iloc[a], Ze.iloc[a])
        p = m.predict(Xe.iloc[b])
        num += ((Ze.iloc[b].values - p) ** 2).sum()
        den += ((Ze.iloc[b].values - Ze.iloc[a].values.mean(0)) ** 2).sum()
    return 1 - num / den


curves = {}
for e in ("A", "B"):
    Xe, Ze = X[era == e], Z[era == e]
    curves[e] = {k: round(np.mean([cv_r2(Xe, Ze, k, s) for s in (0, 1)]), 4) for k in range(1, 13)}
    print(f"era {e}: {len(Xe)} train lines | CV R2 of conventions by #leaves:", curves[e])

# choose leaves: largest total <= MAX_TOTAL where every added leaf still adds >= 0.002 CV R2 (marginal)
def pick(c, gain=0.002):
    k = 1
    while k + 1 in c and c[k + 1] - c[k] >= gain:
        k += 1
    return k
kA, kB = pick(curves["A"]), pick(curves["B"])
while kA + kB > MAX_TOTAL:
    if curves["A"][kA] - curves["A"][kA - 1] < curves["B"][kB] - curves["B"][kB - 1]:
        kA -= 1
    else:
        kB -= 1
print(f"chosen leaves: A={kA}, B={kB} (total {kA + kB})")


def tree_rules(model, cols, prefix):
    """nested if/else source with thresholds rounded to 4 significant digits; returns (code, leaf names)."""
    t = model.tree_
    names, lines = {}, []
    order = []

    def rec(node, depth, path):
        ind = "    " * depth
        if t.children_left[node] == -1:
            order.append(node)
            nm = f"{prefix}{len(order):02d}"
            names[node] = nm
            lines.append(f"{ind}return {nm!r}")
            return
        f, thr = cols[t.feature[node]], float(f"{t.threshold[node]:.4g}")
        lines.append(f"{ind}if f[{f!r}] <= {thr!r}:")
        rec(t.children_left[node], depth + 1, path)
        lines.append(f"{ind}else:")
        rec(t.children_right[node], depth + 1, path)
    rec(0, 1, [])
    return lines, names


mA = DecisionTreeRegressor(max_leaf_nodes=kA, min_samples_leaf=MIN_LEAF, random_state=0).fit(X[era == "A"], Z[era == "A"])
mB = DecisionTreeRegressor(max_leaf_nodes=kB, min_samples_leaf=MIN_LEAF, random_state=0).fit(X[era == "B"], Z[era == "B"])
la, _ = tree_rules(mA, FEATS, "A")
lb, _ = tree_rules(mB, FEATS, "B")
code = ["def subclass_of(f, height_split=150):",
        '    """pixel-only sub-class rule (fitted by work/10_subclass_tree.py). f = line_features(path)."""',
        "    if f['height'] > height_split:",
        "        return _subclass_B(f)",
        "    return _subclass_A(f)", "", "", "def _subclass_A(f):"] + la + ["", "", "def _subclass_B(f):"] + lb
src = "\n".join(code) + "\n"
open(f"{ROOT}/work/subclass_rules.py", "w").write(src)
ns = {}
exec(src, ns)
sub = {i: ns["subclass_of"](F.loc[i].to_dict()) for i in F.index}
A = pd.DataFrame({"ID": list(sub), "subclass": list(sub.values())})
A["split"] = np.where(A.ID.isin(tr.ID), "train", np.where(A.ID.isin(te.ID), "test", np.where(A.ID.isin(CORRUPT), "corrupt", "unlisted")))
old = pd.read_csv(f"{ROOT}/work/image_features_plus.csv")[["ID", "height", "bg_mean"]]
old["group4"] = np.where(old.height > 150, "B", np.where(old.height <= 58, "A1", np.where(old.bg_mean <= 194, "A2", "A3")))
A = A.merge(old[["ID", "group4"]], on="ID", how="left")
A.to_csv(f"{ROOT}/work/subclass_assign.csv", index=False)
print("\n" + src)
print(pd.crosstab(A.subclass, A.split).assign(total=lambda d: d.sum(1)).to_string())
print("\nsub-class vs old 4 groups:\n", pd.crosstab(A.subclass, A.group4).to_string())

# ---- explained convention variance (in-sample, train) : era-only vs 4 groups vs new sub-classes
def r2(labels):
    lab = pd.Series(labels, index=Z.index)
    fitted = Z.groupby(lab).transform("mean")
    return 1 - ((Z - fitted) ** 2).values.sum() / ((Z - Z.mean()) ** 2).values.sum()
tr_sub = A.set_index("ID").loc[tr.ID]
print(f"\nconvention variance explained (train, in-sample): era A/B {r2(era):.4f} | A1/A2/A3/B {r2(tr_sub.group4.values):.4f} "
      f"| {kA + kB} sub-classes {r2(tr_sub.subclass.values):.4f}")
json.dump({"curves": curves, "kA": kA, "kB": kB}, open(f"{ROOT}/work/subclass_tree_meta.json", "w"), indent=1)
