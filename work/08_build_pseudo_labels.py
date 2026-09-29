"""Build the final pseudo-label CSV (test + unlisted) from a GPU run of barbados-2-enhanced.ipynb.

usage:  python work/08_build_pseudo_labels.py <downloaded /kaggle/working dir> [--session EXP_002] [--variant D2_b5nb]
                                              [--keep 0.7] [--out pseudo_labels_final.csv]

1. Pools every shared model's N-best list per image (trial_predictions/<session>__<model>__<variant>__{val,test,unlisted}.csv)
   and picks the MBR transcription under the official per-line cost; its expected cost = "risk".
2. CALIBRATION, measured on the Fold-0 validation lines (true labels): per class (A / B) an isotonic map
   risk -> expected official line score, and risk -> P(line perfect). Quality is reported with 2-fold cross-fitting.
3. Applies the maps to test + unlisted, merges the pixel-only issue scaffold (test_issue_scaffold.csv), writes
   ID, split, group, pseudo_label, confidence, p_perfect, keep_for_training, comments (+ Target/risk for the notebook).
Nothing here reads or writes a test label by hand: every value is computed from model outputs and training evidence.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from road_eval import cost, mbr_select, nbest_posteriors, norm  # noqa: E402

ROOT = "D:/HANAFY/ROAD"
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--session", default=None)
ap.add_argument("--variant", default="D2_b5nb")
ap.add_argument("--keep", type=float, default=0.7)
ap.add_argument("--min_conf", type=float, default=None, help="absolute floor on calibrated confidence, e.g. 0.85")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--out", default=f"{ROOT}/pseudo_labels_final.csv")
a = ap.parse_args()

pdir = os.path.join(a.src, "trial_predictions") if os.path.isdir(os.path.join(a.src, "trial_predictions")) else a.src
files = sorted(glob.glob(os.path.join(pdir, f"*__{a.variant}__*.csv")))
files = [f for f in files if "@" not in os.path.basename(f) and "ENS[" not in os.path.basename(f)]
sessions = sorted({os.path.basename(f).split("__")[0] for f in files})
session = a.session or (sessions[-1] if sessions else None)
files = [f for f in files if os.path.basename(f).startswith(session + "__")]
by_split = collections.defaultdict(dict)
for f in files:
    parts = os.path.basename(f)[:-4].split("__")
    by_split[parts[-1]][parts[1]] = pd.read_csv(f)
print(f"session {session} | variant {a.variant} | models per split:",
      {k: sorted(v) for k, v in by_split.items()})
assert by_split.get("test"), "no test N-best files found for this session/variant"


def pooled(split):
    """-> DataFrame ID, pseudo_label, risk, n_systems, top1_agreement, (GroundTruth for val)"""
    sysd = by_split[split]
    names = sorted(sysd)
    ids = sysd[names[0]].ID.tolist()
    rows = []
    idx = {n: sysd[n].set_index("ID") for n in names}
    for i in ids:
        cands, w, tops = [], [], []
        for n in names:
            nb = json.loads(idx[n].at[i, "nbest"]) if isinstance(idx[n].at[i, "nbest"], str) else []
            nb = [(t, s) for t, s in nb if s is not None]
            if not nb:
                continue
            p = nbest_posteriors([s for _, s in nb], a.temperature)
            cands += [t for t, _ in nb]
            w += list(p / len(names))
            tops.append(norm(nb[0][0]))
        if not cands:
            rows.append({"ID": i, "pseudo_label": "", "risk": 1.0, "n_systems": 0, "top1_agreement": 0.0})
            continue
        txt, risk = mbr_select(cands, w, return_risk=True)
        rows.append({"ID": i, "pseudo_label": txt, "risk": risk, "n_systems": len(tops),
                     "top1_agreement": float(np.mean([t == txt for t in tops]))})
    out = pd.DataFrame(rows)
    if "GroundTruth" in sysd[names[0]]:
        out["GroundTruth"] = sysd[names[0]].set_index("ID").loc[out.ID, "GroundTruth"].astype(str).values
    return out


# ---------------------------------------------------------------- groups (same pixel rule as the notebook)
F = pd.read_csv(f"{ROOT}/work/image_features_plus.csv")[["ID", "height", "bg_mean"]]
F["group"] = np.where(F.height > 150, "B", np.where(F.height <= 58, "A1", np.where(F.bg_mean <= 194, "A2", "A3")))
grp = F.set_index("ID").group

# ---------------------------------------------------------------- calibration on validation (measured)
cal = {}
report = []
if by_split.get("val"):
    V = pooled("val")
    V["loss"] = [cost(r, h) for r, h in zip(V.GroundTruth, V.pseudo_label)]
    V["perfect"] = (V.loss == 0).astype(int)
    V["group"] = grp.loc[V.ID].values
    V["cls"] = V.group.str[0]
    rho = stats.spearmanr(V.risk, V.loss).statistic
    auc = roc_auc_score(V.perfect, -V.risk) if 0 < V.perfect.mean() < 1 else float("nan")
    report.append(f"validation lines: {len(V)} | pooled-MBR score {1 - V.loss.mean():.5f} | "
                  f"Spearman(risk, line loss) {rho:+.3f} | AUROC(risk -> perfect line) {auc:.3f}")
    for c, d in V.groupby("cls"):
        use = d if len(d) >= 50 else V                     # fall back to pooled data for tiny classes
        iso_s = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(use.risk, 1 - use.loss)
        iso_p = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0, y_max=1).fit(use.risk, use.perfect)
        cal[c] = (iso_s, iso_p)
        # honest check: 2-fold cross-fitted calibration error inside this class
        rng = np.random.RandomState(0); perm = rng.permutation(len(use)); h1, h2 = perm[::2], perm[1::2]
        err = []
        for x, y in ((h1, h2), (h2, h1)):
            m = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(use.risk.values[x], 1 - use.loss.values[x])
            err.append(np.abs(m.predict(use.risk.values[y]) - (1 - use.loss.values[y])).mean())
        report.append(f"class {c}: n={len(d)} val score {1 - d.loss.mean():.5f}, perfect lines {d.perfect.mean():.1%}, "
                      f"cross-fitted mean |expected - actual line score| = {np.mean(err):.4f}")
    V["decile"] = pd.qcut(V.risk.rank(method="first"), 10, labels=False)
    rel = V.groupby("decile").agg(risk=("risk", "mean"), line_score=("loss", lambda s: 1 - s.mean()),
                                  perfect=("perfect", "mean"), n=("ID", "size"))
    report.append("reliability by risk decile (validation):\n" + rel.round(4).to_string())
else:
    report.append("NO validation N-best files -> confidence = 1 - risk (uncalibrated)")

# ---------------------------------------------------------------- test + unlisted
scaf = pd.read_csv(f"{ROOT}/test_issue_scaffold.csv").set_index("ID")
train_vocab = collections.Counter(w for t in pd.read_csv(f"{ROOT}/Train.csv").Target.astype(str) for w in t.split())
outs = []
for split in ("test", "unlisted"):
    if not by_split.get(split):
        continue
    P = pooled(split)
    P["split"] = split
    P["group"] = grp.loc[P.ID].values
    P["cls"] = P.group.str[0]
    if cal:
        P["confidence"] = [float(cal.get(c, next(iter(cal.values())))[0].predict([r])[0]) for c, r in zip(P.cls, P.risk)]
        P["p_perfect"] = [float(cal.get(c, next(iter(cal.values())))[1].predict([r])[0]) for c, r in zip(P.cls, P.risk)]
    else:
        P["confidence"] = 1 - P.risk
        P["p_perfect"] = np.nan
    outs.append(P)
P = pd.concat(outs, ignore_index=True)
P["n_chars"] = P.pseudo_label.str.len()
P["n_words"] = P.pseudo_label.str.split().str.len().fillna(0)
lo, hi = scaf.loc[P.ID, "pred_chars_lo"].values, scaf.loc[P.ID, "pred_chars_hi"].values
P["length_ok"] = (P.n_chars >= lo) & (P.n_chars <= hi)
# keep for self-training: within each class, the most confident KEEP fraction, sane length, >= 3 words
P["keep_for_training"] = False
for c, d in P.groupby("cls"):
    thr = d.confidence.quantile(1 - a.keep)
    P.loc[d.index, "keep_for_training"] = ((d.confidence >= thr) & d.length_ok & (d.n_words >= 3)
                                          & (d.confidence >= (a.min_conf if a.min_conf is not None else -np.inf)))


def comments(r):
    c = [scaf.at[r.ID, "comments"]] if isinstance(scaf.at[r.ID, "comments"], str) else []
    if r.n_systems:
        c.append("all systems agree" if r.top1_agreement == 1 else
                 f"systems disagree (top-1 agreement {r.top1_agreement:.0%} of {r.n_systems})")
    if not r.length_ok:
        c.append(f"length {r.n_chars} outside expected {scaf.at[r.ID, 'pred_chars_lo']:.0f}-{scaf.at[r.ID, 'pred_chars_hi']:.0f} chars")
    unseen = [w for w in str(r.pseudo_label).split() if w not in train_vocab]
    if unseen:
        c.append("words unseen in train: " + " ".join(unseen[:6]) + (" ..." if len(unseen) > 6 else ""))
    c.append("KEEP for self-training" if r.keep_for_training else "exclude from self-training")
    return "; ".join(c)
P["comments"] = P.apply(comments, axis=1)
P["Target"] = P.pseudo_label                    # columns the notebook's PSEUDO_LABEL_CSV loader reads
out = P[["ID", "split", "group", "pseudo_label", "confidence", "p_perfect", "risk", "n_systems", "top1_agreement",
         "keep_for_training", "comments", "Target"]].round(4)
out.to_csv(a.out, index=False)
print("\n".join(report))
print(f"\nwrote {a.out}: {len(out)} rows | kept for training: {int(out.keep_for_training.sum())} "
      f"({out.keep_for_training.mean():.0%}) | by group:\n" + out.groupby(["split", "group"]).keep_for_training.agg(["size", "sum"]).to_string())
