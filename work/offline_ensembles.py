"""Offline ensembling of a finished GPU run (CPU only): cross-model MBR / ROVER / loop guard on Fold-0 val,
paired bootstrap vs the best single system, oracle bounds, and the test predictions of every candidate.

usage: python work/offline_ensembles.py <folder with *__val.csv / *__test.csv> [--session EXP_002]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from road_eval import cost, edits, mbr_select, nbest_posteriors, norm, official, paired_delta, word_rover  # noqa: E402

ROOT = "D:/HANAFY/ROAD"
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--session", nargs="+", default=["EXP_002"],
                help="one or more session prefixes; systems of later sessions are tagged <size>@<session>")
a = ap.parse_args()

F = pd.read_csv(f"{ROOT}/work/image_features_plus.csv")[["ID", "height", "bg_mean"]]
F["group"] = np.where(F.height > 150, "B", np.where(F.height <= 58, "A1", np.where(F.bg_mean <= 194, "A2", "A3")))
GRP = F.set_index("ID").group


def load(split):
    out = {}
    for sess in a.session:
        for f in sorted(glob.glob(os.path.join(a.src, f"{sess}__*__{split}.csv"))):
            parts = os.path.basename(f)[:-4].split("__")
            if parts[1].startswith(("ENS", "ROUTED")) or len(parts) != 4:
                continue
            d = pd.read_csv(f)
            d["pred"] = d.pred.fillna("").astype(str)
            out[(parts[1] if sess == a.session[0] else f"{sess}:{parts[1]}", parts[2])] = d
    return out


V, T = load("val"), load("test")
ids = V[next(iter(V))].ID.tolist()
gt = V[next(iter(V))].GroundTruth.astype(str).map(norm).tolist()
grp = GRP.loc[ids].values
cls = np.array([g[0] for g in grp])
tids = T[next(iter(T))].ID.tolist() if T else []
tgrp = GRP.loc[tids].values if tids else []
models = sorted({m for m, _ in V})
def tag(m):
    sess, _, name = m.rpartition(":")
    t = re.search(r"(\d+b)", name).group(1) if re.search(r"(\d+b)", name) else name[:8]
    return t + (f"@{sess}" if sess else "")
print("systems (val):", sorted(V), "\nsystems (test):", sorted(T))


def nb_list(d):
    return [[(t, s) for t, s in json.loads(x)] if isinstance(x, str) else [] for x in d.nbest]


def loop_guard(text, max_run=3):
    """collapse runaway repetitions (same token repeated > max_run times in a row) produced by decoding loops."""
    ws = norm(text).split()
    out, run = [], 0
    for i, w in enumerate(ws):
        run = run + 1 if (i and w == ws[i - 1]) else 1
        if run <= max_run:
            out.append(w)
    return " ".join(out)


C = {}                                              # name -> (val hyps, test hyps or None)
for (m, v), d in V.items():
    th = T[(m, v)].set_index("ID").loc[tids, "pred"].tolist() if (m, v) in T else None
    C[f"{tag(m)}__{v}"] = (d.set_index("ID").loc[ids, "pred"].tolist(), th)
# single-model MBR on test (val MBR files already exist as *_mbr); build test for D2 MBR
def mbr_of(nbs):
    return [mbr_select([t for t, _ in x], nbest_posteriors([s for _, s in x])) if x else "" for x in nbs]
for m in models:
    if (m, "D2_b5nb") in T:
        C[f"{tag(m)}__D2_b5nb_mbr"] = (C[f"{tag(m)}__D2_b5nb_mbr"][0], mbr_of(nb_list(T[(m, "D2_b5nb")].set_index("ID").loc[tids].reset_index())))

# ---- cross-model ensembles on the 5-best lists
def pooled(nb_by_sys, w_sys, extra_top1=()):
    out = []
    for i in range(len(nb_by_sys[0])):
        cands, w = [], []
        for nb, ws in zip(nb_by_sys, w_sys):
            x = nb[i]
            if not x:
                continue
            p = nbest_posteriors([s for _, s in x]) * ws
            cands += [t for t, _ in x]; w += list(p)
        for hyps, ws in extra_top1:
            cands.append(hyps[i]); w.append(ws)
        out.append(mbr_select(cands, w) if cands else "")
    return out

d2 = [m for m in models if (m, "D2_b5nb") in V]
if len(d2) >= 2:
    vnb = [nb_list(V[(m, "D2_b5nb")].set_index("ID").loc[ids].reset_index()) for m in d2]
    tnb = [nb_list(T[(m, "D2_b5nb")].set_index("ID").loc[tids].reset_index()) for m in d2] if all((m, "D2_b5nb") in T for m in d2) else None
    names = "+".join(tag(m) for m in d2)
    C[f"ENS[{names}]__D2__mbr_equal"] = (pooled(vnb, [1 / len(d2)] * len(d2)), pooled(tnb, [1 / len(d2)] * len(d2)) if tnb else None)
    # quality-weighted: weights fixed a priori from the single-model D2 scores (ratio of lost points), not tuned
    lost = np.array([1 - official(gt, C[f"{tag(m)}__D2_b5nb"][0])["score"] for m in d2])
    wq = (1 / lost) / (1 / lost).sum()
    C[f"ENS[{names}]__D2__mbr_quality_w"] = (pooled(vnb, wq), pooled(tnb, wq) if tnb else None)
    if len(d2) >= 3:
        import itertools
        for k in range(2, len(d2)):
            for sub in itertools.combinations(d2, k):
                ii = [d2.index(m) for m in sub]
                lw = np.array([lost[j] for j in ii]); ww = (1 / lw) / (1 / lw).sum()
                C[f"ENS[{'+'.join(tag(m) for m in sub)}]__D2__mbr_quality_w"] = (
                    pooled([vnb[j] for j in ii], ww), pooled([tnb[j] for j in ii], ww) if tnb else None)
    # + D0 top-1 of each model as an extra vote
    d0v = [(C[f"{tag(m)}__D0_orig"][0], 0.25 / len(d2)) for m in d2 if f"{tag(m)}__D0_orig" in C]
    d0t = [(C[f"{tag(m)}__D0_orig"][1], 0.25 / len(d2)) for m in d2 if C.get(f"{tag(m)}__D0_orig", (0, None))[1] is not None]
    if d0v:
        C[f"ENS[{names}]__D2+D0__mbr"] = (pooled(vnb, [0.75 / len(d2)] * len(d2), d0v),
                                          pooled(tnb, [0.75 / len(d2)] * len(d2), d0t) if (tnb and len(d0t) == len(d0v)) else None)
    # word ROVER over the single-model MBR outputs (+ D0 of the best model), best system first
    order = sorted(d2, key=lambda m: -official(gt, C[f"{tag(m)}__D2_b5nb_mbr"][0])["score"])
    rv = [C[f"{tag(m)}__D2_b5nb_mbr"][0] for m in order] + [C[f"{tag(order[0])}__D0_orig"][0]]
    C[f"ROVER[{names}]__mbr_outputs+D0"] = ([word_rover([h[i] for h in rv], [1.2, 1.0, 1.1]) for i in range(len(ids))], None)
# loop guard on every candidate (tests the repetition-loop failure measured on 7B D1)
for k in list(C):
    vh, th = C[k]
    g = [loop_guard(h) for h in vh]
    if any(x != norm(y) for x, y in zip(g, vh)):
        C[k + "+loopguard"] = (g, [loop_guard(h) for h in th] if th else None)

# ---- scores, per class, paired vs best single
rows = []
for k, (vh, th) in C.items():
    m = official(gt, vh)
    r = {"candidate": k, "score": m["score"], "w/line": m["word_edits_per_line"], "c/line": m["char_edits_per_line"],
         "has_test": th is not None}
    for c in ("A", "B"):
        idx = np.where(cls == c)[0]
        r[f"score_{c}"] = official([gt[i] for i in idx], [vh[i] for i in idx])["score"]
    for g in ("A1", "A2", "A3"):
        idx = np.where(grp == g)[0]
        r[g] = official([gt[i] for i in idx], [vh[i] for i in idx])["score"]
    rows.append(r)
R = pd.DataFrame(rows).sort_values("score", ascending=False)
singles = [k for k in C if not k.startswith(("ENS[", "ROVER[")) and "+loopguard" not in k]
best_single = max(singles, key=lambda k: official(gt, C[k][0])["score"])
dl = []
for k in R.candidate:
    d = paired_delta(gt, C[best_single][0], C[k][0], n_boot=5000)
    dl.append((d["delta_score"], f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]", d["p_b_better"], d["improved"], d["worsened"]))
R["delta_vs_best_single"], R["ci95"], R["P(better)"], R["improved"], R["worsened"] = zip(*dl)
pd.set_option("display.width", 250)
print(f"\nbest single system: {best_single}\n")
print(R.round(5).to_string(index=False))

# ---- oracle bounds
sys_top = [C[f"{tag(m)}__D2_b5nb_mbr"][0] for m in d2]
orc = np.mean([min(cost(g, h[i]) for h in sys_top) for i, g in enumerate(gt)])
nbo = {tag(m): 1 - np.mean([min((cost(g, t) for t, _ in x), default=1) for g, x in zip(gt, nb_list(V[(m, "D2_b5nb")].set_index("ID").loc[ids].reset_index()))]) for m in d2}
print(f"\noracle: best of {len(d2)} systems per line = {1 - orc:.5f} | 5-best oracle per model = " + ", ".join(f"{k} {v:.5f}" for k, v in nbo.items()))
L = pd.DataFrame({tag(m): [cost(g, h) for g, h in zip(gt, C[f"{tag(m)}__D2_b5nb_mbr"][0])] for m in d2})
print("line-loss correlation between systems (Spearman):\n", L.corr(method="spearman").round(3).to_string())
print("lines wrong in all systems:", int((L > 0).all(1).sum()), "| wrong in exactly one:", int(((L > 0).sum(1) == 1).sum()), "of", len(L))

# ---- write test predictions of the best candidate that has them
best = next(k for k in R.candidate if C[k][1] is not None)
sub = pd.DataFrame({"ID": tids, "Target": [norm(t) or "the" for t in C[best][1]]})
out = f"{ROOT}/submission_offline_best.csv"
sub.to_csv(out, index=False)
R.to_csv(f"{a.src}/offline_ensembles_val.csv", index=False)
print(f"\nwrote {out} from '{best}' (val {R.set_index('candidate').at[best, 'score']:.5f}) | rows {len(sub)} | "
      f"empty {int((sub.Target == 'the').sum())}")
