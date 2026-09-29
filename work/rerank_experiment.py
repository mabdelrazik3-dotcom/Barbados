"""Candidate reranker, measured by cross-validation on the 819 Fold-0 validation lines (CPU only).

Pool per line = union of the 7B and 8B beam-5 lists (up to 10 distinct transcriptions). A gradient-boosted model
predicts each candidate's REGRET (its official line cost minus the best cost in the pool) from features that exist
at test time (model log-probs / ranks / posteriors, MBR risk, length and markup deviations from the pool, training-
vocabulary coverage, scan group). The candidate with the lowest predicted regret is chosen.
Honest estimate: 5-fold cross-validation over LINES, repeated with 3 seeds; paired bootstrap vs pooled MBR.
Training vocabulary uses Fold-0 TRAIN labels only (never the validation labels).
If the gain is positive, a final model trained on all 819 lines is applied to the test pools -> submission_reranked.csv
"""
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from road_eval import cost, mbr_select, nbest_posteriors, norm, official, paired_delta  # noqa: E402

ROOT = "D:/HANAFY/ROAD"
S = f"{ROOT}/trils"
M7, M8 = "qwen2_5_vl_transformers_7b_instruct_2", "qwen_3_vl_transformers_8b_instruct_1"
fa = pd.read_csv(f"{ROOT}/work/fold_assignments.csv")
vocab = collections.Counter(w for t in fa[fa.fold != 0].Target for w in str(t).split())
F = pd.read_csv(f"{ROOT}/work/image_features_plus.csv")[["ID", "height", "bg_mean"]]
F["group"] = np.where(F.height > 150, "B", np.where(F.height <= 58, "A1", np.where(F.bg_mean <= 194, "A2", "A3")))
GRP = F.set_index("ID").group


def load(split):
    a = pd.read_csv(f"{S}/EXP_002__{M7}__D2_b5nb__{split}.csv")
    b = pd.read_csv(f"{S}/EXP_002__{M8}__D2_b5nb__{split}.csv").set_index("ID").loc[a.ID].reset_index()
    return a, b


def feats_for_line(nb7, nb8, grp):
    L7 = [(norm(t), s) for t, s in json.loads(nb7)] if isinstance(nb7, str) else []
    L8 = [(norm(t), s) for t, s in json.loads(nb8)] if isinstance(nb8, str) else []
    p7 = nbest_posteriors([s for _, s in L7]) if L7 else np.array([])
    p8 = nbest_posteriors([s for _, s in L8]) if L8 else np.array([])
    post7, post8, lp7, lp8, rk7, rk8 = {}, {}, {}, {}, {}, {}
    for r, ((t, s), p) in enumerate(zip(L7, p7)):
        post7[t] = post7.get(t, 0) + p; lp7.setdefault(t, s); rk7.setdefault(t, r)
    for r, ((t, s), p) in enumerate(zip(L8, p8)):
        post8[t] = post8.get(t, 0) + p; lp8.setdefault(t, s); rk8.setdefault(t, r)
    cands = list(dict.fromkeys([t for t, _ in L8] + [t for t, _ in L7]))
    w = {c: 0.5 * post7.get(c, 0) + 0.5 * post8.get(c, 0) for c in cands}
    tot = sum(w.values()) or 1.0
    w = {c: v / tot for c, v in w.items()}
    risk = {h: sum(w[c] * cost(c, h) for c in cands) for h in cands}
    mbr = min(cands, key=risk.get)
    mlen = sum(w[c] * len(c) for c in cands); mw = sum(w[c] * len(c.split()) for c in cands)
    cnt = lambda c, pat: len(re.findall(pat, c))
    pool_mean = lambda pat: sum(w[c] * cnt(c, pat) for c in cands)
    m_caps, m_car, m_col, m_pun = pool_mean(r"\b[A-Z]"), pool_mean(r"\^"), pool_mean(":"), pool_mean(r"[,.;\-~]")
    min7 = min(lp7.values()) if lp7 else -50.0; min8 = min(lp8.values()) if lp8 else -50.0
    rows = []
    for c in cands:
        ws = c.split()
        rows.append({
            "cand": c, "post7": post7.get(c, 0.0), "post8": post8.get(c, 0.0), "wpost": w[c],
            "in7": int(c in lp7), "in8": int(c in lp8), "in_both": int(c in lp7 and c in lp8),
            "lp7": lp7.get(c, min7 - 2), "lp8": lp8.get(c, min8 - 2), "rk7": rk7.get(c, 6), "rk8": rk8.get(c, 6),
            "risk": risk[c], "risk_gap": risk[c] - risk[mbr], "is_mbr": int(c == mbr), "n_cands": len(cands),
            "len": len(c), "dlen": len(c) - mlen, "dwords": len(ws) - mw,
            "dcaps": cnt(c, r"\b[A-Z]") - m_caps, "dcaret": cnt(c, r"\^") - m_car, "dcolon": cnt(c, ":") - m_col,
            "dpunct": cnt(c, r"[,.;\-~]") - m_pun,
            "vocab_cov": np.mean([w_ in vocab for w_ in ws]) if ws else 0.0,
            "oov": sum(w_ not in vocab for w_ in ws),
            "g_A1": int(grp == "A1"), "g_A2": int(grp == "A2"), "g_A3": int(grp == "A3"), "g_B": int(grp == "B"),
        })
    return rows


def build(split, with_gt):
    a, b = load(split)
    rows = []
    for i, (rid, n7, n8) in enumerate(zip(a.ID, a.nbest, b.nbest)):
        for r in feats_for_line(n7, n8, GRP.get(rid, "A1")):
            r["line"] = i; r["ID"] = rid
            if with_gt:
                r["gt"] = norm(a.GroundTruth.iloc[i]); r["cost"] = cost(r["gt"], r["cand"])
            rows.append(r)
    D = pd.DataFrame(rows)
    if with_gt:
        D["regret"] = D.cost - D.groupby("line").cost.transform("min")
    return D


FEATS = ["post7", "post8", "wpost", "in7", "in8", "in_both", "lp7", "lp8", "rk7", "rk8", "risk", "risk_gap", "is_mbr",
         "n_cands", "len", "dlen", "dwords", "dcaps", "dcaret", "dcolon", "dpunct", "vocab_cov", "oov",
         "g_A1", "g_A2", "g_A3", "g_B"]
D = build("val", True)
n_lines = D.line.nunique()
gt = D.groupby("line").gt.first().tolist()
mbr_pick = D[D.is_mbr == 1].sort_values("line").cand.tolist()
oracle = D.loc[D.groupby("line").cost.idxmin()].sort_values("line").cand.tolist()
print(f"val lines {n_lines} | candidates/line {len(D)/n_lines:.1f} | pooled MBR {official(gt, mbr_pick)['score']:.5f} "
      f"| pool oracle {official(gt, oracle)['score']:.5f}")

picks_all, scores = [], []
for seed in (0, 1, 2):
    perm = np.random.RandomState(seed).permutation(n_lines)
    groups = perm[D.line.values]                       # shuffled line ids -> different folds per seed
    pick = [None] * n_lines
    for tr, te in GroupKFold(5).split(D, groups=groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(D.iloc[tr][FEATS], D.iloc[tr].regret)
        te_df = D.iloc[te].copy(); te_df["pred"] = m.predict(te_df[FEATS])
        for ln, g in te_df.groupby("line"):
            pick[ln] = g.loc[g.pred.idxmin(), "cand"]
    s = official(gt, pick)["score"]; scores.append(s); picks_all.append(pick)
    d = paired_delta(gt, mbr_pick, pick, n_boot=5000)
    print(f"seed {seed}: reranker CV score {s:.5f} | vs MBR delta {d['delta_score']:+.5f} CI95 "
          f"[{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}] P={d['p_b_better']:.3f} improved {d['improved']} worsened {d['worsened']}")
print(f"reranker CV mean {np.mean(scores):.5f} ± {np.std(scores):.5f}")
cls = np.array([GRP.get(i, "A1")[0] for i in D.groupby("line").ID.first()])
for c in ("A", "B"):
    idx = np.where(cls == c)[0]
    print(f"  class {c}: MBR {official([gt[i] for i in idx], [mbr_pick[i] for i in idx])['score']:.5f} -> "
          f"reranker {np.mean([official([gt[i] for i in idx], [p[i] for i in idx])['score'] for p in picks_all]):.5f}")

# final model on all val lines -> test (only if the CV gain is positive and consistent)
if min(scores) > official(gt, mbr_pick)["score"]:
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                      l2_regularization=1.0, random_state=0).fit(D[FEATS], D.regret)
    T = build("test", False)
    T["pred"] = m.predict(T[FEATS])
    tp = T.loc[T.groupby("line").pred.idxmin()].sort_values("line")
    sub = pd.DataFrame({"ID": tp.ID.values, "Target": [t or "the" for t in tp.cand.values]})
    sub.to_csv(f"{ROOT}/submission_reranked.csv", index=False)
    imp = pd.Series(np.abs(np.corrcoef(np.c_[D[FEATS].values, D.regret.values].T)[-1, :-1]), index=FEATS).sort_values(ascending=False)
    print("wrote submission_reranked.csv | |corr(feature, regret)| top:", imp.head(6).round(3).to_dict())
else:
    print("reranker not better than MBR in every seed -> no submission written")
