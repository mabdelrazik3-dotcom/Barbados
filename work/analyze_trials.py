"""Phase 8/15/16/20 — offline analysis of a returned Kaggle session.

usage:  python work/analyze_trials.py <dir with trial_results.csv + trial_predictions/>  [out.md]

Joins every Fold-0 validation prediction with the full-data image/label forensics and writes a
markdown report: trial table, paired bootstrap deltas, error forensics of the champion, where the
loss lives (scan batch / visual cluster / length / template-ness / markup), confidence calibration,
model diversity + oracle bounds, and label-noise suspects (all systems agree, GT disagrees)."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from road_eval import cost, edits, forensics, official, paired_delta, norm  # noqa: E402

ROOT = "D:/HANAFY/ROAD"


def md_table(df, floatfmt=".4f", index=True):
    """minimal DataFrame -> GitHub markdown (avoids the optional tabulate dependency)."""
    d = df.reset_index() if index else df
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, floatfmt.lstrip("+") if not floatfmt.startswith("+") else floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(f(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


src = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/work/dryrun"
out_md = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "TRIAL_ANALYSIS.md")
res = pd.read_csv(os.path.join(src, "trial_results.csv"))
pdir = os.path.join(src, "trial_predictions") if os.path.isdir(os.path.join(src, "trial_predictions")) else src
val = {}
for f in sorted(os.listdir(pdir)):
    if f.endswith("__val.csv"):
        d = pd.read_csv(f).set_index("ID") if False else pd.read_csv(os.path.join(pdir, f)).set_index("ID")
        val[f[:-9]] = d
names = list(val)
ids = val[names[0]].index.tolist()
gt = val[names[0]].GroundTruth.astype(str).map(norm).tolist()
H = {k: [norm(x) for x in v.loc[ids, "pred"].fillna("").astype(str)] for k, v in val.items()}
L = {k: np.array([cost(r, h) for r, h in zip(gt, hs)]) for k, hs in H.items()}

# context features (full-data forensics)
F = pd.read_csv(f"{ROOT}/work/image_features_plus.csv").set_index("ID")
J = pd.read_csv(f"{ROOT}/work/train_joint.csv").set_index("ID")
ctx = pd.DataFrame(index=ids)
ctx["tall"] = F.loc[ids, "tall"].values
ctx["vcluster"] = F.loc[ids, "vcluster"].values
ctx["n_row_peaks"] = F.loc[ids, "n_row_peaks"].clip(upper=3).values
ctx["contrast_q"] = pd.qcut(F.loc[ids, "contrast"], 4, labels=["c1_low", "c2", "c3", "c4_high"]).astype(str).values
ctx["sharp_q"] = pd.qcut(F.loc[ids, "lap_var_norm"].rank(method="first"), 4, labels=["s1_blur", "s2", "s3", "s4_sharp"]).astype(str).values
ctx["len_bucket"] = pd.cut([len(t) for t in gt], [0, 50, 60, 70, 80, 200], labels=["<50", "50-59", "60-69", "70-79", "80+"]).astype(str)
ctx["template_q"] = pd.cut(J.loc[ids, "template_cov"], [-0.01, 0.0, 0.5, 1.01], labels=["none", "partial", "template"]).astype(str).values
ctx["has_caret"] = [int("^" in t) for t in gt]
ctx["has_digit"] = [int(any(c.isdigit() for c in t)) for t in gt]
ctx["hapax_q"] = pd.cut(J.loc[ids, "hapax_frac"], [-0.01, 0.0, 0.15, 1.01], labels=["0", "<=15%", ">15%"]).astype(str).values

md = [f"# Trial analysis — `{os.path.basename(os.path.abspath(src))}`\n",
      f"Fold-0 validation lines: **{len(ids)}**. Metric: official leaderboard score (higher = better). "
      f"Paired bootstrap over whole lines (10k resamples).\n"]
cols = [c for c in ["trial", "model", "variant", "score", "word_edits_per_line", "char_edits_per_line", "delta_baseline",
                    "delta_ci95_lo", "delta_ci95_hi", "p_better_baseline", "runtime_min", "status"] if c in res]
md.append("## Trial table\n")
md.append(md_table(res.sort_values("score", ascending=False)[cols], ".5f", index=False))

scores = {k: official(gt, h)["score"] for k, h in H.items()}
champ = max(scores, key=scores.get)
base = next((k for k in names if k.endswith("D0_orig")), names[0])
md.append(f"\n**Champion:** `{champ}` = {scores[champ]:.5f} · **Baseline:** `{base}` = {scores[base]:.5f}\n")

# ---- controlled pairwise deltas (one change each) ----
md.append("## Controlled deltas (paired bootstrap)\n")
pairs = []
for k in names:
    stem = k.rsplit("__", 1)[0]
    for a, b in [("D0_orig", "D1_norep"), ("D1_norep", "D2_b5nb"), ("D2_b5nb", "D2_b5nb_mbr")]:
        if k == f"{stem}__{b}" and f"{stem}__{a}" in H:
            pairs.append((f"{stem}__{a}", k))
if base != champ:
    pairs.append((base, champ))
rows = []
for a, b in pairs:
    d = paired_delta(gt, H[a], H[b])
    rows.append({"A": a, "B": b, "delta": d["delta_score"], "ci95": f"[{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]",
                 "P(B>A)": d["p_b_better"], "improved": d["improved"], "worsened": d["worsened"],
                 "word_edits_saved": d["net_word_edits_saved"], "char_edits_saved": d["net_char_edits_saved"]})
if rows:
    md.append(md_table(pd.DataFrame(rows), "+.5f", index=False))

# ---- where does the champion lose? ----
md.append("\n## Where the champion's loss lives\n")
lc = L[champ]
tot = lc.sum() + 1e-12
for g in ctx.columns:
    t = pd.DataFrame({"g": ctx[g].astype(str).values, "loss": lc, "base": L[base]})
    agg = t.groupby("g").agg(n=("loss", "size"), score=("loss", lambda x: 1 - x.mean()),
                             loss_share=("loss", lambda x: x.sum() / tot),
                             delta_vs_base=("loss", lambda x: (t.loc[x.index, "base"] - x).mean()))
    agg["line_share"] = agg.n / len(t)
    md.append(f"\n**{g}**\n\n" + md_table(agg.round(4)))

# ---- error forensics ----
fx = forensics(gt, H[champ])
md.append("\n## Error forensics (champion)\n")
ops = fx["ops"]
wtot = sum(v for k, v in ops.items() if k.startswith("word_")) or 1
ctot = sum(v for k, v in ops.items() if k.startswith("char_")) or 1
md.append(f"- word ops: " + ", ".join(f"{k[5:]} {v} ({v / wtot:.0%})" for k, v in ops.items() if k.startswith("word_")))
md.append(f"- char ops: " + ", ".join(f"{k[5:]} {v} ({v / ctot:.0%})" for k, v in ops.items() if k.startswith("char_")))
wc = fx["word_sub_classes"]; wct = sum(wc.values()) or 1
md.append(f"- word-substitution classes: " + ", ".join(f"{k} {v} ({v / wct:.0%})" for k, v in sorted(wc.items(), key=lambda kv: -kv[1])))
md.append(f"- space-related char edits: {fx['space_char_edit_frac']:.1%}; perfect lines: {fx['lines_perfect']}/{len(gt)}; "
          f"top-5% lines carry {fx['loss_share_top5pct_lines']:.1%} of loss; mean len diff {fx['len_diff_mean']:+.2f} chars")
for k in ["word_sub_top", "word_del_top", "word_ins_top", "char_sub_top", "char_del_top", "char_ins_top"]:
    md.append(f"- **{k}**: " + "; ".join(f"`{a}` ×{n}" if not isinstance(a, tuple) else f"`{a[0]!r}->{a[1]!r}` ×{n}" for a, n in fx[k][:18]))
# markup ratios
for mk in ["&", "^", ":", ","]:
    g_ = sum(t.count(mk) for t in gt); p_ = sum(t.count(mk) for t in H[champ])
    md.append(f"- `{mk}` emitted/true = {p_}/{g_} = {p_ / max(g_, 1):.3f}")

# ---- confidence calibration ----
md.append("\n## Confidence calibration (champion's own log-prob)\n")
dch = val[champ]
if "logprob" in dch and dch.logprob.notna().any():
    lp = dch.loc[ids, "logprob"].astype(float).values
    nt = dch.loc[ids, "n_tokens"].astype(float).clip(lower=1).values
    for nm, conf in [("total logprob", lp), ("mean logprob/token", lp / nt)]:
        ok = np.isfinite(conf)
        rho = stats.spearmanr(conf[ok], lc[ok]).statistic
        perfect = (lc == 0)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(perfect[ok], conf[ok]) if perfect[ok].any() and (~perfect[ok]).any() else float("nan")
        q = np.quantile(conf[ok], [0.2, 0.5])
        md.append(f"- {nm}: Spearman(conf, line loss) = {rho:+.3f}; AUROC(perfect line) = {auc:.3f}; "
                  f"score of least-confident 20% = {1 - lc[ok][conf[ok] <= q[0]].mean():.4f} vs rest {1 - lc[ok][conf[ok] > q[0]].mean():.4f}")

# ---- diversity + oracle ----
md.append("\n## Diversity and oracle bounds\n")
top1 = [k for k in names if not k.startswith("EXP") or "ENS[" not in k]
singles = [k for k in names if "ENS[" not in k and k.endswith(("D1_norep", "D2_b5nb"))]
if len(singles) >= 2:
    M = pd.DataFrame({k: L[k] for k in singles})
    md.append("Line-loss Spearman correlation between systems:\n\n" + md_table(M.corr(method="spearman").round(3), ".3f"))
    md.append(f"\n- oracle best-of-{len(singles)} systems per line: **{1 - M.min(axis=1).mean():.5f}** "
              f"(best single {max(1 - M.mean()):.5f})")
    err = (M > 0)
    md.append(f"- lines wrong in ALL systems: {int(err.all(1).sum())}; wrong in exactly one: {int((err.sum(1) == 1).sum())}")
for k in names:
    if k.endswith("D2_b5nb") and "nbest" in val[k]:
        nb = [json.loads(x) for x in val[k].loc[ids, "nbest"]]
        orc = np.array([min(cost(r, c[0]) for c in n) for r, n in zip(gt, nb)])
        md.append(f"- `{k}` N-best oracle (best of 5 beams per line): **{1 - orc.mean():.5f}** vs top-1 {scores[k]:.5f}")

# ---- label-noise suspects ----
md.append("\n## Label-noise suspects (systems agree with each other, disagree with GT)\n")
if len(singles) >= 2:
    agree = np.array([np.mean([cost(H[a][i], H[b][i]) for a in singles for b in singles if a < b]) for i in range(len(ids))])
    sus = np.where((agree < 0.02) & (lc > 0.25))[0]
    md.append(f"{len(sus)} lines (inter-system cost < 0.02 and loss vs GT > 0.25); they carry {lc[sus].sum() / tot:.1%} of the champion's loss.\n")
    for i in sus[:25]:
        md.append(f"- `{ids[i]}` GT: {gt[i]}  \n  pred: {H[champ][i]}")
open(out_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("wrote", out_md)
