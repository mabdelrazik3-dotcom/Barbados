"""Issue taxonomy over ALL training lines + pixel-only issue classifier applied to test / unlisted images.

1. Every training line gets measured flags from its LABEL (conventions, punctuation, casing, rarity, formula) and
   its IMAGE (scan group, multi-line crop, edge clipping, contrast, blur).
2. Lines are grouped into "issue groups" = scan group x convention profile; purity of each group is measured.
3. For every flag, a classifier using IMAGE features only is cross-validated on train (measured AUC) and applied
   to test + unlisted images, so each test image gets the same group / issue description as its train twins.
Outputs: work/train_issue_table.csv, work/issue_groups.csv, work/issue_flag_auc.csv, test_issue_scaffold.csv
"""
import collections
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

ROOT = "D:/HANAFY/ROAD"
W = f"{ROOT}/work"
F = pd.read_csv(f"{W}/image_features_plus.csv").drop(columns=["label"], errors="ignore")
tr = pd.read_csv(f"{ROOT}/Train.csv")
tr["label"] = [" ".join(str(t).split()) for t in tr.Target]
tr["raw_ws_issue"] = [int(str(t) != " ".join(str(t).split())) for t in tr.Target]
lt = pd.read_csv(f"{W}/label_table.csv")[["ID", "template_cov", "nn_ratio"]]
CORRUPT = {"79tMUVyfIdy3GzkG", "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z", "VmrEALeZiP1Y6nF9", "t0UrASljcgzvBAnO"}

# ---------------------------------------------------------------- image group (same rule as the notebook)
F["group"] = np.where(F.height > 150, "B", np.where(F.height <= 58, "A1", np.where(F.bg_mean <= 194, "A2", "A3")))
F["multi_line_crop"] = (F.n_row_peaks >= 2).astype(int)
F["edge_clipped"] = ((F.ink_touch_top > 0.15) | (F.ink_touch_bottom > 0.15)).astype(int)
F["low_contrast"] = (F.contrast < F.contrast.quantile(0.2)).astype(int)
F["blurry"] = (F.lap_var_norm < F.groupby("group").lap_var_norm.transform(lambda s: s.quantile(0.2))).astype(int)

T = tr.merge(F, on="ID", how="left").merge(lt, on="ID", how="left")

# ---------------------------------------------------------------- label flags (measured on every train line)
words_all = [w for t in T.label for w in t.split()]
wc = collections.Counter(words_all)
forms = collections.defaultdict(collections.Counter)
for w in words_all:
    forms[w.lower()][w] += 1
majority = {k: c.most_common(1)[0][0] for k, c in forms.items()}
ambig_types = {k for k, c in forms.items() if len(c) > 1 and sum(c.values()) >= 10}

def flags(t):
    ws = t.split()
    return {
        "has_amp": int("&" in t),
        "has_thorn": int(bool(re.search(r"\by\^?[etu]\b|\by\.\^e\b|\by\^t\b", t))),     # ye / yt / y^e ...
        "has_caret": int("^" in t),
        "has_colon_susp": int(bool(re.search(r"\w:", t))),
        "has_ff": int(bool(re.search(r"\b[fF]f", t))),
        "has_digit": int(bool(re.search(r"\d", t))),
        "has_filler": int(bool(re.search(r"[~_]{1,}", t))),
        "has_bracket": int(bool(re.search(r"[()\[\]]", t))),
        "has_apostrophe": int("'" in t),
        "punct_per_word": sum(bool(re.search(r"[,.;:]", w)) for w in ws) / max(1, len(ws)),
        "cap_frac": sum(w[:1].isupper() for w in ws) / max(1, len(ws)),
        "starts_upper": int(t[:1].isupper()),
        "n_chars": len(t), "n_words": len(ws),
        "hapax_frac": sum(wc[w] == 1 for w in ws) / max(1, len(ws)),
        "casing_ambig_words": sum(w.lower() in ambig_types for w in ws),
        "minority_casing_words": sum(w.lower() in ambig_types and w != majority[w.lower()] for w in ws),
    }

L = pd.DataFrame([flags(t) for t in T.label])
T = pd.concat([T.reset_index(drop=True), L], axis=1)
T["is_formula"] = (T.template_cov >= 0.5).astype(int)
T["rare_heavy"] = (T.hapax_frac > 0.15).astype(int)
T["cap_heavy"] = (T.cap_frac > 0.3).astype(int)
T["punct_heavy"] = (T.punct_per_word > 0.15).astype(int)
T["known_corrupt"] = T.ID.isin(CORRUPT).astype(int)
# possible multi-line LABEL: image shows >=2 text rows AND label much longer than its group's typical line
p90 = T.groupby("group").n_chars.transform(lambda s: s.quantile(0.9))
T["possible_multiline_label"] = ((T.multi_line_crop == 1) & (T.n_chars > p90)).astype(int)

# ---------------------------------------------------------------- convention profile (label side, interpretable, priority order)
def profile(r):
    if r.has_caret or r.has_colon_susp:
        return "P1_superscript/suspension"      # Adm^rs, W^m, Exec:^rs, Tho:
    if r.has_thorn or r.has_amp:
        return "P2_thorn/ampersand"            # ye, yt, &
    if r.has_digit:
        return "P3_dates/numbers"
    if r.is_formula:
        return "P4_legal_formula"              # To all Christian people ...
    return "P5_plain_prose"
T["profile"] = T.apply(profile, axis=1)
T["issue_group"] = T.group + " | " + T.profile

# ---------------------------------------------------------------- group table + purity
flag_cols = ["has_amp", "has_thorn", "has_caret", "has_colon_susp", "has_ff", "has_digit", "has_filler", "has_bracket",
             "is_formula", "rare_heavy", "cap_heavy", "punct_heavy", "multi_line_crop", "edge_clipped", "low_contrast",
             "blurry", "possible_multiline_label", "raw_ws_issue"]
def purity(d):
    """mean over flags of max(p, 1-p): 1.0 = every line in the group agrees on every flag"""
    return float(np.mean([max(d[c].mean(), 1 - d[c].mean()) for c in flag_cols]))
G = T.groupby("issue_group").agg(
    n=("ID", "size"), mean_chars=("n_chars", "mean"), mean_words=("n_words", "mean"),
    amp=("has_amp", "mean"), thorn=("has_thorn", "mean"), caret=("has_caret", "mean"), colon=("has_colon_susp", "mean"),
    ff=("has_ff", "mean"), digit=("has_digit", "mean"), formula=("is_formula", "mean"), rare=("hapax_frac", "mean"),
    cap=("cap_frac", "mean"), minority_casing_per_line=("minority_casing_words", "mean"),
    multi_line_crop=("multi_line_crop", "mean"), edge_clipped=("edge_clipped", "mean"),
    possible_multiline_label=("possible_multiline_label", "mean"))
G["purity"] = T.groupby("issue_group").apply(purity)
G["share_train"] = G.n / G.n.sum()
top_words = {}
for g, d in T.groupby("issue_group"):
    cw = collections.Counter(w for t in d.label for w in t.split())
    top_words[g] = " ".join(w for w, _ in cw.most_common(12))
G["top_words"] = pd.Series(top_words)
G = G.sort_values("n", ascending=False)
G.round(3).to_csv(f"{W}/issue_groups.csv")
T.to_csv(f"{W}/train_issue_table.csv", index=False)
print(f"issue groups: {len(G)}  (weighted purity {np.average(G.purity, weights=G.n):.3f} vs "
      f"scan-group-only purity {np.average(T.groupby('group').apply(purity), weights=T.groupby('group').size()):.3f})")
print(G[["n", "share_train", "purity", "mean_chars", "amp", "thorn", "caret", "colon", "digit", "formula",
         "minority_casing_per_line", "multi_line_crop"]].round(3).to_string())

# ---------------------------------------------------------------- pixel-only classifiers, measured by 5-fold CV on train
img_cols = [c for c in F.select_dtypes(include=[np.number]).columns
            if c not in ("vcluster",) and not c.startswith(("q25_", "q3_")) and c not in ("ink_x0", "ink_x1")]
Z = np.load(f"{W}/thumb_pca64.npy")
zid = pd.Series(np.arange(len(F)), index=F.ID.values)
def X_of(ids):
    base = F.set_index("ID").loc[ids, img_cols].values.astype(np.float32)
    return np.c_[base, Z[zid.loc[ids].values]]
X = X_of(T.ID.values)
targets = ["has_caret", "has_colon_susp", "has_amp", "has_thorn", "has_digit", "has_ff", "is_formula", "rare_heavy",
           "cap_heavy", "punct_heavy", "starts_upper", "possible_multiline_label", "profile_is_P1", "profile_is_P2"]
T["profile_is_P1"] = (T.profile == "P1_superscript/suspension").astype(int)
T["profile_is_P2"] = (T.profile == "P2_thorn/ampersand").astype(int)
res, models = [], {}
for t in targets:
    y = T[t].values
    if y.sum() < 20:
        continue
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    # how much better than knowing the scan group alone?
    pg = T.groupby("group")[t].transform("mean").values
    auc_g = roc_auc_score(y, pg)
    res.append({"flag": t, "base_rate": y.mean(), "cv_auc_pixels": auc, "auc_scan_group_only": auc_g,
                "usable": auc >= 0.70})
    models[t] = clf.fit(X, y)
R = pd.DataFrame(res).sort_values("cv_auc_pixels", ascending=False)
R.round(3).to_csv(f"{W}/issue_flag_auc.csv", index=False)
print("\npixel-only prediction of label issues (5-fold CV on all 4,098 train lines):")
print(R.round(3).to_string(index=False))
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
pl = cross_val_predict(reg, X, T.n_chars.values, cv=KFold(5, shuffle=True, random_state=0))
mae = mean_absolute_error(T.n_chars, pl)
print(f"\nlength from pixels: CV MAE {mae:.1f} chars (label mean {T.n_chars.mean():.1f}); "
      f"Spearman {pd.Series(pl).corr(T.n_chars, method='spearman'):.3f}")
reg.fit(X, T.n_chars.values)
T["cv_pred_chars"] = pl
resid_sd = T.groupby("group").apply(lambda d: float(np.std(d.n_chars - d.cv_pred_chars)))

# ---------------------------------------------------------------- apply to test + unlisted -> scaffold
tst = pd.read_csv(f"{ROOT}/Test.csv")
U = F[F.split.isin(["test", "unlisted"])].copy()
Xu = X_of(U.ID.values)
for t, m in models.items():
    U[f"p_{t}"] = m.predict_proba(Xu)[:, 1]
U["pred_chars"] = reg.predict(Xu)
U["pred_chars_lo"] = U.pred_chars - 1.64 * U.group.map(resid_sd)
U["pred_chars_hi"] = U.pred_chars + 1.64 * U.group.map(resid_sd)
usable = set(R[R.usable].flag)
GROUP_NOTE = {
    "B": "class B (1670s-1710s hand, high-res scan): expect ^ superscripts and ':' suspensions; heyres/saide spellings; "
         "stained parchment; train score of this class is the lowest",
    "A1": "class A1 (tight 1640s crop): expect '&' and ye/yt abbreviations; sd for said",
    "A2": "class A2 (dark parchment, larger hand): '&' and ye are rare here; shorter lines",
    "A3": "class A3 (loose crop): neighbouring lines visible - transcribe the centre line only; longest lines",
}
def comment(r):
    c = [GROUP_NOTE[r.group]]
    if r.multi_line_crop:
        c.append("multi-line crop (>=2 text rows visible): centre-line convention")
    if r.edge_clipped:
        c.append("ink touches top/bottom edge (clipped ascenders/descenders)")
    if r.low_contrast:
        c.append("low ink contrast")
    if r.blurry:
        c.append("blurry for its group")
    for t, name in [("has_caret", "'^' superscripts"), ("has_colon_susp", "':' suspensions"), ("has_amp", "'&'"),
                    ("has_thorn", "ye/yt"), ("has_digit", "digits/dates"), ("is_formula", "legal formula")]:
        k = f"p_{t}"
        if t in usable and k in r and r[k] >= 0.6:
            c.append(f"likely {name} (p={r[k]:.2f})")
    return "; ".join(c)
U["comments"] = U.apply(comment, axis=1)
U["pseudo_label"] = ""               # filled by work/08_build_pseudo_labels.py from the notebook's model output
U["confidence"] = np.nan
keep = ["ID", "split", "group"] + [f"p_{t}" for t in models] + ["pred_chars", "pred_chars_lo", "pred_chars_hi",
        "multi_line_crop", "edge_clipped", "low_contrast", "blurry", "pseudo_label", "confidence", "comments"]
out = U[keep].copy()
out["split"] = np.where(out.ID.isin(tst.ID), "test", "unlisted")
out = out.round(3)
out.to_csv(f"{ROOT}/test_issue_scaffold.csv", index=False)
print(f"\nwrote test_issue_scaffold.csv: {len(out)} rows ({(out.split == 'test').sum()} test, {(out.split == 'unlisted').sum()} unlisted)")
print(out.groupby(["split", "group"]).size().unstack(fill_value=0))
print(out.comments.head(5).to_string())
