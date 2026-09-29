"""Image <-> label joint analysis, batches/clusters, duplicates, train/test shift, fold-leakage diagnostic.
Consumes image_features.csv (all 6,159 images) + Train.csv + fold_assignments.csv."""
import collections
import json
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = "D:/HANAFY/ROAD"
W = f"{ROOT}/work"
F = pd.read_csv(f"{W}/image_features.csv")
tr = pd.read_csv(f"{ROOT}/Train.csv")
fa = pd.read_csv(f"{W}/fold_assignments.csv")
tr["label"] = [" ".join(str(t).split()) for t in tr.Target]
F = F.copy()
F = F.merge(tr[["ID", "label"]], on="ID", how="left").merge(fa[["ID", "fold", "role"]], on="ID", how="left")
rep = {}
L = F[F.split == "train"].copy()
L["n_chars"] = L.label.str.len()
L["n_words"] = L.label.str.split().str.len()
L["n_upper"] = L.label.str.count(r"[A-Z]")
L["upper_frac"] = L.n_upper / L.n_chars
L["n_caret"] = L.label.str.count(r"\^")
L["n_colon"] = L.label.str.count(":")
L["n_amp"] = L.label.str.count("&")
L["n_digit"] = L.label.str.count(r"\d")
L["n_punct"] = L.label.str.count(r"[^A-Za-z0-9 ]")
L["mean_wlen"] = L.label.map(lambda t: np.mean([len(w) for w in t.split()]))
wc = collections.Counter(w for t in L.label for w in t.split())
L["hapax_frac"] = L.label.map(lambda t: np.mean([wc[w] == 1 for w in t.split()]))
lt = pd.read_csv(f"{W}/label_table.csv")[["ID", "template_cov", "nn_ratio"]]
L = L.merge(lt, on="ID", how="left")

# ---------------- scan batches ----------------
F["tall"] = (F.height > 150).astype(int)
rep["tall_batch_counts"] = F.groupby(["split", "tall"]).size().unstack().to_dict()
bat = F.groupby("tall")[["width", "height", "aspect", "jpeg_quality_est", "bg_mean", "bg_tint_RB", "core_h", "stroke_w",
                         "ink_h_frac", "n_row_peaks", "lap_var_norm", "noise_sigma", "file_bytes", "q25_kaggle_tokens",
                         "q25_kaggle_scale", "q3_kaggle_token_rows"]].median()
rep["tall_batch_medians"] = bat.round(3).to_dict()
for c in ["jpeg_quality_est", "jpeg_q0_dc", "subsampling", "mode", "progressive", "dpi", "has_icc", "is_grayscale"]:
    if c in F:
        rep[f"value_counts_{c}"] = {str(k): int(v) for k, v in F[c].value_counts(dropna=False).head(12).items()}
# resolution proxy: x-height (core_h) in px -> glyph scale; after notebook preprocessing
F["core_h_after_q25"] = F.core_h * F.q25_kaggle_scale
F["core_h_after_q3"] = F.core_h * F.q3_kaggle_scale
rep["xheight_px_raw_by_batch"] = F.groupby("tall").core_h.describe().round(2).to_dict()
rep["xheight_px_after_q25_by_batch"] = F.groupby("tall").core_h_after_q25.describe().round(2).to_dict()
L2 = L.merge(F[["ID", "core_h_after_q25", "tall"]], on="ID")
L2["px_per_char_q25"] = L2.q25_kaggle_w * (L2.ink_w_frac) / L2.n_chars
L2["tokens_per_char_q25"] = L2.q25_kaggle_tokens / L2.n_chars
rep["px_per_char_q25_by_batch"] = L2.groupby("tall").px_per_char_q25.describe().round(2).to_dict()
rep["tokens_per_char_q25_by_batch"] = L2.groupby("tall").tokens_per_char_q25.describe().round(3).to_dict()
rep["visual_tokens_q25_kaggle"] = F.q25_kaggle_tokens.describe().round(1).to_dict()
rep["visual_tokens_q3_kaggle"] = F.q3_kaggle_tokens.describe().round(1).to_dict()
rep["token_rows_q25_kaggle"] = {int(k): int(v) for k, v in F.q25_kaggle_token_rows.value_counts().sort_index().items()}
rep["token_rows_q3_kaggle"] = {int(k): int(v) for k, v in F.q3_kaggle_token_rows.value_counts().sort_index().items()}
rep["token_rows_q25_default"] = {int(k): int(v) for k, v in F.q25_default_token_rows.value_counts().sort_index().items()}
rep["downscaled_by_ladder_frac"] = float(((F.ladder_w < F.width)).mean())
rep["ladder_downscale_factor_tall"] = F[F.tall == 1].eval("ladder_w/width").describe().round(3).to_dict()

# ---------------- image <-> label relationships ----------------
L = L.merge(F[["ID", "tall"]], on="ID")
L["ink_w_xh"] = L.ink_w / L.core_h.clip(lower=3)       # ink width in x-height units
L["w_xh"] = L.width / L.core_h.clip(lower=3)
img_cols = [c for c in F.select_dtypes(include=[np.number]).columns if c not in ("fold",) and not c.startswith("q")]
lab_cols = ["n_chars", "n_words", "upper_frac", "n_caret", "n_colon", "n_amp", "n_digit", "n_punct", "mean_wlen",
            "hapax_frac", "template_cov"]
corr = []
for a in img_cols + ["ink_w_xh", "w_xh"]:
    if a not in L:
        continue
    for b in lab_cols:
        x, y = L[a].values, L[b].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 100 and np.nanstd(x[ok]) > 0:
            rho = stats.spearmanr(x[ok], y[ok]).statistic
            corr.append((a, b, float(rho)))
corr = pd.DataFrame(corr, columns=["image_feat", "label_feat", "spearman"])
corr["abs"] = corr.spearman.abs()
corr.sort_values("abs", ascending=False).to_csv(f"{W}/image_label_correlations.csv", index=False)
rep["top_image_label_correlations"] = corr.sort_values("abs", ascending=False).head(40)[["image_feat", "label_feat", "spearman"]].round(3).values.tolist()
# per-batch length models
fits = {}
L["pred_chars"] = np.nan
for t in [0, 1]:
    s = L[L.tall == t]
    X = np.c_[s.ink_w_xh.values, s.n_gaps.values]
    ok = np.isfinite(X).all(1)
    hr = HuberRegressor().fit(X[ok], s.n_chars.values[ok])
    pred = hr.predict(X)
    L.loc[s.index, "pred_chars"] = pred
    r = s.n_chars.values - pred
    fits[t] = {"coef": hr.coef_.round(3).tolist(), "intercept": round(float(hr.intercept_), 2),
               "spearman_pred_true": round(float(stats.spearmanr(pred, s.n_chars).statistic), 3),
               "resid_mad": round(float(np.median(np.abs(r - np.median(r)))), 2),
               "rel_err_p50": round(float(np.median(np.abs(r) / s.n_chars)), 3),
               "rel_err_p90": round(float(np.percentile(np.abs(r) / s.n_chars, 90)), 3)}
rep["length_model_by_batch"] = fits
L["len_resid"] = (L.n_chars - L.pred_chars) / L.pred_chars
rep["len_outliers_resid_gt_0.6"] = int((L.len_resid.abs() > 0.6).sum())
out = L.reindex(L.len_resid.abs().sort_values(ascending=False).index)
out[["ID", "label", "width", "height", "tall", "core_h", "ink_w", "n_row_peaks", "n_chars", "pred_chars", "len_resid", "fold"]].head(150).to_csv(
    f"{W}/length_outliers.csv", index=False)
rep["len_outlier_examples"] = out[["ID", "label", "width", "height", "n_chars", "pred_chars"]].head(15).round(1).values.tolist()
# multi-line crops
rep["n_row_peaks_by_split"] = F.groupby("split").n_row_peaks.value_counts().unstack(fill_value=0).to_dict()
rep["aspect_lt6_by_split"] = F.groupby("split").apply(lambda d: int((d.aspect < 6).sum())).to_dict()

# ---------------- visual clusters ----------------
feat_cols = ["width", "height", "aspect", "g_mean", "g_std", "entropy", "otsu", "ink_frac", "ink_mean", "bg_mean", "bg_std",
             "contrast", "fisher_sep", "bg_R", "bg_G", "bg_B", "bg_tint_RB", "bg_sat", "ink_R", "ink_G", "ink_B", "ink_sat",
             "midtone_frac", "illum_cv", "lap_var_norm", "edge_density", "edge_grad_rel", "noise_sigma", "blockiness",
             "margin_t", "margin_b", "ink_h_frac", "core_h", "core_center", "cc_height_med", "stroke_w", "stroke_w_rel",
             "tiny_cc_frac", "slant_shear", "skew_deg", "file_bytes", "jpeg_quality_est"]
feat_cols = [c for c in feat_cols if c in F]
X = F[feat_cols].copy()
for c in ["width", "height", "aspect", "file_bytes", "lap_var_norm", "core_h", "noise_sigma"]:
    if c in X:
        X[c] = np.log1p(X[c].clip(lower=0))
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)
best = None
bics = {}
for k in range(2, 13):
    gm = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=2).fit(Xs)
    bics[k] = gm.bic(Xs)
kb = min(bics, key=bics.get)
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(Xs)
F["vcluster"] = km.labels_
rep["gmm_bic_best_k"] = kb
rep["gmm_bic"] = {k: round(v) for k, v in bics.items()}
cl = F.groupby("vcluster").agg(n=("ID", "size"), train=("split", lambda s: (s == "train").sum()),
                               test=("split", lambda s: (s == "test").sum()),
                               unlisted=("split", lambda s: (s == "unlisted").sum()),
                               height=("height", "median"), width=("width", "median"), bg_tint=("bg_tint_RB", "median"),
                               contrast=("contrast", "median"), stroke=("stroke_w", "median"),
                               sharp=("lap_var_norm", "median"), core_h=("core_h", "median"))
rep["visual_clusters_k8"] = cl.round(2).reset_index().values.tolist()
rep["visual_clusters_cols"] = ["vcluster"] + list(cl.columns)
# distinctive words per cluster (log-odds with prior)
Lc = L.merge(F[["ID", "vcluster"]], on="ID")
allw = collections.Counter(w.lower() for t in Lc.label for w in t.split())
tot = sum(allw.values())
dist = {}
for c, g in Lc.groupby("vcluster"):
    cw = collections.Counter(w.lower() for t in g.label for w in t.split())
    n = sum(cw.values())
    sc = {w: np.log((cw[w] + 1) / (n + len(allw))) - np.log((allw[w] + 1) / (tot + len(allw))) for w in cw if cw[w] >= 5}
    dist[int(c)] = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:12]]
    dist[f"{int(c)}_len"] = round(float(g.n_chars.mean()), 1)
rep["cluster_distinctive_words"] = dist
F[["ID", "split", "vcluster", "tall"]].to_csv(f"{W}/visual_clusters.csv", index=False)

# ---------------- duplicates via pHash / dHash ----------------
ph = np.array([int(h, 16) for h in F.phash.astype(str)], dtype=np.uint64)
def popcount64(a):
    a = a.copy()
    c = np.zeros(a.shape, np.uint8)
    for _ in range(64):
        c += (a & np.uint64(1)).astype(np.uint8)
        a >>= np.uint64(1)
    return c
N = len(ph)
pairs = []
for i in range(N):
    d = popcount64(ph[i] ^ ph[i + 1:])
    j = np.where(d <= 6)[0]
    for jj in j:
        pairs.append((i, i + 1 + jj, int(d[jj])))
dup = pd.DataFrame(pairs, columns=["i", "j", "ham"])
if len(dup):
    dup["id_i"] = F.ID.values[dup.i]; dup["id_j"] = F.ID.values[dup.j]
    dup["split_i"] = F.split.values[dup.i]; dup["split_j"] = F.split.values[dup.j]
    dup["lab_i"] = F.label.values[dup.i]; dup["lab_j"] = F.label.values[dup.j]
    dup["wh_i"] = F.width.astype(str).values[dup.i] + "x" + F.height.astype(str).values[dup.i]
    dup["wh_j"] = F.width.astype(str).values[dup.j] + "x" + F.height.astype(str).values[dup.j]
dup.to_csv(f"{W}/phash_pairs.csv", index=False)
rep["phash_pairs_ham_le6"] = int(len(dup))
rep["phash_pairs_by_split"] = dup.groupby(["split_i", "split_j"]).size().to_dict() if len(dup) else {}
rep["phash_pairs_ham0"] = int((dup.ham == 0).sum()) if len(dup) else 0

# thumbnail-based similarity (normalised grey 32x320) for re-crops
th = np.load(f"{W}/thumbs_32x320.npy").astype(np.float32).reshape(N, -1)
th = (th - th.mean(1, keepdims=True)) / (th.std(1, keepdims=True) + 1e-6)
pca = PCA(64, random_state=0).fit(th)
Z = pca.transform(th)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
np.save(f"{W}/thumb_pca64.npy", Z)
nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Z)
dist_, idx_ = nn.kneighbors(Z)
F["thumb_nn_cos"] = 1 - dist_[:, 1]
F["thumb_nn_split"] = F.split.values[idx_[:, 1]]
rep["thumb_nn_cos_quantiles"] = F.thumb_nn_cos.describe(percentiles=[.5, .9, .99]).round(4).to_dict()
hi = F[F.thumb_nn_cos > 0.97]
rep["thumb_nn_gt_0.97_by_split_pair"] = hi.groupby(["split", "thumb_nn_split"]).size().to_dict()

# ---------------- train/test/unlisted shift: adversarial validation ----------------
def adv(maskA, maskB, name):
    Xa = X[maskA | maskB].values
    y = maskB[maskA | maskB].astype(int).values
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    p = cross_val_predict(clf, Xa, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    rep[f"adv_auc_{name}"] = round(float(roc_auc_score(y, p)), 4)
adv(F.split == "train", F.split == "test", "train_vs_test")
adv(F.split == "train", F.split == "unlisted", "train_vs_unlisted")
adv(F.split == "test", F.split == "unlisted", "test_vs_unlisted")
rep["split_medians"] = F.groupby("split")[["width", "height", "aspect", "tall", "bg_mean", "contrast", "core_h", "stroke_w",
                                           "lap_var_norm", "n_row_peaks"]].median().round(3).to_dict()
rep["split_tall_frac"] = F.groupby("split").tall.mean().round(4).to_dict()

# ---------------- fold-leakage diagnostic: NN distance val->train vs test->train ----------------
Zf = np.c_[Xs, Z * 3]
tr_mask = (F.split == "train") & (F.fold != 0)
va_mask = (F.split == "train") & (F.fold == 0)
te_mask = F.split == "test"
nnm = NearestNeighbors(n_neighbors=1).fit(Zf[tr_mask.values])
dv = nnm.kneighbors(Zf[va_mask.values])[0].ravel()
dt = nnm.kneighbors(Zf[te_mask.values])[0].ravel()
rep["nn_dist_val0_to_train0"] = pd.Series(dv).describe().round(3).to_dict()
rep["nn_dist_test_to_train0"] = pd.Series(dt).describe().round(3).to_dict()
rep["ks_val_vs_test_nn_dist"] = [round(float(v), 4) for v in stats.ks_2samp(dv, dt)]

F.to_csv(f"{W}/image_features_plus.csv", index=False)
L.to_csv(f"{W}/train_joint.csv", index=False)
def _fix(o):
    if isinstance(o, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): _fix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix(v) for v in o]
    return o
rep = _fix(rep)
json.dump(rep, open(f"{W}/joint_report.json", "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)
for k, v in rep.items():
    print(f"{k}: {json.dumps(v, default=str, ensure_ascii=False)[:900]}")
