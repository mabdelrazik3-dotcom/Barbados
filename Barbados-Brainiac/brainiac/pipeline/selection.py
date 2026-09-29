"""Approach B, part 3 — Pick, Stabilise, Gates (the workflow diagram, layers 4-5).

    b_baseline  CONFIG "judge mix -> MBR -> CTC overlay": a weighted mix of judge log-probs gives a
                posterior over a line's candidates; the pick minimises expected metric loss (MBR),
                nudged by the CTC reader's per-character likelihood
    b_rank      GBDT ranker, target = exact metric loss per candidate, trained on the holdout-fold
                lines with nested CV (outer folds -> out-of-fold score; inner split -> boosting rounds);
                members x seeds are averaged within each line ("3 CatBoost members x 5 seeds");
                gates decide whether the ranker may replace the baseline:
                twin -> tier 1 (bootstrap) -> Holm -> K5 fold consistency -> seed flip on test
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import Cfg
from ..metric import loss_matrix, paired_bootstrap
from ..utils import log, save_json, work_path
from .common import write_submission
from .features import feature_columns, load_features, oracle_score, safe


# ------------------------------------------------------------------ helpers
def within_z(ids: pd.Series, values: np.ndarray) -> np.ndarray:
    s = pd.Series(values, index=ids.index)
    g = s.groupby(ids)
    std = g.transform("std").replace(0, np.nan)
    return ((s - g.transform("mean")) / std).fillna(0.0).to_numpy()


def pick(df: pd.DataFrame, badness: np.ndarray) -> pd.Series:
    """Index (into df) of the lowest-badness candidate of each line."""
    tmp = pd.DataFrame({"ID": df.ID.values, "b": np.nan_to_num(badness, nan=np.inf)}, index=df.index)
    return tmp.groupby("ID").b.idxmin()


def picked_loss(df: pd.DataFrame, badness: np.ndarray) -> pd.Series:
    idx = pick(df, badness)
    return pd.Series(df.loc[idx.values, "loss"].values, index=idx.index)


def picks_text(df: pd.DataFrame, badness: np.ndarray) -> dict[str, str]:
    idx = pick(df, badness)
    return dict(zip(idx.index, df.loc[idx.values, "text"].values))


# ------------------------------------------------------------------ baseline
def baseline_badness(df: pd.DataFrame, cfg: Cfg) -> np.ndarray:
    bc = cfg.approach_b.baseline
    stat = bc.get("use", "mean")
    mix = np.zeros(len(df))
    used = 0
    for name, w in dict(bc.judge_weights).items():
        col = f"j_{safe(name)}_{stat}"
        if col in df.columns:
            mix += float(w) * df[col].fillna(df[col].min()).to_numpy()
            used += 1
    bad = np.zeros(len(df))
    temp = float(bc.temperature)
    wx, cx = float(cfg.metric.word_xmax), float(cfg.metric.char_xmax)
    for image_id, idx in df.groupby("ID").indices.items():
        texts = list(df.text.iloc[idx])
        m = loss_matrix(texts, wx, cx)
        if used:
            z = mix[idx] / max(temp, 1e-6)
            p = np.exp(z - z.max())
            p /= p.sum()
        else:
            p = np.full(len(idx), 1.0 / len(idx))
        bad[idx] = p @ m
    if "ctc_nll_pc" in df.columns and float(bc.ctc_overlay):
        bad = bad + float(bc.ctc_overlay) * within_z(df.ID, df.ctc_nll_pc.fillna(df.ctc_nll_pc.max()).to_numpy())
    return bad


def stage_b_baseline(cfg: Cfg, force: bool = False) -> None:
    if not cfg.approach_b.enabled:
        return
    out = work_path(cfg, "selection", "baseline.json")
    if out.exists() and not force:
        log.info("b_baseline: %s exists", out)
        return
    res = {}
    for scope in cfg.scopes:
        df = load_features(cfg, scope)
        bad = baseline_badness(df, cfg)
        df[["ID", "cand_id"]].assign(badness=bad).to_csv(work_path(cfg, "selection", f"baseline__{scope}.csv"),
                                                          index=False)
        if scope == "oof":
            res["oof_score"] = float(1 - picked_loss(df, bad).mean())
            res["oof_oracle"] = oracle_score(df)
        else:
            write_submission(cfg, "baseline", picks_text(df, bad))
    save_json(res, out)
    log.info("b_baseline: %s", res)


# ------------------------------------------------------------------ ranker members
RANKING_LOSSES = {"YetiRank", "YetiRankPairwise", "PairLogit", "PairLogitPairwise", "LambdaMart", "StochasticRank"}


def relevance(loss: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(loss / 0.5, 0.0, 1.0)


def group_sizes(ids) -> np.ndarray:
    """Run lengths of consecutive equal IDs (rows are sorted by ID)."""
    ids = np.asarray(ids)
    edges = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    return np.diff(edges)


class Member:
    """One GBDT; `predict` returns badness (lower = better candidate)."""

    def __init__(self, spec: dict, seed: int, threads: int, iterations: int | None = None):
        self.spec, self.seed, self.threads = dict(spec), int(seed), int(threads)
        self.iterations = int(iterations or self.spec.get("iterations", 1000))
        self.best_iteration = self.iterations
        self.kind = self.spec["kind"]
        self.flip = False
        self.model = None

    def fit(self, X, y, ids, Xv=None, yv=None, idsv=None, es: int | None = None) -> "Member":
        s, kind = self.spec, self.kind
        if kind in ("catboost_regressor", "catboost_ranker"):
            from catboost import CatBoostRanker, CatBoostRegressor, Pool

            params = dict(iterations=self.iterations, learning_rate=float(s.get("lr", 0.05)), depth=int(s.get("depth", 6)),
                          l2_leaf_reg=float(s.get("l2", 3.0)), random_seed=self.seed, thread_count=self.threads,
                          verbose=False, allow_writing_files=False)
            loss = s.get("loss", "RMSE" if kind == "catboost_regressor" else "QueryRMSE")
            self.flip = loss in RANKING_LOSSES
            target = relevance(y) if self.flip else y
            if kind == "catboost_regressor":
                self.model = CatBoostRegressor(loss_function=loss, **params)
                evalset = (Xv, yv) if Xv is not None else None
                self.model.fit(X, target, eval_set=evalset, early_stopping_rounds=es if evalset else None,
                               use_best_model=evalset is not None)
            else:
                self.model = CatBoostRanker(loss_function=loss, **params)
                train = Pool(X, label=target, group_id=list(ids))
                evalset = Pool(Xv, label=relevance(yv) if self.flip else yv, group_id=list(idsv)) if Xv is not None else None
                self.model.fit(train, eval_set=evalset, early_stopping_rounds=es if evalset else None,
                               use_best_model=evalset is not None)
            if Xv is not None:
                best = self.model.get_best_iteration()  # 0-based; None without an eval set
                self.best_iteration = self.iterations if best is None else int(best) + 1
        elif kind == "lightgbm_regressor":
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(n_estimators=self.iterations, learning_rate=float(s.get("lr", 0.05)),
                                           num_leaves=int(s.get("num_leaves", 31)), min_child_samples=int(s.get("min_child", 20)),
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           random_state=self.seed, n_jobs=self.threads, verbose=-1)
            cb = [lgb.early_stopping(es, verbose=False)] if Xv is not None and es else None
            self.model.fit(X, y, eval_set=[(Xv, yv)] if Xv is not None else None, callbacks=cb)
            if Xv is not None:
                self.best_iteration = max(1, int(self.model.best_iteration_ or self.iterations))
        elif kind == "lightgbm_lambdarank":
            import lightgbm as lgb

            self.flip = True

            def to_int(v):
                return np.rint(relevance(v) * 10).astype(int)

            self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=self.iterations,
                                        learning_rate=float(s.get("lr", 0.05)), num_leaves=int(s.get("num_leaves", 31)),
                                        random_state=self.seed, n_jobs=self.threads, verbose=-1)
            kw = {}
            if Xv is not None:
                kw = dict(eval_set=[(Xv, to_int(yv))], eval_group=[group_sizes(idsv)], eval_at=[1],
                          callbacks=[lgb.early_stopping(es, verbose=False)] if es else None)
            self.model.fit(X, to_int(y), group=group_sizes(ids), **kw)
            if Xv is not None:
                self.best_iteration = max(1, int(self.model.best_iteration_ or self.iterations))
        elif kind == "sklearn_hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(learning_rate=float(s.get("lr", 0.05)), max_iter=self.iterations,
                                                       random_state=self.seed, early_stopping=False)
            self.model.fit(X, y)
        else:
            raise ValueError(f"unknown ranker kind '{kind}'")
        return self

    def predict(self, X) -> np.ndarray:
        p = np.asarray(self.model.predict(X), dtype=float)
        return -p if self.flip else p


# ------------------------------------------------------------------ nested CV
def outer_folds(df: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Fold per candidate row; lines of one near-duplicate group stay together."""
    groups = df.group.where(df.group >= 0, other=df.ID.factorize()[0] + 10**6)
    uniq = pd.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    return groups.map(fold_of).to_numpy()


def nested_cv(df: pd.DataFrame, feats: list[str], spec: dict, seed: int, rc: Cfg) -> tuple[np.ndarray, list[int]]:
    X, y, ids = df[feats].to_numpy(float), df.loss.to_numpy(float), df.ID.to_numpy()
    folds = outer_folds(df, int(rc.outer_folds), seed)
    preds = np.full(len(df), np.nan)
    iters = []
    rng = np.random.default_rng(seed + 1)
    for k in range(int(rc.outer_folds)):
        te, tr = folds == k, folds != k
        tr_groups = pd.unique(df.group[tr])
        n_val = max(1, int(len(tr_groups) * float(rc.inner_valid_frac)))
        val_groups = set(rng.choice(tr_groups, size=n_val, replace=False))
        iv = tr & df.group.isin(val_groups).to_numpy()
        it = tr & ~iv
        m = Member(spec, seed, int(rc.threads)).fit(X[it], y[it], ids[it], X[iv], y[iv], ids[iv],
                                                    int(rc.early_stopping_rounds))
        iters.append(m.best_iteration)
        refit = Member(spec, seed, int(rc.threads), iterations=m.best_iteration).fit(X[tr], y[tr], ids[tr])
        preds[te] = refit.predict(X[te])
    return preds, iters


def seed_flip_rate(df: pd.DataFrame, badness_by_seed: list[np.ndarray]) -> float:
    if len(badness_by_seed) < 2:
        return 0.0
    consensus = pick(df, np.mean(badness_by_seed, axis=0))
    rates = [float((pick(df, b) != consensus).mean()) for b in badness_by_seed]
    return float(np.mean(rates))


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, bool]:
    order = sorted(pvalues, key=pvalues.get)
    passed, m = {}, len(order)
    still = True
    for i, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (m - i)
        passed[name] = still
    return passed


# ------------------------------------------------------------------ stage
def stage_b_rank(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    out = work_path(cfg, "selection", "ranker.json")
    if out.exists() and not force:
        log.info("b_rank: %s exists", out)
        return
    rc, gc = b.ranker, b.gates
    oof = load_features(cfg, "oof").sort_values(["ID", "cand_id"]).reset_index(drop=True)
    test = load_features(cfg, "full").sort_values(["ID", "cand_id"]).reset_index(drop=True)
    feats = [c for c in feature_columns(oof) if c in test.columns]
    X_test = test[feats].to_numpy(float)
    oof_by, test_by, iters_by = {}, {}, {}
    for spec in rc.members:
        for seed in rc.seeds:
            key = (spec["name"], int(seed))
            preds, iters = nested_cv(oof, feats, dict(spec), int(seed), rc)
            n_iter = int(np.median(iters))
            final = Member(dict(spec), int(seed), int(rc.threads), iterations=n_iter).fit(
                oof[feats].to_numpy(float), oof.loss.to_numpy(float), oof.ID.to_numpy())
            oof_by[key], test_by[key], iters_by[key] = preds, final.predict(X_test), n_iter
            log.info("ranker %s seed %s: OOF score %.5f (%d rounds)", spec["name"], seed,
                     1 - picked_loss(oof, preds).mean(), n_iter)

    # stabilise: z-score within line, average over seeds, then over members
    members = [s["name"] for s in rc.members]
    seeds = [int(s) for s in rc.seeds]

    def ensemble(frame: pd.DataFrame, by: dict, member_list, seed_list) -> np.ndarray:
        return np.mean([within_z(frame.ID, by[(m, s)]) for m in member_list for s in seed_list], axis=0)

    oof_stab = ensemble(oof, oof_by, members, seeds)
    test_stab = ensemble(test, test_by, members, seeds)
    per_seed_test = [ensemble(test, test_by, members, [s]) for s in seeds]
    per_seed_oof = [ensemble(oof, oof_by, members, [s]) for s in seeds]

    base = pd.read_csv(work_path(cfg, "selection", "baseline__oof.csv"), dtype={"ID": str, "cand_id": str})
    base_bad = oof[["cand_id"]].merge(base, on="cand_id", how="left").badness.to_numpy(float)
    base_loss = picked_loss(oof, base_bad)
    stab_loss = picked_loss(oof, oof_stab).reindex(base_loss.index)

    # ---- gates
    boot = paired_bootstrap(base_loss.to_numpy(), stab_loss.to_numpy(), int(gc.bootstrap), int(cfg.seed))
    gates: dict = {"tier1": boot["p_b_better"] >= float(gc.tier1_p) and boot["mean_delta_loss"] > 0, "bootstrap": boot}
    if gc.twin and len(seeds) >= 2:
        twin = abs(float(picked_loss(oof, per_seed_oof[0]).mean() - picked_loss(oof, per_seed_oof[1]).mean()))
        gates["twin_delta"] = twin
        gates["twin"] = boot["mean_delta_loss"] > float(gc.twin_factor) * twin
    else:
        gates["twin"] = True
    pvals = {"stabilised": 1 - boot["p_b_better"]}
    for m in members:
        lm = picked_loss(oof, ensemble(oof, oof_by, [m], seeds)).reindex(base_loss.index)
        pvals[m] = 1 - paired_bootstrap(base_loss.to_numpy(), lm.to_numpy(), int(gc.bootstrap), int(cfg.seed))["p_b_better"]
    holm_pass = holm(pvals, float(gc.holm_alpha))
    gates["holm"] = holm_pass["stabilised"]
    gates["holm_detail"] = {"p": pvals, "pass": holm_pass}
    fold_of_line = pd.Series(outer_folds(oof, int(rc.outer_folds), int(cfg.seed)), index=oof.ID).groupby(level=0).first()
    wins = 0
    for k in range(int(rc.outer_folds)):
        ids_k = fold_of_line[fold_of_line == k].index
        if len(ids_k) and (base_loss.reindex(ids_k) - stab_loss.reindex(ids_k)).mean() > 0:
            wins += 1
    gates["k_folds_better"] = wins
    gates["k5"] = wins >= int(gc.k_folds_min)
    flip = seed_flip_rate(test, per_seed_test)
    gates["seed_flip"] = flip
    gates["flip"] = flip <= float(gc.max_seed_flip)
    gates["all_pass"] = bool(gates["tier1"] and gates["twin"] and gates["holm"] and gates["k5"] and gates["flip"])

    summary = {
        "features": len(feats),
        "oof_lines": int(oof.ID.nunique()),
        "oof_oracle": oracle_score(oof),
        "oof_baseline": float(1 - base_loss.mean()),
        "oof_stabilised": float(1 - stab_loss.mean()),
        "oof_members": {m: float(1 - picked_loss(oof, ensemble(oof, oof_by, [m], seeds)).mean()) for m in members},
        "rounds": {f"{k[0]}_s{k[1]}": v for k, v in iters_by.items()},
        "gates": gates,
    }
    oof[["ID", "cand_id"]].assign(badness=oof_stab).to_csv(work_path(cfg, "selection", "ranker__oof.csv"), index=False)
    test[["ID", "cand_id"]].assign(badness=test_stab).to_csv(work_path(cfg, "selection", "ranker__full.csv"), index=False)
    write_submission(cfg, "ranker", picks_text(test, test_stab))
    save_json(summary, out)
    log.info("b_rank: baseline %.5f -> stabilised %.5f (oracle %.5f); gates %s", summary["oof_baseline"],
             summary["oof_stabilised"], summary["oof_oracle"], {k: v for k, v in gates.items() if isinstance(v, bool)})
    if math.isnan(summary["oof_stabilised"]):
        log.warning("b_rank: no out-of-fold score (empty holdout?)")
