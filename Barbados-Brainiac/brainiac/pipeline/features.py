"""Approach B, part 2 — the ranker's features (diagram: "Features F0-F3 + reader blocks").

    F0  candidate and source meta: which sources wrote it, their ranks and generation scores,
        length and mark counts, the line's family and image size
    F1  judges: per-judge forced-decode log-probs (sum / per token / weakest token), in-line ranks
    F2  consensus: MBR expected loss (uniform and judge-weighted), loss to each key source's line
    F3  character LM: log-prob per character, weakest character, unseen characters, word coverage
    R   readers: CTC negative log-likelihood per character, loss to the CTC / external readings
    target (holdout lines only): the candidate's exact metric loss against the reference
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import Cfg
from ..data.dataset import load_lines, scope_rows
from ..metric import line_loss, loss_matrix
from ..readers.crnn import CTCReader
from ..utils import log, read_jsonl
from .approach_b import crnn_path, scope_lm
from .common import features_path, judge_path, pool_path

META_COLS = ["ID", "cand_id", "text", "group", "loss"]


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = (x - np.nanmax(x)) / max(t, 1e-6)
    z = np.nan_to_num(z, nan=-50.0)
    e = np.exp(z)
    return e / e.sum()


def _within_line(df: pd.DataFrame, col: str, higher_is_better: bool, kinds: list[str]) -> None:
    g = df.groupby("ID")[col]
    signed = df[col] if higher_is_better else -df[col]
    if "rank" in kinds:
        df[f"{col}_rank"] = signed.groupby(df.ID).rank(ascending=False, method="min")
    if "z" in kinds:
        std = g.transform("std").replace(0, np.nan)
        df[f"{col}_z"] = ((df[col] - g.transform("mean")) / std).fillna(0.0) * (1 if higher_is_better else -1)
    if "margin" in kinds:
        best = signed.groupby(df.ID).transform("max")
        df[f"{col}_gap"] = best - signed


def build_features(cfg: Cfg, scope: str) -> pd.DataFrame:
    b = cfg.approach_b
    lines = load_lines(cfg)
    meta = lines.set_index("ID")
    pool = read_jsonl(pool_path(cfg, scope))
    df = pd.DataFrame({"ID": [r["ID"] for r in pool], "cand_id": [r["cand_id"] for r in pool],
                       "text": [r["text"] for r in pool]})
    df["group"] = df.ID.map(meta.group).fillna(-1).astype(int)

    # ---- F0: sources and shape
    sources = sorted({s for r in pool for s in r["sources"]})
    for s in sources:
        col = safe(s)
        df[f"src_{col}"] = [int(s in r["sources"]) for r in pool]
        df[f"rank_{col}"] = [r["sources"][s]["rank"] if s in r["sources"] else 99 for r in pool]
        df[f"greedy_{col}"] = [r["sources"][s]["greedy"] if s in r["sources"] else 0 for r in pool]
        df[f"gmean_{col}"] = [r["sources"][s].get("gen_mean") if s in r["sources"] else np.nan for r in pool]
    df["n_sources"] = [len(r["sources"]) for r in pool]
    df["n_methods"] = [len(r["methods"]) for r in pool]
    df["frac_sources"] = df.n_sources / max(len(sources), 1)
    t = df.text.fillna("")
    df["len_c"] = t.str.len()
    df["len_w"] = t.str.split().str.len().fillna(0)
    for name, ch in [("caret", "^"), ("colon", ":"), ("tilde", "~"), ("amp", "&"), ("dash", "-"), ("period", "."),
                     ("comma", ","), ("quote", '"'), ("apos", "'"), ("semi", ";")]:
        df[f"n_{name}"] = t.str.count(re.escape(ch))
    df["n_caps"] = t.str.count(r"[A-Z]")
    df["n_digits"] = t.str.count(r"[0-9]")
    df["frac_caps"] = df.n_caps / df.len_c.clip(lower=1)
    df["family_B"] = (df.ID.map(meta.family) == "B").astype(int)
    df["img_w"] = df.ID.map(meta.w)
    df["img_h"] = df.ID.map(meta.h)
    df["aspect"] = df.img_w / df.img_h
    df["len_ratio"] = df.len_c / df.groupby("ID").len_c.transform("median").clip(lower=1)
    df["len_w_diff"] = df.len_w - df.groupby("ID").len_w.transform("median")
    df["pool_size"] = df.groupby("ID").cand_id.transform("count")

    # ---- F1: judges
    kinds = list(b.features.within_line)
    judge_means = []
    for j in b.judges:
        path = judge_path(cfg, j.name, scope)
        if not path.exists():
            log.warning("features: judge %s [%s] missing", j.name, scope)
            continue
        scores = {r["cand_id"]: r for r in read_jsonl(path)}
        col = f"j_{safe(j.name)}"
        for stat in ("sum", "mean", "min"):
            df[f"{col}_{stat}"] = df.cand_id.map(lambda c, s=stat: scores.get(c, {}).get(s, np.nan))
        _within_line(df, f"{col}_mean", True, kinds)
        _within_line(df, f"{col}_sum", True, kinds)
        judge_means.append(f"{col}_mean")
    if judge_means:
        df["judges_mean"] = df[judge_means].mean(axis=1)
        _within_line(df, "judges_mean", True, kinds)
        df["judges_top1"] = sum((df[f"{c}_rank"] == 1).astype(int) for c in judge_means) if "rank" in kinds else 0

    # ---- R: CTC reader
    if b.ctc.enabled and crnn_path(cfg, scope).exists():
        reader = CTCReader(crnn_path(cfg, scope))
        nll, pc, unk = {}, {}, {}
        for image_id, sub in df.groupby("ID"):
            logp = reader.log_probs(meta.loc[image_id, "image_path"])
            for cid, s in zip(sub.cand_id, reader.score(logp, list(sub.text))):
                nll[cid], pc[cid], unk[cid] = s["nll"], s["nll"] / max(s["n"], 1), s["unknown"]
        df["ctc_nll"] = df.cand_id.map(nll)
        df["ctc_nll_pc"] = df.cand_id.map(pc)
        df["ctc_unknown"] = df.cand_id.map(unk)
        _within_line(df, "ctc_nll_pc", False, kinds)
        _within_line(df, "ctc_nll", False, kinds)

    # ---- F3: character LM and word coverage
    lm = scope_lm(cfg, scope, lines)
    lm_scores = [lm.score(x) for x in t]
    for key in ("lm_lp", "lm_lp_pc", "lm_min", "lm_oov"):
        df[key] = [s[key] for s in lm_scores]
    _within_line(df, "lm_lp_pc", True, kinds)
    vocab = {w for x in scope_rows(lines, cfg, scope)[0].Target for w in str(x).split()}
    vocab_lc = {w.lower() for w in vocab}
    df["vocab_cov"] = [np.mean([w in vocab for w in x.split()]) if x.split() else 0.0 for x in t]
    df["vocab_cov_lc"] = [np.mean([w.lower() in vocab_lc for w in x.split()]) if x.split() else 0.0 for x in t]

    # ---- F2: consensus / MBR and agreement with key sources
    key_sources = [s for s in b.features.key_sources if s in sources] + [s for s in sources if s.startswith("ext:")]
    top1: dict[str, dict[str, str]] = defaultdict(dict)
    for r in pool:
        for s, info in r["sources"].items():
            if s in key_sources and (info["greedy"] or s not in top1[r["ID"]]):
                if info["greedy"] or info["rank"] == 0:
                    top1[r["ID"]][s] = r["text"]
    wx, cx = float(cfg.metric.word_xmax), float(cfg.metric.char_xmax)
    mbr_u, mbr_j, lmin, lmean = {}, {}, {}, {}
    lt1: dict[str, dict] = {s: {} for s in key_sources}
    temp = float(b.features.mbr_temperature)
    for image_id, sub in df.groupby("ID"):
        texts = list(sub.text)
        m = loss_matrix(texts, wx, cx)
        k = len(texts)
        w = _softmax(sub["judges_mean"].to_numpy(float), temp) if judge_means else np.full(k, 1.0 / k)
        exp_j = w @ m                                      # expected loss of each candidate
        exp_u = m.mean(axis=0) if k > 1 else np.zeros(k)
        off = m + np.eye(k) * 1e9
        for idx, cid in enumerate(sub.cand_id):
            mbr_u[cid], mbr_j[cid] = exp_u[idx], exp_j[idx]
            lmin[cid] = off[:, idx].min() if k > 1 else 0.0
            lmean[cid] = m[:, idx].sum() / max(k - 1, 1)
        for s in key_sources:
            ref = top1[image_id].get(s)
            for cid, text in zip(sub.cand_id, texts):
                lt1[s][cid] = line_loss(ref, text, wx, cx) if ref is not None else np.nan
    df["mbr_uniform"] = df.cand_id.map(mbr_u)
    df["mbr_judge"] = df.cand_id.map(mbr_j)
    df["loss_min_other"] = df.cand_id.map(lmin)
    df["loss_mean_other"] = df.cand_id.map(lmean)
    _within_line(df, "mbr_judge", False, kinds)
    _within_line(df, "mbr_uniform", False, kinds)
    for s in key_sources:
        df[f"lt1_{safe(s)}"] = df.cand_id.map(lt1[s])

    # ---- target
    if scope == "oof":
        refs = df.ID.map(meta.Target)
        df["loss"] = [line_loss(r, h, wx, cx) for r, h in zip(refs, df.text)]
    else:
        df["loss"] = np.nan
    return df


def stage_b_features(cfg: Cfg, force: bool = False) -> None:
    if not cfg.approach_b.enabled:
        return
    for scope in cfg.scopes:
        out = features_path(cfg, scope)
        if out.exists() and not force:
            log.info("b_features: %s exists", out)
            continue
        df = build_features(cfg, scope)
        df.to_csv(out, index=False)
        log.info("b_features [%s]: %d candidates x %d columns -> %s", scope, len(df), df.shape[1], out)


def load_features(cfg: Cfg, scope: str) -> pd.DataFrame:
    df = pd.read_csv(features_path(cfg, scope), dtype={"ID": str, "cand_id": str, "text": str}, keep_default_na=False,
                     na_values=[""])
    df["text"] = df.text.fillna("")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def oracle_score(df: pd.DataFrame) -> float:
    if df.loss.isna().all():
        return math.nan
    return float(1 - df.groupby("ID").loss.min().mean())
