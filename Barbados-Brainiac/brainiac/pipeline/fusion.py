"""Fusion of approach A and approach B, final submissions, and the out-of-fold report.

Fusion happens at two levels:
  1. candidate level (main): approach-A models write candidates into the same pool as the
     approach-B generators and the CTC reader; every judge scores them and the ranker can pick
     them (their source flags, ranks and generation scores are ranker features);
  2. decision level (here): per-method submissions, gaps filled along `fusion.fallback_order`,
     and the primary submission chosen by the gates (ranker if every gate passes, else baseline).

Submissions written below <run>/submissions/: ranker, blend ("gblmx"), baseline, a_<backbone>,
and <run>/submission.csv = the primary one.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from ..config import Cfg
from ..data.dataset import load_lines, scope_rows
from ..metric import score
from ..utils import load_json, log, read_jsonl, save_json, work_path
from .common import candidates_path, submission_path, top1_by_source, write_submission
from .features import load_features, oracle_score
from .selection import picks_text, within_z


def _badness(cfg: Cfg, name: str, df: pd.DataFrame) -> np.ndarray | None:
    path = work_path(cfg, "selection", name)
    if not path.exists():
        return None
    b = pd.read_csv(path, dtype={"ID": str, "cand_id": str})
    return df[["cand_id"]].merge(b[["cand_id", "badness"]], on="cand_id", how="left").badness.to_numpy(float)


def blend_badness(df: pd.DataFrame, ranker_bad: np.ndarray, cfg: Cfg) -> np.ndarray:
    """"gblmx": ranker score mixed with MBR expected loss, char-LM and CTC evidence (all z-scored per line)."""
    w = cfg.fusion.blend
    out = float(w.ranker) * within_z(df.ID, ranker_bad)
    if "mbr_judge" in df.columns:
        out = out + float(w.mbr) * within_z(df.ID, df.mbr_judge.fillna(df.mbr_judge.max()).to_numpy())
    if "lm_lp_pc" in df.columns:
        out = out + float(w.charlm) * within_z(df.ID, -df.lm_lp_pc.fillna(df.lm_lp_pc.min()).to_numpy())
    if "ctc_nll_pc" in df.columns:
        out = out + float(w.ctc) * within_z(df.ID, df.ctc_nll_pc.fillna(df.ctc_nll_pc.max()).to_numpy())
    return out


def _approach_a_picks(cfg: Cfg, scope: str) -> dict[str, dict[str, str]]:
    picks = {}
    if cfg.approach_a.enabled:
        for backbone in cfg.approach_a.backbones:
            source = f"a:{backbone}"
            path = candidates_path(cfg, source, scope)
            if path.exists():
                picks[source] = top1_by_source(read_jsonl(path), source)
    return picks


def stage_fuse(cfg: Cfg, force: bool = False) -> None:
    subs: dict[str, dict[str, str]] = {}
    if cfg.approach_b.enabled:
        test = load_features(cfg, "full").sort_values(["ID", "cand_id"]).reset_index(drop=True)
        rb = _badness(cfg, "ranker__full.csv", test)
        if rb is not None:
            subs["ranker"] = picks_text(test, rb)
            subs["blend"] = picks_text(test, blend_badness(test, rb, cfg))
        bb = _badness(cfg, "baseline__full.csv", test)
        if bb is not None:
            subs["baseline"] = picks_text(test, bb)
    subs.update(_approach_a_picks(cfg, "full"))
    if not subs:
        raise RuntimeError("fuse: nothing to fuse - run approach A and/or B first")

    fallback = [n for n in cfg.fusion.fallback_order if n in subs]
    for name, preds in subs.items():
        for image_id in list(preds):
            if not preds[image_id]:
                preds[image_id] = next((subs[f][image_id] for f in fallback if subs[f].get(image_id)), "")
    for name in cfg.fusion.write:
        if name in subs:
            write_submission(cfg, name, subs[name])

    primary = cfg.fusion.primary
    if primary == "auto":
        gates = {}
        ranker_json = work_path(cfg, "selection", "ranker.json")
        if ranker_json.exists():
            gates = load_json(ranker_json).get("gates", {})
        primary = "ranker" if gates.get("all_pass") and "ranker" in subs else next(
            (n for n in ["baseline", *fallback, *subs] if n in subs), None)
    if primary not in subs:
        raise RuntimeError(f"fuse: primary '{primary}' has no predictions")
    if not submission_path(cfg, primary).exists():
        write_submission(cfg, primary, subs[primary])
    final = work_path(cfg, "submission.csv")
    shutil.copy2(submission_path(cfg, primary), final)
    save_json({"primary": primary, "available": sorted(subs)}, work_path(cfg, "selection", "fusion.json"))
    log.info("fuse: primary submission = %s -> %s", primary, final)


def stage_report(cfg: Cfg, force: bool = False) -> None:
    """Holdout-fold scores of every source's top line, the pool oracle and each selection method."""
    lines = load_lines(cfg)
    hold = scope_rows(lines, cfg, "oof")[1]
    refs = dict(zip(hold.ID, hold.Target))
    wx, cx = float(cfg.metric.word_xmax), float(cfg.metric.char_xmax)
    table = {}

    def add(name: str, preds: dict[str, str]) -> None:
        ids = [i for i in refs if i in preds]
        if ids:
            s = score([refs[i] for i in ids], [preds[i] for i in ids], wx, cx)
            table[name] = {"score": round(s["score"], 5), "lines": len(ids),
                           "word_edits": round(s["word_edits_per_line"], 3), "char_edits": round(s["char_edits_per_line"], 3)}

    cand_files = sorted(work_path(cfg, "candidates").glob("*__oof.jsonl"))
    for path in cand_files:
        rows = read_jsonl(path)
        for source in sorted({r["source"] for r in rows}):
            add(f"source {source}", top1_by_source(rows, source))
    if cfg.approach_b.enabled and work_path(cfg, "features", "features__oof.csv").exists():
        oof = load_features(cfg, "oof").sort_values(["ID", "cand_id"]).reset_index(drop=True)
        table["pool oracle"] = {"score": round(oracle_score(oof), 5), "lines": int(oof.ID.nunique())}
        for name, file in [("baseline (judge mix -> MBR -> CTC)", "baseline__oof.csv"),
                           ("ranker (stabilised, nested CV)", "ranker__oof.csv")]:
            bad = _badness(cfg, file, oof)
            if bad is not None:
                add(name, picks_text(oof, bad))
        rb = _badness(cfg, "ranker__oof.csv", oof)
        if rb is not None:
            add("blend (gblmx)", picks_text(oof, blend_badness(oof, rb, cfg)))
    out = work_path(cfg, "report.json")
    save_json(table, out)
    width = max((len(k) for k in table), default=10)
    lines_out = [f"{'method':<{width}}  score     lines"]
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["score"]):
        lines_out.append(f"{name:<{width}}  {row['score']:.5f}  {row['lines']}")
    (work_path(cfg, "report.txt")).write_text("\n".join(lines_out) + "\n")
    log.info("report (holdout fold):\n%s", "\n".join(lines_out))
