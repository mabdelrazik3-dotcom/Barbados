"""Approach A — the 1st-place text-extraction pipeline, adapted to line transcription.

1st place (Barbados survey plans)                  here
  automated_label_correction.py (base Qwen3-VL)      stage_label_correction
  train_and_create_pseudo.py: fine-tune ...          stage_a_train_first
  ... + pseudo labels on the test plans              stage_a_pseudo (test + unlabelled images)
  train_with_pseudos.py                              stage_a_train_final
  text_inference.py + clean_text_preds.py            stage_a_infer (cleaning in text/clean.py)

The 1st place trained each field twice in one answer — the corrected value ("Land Surveyor")
and the raw annotator value ("Land Surveyor2") — and submitted the raw-style one, because the
leaderboard scores against annotator-style references. The same idea here: with dual_target
the model answers {"ink": corrected line, "transcription": annotator-style line}, and the
"transcription" field is what gets submitted (the "ink" field joins the candidate pool).
"""
from __future__ import annotations

import math

import pandas as pd

from ..config import Cfg
from ..data.dataset import load_lines, scope_rows
from ..data.views import ViewSpec, view_instruction
from ..metric import line_edits
from ..text.prompts import answer_json, correction_instruction, correction_prompt
from ..utils import chunks, log, work_path
from ..vlm.base import DecodeSpec, make_backend
from ..vlm.sft import train_lora
from .common import (a_model_dir, build_requests, candidates_path, generate_candidates, ordered, pseudo_path,
                     save_candidates, top1_by_source, write_submission)

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def corrections_path(cfg: Cfg):
    return work_path(cfg, "data", "label_corrections.csv")


def load_corrections(cfg: Cfg) -> pd.DataFrame | None:
    path = corrections_path(cfg)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"ID": str, "Target": str, "corrected": str}, keep_default_na=False)
    df["accepted"] = df.accepted.astype(str).str.lower().isin(["true", "1"])
    df["noisy"] = df.noisy.astype(str).str.lower().isin(["true", "1"])
    return df


# ------------------------------------------------------------------ label correction
def _external_corrections(cfg: Cfg) -> dict[str, str]:
    ext = cfg.approach_a.label_correction.get("external") or {}
    if not ext.get("csv"):
        return {}
    df = pd.read_csv(ext.csv, dtype=str, keep_default_na=False)
    if ext.get("verdict_col") in df.columns:
        df = df[df[ext.verdict_col].str.upper() == "CORRECTED"]
    if ext.get("confidence_col") in df.columns:
        floor = _CONF_RANK.get(str(ext.get("min_confidence", "high")), 2)
        df = df[df[ext.confidence_col].map(lambda c: _CONF_RANK.get(str(c).lower(), -1)) >= floor]
    return dict(zip(df[ext.id_col], df[ext.text_col]))


def stage_label_correction(cfg: Cfg, force: bool = False) -> None:
    out = corrections_path(cfg)
    if out.exists() and not force:
        log.info("label_correction: %s exists", out)
        return
    lc = cfg.approach_a.label_correction
    train = load_lines(cfg).query("split == 'train'")
    table = pd.DataFrame({"ID": train.ID, "Target": train.Target, "corrected": train.Target, "source": "label",
                          "char_edits": 0, "accepted": False, "noisy": False}).set_index("ID")
    external = _external_corrections(cfg)
    for image_id, text in external.items():
        if image_id in table.index and text:
            table.loc[image_id, ["corrected", "source", "accepted"]] = [text, "external", True]
            table.loc[image_id, "char_edits"] = line_edits(table.loc[image_id, "Target"], text)[1]
    log.info("label_correction: %d external corrections", len(external))

    if lc.enabled and cfg.approach_a.enabled:
        view = ViewSpec.from_cfg(cfg, lc.views)
        prompt = correction_prompt(cfg, lc.prompt)
        backend = make_backend(cfg, lc.model, lc.get("adapter"))
        decode = DecodeSpec(greedy=True, max_new_tokens=int(cfg.hf.max_new_tokens))
        todo = ordered(train[~train.ID.isin(list(external))])
        try:
            for part in chunks(list(range(len(todo))), int(lc.batch_size) * 16):
                sub = todo.iloc[part]
                reqs = build_requests(cfg, sub, prompt, view, with_reference=cfg.backend == "mock")
                for req, row in zip(reqs, sub.itertuples()):
                    req.instruction = correction_instruction(view_instruction(view), row.Target)
                for req, cands in zip(reqs, backend.generate(reqs, decode)):
                    fixed = cands[0].text if cands else ""
                    edits = line_edits(table.loc[req.id, "Target"], fixed)[1] if fixed else 0
                    table.loc[req.id, "char_edits"] = edits
                    if fixed and 0 < edits <= int(lc.max_char_edits):
                        table.loc[req.id, ["corrected", "source", "accepted"]] = [fixed, "vlm", True]
                    elif fixed and edits >= int(lc.noisy_char_edits):
                        table.loc[req.id, "noisy"] = True
                log.info("label_correction: %d / %d", min(part[-1] + 1, len(todo)), len(todo))
        finally:
            backend.close()
    noisy = table[table.noisy].sort_values("char_edits", ascending=False)
    if len(noisy) > int(lc.max_noisy_rows):  # keep only the most extreme disagreements
        table.loc[noisy.index[int(lc.max_noisy_rows):], "noisy"] = False
    table.reset_index().to_csv(out, index=False)
    log.info("label_correction: %d accepted corrections, %d noisy rows -> %s",
             int(table.accepted.sum()), int(table.noisy.sum()), out)


# ------------------------------------------------------------------ training data
def training_examples(cfg: Cfg, rows: pd.DataFrame, *, variant: str, dual: bool, dual_order: str,
                      pseudo: pd.DataFrame | None = None) -> list[dict]:
    """Rows -> SFT examples. variant: raw | corrected | dual (answers), noisy rows dropped when configured."""
    corr = load_corrections(cfg)
    corr = corr.set_index("ID") if corr is not None else None
    drop_noisy = bool(cfg.approach_a.label_correction.drop_noisy)
    examples = []
    for r in rows.itertuples():
        target, ink = r.Target, r.Target
        if corr is not None and r.ID in corr.index:
            c = corr.loc[r.ID]
            if drop_noisy and c.noisy:
                continue
            if c.accepted:
                ink = c.corrected
        if variant == "corrected":
            target = ink
        answer = answer_json(target, ink=ink, dual=dual, dual_order=dual_order)
        examples.append({"ID": r.ID, "image_path": r.image_path, "family": r.family, "answer": answer})
    if pseudo is not None:
        for r in pseudo.itertuples():
            ink = r.ink if isinstance(r.ink, str) and r.ink else r.transcription
            answer = answer_json(r.transcription, ink=ink, dual=dual, dual_order=dual_order)
            examples.append({"ID": r.ID, "image_path": r.image_path, "family": r.family, "answer": answer})
    return examples


def load_pseudo(cfg: Cfg, backbone: str, scope: str) -> pd.DataFrame | None:
    path = pseudo_path(cfg, backbone, scope)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"ID": str, "transcription": str, "ink": str}, keep_default_na=False)


# ------------------------------------------------------------------ stages
def _a_cfg(cfg: Cfg):
    a = cfg.approach_a
    return a, a.views, dict(a.prompt), bool(a.dual_target), a.get("dual_order", "ink_first")


def stage_a_train_first(cfg: Cfg, force: bool = False) -> None:
    a, view, prompt, dual, order = _a_cfg(cfg)
    if not a.enabled:
        return
    lines = load_lines(cfg)
    for backbone in a.backbones:
        for scope in cfg.scopes:
            rows = scope_rows(lines, cfg, scope)[0]
            examples = training_examples(cfg, rows, variant="dual" if dual else "raw", dual=dual, dual_order=order)
            train_lora(cfg, model_name=backbone, lora=a.lora, train=a.train, examples=examples,
                       out_dir=a_model_dir(cfg, 1, backbone, scope), seed=int(cfg.seed), prompt_spec=prompt,
                       view_name=view, dual=dual, dual_order=order, force=force)


def stage_a_pseudo(cfg: Cfg, force: bool = False) -> None:
    a = cfg.approach_a
    if not (a.enabled and a.pseudo.enabled):
        return
    lines = load_lines(cfg)
    for backbone in a.backbones:
        for scope in cfg.scopes:
            out = pseudo_path(cfg, backbone, scope)
            if out.exists() and not force:
                log.info("pseudo: %s exists", out)
                continue
            # Each scope pseudo-labels its own prediction lines too (images only, labels unused): the full
            # models self-train on the test lines, so the oof models self-train on the holdout lines, and the
            # ranker learns from candidates made under the same conditions as the test candidates.
            own = scope_rows(lines, cfg, scope)[1].ID
            pool = lines[lines.split.isin(list(a.pseudo.splits)) | lines.ID.isin(own)]
            if a.pseudo.get("max_rows"):
                pool = pool.head(int(a.pseudo.max_rows))
            cands = generate_candidates(cfg, backbone, a_model_dir(cfg, 1, backbone, scope), pool,
                                        dict(a.pseudo.decode, include_ink=False), f"pseudo:{backbone}", scope)
            greedy = {c["ID"]: c for c in cands if c["method"] == "greedy"}
            rows = []
            for r in pool.itertuples():
                c = greedy.get(r.ID)
                if not c or not c["text"]:
                    continue
                mean = c.get("gen_mean", math.nan)
                floor = a.pseudo.get("min_mean_logprob")
                if floor is not None and (mean is None or math.isnan(mean) or mean < float(floor)):
                    continue
                rows.append({"ID": r.ID, "split": r.split, "image_path": r.image_path, "family": r.family,
                             "transcription": c["text"], "ink": c.get("ink", ""), "gen_mean": mean})
            pd.DataFrame(rows, columns=["ID", "split", "image_path", "family", "transcription", "ink",
                                        "gen_mean"]).to_csv(out, index=False)
            log.info("pseudo %s [%s]: kept %d of %d lines -> %s", backbone, scope, len(rows), len(pool), out)


def stage_a_train_final(cfg: Cfg, force: bool = False) -> None:
    a, view, prompt, dual, order = _a_cfg(cfg)
    if not a.enabled:
        return
    lines = load_lines(cfg)
    for backbone in a.backbones:
        for scope in cfg.scopes:
            rows = scope_rows(lines, cfg, scope)[0]
            pseudo = load_pseudo(cfg, backbone, scope) if a.pseudo.enabled else None
            examples = training_examples(cfg, rows, variant="dual" if dual else "raw", dual=dual, dual_order=order,
                                         pseudo=pseudo)
            train_lora(cfg, model_name=backbone, lora=a.lora, train=a.train, examples=examples,
                       out_dir=a_model_dir(cfg, 2, backbone, scope), seed=int(cfg.seed) + 1, prompt_spec=prompt,
                       view_name=view, dual=dual, dual_order=order, force=force)


def stage_a_infer(cfg: Cfg, force: bool = False) -> None:
    a = cfg.approach_a
    if not a.enabled:
        return
    lines = load_lines(cfg)
    for backbone in a.backbones:
        source = f"a:{backbone}"
        for scope in cfg.scopes:
            if candidates_path(cfg, source, scope).exists() and not force:
                log.info("a_infer: %s [%s] exists", source, scope)
                continue
            rows = scope_rows(lines, cfg, scope)[1]
            cands = generate_candidates(cfg, backbone, a_model_dir(cfg, 2, backbone, scope), rows,
                                        dict(a.infer.decode), source, scope)
            save_candidates(cfg, cands, source, scope)
            if scope == "full":
                write_submission(cfg, source, top1_by_source(cands, source))
