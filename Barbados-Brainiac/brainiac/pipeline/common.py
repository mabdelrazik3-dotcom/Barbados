"""Helpers shared by the stages: adapter settings, requests, generation, file conventions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Cfg
from ..data.views import ViewSpec, get_views, view_instruction
from ..text.prompts import answer_json, build_prompt
from ..utils import chunks, log, models_path, read_jsonl, work_path, write_jsonl
from ..vlm.base import DecodeSpec, Request, make_backend
from ..vlm.sft import adapter_meta

INFER_CHUNK = 256  # lines whose views are held in memory at once


# ------------------------------------------------------------------ file conventions
def a_model_dir(cfg: Cfg, round_: int, backbone: str, scope: str) -> Path:
    return models_path(cfg, f"a{round_}_{backbone}_{scope}")


def gen_model_dir(cfg: Cfg, name: str, scope: str, seed: int | None = None) -> Path:
    return models_path(cfg, f"gen_{name}_{scope}" + (f"_s{seed}" if seed is not None else "_soup"))


def candidates_path(cfg: Cfg, source: str, scope: str) -> Path:
    return work_path(cfg, "candidates", f"{source.replace(':', '_')}__{scope}.jsonl")


def pool_path(cfg: Cfg, scope: str) -> Path:
    return work_path(cfg, "pool", f"pool__{scope}.jsonl")


def judge_path(cfg: Cfg, judge: str, scope: str) -> Path:
    return work_path(cfg, "judges", f"{judge}__{scope}.jsonl")


def features_path(cfg: Cfg, scope: str) -> Path:
    return work_path(cfg, "features", f"features__{scope}.csv")


def pseudo_path(cfg: Cfg, backbone: str, scope: str) -> Path:
    return work_path(cfg, "data", f"pseudo_a_{backbone}__{scope}.csv")


def submission_path(cfg: Cfg, name: str) -> Path:
    return work_path(cfg, "submissions", f"{name.replace(':', '_')}.csv")


# ------------------------------------------------------------------ adapters
def generator_adapter(cfg: Cfg, name: str, scope: str) -> tuple[str, Path]:
    """(backbone, adapter dir) of a pool source: 'a:<backbone>' or a generator name."""
    if name.startswith("a:"):
        backbone = name.split(":", 1)[1]
        return backbone, a_model_dir(cfg, 2, backbone, scope)
    for g in cfg.approach_b.generators:
        if g.name == name:
            if g.get("soup", "none") != "none" and len(g.seeds) > 1:
                return g.model, gen_model_dir(cfg, name, scope)
            return g.model, gen_model_dir(cfg, name, scope, int(g.seeds[0]))
    raise KeyError(f"unknown generator '{name}'")


def adapter_settings(adapter: Path) -> dict:
    meta = adapter_meta(adapter)
    return {"prompt": meta["prompt"], "view": meta["view"], "dual": bool(meta.get("dual")),
            "dual_order": meta.get("dual_order", "ink_first")}


def answer_for(settings: dict, text: str) -> str:
    """The answer JSON a model trained with `settings` would write for the line `text`."""
    return answer_json(text, ink=text, dual=settings["dual"], dual_order=settings["dual_order"])


# ------------------------------------------------------------------ requests / generation
def build_requests(cfg: Cfg, rows: pd.DataFrame, prompt: str, view: ViewSpec, instruction: str | None = None,
                   with_reference: bool = False) -> list[Request]:
    instruction = instruction or view_instruction(view)
    reqs = []
    for r in rows.itertuples():
        meta = {"reference": r.Target} if with_reference and getattr(r, "Target", "") else {}
        reqs.append(Request(id=r.ID, prompt=prompt, images=get_views(cfg, view, r.ID, r.image_path, r.family),
                            instruction=instruction, meta=meta))
    return reqs


def ordered(rows: pd.DataFrame) -> pd.DataFrame:
    """Similar image sizes next to each other -> less padding per batch."""
    return rows.sort_values(["family", "w"]).reset_index(drop=True)


def generate_candidates(cfg: Cfg, model_name: str, adapter: Path, rows: pd.DataFrame, decode_spec: dict,
                        source: str, scope: str) -> list[dict]:
    settings = adapter_settings(adapter)
    prompt = build_prompt(cfg, settings["prompt"], dual=settings["dual"], dual_order=settings["dual_order"])
    view = ViewSpec.from_cfg(cfg, settings["view"])
    decode = DecodeSpec.from_cfg(decode_spec, int(cfg.hf.max_new_tokens))
    backend = make_backend(cfg, model_name, adapter)
    out: list[dict] = []
    try:
        rows = ordered(rows)
        for part in chunks(list(range(len(rows))), INFER_CHUNK):
            sub = rows.iloc[part]
            reqs = build_requests(cfg, sub, prompt, view, with_reference=cfg.backend == "mock")
            for req, cands in zip(reqs, backend.generate(reqs, decode)):
                for c in cands:
                    out.append(dict(c.as_row(req.id, source), scope=scope))
                    if decode.include_ink and c.ink and c.ink != c.text:
                        out.append(dict(c.as_row(req.id, source), scope=scope, text=c.ink, method="ink"))
            log.info("%s [%s]: %d / %d lines", source, scope, min(part[-1] + 1, len(rows)), len(rows))
    finally:
        backend.close()
    return out


def save_candidates(cfg: Cfg, rows: list[dict], source: str, scope: str) -> Path:
    path = candidates_path(cfg, source, scope)
    write_jsonl(rows, path)
    log.info("candidates %s [%s]: %d rows -> %s", source, scope, len(rows), path)
    return path


def load_candidate_files(cfg: Cfg, scope: str) -> list[dict]:
    rows = []
    for p in sorted(work_path(cfg, "candidates").glob(f"*__{scope}.jsonl")):
        rows.extend(read_jsonl(p))
    return rows


def top1_by_source(cands: list[dict], source: str) -> dict[str, str]:
    """ID -> the source's greedy (or best-ranked) line."""
    best: dict[str, tuple] = {}
    for c in cands:
        if c["source"] != source or c.get("method") == "ink":
            continue
        key = (0 if c["method"] == "greedy" else 1, c["rank"])
        if c["ID"] not in best or key < best[c["ID"]][0]:
            best[c["ID"]] = (key, c["text"])
    return {k: v[1] for k, v in best.items()}


def write_submission(cfg: Cfg, name: str, preds: dict[str, str]) -> Path:
    sample = pd.read_csv(cfg.paths.sample_submission, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sample.columns = [c.strip().lstrip("﻿") for c in sample.columns]
    sub = pd.DataFrame({"ID": sample.ID, "Target": [preds.get(i, "") for i in sample.ID]})
    path = submission_path(cfg, name)
    sub.to_csv(path, index=False)
    missing = int((sub.Target == "").sum())
    log.info("submission %s: %d lines (%d empty) -> %s", name, len(sub), missing, path)
    return path
