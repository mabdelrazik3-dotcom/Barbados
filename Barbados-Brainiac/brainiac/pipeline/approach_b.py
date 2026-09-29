"""Approach B, part 1 — Generate and Score (the workflow diagram, layers 1-3).

    b_train_generators  LoRA generators per scope (several seeds for a soup)
    b_soups             average the seed LoRAs ("8B soup — 3 LoRAs averaged", "soup3d3")
    b_generate          ~14 candidate lines per generator and line (greedy + beam n-best + samples)
    b_readers           CRNN CTC reader: train per scope, beam-8 strings; external readers
    b_pool              union of all candidates per line: approach A + B generators + readers
    b_judges            each judge scores every pooled candidate by forced decoding
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..config import Cfg
from ..data.dataset import load_lines, scope_rows
from ..data.views import ViewSpec
from ..lm.charlm import CharNGramLM
from ..readers.crnn import CTCReader, greedy_decode, prefix_beam_search, train_crnn
from ..readers.external import ExternalReader
from ..text.clean import clean_line
from ..text.prompts import build_prompt
from ..utils import chunks, log, models_path, read_jsonl, write_jsonl
from ..vlm.base import make_backend
from ..vlm.sft import train_lora
from ..vlm.soup import soup_adapters
from .approach_a import load_pseudo, training_examples
from .common import (INFER_CHUNK, adapter_settings, answer_for, backend_kind, build_requests, candidates_path,
                     gen_model_dir, generate_candidates, generator_adapter, judge_path, load_candidate_files, ordered,
                     pool_path, save_candidates)


# ------------------------------------------------------------------ generators
def generator_examples(cfg: Cfg, g, rows: pd.DataFrame, scope: str) -> tuple[list[dict], bool]:
    """Training examples of a generator's data variant: raw | corrected | dual, + pseudo."""
    parts = set(str(g.get("data", "raw")).split("+"))
    dual = "dual" in parts
    variant = "corrected" if "corrected" in parts else ("dual" if dual else "raw")
    pseudo = None
    if "pseudo" in parts:
        source = g.get("pseudo_from") or f"a:{cfg.approach_a.backbones[0]}"
        pseudo = load_pseudo(cfg, source.split(":", 1)[1], scope)
        if pseudo is None:
            log.warning("generator %s: no pseudo labels for %s [%s]; training without them", g.name, source, scope)
    return training_examples(cfg, rows, variant=variant, dual=dual, dual_order="ink_first", pseudo=pseudo), dual


def stage_b_train_generators(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    lines = load_lines(cfg)
    for g in b.generators:
        for scope in cfg.scopes:
            rows = scope_rows(lines, cfg, scope)[0]
            examples, dual = generator_examples(cfg, g, rows, scope)
            for seed in g.seeds:
                train_lora(cfg, model_name=g.model, lora=g.lora, train=g.train, examples=examples,
                           out_dir=gen_model_dir(cfg, g.name, scope, int(seed)), seed=int(seed),
                           prompt_spec=dict(g.prompt), view_name=g.get("views", "single"), dual=dual, force=force)


def stage_b_soups(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    for g in b.generators:
        if g.get("soup", "none") == "none" or len(g.seeds) < 2:
            continue
        for scope in cfg.scopes:
            members = [gen_model_dir(cfg, g.name, scope, int(s)) for s in g.seeds]
            soup_adapters(members, gen_model_dir(cfg, g.name, scope), method=g.soup, force=force)


def stage_b_generate(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    lines = load_lines(cfg)
    for g in b.generators:
        for scope in cfg.scopes:
            if candidates_path(cfg, g.name, scope).exists() and not force:
                log.info("b_generate: %s [%s] exists", g.name, scope)
                continue
            model, adapter = generator_adapter(cfg, g.name, scope)
            rows = scope_rows(lines, cfg, scope)[1]
            save_candidates(cfg, generate_candidates(cfg, model, adapter, rows, dict(g.decode), g.name, scope),
                            g.name, scope)


# ------------------------------------------------------------------ readers
def scope_lm(cfg: Cfg, scope: str, lines: pd.DataFrame | None = None) -> CharNGramLM:
    lines = load_lines(cfg) if lines is None else lines
    texts = list(scope_rows(lines, cfg, scope)[0].Target)
    lm_cfg = cfg.approach_b.charlm
    if lm_cfg.get("include_pseudo"):
        pseudo = load_pseudo(cfg, cfg.approach_a.backbones[0], scope)
        if pseudo is not None:
            texts += list(pseudo.transcription)
    return CharNGramLM(int(lm_cfg.order), float(lm_cfg.discount)).fit(texts)


def crnn_path(cfg: Cfg, scope: str):
    return models_path(cfg, f"crnn_{scope}.pt")


def stage_b_readers(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    lines = load_lines(cfg)
    c = b.ctc
    for scope in cfg.scopes:
        train_rows, pred_rows = scope_rows(lines, cfg, scope)
        if c.enabled:
            path = train_crnn(cfg, train_rows, crnn_path(cfg, scope), int(cfg.seed), force=force)
            if candidates_path(cfg, "ctc", scope).exists() and not force:
                log.info("b_readers: ctc [%s] candidates exist", scope)
            else:
                reader = CTCReader(path)
                lm = scope_lm(cfg, scope, lines) if float(c.lm_weight) > 0 else None
                rows = []
                for r in pred_rows.itertuples():
                    logp = reader.log_probs(r.image_path)
                    greedy = greedy_decode(logp, reader.charset)
                    rows.append({"ID": r.ID, "source": "ctc", "text": clean_line(greedy), "raw": greedy,
                                 "method": "greedy", "rank": 0, "gen_logprob": float("nan"), "scope": scope})
                    beams = prefix_beam_search(logp, reader.charset, int(c.beam_width), float(c.prune), lm,
                                               float(c.lm_weight))
                    for k, (text, lp) in enumerate(beams[: int(c.candidates)]):
                        rows.append({"ID": r.ID, "source": "ctc", "text": clean_line(text), "raw": text,
                                     "method": "beam", "rank": k, "gen_logprob": lp, "scope": scope})
                save_candidates(cfg, rows, "ctc", scope)
        for spec in b.get("external_readers") or []:
            source = f"ext:{spec['name']}"
            if candidates_path(cfg, source, scope).exists() and not force:
                continue
            texts = ExternalReader(dict(spec), cfg).predict(pred_rows)
            save_candidates(cfg, [{"ID": i, "source": source, "text": t, "raw": t, "method": "greedy", "rank": 0,
                                   "gen_logprob": float("nan"), "scope": scope} for i, t in texts.items()],
                            source, scope)


# ------------------------------------------------------------------ pool
def stage_b_pool(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    lines = load_lines(cfg)
    for scope in cfg.scopes:
        out = pool_path(cfg, scope)
        if out.exists() and not force:
            log.info("b_pool: %s exists", out)
            continue
        ids = list(scope_rows(lines, cfg, scope)[1].ID)
        cands = load_candidate_files(cfg, scope)
        if not b.get("include_approach_a", True):
            cands = [c for c in cands if not c["source"].startswith("a:")]
        per_line: dict[str, dict[str, dict]] = defaultdict(dict)
        for c in cands:
            text = c["text"]
            entry = per_line[c["ID"]].setdefault(text, {"sources": {}, "methods": set()})
            src = entry["sources"].setdefault(c["source"], {"rank": 99, "greedy": 0, "gen_mean": None})
            src["rank"] = min(src["rank"], int(c.get("rank", 0)))
            src["greedy"] = max(src["greedy"], int(c.get("method") == "greedy"))
            gm = c.get("gen_mean")
            if gm is not None and gm == gm and (src["gen_mean"] is None or gm > src["gen_mean"]):
                src["gen_mean"] = gm
            entry["methods"].add(c.get("method", ""))
        rows = []
        for image_id in ids:
            entries = per_line.get(image_id) or {"": {"sources": {}, "methods": set()}}
            items = sorted(entries.items(), key=lambda kv: (-len(kv[1]["sources"]),
                                                            -sum(s["greedy"] for s in kv[1]["sources"].values()),
                                                            min([s["rank"] for s in kv[1]["sources"].values()] or [99])))
            for k, (text, e) in enumerate(items[: int(b.pool.max_per_line)]):
                rows.append({"ID": image_id, "cand_id": f"{image_id}#{k}", "text": text,
                             "sources": e["sources"], "methods": sorted(e["methods"])})
        write_jsonl(rows, out)
        n_lines = max(len(ids), 1)
        log.info("b_pool [%s]: %d candidates for %d lines (%.1f per line) -> %s", scope, len(rows), len(ids),
                 len(rows) / n_lines, out)


def read_pool(cfg: Cfg, scope: str) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(pool_path(cfg, scope)):
        by_id[row["ID"]].append(row)
    return by_id


# ------------------------------------------------------------------ judges
def stage_b_judges(cfg: Cfg, force: bool = False) -> None:
    b = cfg.approach_b
    if not b.enabled:
        return
    lines = load_lines(cfg)
    for j in b.judges:
        for scope in cfg.scopes:
            out = judge_path(cfg, j.name, scope)
            if out.exists() and not force:
                log.info("b_judges: %s [%s] exists", j.name, scope)
                continue
            model, adapter = generator_adapter(cfg, j.generator, scope)
            settings = adapter_settings(adapter)
            if j.get("prompt"):
                settings["prompt"] = dict(j.prompt)
            prompt = build_prompt(cfg, settings["prompt"], dual=settings["dual"], dual_order=settings["dual_order"])
            view = ViewSpec.from_cfg(cfg, settings["view"])
            pool = read_pool(cfg, scope)
            rows = ordered(scope_rows(lines, cfg, scope)[1])
            kind = backend_kind(cfg, j.generator)
            backend = make_backend(cfg, model, adapter, kind=kind)
            results = []
            try:
                for part in chunks(list(range(len(rows))), INFER_CHUNK):
                    sub = rows.iloc[part]
                    reqs = build_requests(cfg, sub, prompt, view, with_reference=kind == "mock")
                    answers = [[answer_for(settings, c["text"]) for c in pool.get(r.id, [])] for r in reqs]
                    for req, scores in zip(reqs, backend.score(reqs, answers)):
                        for c, s in zip(pool.get(req.id, []), scores):
                            results.append({"ID": req.id, "cand_id": c["cand_id"], **s})
                    log.info("judge %s [%s]: %d / %d lines", j.name, scope, min(part[-1] + 1, len(rows)), len(rows))
            finally:
                backend.close()
            write_jsonl(results, out)
