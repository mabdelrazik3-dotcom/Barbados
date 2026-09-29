"""Barbados-Brainiac stage runner.

    python -m brainiac.run                                    # every stage in configs/brainiac.yaml
    python -m brainiac.run --stages b_features,b_rank,fuse    # some stages
    python -m brainiac.run --from-stage b_pool                # resume from a stage
    python -m brainiac.run --force --stages b_rank            # recompute outputs that exist
    python -m brainiac.run approach_b.ranker.threads=8 seed=7 # config overrides (key=value)

Stages run strictly one after another in this process; one model is in GPU memory at a time.
A stage skips work whose outputs already exist unless --force is given.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

import yaml

from .config import load_config, to_plain
from .utils import free_gpu, get_logger, set_seed, stage_done, work_path

STAGES = {
    "prepare": "brainiac.data.dataset:stage_prepare",
    "views": "brainiac.data.views:stage_views",
    "label_correction": "brainiac.pipeline.approach_a:stage_label_correction",
    "a_train_first": "brainiac.pipeline.approach_a:stage_a_train_first",
    "a_pseudo": "brainiac.pipeline.approach_a:stage_a_pseudo",
    "a_train_final": "brainiac.pipeline.approach_a:stage_a_train_final",
    "a_infer": "brainiac.pipeline.approach_a:stage_a_infer",
    "b_train_generators": "brainiac.pipeline.approach_b:stage_b_train_generators",
    "b_soups": "brainiac.pipeline.approach_b:stage_b_soups",
    "b_generate": "brainiac.pipeline.approach_b:stage_b_generate",
    "b_readers": "brainiac.pipeline.approach_b:stage_b_readers",
    "b_pool": "brainiac.pipeline.approach_b:stage_b_pool",
    "b_judges": "brainiac.pipeline.approach_b:stage_b_judges",
    "b_features": "brainiac.pipeline.features:stage_b_features",
    "b_baseline": "brainiac.pipeline.selection:stage_b_baseline",
    "b_rank": "brainiac.pipeline.selection:stage_b_rank",
    "fuse": "brainiac.pipeline.fusion:stage_fuse",
    "report": "brainiac.pipeline.fusion:stage_report",
}


def resolve(name: str):
    module, func = STAGES[name].split(":")
    return getattr(importlib.import_module(module), func)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Barbados-Brainiac pipeline")
    p.add_argument("--config", default="configs/brainiac.yaml")
    p.add_argument("--stages", default=None, help="comma-separated stage names (default: config `stages`)")
    p.add_argument("--from-stage", default=None)
    p.add_argument("--to-stage", default=None)
    p.add_argument("--force", action="store_true", help="recompute stage outputs that already exist")
    p.add_argument("--list", action="store_true", help="print the stages and exit")
    p.add_argument("overrides", nargs="*", help="config overrides, key.sub=value")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    log = get_logger(level=cfg.get("log_level", "INFO"))
    names = args.stages.split(",") if args.stages else list(cfg.stages)
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        p.error(f"unknown stages {unknown}; known: {list(STAGES)}")
    if args.from_stage:
        names = names[names.index(args.from_stage):]
    if args.to_stage:
        names = names[: names.index(args.to_stage) + 1]
    if args.list:
        print("\n".join(names))
        return 0

    set_seed(int(cfg.seed), bool(cfg.deterministic))
    with open(work_path(cfg, "config_resolved.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False, allow_unicode=True)
    log.info("experiment %s -> %s", cfg.experiment, cfg.paths.work_dir)
    for name in names:
        start = time.time()
        log.info("=== stage %s ===", name)
        resolve(name)(cfg, force=args.force)
        stage_done(cfg, name).write_text(f"{time.time() - start:.1f}s\n")
        free_gpu()
        log.info("=== stage %s finished in %.1f s ===", name, time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
