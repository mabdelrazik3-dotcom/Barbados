"""Shared helpers: seeding, logging, run-directory paths, JSON/JSONL I/O, GPU memory."""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .config import Cfg

_LOGGER_READY = False


def get_logger(name: str = "brainiac", level: str = "INFO") -> logging.Logger:
    global _LOGGER_READY
    if not _LOGGER_READY:
        logging.basicConfig(
            level=getattr(logging, str(level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGER_READY = True
    return logging.getLogger(name)


log = get_logger()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG; optionally make torch deterministic (the 1st place's make_deterministic)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: CTC loss and a few attention kernels have no deterministic CUDA backward
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def work_path(cfg: Cfg, *parts: str, mkdir: bool = True) -> Path:
    """A path below the experiment's work directory; parent directories are created."""
    path = Path(cfg.paths.work_dir).joinpath(*parts)
    if mkdir:
        (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
    return path


def models_path(cfg: Cfg, *parts: str) -> Path:
    path = Path(cfg.paths.models_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_default)
    tmp.replace(path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), max(1, size)):
        yield seq[i : i + size]


@contextmanager
def timed(what: str) -> Iterator[None]:
    start = time.time()
    log.info("%s ...", what)
    yield
    log.info("%s done in %.1f s", what, time.time() - start)


def stage_done(cfg: Cfg, stage: str) -> Path:
    return work_path(cfg, "stages", f"{stage}.done")


def outputs_exist(paths: Iterable[str | Path]) -> bool:
    paths = list(paths)
    return bool(paths) and all(Path(p).exists() for p in paths)
