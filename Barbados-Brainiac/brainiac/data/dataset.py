"""The line table: one row per image with split, family, near-duplicate group and fold.

Columns of data/lines.csv (written by the `prepare` stage):
    ID, split (train | test | extra), image_path, w, h, family (A | B), Target (train only),
    group (train only), fold (train only, else -1)

Neighbouring Train.csv rows are often copies of the same formula in different hands, and some
rows are two crops of the same physical line. Such rows share a `group` and therefore a fold,
so the holdout fold never contains a near-copy of a training line.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Cfg
from ..utils import log, work_path

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    _fuzz = None


def read_train(cfg: Cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.paths.train_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df[["ID", "Target"]].drop_duplicates("ID").reset_index(drop=True)


def read_test(cfg: Cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.paths.test_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df[["ID"]].drop_duplicates("ID").reset_index(drop=True)


def list_images(images_dir: str | Path) -> dict[str, str]:
    """ID -> image path for the image files in `images_dir`."""
    return {
        p.stem: str(p)
        for p in sorted(Path(images_dir).iterdir())
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }


def image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def family_of(w: int, h: int, cfg: Cfg) -> str:
    return "B" if (w > cfg.data.family_width or h > cfg.data.family_height) else "A"


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    if _fuzz is not None:
        return _fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def build_groups(targets: list[str], window: int, threshold: float) -> np.ndarray:
    """Union-find over (a) identical normalised labels anywhere and (b) similar labels within `window` rows."""
    n = len(targets)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    keys = [_key(t) for t in targets]
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first:
            union(i, first[k])
        else:
            first[k] = i
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + window)):
            if keys[i] and keys[j] and _similarity(keys[i], keys[j]) >= threshold:
                union(i, j)
    roots = [find(i) for i in range(n)]
    remap = {r: g for g, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def assign_folds(groups: np.ndarray, families: list[str], n_folds: int, seed: int) -> np.ndarray:
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = np.full(len(groups), -1)
        for k, (_, idx) in enumerate(skf.split(np.zeros(len(groups)), families, groups)):
            folds[idx] = k
        return folds
    except ImportError:  # pragma: no cover
        rng = np.random.default_rng(seed)
        order = rng.permutation(np.unique(groups))
        fold_of_group = {g: i % n_folds for i, g in enumerate(order)}
        return np.array([fold_of_group[g] for g in groups])


def build_lines(cfg: Cfg) -> pd.DataFrame:
    train, test = read_train(cfg), read_test(cfg)
    images = list_images(cfg.paths.images_dir)
    missing = [i for i in list(train.ID) + list(test.ID) if i not in images]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Train/Test IDs have no image, e.g. {missing[:5]}")
    parts = [train.assign(split="train"), test.assign(split="test", Target="")]
    if cfg.data.use_extra_images:
        known = set(train.ID) | set(test.ID)
        extra = sorted(i for i in images if i not in known)
        parts.append(pd.DataFrame({"ID": extra, "split": "extra", "Target": ""}))
    lines = pd.concat(parts, ignore_index=True)
    if cfg.data.max_rows:
        lines = lines.groupby("split", sort=False).head(int(cfg.data.max_rows)).reset_index(drop=True)
    lines["image_path"] = lines.ID.map(images)
    sizes = [image_size(p) for p in lines.image_path]
    lines["w"] = [s[0] for s in sizes]
    lines["h"] = [s[1] for s in sizes]
    lines["family"] = [family_of(w, h, cfg) for w, h in sizes]
    lines["group"] = -1
    lines["fold"] = -1
    tr = lines.split == "train"
    groups = build_groups(list(lines.loc[tr, "Target"]), int(cfg.data.group_window), float(cfg.data.group_similarity))
    lines.loc[tr, "group"] = groups
    lines.loc[tr, "fold"] = assign_folds(groups, list(lines.loc[tr, "family"]), int(cfg.data.folds), int(cfg.seed))
    log.info(
        "lines: %d train (%d groups), %d test, %d extra; family B share %.2f",
        tr.sum(), len(set(groups)), (lines.split == "test").sum(), (lines.split == "extra").sum(),
        (lines.family == "B").mean(),
    )
    return lines


def lines_path(cfg: Cfg) -> Path:
    return work_path(cfg, "data", "lines.csv")


def load_lines(cfg: Cfg) -> pd.DataFrame:
    df = pd.read_csv(lines_path(cfg), dtype={"ID": str, "Target": str}, keep_default_na=False)
    for col in ("w", "h", "group", "fold"):
        df[col] = df[col].astype(int)
    return df


def stage_prepare(cfg: Cfg, force: bool = False) -> None:
    out = lines_path(cfg)
    if out.exists() and not force:
        log.info("prepare: %s exists", out)
        return
    build_lines(cfg).to_csv(out, index=False)
    log.info("prepare: wrote %s", out)


# ------------------------------------------------------------------ scopes
def scope_rows(lines: pd.DataFrame, cfg: Cfg, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, prediction rows) of a scope.

    oof:  train on every fold except the holdout fold; predict the holdout fold (ranker data)
    full: train on all labelled rows; predict the test split
    """
    train = lines[lines.split == "train"]
    hold = int(cfg.data.holdout_fold)
    if scope == "oof":
        return train[train.fold != hold], train[train.fold == hold]
    if scope == "full":
        return train, lines[lines.split == "test"]
    raise ValueError(f"unknown scope '{scope}'")
