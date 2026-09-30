"""Configuration: YAML files with includes, ${dotted.key} interpolation and CLI overrides."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
_VAR = re.compile(r"\$\{([A-Za-z0-9_.\-]+)\}")


class Cfg(dict):
    """A dict with attribute access; nested dicts are Cfg too."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def to_plain(obj: Any) -> Any:
    """Cfg -> plain dicts/lists (for yaml.safe_dump)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [wrap(v) for v in obj]
    return obj


def deep_merge(base: dict, other: dict) -> dict:
    """Merge `other` into a copy of `base`; nested dicts merge, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    merged: dict = {}
    for inc in raw.pop("include", None) or []:
        merged = deep_merge(merged, _load_raw((path.parent / inc).resolve()))
    return deep_merge(merged, raw)


def _lookup(tree: dict, dotted: str) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config interpolation: unknown key '{dotted}'")
        node = node[part]
    return node


def _interpolate(node: Any, tree: dict) -> Any:
    if isinstance(node, dict):
        return {k: _interpolate(v, tree) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, tree) for v in node]
    if isinstance(node, str) and "${" in node:
        whole = _VAR.fullmatch(node)
        if whole:  # the whole value is one reference: keep the referenced type (None, int, ...)
            return _interpolate(_lookup(tree, whole.group(1)), tree)
        return _VAR.sub(lambda m: str(_interpolate(_lookup(tree, m.group(1)), tree)), node)
    return node


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Cfg:
    """Load a config file (with its includes), apply `key.sub=value` overrides, resolve paths."""
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else (PROJECT_DIR / path)
    tree = _load_raw(path.resolve())
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not key=value")
        key, value = item.split("=", 1)
        _set_dotted(tree, key.strip(), yaml.safe_load(value))
    # paths first, so ${paths.x} elsewhere receives the resolved absolute path
    paths = _interpolate(tree.get("paths", {}), tree)
    for key, value in paths.items():
        if isinstance(value, str) and value:
            p = Path(value)
            paths[key] = str(p if p.is_absolute() else (PROJECT_DIR / p).resolve())
    tree["paths"] = paths
    cfg = wrap(_interpolate(tree, tree))
    cfg["config_file"] = str(path.resolve())
    return cfg


def preset(cfg: Cfg, group: str, spec: Any) -> Cfg:
    """Resolve a preset reference: a name from cfg[group], or an inline dict (optionally with `base`)."""
    if isinstance(spec, str):
        if spec not in cfg[group]:
            raise KeyError(f"unknown {group} preset '{spec}'")
        return wrap(copy.deepcopy(dict(cfg[group][spec])))
    spec = dict(spec or {})
    base = spec.pop("base", None)
    merged = deep_merge(dict(cfg[group][base]) if base else {}, spec)
    return wrap(merged)
