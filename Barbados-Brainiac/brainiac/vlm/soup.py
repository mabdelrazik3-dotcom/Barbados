"""LoRA soups: average several adapters of the same backbone into one adapter.

    cat     exact: dW_soup = sum_i w_i * dW_i. The members' A matrices (scaled by w_i * alpha_i / r_i)
            and B matrices are concatenated along the rank axis; rank = sum of ranks, alpha = rank.
    linear  approximate: A = sum_i w_i A_i, B = sum_i w_i B_i (members must share rank and alpha).

Works on the saved adapter files only (no base model is loaded).
"""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

from ..utils import load_json, log, save_json
from .sft import META, adapter_ready


def _load_state(d: Path) -> dict:
    import torch

    st = d / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    return torch.load(str(d / "adapter_model.bin"), map_location="cpu")


def _scaling(conf: dict, r: int) -> float:
    if conf.get("alpha_pattern") or conf.get("rank_pattern"):
        log.warning("soup: rank/alpha patterns are ignored; using the global lora_alpha")
    alpha = float(conf["lora_alpha"])
    return alpha / math.sqrt(r) if conf.get("use_rslora") else alpha / r


def soup_adapters(adapter_dirs: list[str | Path], out_dir: str | Path, method: str = "cat",
                  weights: list[float] | None = None, force: bool = False) -> Path:
    import torch

    dirs = [Path(d) for d in adapter_dirs]
    out_dir = Path(out_dir)
    if adapter_ready(out_dir) and not force:
        log.info("soup exists: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = list(weights) if weights else [1.0 / len(dirs)] * len(dirs)
    metas = [load_json(d / META) for d in dirs]
    meta = dict(metas[0], soup={"method": method, "members": [str(d) for d in dirs], "weights": weights})
    if all(m.get("backend") == "mock" for m in metas):
        save_json(meta, out_dir / META)
        return out_dir

    confs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    states = [_load_state(d) for d in dirs]
    suffix_a, suffix_b = ".lora_A.weight", ".lora_B.weight"
    modules = sorted(k[: -len(suffix_a)] for k in states[0] if k.endswith(suffix_a))
    out, ranks = {}, {}
    for m in modules:
        As = [s[m + suffix_a].float() for s in states]
        Bs = [s[m + suffix_b].float() for s in states]
        dtype = states[0][m + suffix_a].dtype
        if method == "cat":
            A = torch.cat([w * _scaling(c, a.shape[0]) * a for w, c, a in zip(weights, confs, As)], dim=0)
            B = torch.cat(Bs, dim=1)
        elif method == "linear":
            if len({a.shape for a in As}) != 1 or len({(c["r"], c["lora_alpha"]) for c in confs}) != 1:
                raise ValueError("linear soup needs members with the same rank and alpha; use method 'cat'")
            A = sum(w * a for w, a in zip(weights, As))
            B = sum(w * b for w, b in zip(weights, Bs))
        else:
            raise ValueError(f"unknown soup method '{method}'")
        out[m + suffix_a], out[m + suffix_b] = A.to(dtype).contiguous(), B.to(dtype).contiguous()
        ranks[m] = A.shape[0]
    for key in states[0]:  # anything that is not a LoRA factor (e.g. modules_to_save) is averaged
        if not (key.endswith(suffix_a) or key.endswith(suffix_b)):
            out[key] = sum(w * s[key].float() for w, s in zip(weights, states)).to(states[0][key].dtype)

    conf = dict(confs[0])
    if method == "cat":
        r = Counter(ranks.values()).most_common(1)[0][0]
        conf.update(r=r, lora_alpha=r, use_rslora=False)
        odd = {m.split("base_model.model.", 1)[-1]: v for m, v in ranks.items() if v != r}
        conf["rank_pattern"] = odd
        conf["alpha_pattern"] = dict(odd)
    from safetensors.torch import save_file

    save_file(out, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(conf, indent=2))
    for f in dirs[0].iterdir():  # processor / tokenizer files saved with the first member
        if f.is_file() and f.name not in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", META):
            shutil.copy2(f, out_dir / f.name)
    save_json(meta, out_dir / META)
    log.info("soup (%s) of %d adapters -> %s (rank %s)", method, len(dirs), out_dir, conf.get("r"))
    return out_dir
