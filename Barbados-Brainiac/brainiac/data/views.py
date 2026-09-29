"""Model input views — the 1st place's `patchify_images.py`, adapted to single-line crops.

The 1st place cut every survey plan into 7 crops (full, top/bottom halves, 4 quadrants) and
letterboxed them to 1024 px so small print became legible to Qwen3-VL. A deed-book line has a
single long axis, and the family-A strips are only 40-115 px tall: at their native size the
letters cover 2-3 rows of the vision encoder's patch grid (28 px for Qwen2.5-VL, 32 px for
Qwen3-VL). So the views here rescale by height and cut along the width:

    single     the whole line, rescaled to a family-specific height
    multi      the whole line + `parts` overlapping pieces at a larger height (the 7-crop idea)
    composite  the pieces stacked into one image (one image token block, cheaper prompts)

Views are rendered once by the `views` stage and cached as PNG files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from ..config import Cfg
from ..utils import log, work_path

Image.MAX_IMAGE_PIXELS = None


@dataclass
class ViewSpec:
    name: str
    mode: str = "single"
    height_a: int = 112
    height_b: int = 192
    part_height_a: int = 160
    part_height_b: int = 256
    parts: int = 2
    overlap: float = 0.12
    max_width: int = 2240
    autocontrast: bool = True

    @classmethod
    def from_cfg(cls, cfg: Cfg, name: str) -> "ViewSpec":
        if name not in cfg.views:
            raise KeyError(f"unknown view spec '{name}' (configs: views.*)")
        known = cls.__dataclass_fields__
        return cls(name=name, **{k: v for k, v in dict(cfg.views[name]).items() if k in known})

    @property
    def n_images(self) -> int:
        return 1 + self.parts if self.mode == "multi" else 1


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def fit_height(img: Image.Image, height: int, max_width: int) -> Image.Image:
    w, h = img.size
    scale = height / h
    if w * scale > max_width:
        scale = max_width / w
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.LANCZOS)


def split_parts(img: Image.Image, parts: int, overlap: float) -> list[Image.Image]:
    w, h = img.size
    step = w / parts
    pad = int(step * overlap / 2)
    out = []
    for k in range(parts):
        x0 = max(0, int(k * step) - pad)
        x1 = min(w, int((k + 1) * step) + pad)
        out.append(img.crop((x0, 0, x1, h)))
    return out


def make_views(path: str, family: str, spec: ViewSpec) -> list[Image.Image]:
    img = load_rgb(path)
    if spec.autocontrast:
        img = ImageOps.autocontrast(img, cutoff=1)
    height = spec.height_b if family == "B" else spec.height_a
    if spec.mode == "single":
        return [fit_height(img, height, spec.max_width)]
    part_h = spec.part_height_b if family == "B" else spec.part_height_a
    pieces = split_parts(img, spec.parts, spec.overlap)
    if spec.mode == "multi":
        return [fit_height(img, height, spec.max_width)] + [fit_height(p, part_h, spec.max_width) for p in pieces]
    if spec.mode == "composite":
        pieces = [fit_height(p, height, spec.max_width) for p in pieces]
        gap = max(4, height // 12)
        canvas = Image.new("RGB", (max(p.width for p in pieces), sum(p.height for p in pieces) + gap * (len(pieces) - 1)), "white")
        y = 0
        for p in pieces:
            canvas.paste(p, (0, y))
            y += p.height + gap
        return [canvas]
    raise ValueError(f"unknown view mode '{spec.mode}'")


def view_instruction(spec: ViewSpec) -> str:
    """What the images are, told to the model in the user message."""
    if spec.mode == "multi":
        return (
            f"Image 1 shows the whole crop. Images 2-{spec.parts + 1} show the same line cut from left to right "
            "into overlapping parts, enlarged. Transcribe the target line once, from its first to its last mark."
        )
    if spec.mode == "composite":
        return (
            "The image shows one line cut into overlapping parts, stacked from top (left end of the line) "
            "to bottom (right end). Transcribe the line once, from its first to its last mark."
        )
    return "Transcribe the target line of this image."


def cache_dir(cfg: Cfg, spec: ViewSpec) -> Path:
    return work_path(cfg, "views", spec.name)


def get_views(cfg: Cfg, spec: ViewSpec, image_id: str, path: str, family: str) -> list[Image.Image]:
    """Cached views of one line (rendered on demand when the cache is missing)."""
    d = cache_dir(cfg, spec)
    files = [d / f"{image_id}_{k}.png" for k in range(spec.n_images)]
    if all(f.exists() for f in files):
        return [load_rgb(str(f)) for f in files]
    views = make_views(path, family, spec)
    for f, v in zip(files, views):
        v.save(f)
    return views


def used_view_specs(cfg: Cfg) -> list[str]:
    names = set()
    a = cfg.get("approach_a", {})
    if a.get("enabled"):
        names.add(a.views)
        if a.label_correction.enabled:
            names.add(a.label_correction.views)
    b = cfg.get("approach_b", {})
    if b.get("enabled"):
        for g in b.generators:
            names.add(g.get("views", "single"))
    return sorted(names)


def stage_views(cfg: Cfg, force: bool = False) -> None:
    from .dataset import load_lines

    lines = load_lines(cfg)
    for name in used_view_specs(cfg):
        spec = ViewSpec.from_cfg(cfg, name)
        d = cache_dir(cfg, spec)
        n_new = 0
        for row in lines.itertuples():
            files = [d / f"{row.ID}_{k}.png" for k in range(spec.n_images)]
            if not force and all(f.exists() for f in files):
                continue
            for f, v in zip(files, make_views(row.image_path, row.family, spec)):
                v.save(f)
            n_new += 1
        log.info("views[%s]: %d rendered, cache %s", name, n_new, d)
