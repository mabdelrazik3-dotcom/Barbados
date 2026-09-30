"""Render a reading view for one or more line images.

A view is one PNG (at most 1568 px wide) holding:
  1. the whole crop (overview) with a ruler every 10 % of the width, so the
     target line and its neighbours can be located;
  2. the line cut into overlapping segments, each magnified and
     contrast-stretched, so single strokes can be read.

Usage:
    python3 render_view.py OUT_DIR ID [ID ...]
Prints one line per ID: "<ID> <view path> <w>x<h> <n segments>".
"""
import os
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, "images")

MAX_W = 1568          # widest image the model sees without downscaling
SEG_H_A = 210         # target height of a magnified segment, small crops
SEG_H_B = 300         # target height of a magnified segment, large crops
OVERLAP = 0.10        # overlap between neighbouring segments (fraction)
MAX_SCALE = 4.0


def ruler(draw, x0, y, width, height):
    for k in range(11):
        x = x0 + round(k * (width - 1) / 10)
        draw.line([(x, y), (x, y + height)], fill=(200, 0, 0), width=1)
        draw.text((min(x + 2, x0 + width - 24), y), f"{k * 10}", fill=(200, 0, 0))


def enhance(im):
    return ImageOps.autocontrast(im, cutoff=1)


def render(image_id, out_dir):
    src = Image.open(os.path.join(IMG_DIR, image_id + ".jpg")).convert("RGB")
    w, h = src.size
    family_b = w > 2000 or h > 180

    # 1. overview
    ov_scale = min(1.0, MAX_W / w) if w > MAX_W else min(MAX_W / w, 2.0 if h < 80 else 1.0)
    ov = src.resize((max(1, round(w * ov_scale)), max(1, round(h * ov_scale))), Image.LANCZOS)

    # 2. segments
    seg_h = SEG_H_B if family_b else SEG_H_A
    scale = min(MAX_SCALE, seg_h / h)
    seg_w_src = min(w, int(MAX_W / scale))
    if seg_w_src >= w:
        bounds = [(0, w)]
    else:
        step = int(seg_w_src * (1 - OVERLAP))
        bounds, x = [], 0
        while True:
            x1 = min(w, x + seg_w_src)
            bounds.append((max(0, x1 - seg_w_src), x1))
            if x1 >= w:
                break
            x += step
    segs = []
    for x0, x1 in bounds:
        crop = src.crop((x0, 0, x1, h))
        crop = crop.resize((max(1, round((x1 - x0) * scale)), max(1, round(h * scale))), Image.LANCZOS)
        segs.append(((x0, x1), enhance(crop)))

    head = 14
    total_h = head + ov.height + 6 + sum(head + s.height + 4 for _, s in segs)
    total_w = max([ov.width] + [s.width for _, s in segs])
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((2, 1), f"{image_id}  {w}x{h}  overview (red ruler = % of width)", fill=(0, 0, 160))
    y = head
    canvas.paste(ov, (0, y))
    ruler(d, 0, y, ov.width, 8)
    y += ov.height + 6
    for i, ((x0, x1), s) in enumerate(segs):
        d.line([(0, y), (total_w, y)], fill=(0, 0, 160), width=1)
        d.text((2, y + 1), f"segment {i + 1}/{len(segs)}: {100 * x0 / w:.0f}%-{100 * x1 / w:.0f}% of width  (x{scale:.1f})",
               fill=(0, 0, 160))
        y += head
        canvas.paste(s, (0, y))
        y += s.height + 4

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, image_id + ".png")
    canvas.save(path)
    return path, w, h, len(segs)


if __name__ == "__main__":
    out = sys.argv[1]
    for iid in sys.argv[2:]:
        p, w, h, n = render(iid, out)
        print(iid, p, f"{w}x{h}", n)
