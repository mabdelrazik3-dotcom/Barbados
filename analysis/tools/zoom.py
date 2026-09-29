"""Magnify one region of a line image to settle a doubtful word or mark.

Usage:
    python3 zoom.py OUT_DIR ID X0 X1 [Y0 Y1]
X0 X1 (and optional Y0 Y1) are percentages of the image width (height).
The region is enlarged to fit 1568x700 px (up to 8x), shown twice:
in natural colour and contrast-stretched grey. Prints the output path.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, "images")


def zoom(out_dir, image_id, x0, x1, y0=0.0, y1=100.0):
    src = Image.open(os.path.join(IMG_DIR, image_id + ".jpg")).convert("RGB")
    w, h = src.size
    box = (int(w * x0 / 100), int(h * y0 / 100), max(int(w * x0 / 100) + 1, int(w * x1 / 100)),
           max(int(h * y0 / 100) + 1, int(h * y1 / 100)))
    crop = src.crop(box)
    cw, ch = crop.size
    scale = min(8.0, 1568 / cw, 340 / ch)
    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    colour = crop.resize(size, Image.LANCZOS)
    grey = ImageOps.autocontrast(ImageOps.grayscale(crop), cutoff=1).resize(size, Image.LANCZOS).convert("RGB")
    canvas = Image.new("RGB", (size[0], 2 * size[1] + 30), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((2, 1), f"{image_id} x {x0:g}-{x1:g}% y {y0:g}-{y1:g}% (x{scale:.1f})", fill=(0, 0, 160))
    canvas.paste(colour, (0, 14))
    canvas.paste(grey, (0, 16 + size[1] + 14 - 14))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{image_id}_{x0:g}_{x1:g}_{y0:g}_{y1:g}.png")
    canvas.save(path)
    return path


if __name__ == "__main__":
    a = sys.argv
    args = [float(v) for v in a[3:]]
    print(zoom(a[1], a[2], *args))
