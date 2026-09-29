"""Pixel-only line features used to assign every image (train, test, unlisted) to a sub-class.
The function `line_features` is copied VERBATIM into test_pesudo.ipynb (the builder reads it from this file),
so the offline rules and the notebook compute identical numbers.
usage: python work/subclass_features.py   -> work/subclass_features.csv (all 6,159 images)
"""
import os
import sys

import numpy as np
from PIL import Image


# <<LINE_FEATURES_BEGIN>>
def line_features(path):
    """cheap, label-free description of one line image (numpy + PIL only)."""
    with Image.open(path) as im:
        W, H = im.size
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    g = rgb.mean(2)
    hist = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    w = np.cumsum(hist); s = np.cumsum(np.arange(256) * hist); tot, sall = w[-1], s[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (s * tot - sall * w) ** 2 / (w * (tot - w))
    t = int(np.nanargmax(np.where((w > 0) & (w < tot), between, -1)))
    ink = g <= t
    paper = ~ink
    paper_grey = float(g[paper].mean()) if paper.any() else float(g.mean())
    ink_grey = float(g[ink].mean()) if ink.any() else paper_grey
    tint = float(rgb[..., 0][paper].mean() - rgb[..., 2][paper].mean()) if paper.any() else 0.0
    rows = ink.mean(1)
    k = max(1, H // 40)
    rs = np.convolve(rows, np.ones(k) / k, mode="same")
    peak = int(rs.argmax()); thr = 0.35 * float(rs.max())
    lo = peak
    while lo > 0 and rs[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < H - 1 and rs[hi + 1] >= thr:
        hi += 1
    band = rs[lo:hi + 1].mean() + 1e-6
    e = max(1, int(round(0.18 * H)))
    edge_ratio = float((ink[:e].mean() + ink[H - e:].mean()) / 2 / band)
    d = np.diff(ink.astype(np.int8), axis=1)
    starts = int((d == 1).sum() + ink[:, 0].sum())
    stroke = float(ink.sum() / max(1, starts))
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean() if H > 2 and W > 2 else 0.0
    cols = ink.mean(0)
    xs = np.where(cols > 0.02)[0]
    return {"height": float(H), "width": float(W), "aspect": float(W / H), "paper": paper_grey,
            "contrast": paper_grey - ink_grey, "ink_frac": float(ink.mean()), "tint": tint,
            "band_frac": float((hi - lo + 1) / H), "band_center": float((lo + hi) / 2 / H), "edge_ratio": edge_ratio,
            "xh_px": float(hi - lo + 1), "stroke_rel": stroke / H, "sharp": float(lap) / max(paper_grey - ink_grey, 1.0),
            "ink_w_frac": float((xs[-1] - xs[0] + 1) / W) if len(xs) else 0.0}
# <<LINE_FEATURES_END>>


def _one(p):
    try:
        return {"ID": os.path.basename(p)[:-4], **line_features(p)}
    except Exception as ex:          # noqa: BLE001
        return {"ID": os.path.basename(p)[:-4], "error": repr(ex)}


if __name__ == "__main__":
    import glob
    from multiprocessing import Pool

    import pandas as pd
    paths = sorted(glob.glob("D:/HANAFY/ROAD/images/*.jpg"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = pool.map(_one, paths, chunksize=32)
    df = pd.DataFrame(rows)
    df.to_csv("D:/HANAFY/ROAD/work/subclass_features.csv", index=False)
    print(df.shape, "errors:", int(df.get("error", pd.Series(dtype=str)).notna().sum()))
    print(df.describe().T.round(3).to_string())
