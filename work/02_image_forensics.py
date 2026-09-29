"""Full-dataset image forensics: one row per image for ALL 6,159 images (train + test + unlisted).

Also computes the exact Qwen processor geometry (notebook load_image ladder -> smart_resize) under the
Kaggle transformers-5.0.0 behaviour (MIN_PIXELS=200704, MAX_PIXELS=2e6 honoured) and under the
model-default behaviour (overrides ignored, as in transformers>=5.1x), plus perceptual hashes and a
32x320 grayscale thumbnail for similarity / clustering work.
"""
import math
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = "D:/HANAFY/ROAD"
IMG = f"{ROOT}/images"
OUT = f"{ROOT}/work"


def smart_resize(h, w, factor, min_pixels, max_pixels):
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return max(h_bar, factor), max(w_bar, factor)


def ladder_size(w, h):
    """barbados-2.ipynb load_image(): PIL thumbnail (downscale-only, aspect preserved)."""
    a = w / h
    if a > 10:
        box = (2048, 384)
    elif a > 5:
        box = (2048, 512)
    elif a > 3:
        box = (1600, 640)
    else:
        box = (1536, 768)
    if w <= box[0] and h <= box[1]:
        return w, h
    s = min(box[0] / w, box[1] / h)
    # PIL thumbnail rounds to keep aspect; replicate its rounding closely
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return min(nw, box[0]), min(nh, box[1])


def phash(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]
    med = np.median(d[1:].ravel()) if d.size else 0
    bits = (d > np.median(d)).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(gray, hs=8):
    g = cv2.resize(gray, (hs * 4 + 1, hs), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def jpeg_info(path):
    info = {}
    try:
        im = Image.open(path)
        info["mode"] = im.mode
        q = getattr(im, "quantization", None) or {}
        if q:
            t0 = np.array(q.get(0, [0]), np.float64)
            info["jpeg_q0_mean"] = float(t0.mean())
            info["jpeg_q0_dc"] = float(t0[0])
            info["jpeg_ntables"] = len(q)
            # libjpeg-style quality estimate from luma table
            std_luma = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                                 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                                 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float64)
            s = 100.0 * t0.sum() / std_luma.sum() if len(t0) == 64 else np.nan
            info["jpeg_quality_est"] = float((200 - s) / 2 if s <= 100 else 5000 / s) if s == s else np.nan
        info["progressive"] = int(bool(im.info.get("progressive") or im.info.get("progression")))
        dpi = im.info.get("dpi")
        info["dpi"] = float(dpi[0]) if dpi else np.nan
        try:
            from PIL.JpegImagePlugin import get_sampling
            info["subsampling"] = get_sampling(im)
        except Exception:
            info["subsampling"] = -9
        info["has_exif"] = int(bool(im.info.get("exif")))
        info["has_icc"] = int(bool(im.info.get("icc_profile")))
    except Exception as e:  # noqa
        info["read_error"] = str(e)[:80]
    return info


def shear_score(mask, shears):
    h, w = mask.shape
    best, bs = None, -1
    ys = np.arange(h)[:, None]
    for s in shears:
        M = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
        sh = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        col = sh.sum(axis=0).astype(np.float64)
        v = float((col ** 2).sum())
        if v > bs:
            bs, best = v, s
    return best


def skew_score(mask, angles):
    h, w = mask.shape
    best, bs = 0.0, -1
    for a in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        row = r.sum(axis=1).astype(np.float64)
        v = float((row ** 2).sum())
        if v > bs:
            bs, best = v, a
    return best


def runs(b):
    """lengths of consecutive True runs in 1-D bool array"""
    if b.size == 0:
        return np.array([], int)
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return e - s


def analyse(fn):
    iid = fn[:-4]
    path = os.path.join(IMG, fn)
    r = {"ID": iid, "file_bytes": os.path.getsize(path)}
    r.update(jpeg_info(path))
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        r["read_error"] = "cv2_none"
        return r, None
    h, w = bgr.shape[:2]
    r.update(width=w, height=h, aspect=w / h, pixels=w * h)
    rgb = bgr[..., ::-1].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    r["is_grayscale"] = int(np.abs(rgb[..., 0] - rgb[..., 2]).mean() < 1.0 and np.abs(rgb[..., 0] - rgb[..., 1]).mean() < 1.0)
    pct = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    r.update({f"g_p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct)})
    r["g_mean"], r["g_std"] = float(g.mean()), float(g.std())
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    pr = hist / hist.sum()
    r["entropy"] = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
    r["overexp_frac"] = float((gray >= 250).mean())
    r["underexp_frac"] = float((gray <= 5).mean())
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r["otsu"] = float(otsu)
    # adaptive (Sauvola-like) ink mask, robust to uneven parchment
    blk = max(15, (min(h, w) // 2) | 1)
    blk = min(blk, 51)
    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 10)
    ink_otsu = gray < otsu
    ink = (ada > 0) & (g < (otsu + 0.5 * (pct[5] if False else 0) + 10))
    ink_u8 = ink.astype(np.uint8)
    # remove speckle
    ink_u8 = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink = ink_u8 > 0
    bg = ~ink_otsu
    r["ink_frac"] = float(ink.mean())
    r["ink_frac_otsu"] = float(ink_otsu.mean())
    ink_vals = g[ink] if ink.any() else g.ravel()
    bg_vals = g[bg] if bg.any() else g.ravel()
    r["ink_mean"] = float(ink_vals.mean())
    r["ink_p10"] = float(np.percentile(ink_vals, 10))
    r["bg_mean"] = float(bg_vals.mean())
    r["bg_std"] = float(bg_vals.std())
    r["contrast"] = r["bg_mean"] - r["ink_mean"]
    r["fisher_sep"] = float((bg_vals.mean() - ink_vals.mean()) / (bg_vals.std() + ink_vals.std() + 1e-6))
    for c, name in enumerate("RGB"):
        ch = rgb[..., c]
        r[f"bg_{name}"] = float(ch[bg].mean()) if bg.any() else float(ch.mean())
        r[f"ink_{name}"] = float(ch[ink].mean()) if ink.any() else float(ch.mean())
    r["bg_tint_RB"] = r["bg_R"] - r["bg_B"]
    r["bg_sat"] = float(hsv[..., 1][bg].mean()) if bg.any() else float(hsv[..., 1].mean())
    r["bg_hue"] = float(np.median(hsv[..., 0][bg])) if bg.any() else float(np.median(hsv[..., 0]))
    r["ink_sat"] = float(hsv[..., 1][ink].mean()) if ink.any() else np.nan
    # faint mid-tones between ink and paper: bleed-through / faded ink proxy
    lo, hi = r["ink_mean"], r["bg_mean"] - 2 * r["bg_std"]
    r["midtone_frac"] = float(((g > lo + 0.5 * (hi - lo)) & (g < hi)).mean()) if hi > lo else 0.0
    # illumination variation: heavy blur of background
    bgimg = g.copy()
    if ink.any():
        bgimg[ink] = r["bg_mean"]
    k = int(max(3, (min(h, w) // 2) | 1))
    big = cv2.GaussianBlur(bgimg, (0, 0), sigmaX=max(3.0, w / 20), sigmaY=max(3.0, h / 4))
    r["illum_cv"] = float(big.std() / (big.mean() + 1e-6))
    r["illum_lr_diff"] = float(big[:, : max(1, w // 10)].mean() - big[:, -max(1, w // 10):].mean())
    # sharpness / blur
    lap = cv2.Laplacian(g, cv2.CV_32F)
    r["lap_var"] = float(lap.var())
    r["lap_var_norm"] = float(lap.var() / (r["contrast"] ** 2 + 1e-6))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    r["tenengrad"] = float((gm ** 2).mean())
    # edge sharpness: gradient magnitude at edges relative to contrast
    edges = cv2.Canny(gray, 50, 150)
    r["edge_density"] = float((edges > 0).mean())
    r["edge_grad_rel"] = float(gm[edges > 0].mean() / (r["contrast"] + 1e-6)) if (edges > 0).any() else 0.0
    # noise: MAD of Laplacian over background pixels (Immerkaer-like)
    bgl = lap[bg] if bg.any() else lap.ravel()
    r["noise_sigma"] = float(1.4826 * np.median(np.abs(bgl - np.median(bgl))) / 6.0 ** 0.5)
    # JPEG blockiness: discontinuity at 8-px boundaries vs inside
    if w > 32 and h > 16:
        dcol = np.abs(np.diff(g, axis=1))
        b_idx = np.arange(7, dcol.shape[1], 8)
        nb_idx = np.setdiff1d(np.arange(dcol.shape[1]), b_idx)
        r["blockiness"] = float(dcol[:, b_idx].mean() / (dcol[:, nb_idx].mean() + 1e-6))
    # ----- geometry of ink -----
    colp = ink.sum(axis=0).astype(np.float64)
    rowp = ink.sum(axis=1).astype(np.float64)
    tot = ink.sum()
    if tot > 20:
        cc = np.cumsum(colp) / tot
        x0 = int(np.searchsorted(cc, 0.005)); x1 = int(np.searchsorted(cc, 0.995))
        rc = np.cumsum(rowp) / tot
        y0 = int(np.searchsorted(rc, 0.01)); y1 = int(np.searchsorted(rc, 0.99))
    else:
        x0, x1, y0, y1 = 0, w - 1, 0, h - 1
    r.update(ink_x0=x0, ink_x1=x1, ink_y0=y0, ink_y1=y1,
             ink_w=x1 - x0 + 1, ink_h=y1 - y0 + 1,
             margin_l=x0 / w, margin_r=(w - 1 - x1) / w, margin_t=y0 / h, margin_b=(h - 1 - y1) / h,
             ink_w_frac=(x1 - x0 + 1) / w, ink_h_frac=(y1 - y0 + 1) / h)
    # text core band (x-height region): rows with density > 30% of max smoothed row density
    rs = np.convolve(rowp, np.ones(3) / 3, mode="same")
    if rs.max() > 0:
        core = rs > 0.3 * rs.max()
        cr = runs(core)
        r["core_h"] = int(cr.max()) if cr.size else 0
        r["core_center"] = float(np.average(np.arange(h), weights=rs) / h)
        # line count: peaks in heavily smoothed row profile
        ks = max(3, int(h / 12) | 1)
        rs2 = cv2.GaussianBlur(rowp.reshape(-1, 1), (1, ks), 0).ravel()
        thr = 0.25 * rs2.max()
        pk = [i for i in range(1, h - 1) if rs2[i] >= rs2[i - 1] and rs2[i] > rs2[i + 1] and rs2[i] > thr]
        # merge peaks closer than core height
        merged = []
        for p in pk:
            if not merged or p - merged[-1] > max(8, r["core_h"]):
                merged.append(p)
        r["n_row_peaks"] = len(merged)
    else:
        r["core_h"], r["core_center"], r["n_row_peaks"] = 0, 0.5, 0
    # vertical centroid offset (is the line centered?)
    # connected components
    n, lab, stats, cent = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    big_cc = areas >= 8
    r["cc_count"] = int(big_cc.sum())
    r["cc_area_med"] = float(np.median(areas[big_cc])) if big_cc.any() else 0.0
    r["cc_area_p90"] = float(np.percentile(areas[big_cc], 90)) if big_cc.any() else 0.0
    if n > 1 and big_cc.any():
        hts = stats[1:, cv2.CC_STAT_HEIGHT][big_cc]
        r["cc_height_med"] = float(np.median(hts))
        # components touching image boundary (cropped glyphs / neighbour lines)
        s = stats[1:][big_cc]
        r["cc_touch_top"] = int((s[:, cv2.CC_STAT_TOP] == 0).sum())
        r["cc_touch_bottom"] = int((s[:, cv2.CC_STAT_TOP] + s[:, cv2.CC_STAT_HEIGHT] >= h).sum())
        r["cc_touch_left"] = int((s[:, cv2.CC_STAT_LEFT] == 0).sum())
        r["cc_touch_right"] = int((s[:, cv2.CC_STAT_LEFT] + s[:, cv2.CC_STAT_WIDTH] >= w).sum())
    r["ink_touch_top"] = float(ink[:2].mean()); r["ink_touch_bottom"] = float(ink[-2:].mean())
    r["ink_touch_left"] = float(ink[:, :2].mean()); r["ink_touch_right"] = float(ink[:, -2:].mean())
    # stroke width via distance transform on ink
    if ink.any():
        dt = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 3)
        # local maxima approx: ridge pixels where dt >= neighbours
        dil = cv2.dilate(dt, np.ones((3, 3), np.uint8))
        ridge = (dt > 0) & (dt >= dil - 1e-6)
        sw = 2 * dt[ridge]
        r["stroke_w"] = float(np.median(sw)); r["stroke_w_std"] = float(sw.std())
        r["stroke_w_rel"] = r["stroke_w"] / max(1.0, r["core_h"])
        # broken-stroke proxy: fraction of tiny components among all
        r["tiny_cc_frac"] = float((areas < 8).mean()) if areas.size else 0.0
    # word gaps within ink bbox, using core band
    band = ink[max(0, y0):y1 + 1, x0:x1 + 1]
    cp = band.sum(axis=0) > 0
    gaps = runs(~cp)
    gthr = max(4.0, 0.35 * max(1, r["core_h"]))
    r["n_gaps"] = int((gaps >= gthr).sum())
    r["n_gaps_wide"] = int((gaps >= 2 * gthr).sum())
    r["gap_mean"] = float(gaps[gaps >= gthr].mean()) if (gaps >= gthr).any() else 0.0
    # slant & skew on a normalised mask (height 64)
    if tot > 50:
        sc = 64.0 / h
        m = cv2.resize(ink_u8 * 255, (max(8, int(w * sc)), 64), interpolation=cv2.INTER_AREA)
        m = (m > 64).astype(np.float32)
        r["slant_shear"] = float(shear_score(m, np.linspace(-0.8, 0.8, 33)))
        r["skew_deg"] = float(skew_score(m, np.linspace(-6, 6, 25)))
        # baseline slope: robust fit of per-column lowest ink pixel in core region
        cols = np.where(m.sum(axis=0) > 0)[0]
        if cols.size > 10:
            low = np.array([np.where(m[:, c] > 0)[0].max() for c in cols], np.float64)
            med = np.median(low)
            keep = np.abs(low - med) < 12
            if keep.sum() > 10:
                A = np.vstack([cols[keep], np.ones(keep.sum())]).T
                slope = np.linalg.lstsq(A, low[keep], rcond=None)[0][0]
                r["baseline_slope_deg"] = float(np.degrees(np.arctan(slope)))
    # ----- processor geometry (notebook) -----
    lw, lh = ladder_size(w, h)
    r["ladder_w"], r["ladder_h"] = lw, lh
    for tag, fac, mn, mx in [("q25_kaggle", 28, 256 * 28 * 28, 2_000_000), ("q3_kaggle", 32, 256 * 28 * 28, 2_000_000),
                             ("q25_default", 28, 3136, 12845056), ("q3_default", 32, 65536, 16777216)]:
        rh, rw = smart_resize(lh, lw, fac, mn, mx)
        r[f"{tag}_h"], r[f"{tag}_w"] = rh, rw
        r[f"{tag}_tokens"] = (rh // (fac // 2)) * (rw // (fac // 2)) // 4
        r[f"{tag}_scale"] = rh / h
        r[f"{tag}_token_rows"] = rh // fac
    # hashes + thumbnail
    r["phash"] = format(phash(gray), "016x")
    r["dhash"] = format(dhash(gray), "064x")
    th = cv2.resize(gray, (320, 32), interpolation=cv2.INTER_AREA)
    return r, th


def main():
    files = sorted(f for f in os.listdir(IMG) if f.endswith(".jpg"))
    print("images:", len(files), flush=True)
    with Pool(15) as pool:
        res = pool.map(analyse, files, chunksize=20)
    rows = [a for a, _ in res]
    thumbs = np.stack([t if t is not None else np.zeros((32, 320), np.uint8) for _, t in res])
    df = pd.DataFrame(rows)
    tr = pd.read_csv(f"{ROOT}/Train.csv"); te = pd.read_csv(f"{ROOT}/Test.csv")
    df["split"] = np.where(df.ID.isin(tr.ID), "train", np.where(df.ID.isin(te.ID), "test", "unlisted"))
    df.to_parquet(f"{OUT}/image_features.parquet") if False else df.to_csv(f"{OUT}/image_features.csv", index=False)
    np.save(f"{OUT}/thumbs_32x320.npy", thumbs)
    np.save(f"{OUT}/thumb_ids.npy", df.ID.values)
    print(df.shape, df.split.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
