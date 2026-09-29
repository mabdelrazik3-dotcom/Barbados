"""Shared helpers for config management and polygon geometry alignment."""

import ast
import math
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from box import Box
from shapely import make_valid
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import explain_validity


def load_config(file_path):
    """Load a YAML configuration file into a Box for attribute access."""
    with open(file_path, "r") as file:
        return Box(yaml.safe_load(file))


def wkt_polygon_to_xy_list(wkt: str) -> List[Tuple[float, float]]:
    """
    Parse WKT 'POLYGON' or 'POLYGON Z' and return a list of (x, y) tuples for the outer ring.
    Returns [] for EMPTY/invalid inputs.
    """
    if not isinstance(wkt, str):
        return []
    s = wkt.strip()
    if not s or "EMPTY" in s.upper():
        return []

    # Find start of outer ring: the first '((' if present, otherwise first '('
    start = s.find("((")
    if start != -1:
        start += 2
        depth = 1  # we're inside the first '(' of the ring
    else:
        start = s.find("(")
        if start == -1:
            return []
        start += 1
        depth = 1

    # Scan to the closing ')' of the first ring (handles holes by stopping at depth==0)
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None or end <= start:
        return []

    inner = s[start:end]  # coords string for outer ring: "x y z, x y z, ..."

    # Build (x, y) list; if 'z' exists it is ignored
    out: List[Tuple[float, float]] = []
    for part in inner.split(","):
        token = part.strip()
        if not token:
            continue
        nums = token.split()
        if len(nums) < 2:
            continue
        try:
            x = float(nums[0])
            y = float(nums[1])
            out.append((x, y))
        except ValueError:
            continue
    return out


def parse_geom(v):
    """Standardize stored geometry values (strings, lists) into python objects."""
    if pd.isna(v):
        return []
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return []
        return ast.literal_eval(v)  # safe parse of "[(x, y), ...]"
    if isinstance(v, (list, tuple)):
        return list(v)
    return v  # already parsed or another object you want to keep


def flip_poly_y(img_path, points):
    """Flip polygon Y coordinates to match image coordinate origin."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if not points or len(points) < 1:
        return []
    return [(float(x), float(h - 1 - y)) for x, y in points]


XY = List[Tuple[float, float]]


def _clean_xy(xy: XY) -> XY:
    """Filter out None/invalid points and collapse duplicate vertices."""
    pts = []
    for t in xy:
        if t is None or len(t) < 2:
            continue
        x, y = float(t[0]), float(t[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not pts or (x, y) != pts[-1]:
            pts.append((x, y))
    return pts


def diagnose_polygon(xy: XY) -> Dict[str, Any]:
    """Summarize validity/area metrics for a coordinate sequence."""
    info = {
        "ok": True,
        "reason": None,
        "n_pts": None,
        "n_unique": None,
        "area": None,
        "validity": None,
    }
    if xy is None:
        info.update(ok=False, reason="xy is None")
        return info
    pts = _clean_xy(xy)
    info["n_pts"] = len(xy) if hasattr(xy, "__len__") else None
    info["n_unique"] = len(pts)
    if len(pts) < 3:
        info.update(ok=False, reason="too few unique points (<3)")
        return info
    p = Polygon(pts)
    info["area"] = float(p.area)
    if info["area"] <= 0:
        info.update(ok=False, reason="zero/near-zero area (collinear or collapsed)")
        return info
    if not p.is_valid:
        info["validity"] = explain_validity(p)
        info.update(ok=False, reason=f"invalid geometry: {info['validity']}")
        return info
    return info


def _pick_largest_polygon(g):
    """Return the polygon with greatest area from shapely compound geometries."""
    if isinstance(g, Polygon):
        return g
    if isinstance(g, MultiPolygon):
        return max(
            (geom for geom in g.geoms if isinstance(geom, Polygon)),
            key=lambda q: q.area,
            default=None,
        )
    if isinstance(g, GeometryCollection):
        polys = [geom for geom in g.geoms if isinstance(geom, Polygon)]
        if polys:
            return max(polys, key=lambda q: q.area)
    return None


def _to_poly_safe(
    xy: XY, *, allow_convex_hull: bool = True
) -> Tuple[Optional[Polygon], Dict[str, Any]]:
    """Convert XY lists into valid polygons, repairing degeneracies when possible."""
    dbg = {
        "cleaned": True,
        "made_valid": False,
        "buffer0": False,
        "used_convex_hull": False,
        "validity_before": None,
        "validity_after": None,
    }
    pts = _clean_xy(xy)
    if len(pts) < 3:
        return None, {**dbg, "error": "too few unique points"}
    p = Polygon(pts)
    dbg["validity_before"] = None if p.is_valid else explain_validity(p)
    if not p.is_valid:
        p = make_valid(p)
        dbg["made_valid"] = True
        if not isinstance(p, (Polygon, MultiPolygon, GeometryCollection)):
            return None, {**dbg, "error": "make_valid returned non-polygonal"}
        p2 = _pick_largest_polygon(p)
        if p2 is None:
            return None, {**dbg, "error": "no polygonal component after make_valid"}
        p = p2
    if not p.is_valid:
        p = p.buffer(0)
        dbg["buffer0"] = True
        p2 = _pick_largest_polygon(p)
        if p2 is None:
            return None, {**dbg, "error": "buffer(0) failed to produce polygon"}
        p = p2
    if (p.area is None) or (p.area <= 0):
        if allow_convex_hull:
            ch = Polygon(pts).convex_hull
            if isinstance(ch, Polygon) and ch.area > 0:
                dbg["used_convex_hull"] = True
                p = ch
            else:
                return None, {**dbg, "error": "zero-area even after convex hull"}
        else:
            return None, {**dbg, "error": "zero-area polygon"}
    dbg["validity_after"] = None if p.is_valid else explain_validity(p)
    if not p.is_valid:
        return None, {**dbg, "error": f"still invalid: {dbg['validity_after']}"}
    return p, dbg


def _centroid_xy(p: Polygon) -> Tuple[float, float]:
    """Return centroid coordinates for convenience."""
    c = p.centroid
    return (c.x, c.y)


def _scale_about(p: Polygon, s: float, origin: Tuple[float, float]) -> Polygon:
    """Scale a polygon isotropically around an anchor point."""
    return shp_scale(p, xfact=s, yfact=s, origin=origin)


def _translate_to(
    p: Polygon, target: Tuple[float, float], source: Tuple[float, float]
) -> Polygon:
    """Translate polygon from source anchor to target anchor."""
    return shp_translate(p, xoff=target[0] - source[0], yoff=target[1] - source[1])


def _iou(a: Polygon, b: Polygon) -> float:
    """Compute shapely IoU while guarding against empty shapes."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter == 0.0:
        return 0.0
    uni = a.union(b).area
    return float(inter / uni) if uni > 0 else 0.0


def _boundary_chamfer_mse(a: Polygon, b: Polygon, n: int = 256) -> float:
    """Approximate symmetric Chamfer distance across sampled boundary points."""
    ra, rb = a.boundary, b.boundary
    if ra.length == 0 or rb.length == 0:
        return float("inf")
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    pts_a = [ra.interpolate(t * ra.length) for t in ts]
    pts_b = [rb.interpolate(t * rb.length) for t in ts]
    da = np.array([rb.distance(pt) ** 2 for pt in pts_a]).mean()
    db = np.array([ra.distance(pt) ** 2 for pt in pts_b]).mean()
    return float(0.5 * (da + db))


def _bbox_scale_guess(pixel: Polygon, geo: Polygon) -> float:
    """Estimate a good initial scale factor from bounding-box ratios."""
    minx1, miny1, maxx1, maxy1 = pixel.bounds
    minx2, miny2, maxx2, maxy2 = geo.bounds
    w1, h1 = maxx1 - minx1, maxy1 - miny1
    w2, h2 = maxx2 - minx2, maxy2 - miny2
    if w2 <= 0 or h2 <= 0:
        return 1.0
    sx = w1 / w2 if w2 > 0 else 1.0
    sy = h1 / h2 if h2 > 0 else 1.0
    vals = [v for v in [sx, sy] if np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else 1.0


def _normalize_shape(p: Polygon, mode: Literal["area", "bbox"] = "area") -> Polygon:
    """Normalize polygon size for shape-similarity comparisons."""
    cx, cy = _centroid_xy(p)
    p0 = shp_translate(p, xoff=-cx, yoff=-cy)
    if mode == "area":
        a = p0.area
        s = 1.0 / math.sqrt(a) if a > 0 else 1.0
    else:
        minx, miny, maxx, maxy = p0.bounds
        m = max(maxx - minx, maxy - miny)
        s = 1.0 / m if m > 0 else 1.0
    return shp_scale(p0, xfact=s, yfact=s, origin=(0.0, 0.0))


def align_geo_to_pixel(
    poly_pixel_xy: XY,
    poly_geo_xy: XY,
    metric: Literal["iou", "chamfer_mse"] = "iou",
    n_steps: int = 201,
    search_span: Tuple[float, float] = (0.25, 4.0),
    anchor: Literal["centroid", "first_vertex"] = "centroid",
    refine: bool = True,
    shape_norm: Literal["area", "bbox"] = "area",
    strict: bool = False,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """Align geo-space polygons onto pixel-space masks via scale/translate search."""
    if strict:
        pixel, dpx = _to_poly_safe(poly_pixel_xy)
        geo, dge = _to_poly_safe(poly_geo_xy)
        if pixel is None:
            raise ValueError(f"Invalid pixel polygon: {dpx}")
        if geo is None:
            raise ValueError(f"Invalid geo polygon: {dge}")
    else:
        pixel, dpx = _to_poly_safe(poly_pixel_xy)
        geo, dge = _to_poly_safe(poly_geo_xy)
        if (pixel is None) or (geo is None):
            return {
                "status": "error",
                "error": "invalid input polygon(s)",
                "debug": {"pixel_dbg": dpx, "geo_dbg": dge},
            }

    if anchor == "centroid":
        anchor_geo = _centroid_xy(geo)
        anchor_pix = _centroid_xy(pixel)
    else:
        anchor_geo = tuple(list(geo.exterior.coords)[0][:2])
        anchor_pix = tuple(list(pixel.exterior.coords)[0][:2])

    s0 = _bbox_scale_guess(pixel, geo)
    lo_m, hi_m = search_span
    scales = s0 * np.exp(np.linspace(np.log(lo_m), np.log(hi_m), n_steps))

    def objective(s: float) -> float:
        g_scaled = _scale_about(geo, s, origin=anchor_geo)
        g_mapped = _translate_to(
            g_scaled,
            target=anchor_pix,
            source=tuple(
                _centroid_xy(g_scaled) if anchor == "centroid" else anchor_geo
            ),
        )
        if metric == "iou":
            return _iou(pixel, g_mapped)
        else:
            return -_boundary_chamfer_mse(pixel, g_mapped)

    vals = np.array([objective(s) for s in scales])
    k = int(vals.argmax())
    s_best = float(scales[k])
    v_best = float(vals[k])

    if refine:
        left = max(scales[max(0, k - 1)], s_best / 1.25)
        right = min(scales[min(len(scales) - 1, k + 1)], s_best * 1.25)
        phi = (1 + 5**0.5) / 2
        a, b = left, right
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        for _ in range(40):
            if objective(c) < objective(d):
                a = c
                c = d
                d = a + (b - a) / phi
            else:
                b = d
                d = c
                c = b - (b - a) / phi
        s_ref = (a + b) / 2
        v_ref = objective(s_ref)
        if v_ref > v_best:
            s_best, v_best = float(s_ref), float(v_ref)

    g_scaled = _scale_about(geo, s_best, origin=anchor_geo)
    g_mapped = _translate_to(
        g_scaled,
        target=anchor_pix,
        source=tuple(_centroid_xy(g_scaled) if anchor == "centroid" else anchor_geo),
    )
    out_xy = [
        (float(x), float(y)) for x, y in np.asarray(g_mapped.exterior.coords)[:-1]
    ]
    shape_iou = _iou(
        _normalize_shape(g_mapped, shape_norm), _normalize_shape(geo, shape_norm)
    )

    result = {
        "status": "ok",
        "geo_in_pixel_xy": out_xy,
        "best_scale": s_best,
        "best_metric": ("IoU" if metric == "iou" else "negative_chamfer_mse"),
        "best_metric_value": (v_best if metric == "iou" else -v_best),
        "shape_iou_geo_pixel_vs_geo_orig": float(shape_iou),
    }
    if return_debug:
        result["debug"] = {"pixel_dbg": dpx, "geo_dbg": dge}
    return result


def flip_poly_y(img_path, points, H=None):
    """Flip y-coordinates either via cached height or by reading the image."""
    if not points:
        return []
    if H is None:
        img = cv2.imread(img_path)
        if img is None:
            return []
        H = img.shape[0]
    return [(float(x), float(H - 1 - y)) for x, y in points]


def map_geo_to_pixel_column(
    df: pd.DataFrame,
    pixel_col: str = "poly_e",
    geo_col: str = "poly_b",
    image_col: str = "image_path",
    flip_fn=None,
    align_kwargs: dict = None,
):
    """Vectorized helper to align geo polygons to pixel space for entire DataFrame."""
    if align_kwargs is None:
        align_kwargs = {}

    def _row_op(r):
        poly_pix = r.get(pixel_col, None)
        poly_geo = r.get(geo_col, None)
        img = r.get(image_col, None)
        if poly_pix is None or poly_geo is None or img is None or pd.isna(img):
            return np.nan
        try:
            poly_pix = flip_fn(img, poly_pix) if callable(flip_fn) else poly_pix
            res = align_geo_to_pixel(poly_pix, poly_geo, **align_kwargs)
        except Exception:
            return np.nan
        if isinstance(res, dict):
            if res.get("status") == "ok" and "geo_in_pixel_xy" in res:
                return res["geo_in_pixel_xy"]
            if "geo_in_pixel_xy" in res:
                return res["geo_in_pixel_xy"]
        return np.nan

    return df.apply(_row_op, axis=1)
