"""Segmentation helpers for bias embeddings, geometry metrics, and resizing."""

import ast
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon

from segmentation.train.train_bias import BiasEmbeds

IMG_SIZE = 2048


def reconstruct_exact_decimals(
    row_idx: int, maxV: int, fixed_int_tensor: torch.Tensor, scale_pow: int
) -> List[Tuple[str, str]]:
    """Rebuild exact decimal strings from fixed-point int64 buffers."""
    scale = Decimal(10) ** scale_pow
    arr = fixed_int_tensor[row_idx].view(maxV, 2).cpu().numpy()  # int64
    out: List[Tuple[str, str]] = []
    for x_i, y_i in arr:
        if x_i == 0 and y_i == 0:
            break
        x_str = str(Decimal(int(x_i)) / scale)
        y_str = str(Decimal(int(y_i)) / scale)
        out.append((x_str, y_str))
    return out


def load_bias_model(ckpt_path: Path) -> Dict[str, object]:
    """Load the serialized bias embedding checkpoint and metadata."""
    blob = torch.load(ckpt_path, map_location="cpu")
    meta = blob["meta"]

    maxV = int(meta["maxV"])
    n_ids = len(meta["lt_ids"])
    twoV = 2 * maxV
    name_vocab_size = len(meta["name_vocab"])
    bias_vocab_size = len(meta["bias_vocab"])

    coords_input = torch.zeros((n_ids, twoV), dtype=torch.float64)
    mask_input = torch.zeros((n_ids, maxV), dtype=torch.float64)
    bias_idx_input = torch.zeros(
        (n_ids,), dtype=torch.float64
    )  # vector; class will unsqueeze(1)
    name_dist_input = torch.zeros((n_ids, name_vocab_size), dtype=torch.float64)

    model = BiasEmbeds(
        coords_input, mask_input, bias_idx_input, name_dist_input, bias_vocab_size
    )
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()

    return {
        "model": model,
        "maxV": maxV,
        "lt_ids": meta["lt_ids"],
        "id2idx": meta["id2idx"],
        "idx2bias": meta["idx2bias"],
        "preferred_names_by_bias": meta["preferred_names_by_bias"],
        "coords_fixed_int": meta["coords_fixed_int"],  # int64 [N, 2*maxV]
        "coords_scale_pow": int(meta["coords_scale_pow"]),
    }


def predict_bias(lt_num: str, art: Dict[str, object]) -> Dict[str, object]:
    """Retrieve stored bias geometry and preferred surveyor names for a given LT number."""
    m: BiasEmbeds = art["model"]
    i = art["id2idx"][str(lt_num)]
    with torch.no_grad():
        coords, mask, b_idx, _ = m(torch.tensor([i], dtype=torch.long))
    coords = coords.view(art["maxV"], 2).cpu().numpy()  # float64
    _nonzero_mask = ~np.all(coords == 0, axis=1)
    coords = [tuple(map(float, xy)) for xy in coords[_nonzero_mask]]
    bias_label = art["idx2bias"][int(b_idx.item())]
    names = art["preferred_names_by_bias"].get(bias_label, [])
    exact = reconstruct_exact_decimals(
        row_idx=i,
        maxV=art["maxV"],
        fixed_int_tensor=art["coords_fixed_int"],
        scale_pow=art["coords_scale_pow"],
    )
    return {
        "LT Num": lt_num,
        "bias_mapper": bias_label,
        "bias_names": names,
        "bias_geom": coords,
    }


def polygon_to_mask(polygon, width=256, height=256):
    """
    Convert a polygon to a binary mask array, ensuring correct orientation.
    Parameters:
        polygon: shapely Polygon object
        width, height: dimensions of the output mask
    Returns:
        numpy array (binary mask)
    """
    if polygon is None or polygon.is_empty:
        return np.zeros((height, width), dtype=np.uint8)

    # Get bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Get polygon exterior coordinates
    coords = np.array(polygon.exterior.coords)

    # Transform coordinates to pixel space
    x_scale = width / (maxx - minx) if maxx != minx else 1
    y_scale = height / (maxy - miny) if maxy != miny else 1

    # Convert to pixel coordinates (flip y)
    pixel_coords = np.array(
        [[(x - minx) * x_scale, (maxy - y) * y_scale] for x, y in coords],
        dtype=np.int32,
    )

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill polygon using OpenCV
    cv2.fillPoly(mask, [pixel_coords], 1)
    return mask


def iou_from_polys(
    poly1_coords, poly2_coords, width=256, height=256
) -> Optional[float]:
    """Compute IoU between two polygons given coord lists."""
    if not poly1_coords or not poly2_coords:
        return None
    try:
        p1 = Polygon(poly1_coords)
        p2 = Polygon(poly2_coords)
        if not p1.is_valid or not p2.is_valid:
            return None
        m1 = polygon_to_mask(p1, width, height)
        m2 = polygon_to_mask(p2, width, height)
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return inter / union if union > 0 else 0.0
    except Exception:
        return None


def _to_xy_list(
    val: Union[str, list, tuple, None],
) -> Optional[List[Tuple[float, float]]]:
    """Coerce stored coordinate representations into numeric tuples."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        val = ast.literal_eval(val)
    try:
        pts = [(float(x), float(y)) for x, y in val]
    except Exception:
        return None
    return pts if len(pts) >= 3 else None


def _letterbox_params(h: int, w: int, img_size: int = IMG_SIZE):
    """Compute scale/padding used when resizing into a square canvas."""
    s = img_size / float(max(h, w))
    h1, w1 = int(round(h * s)), int(round(w * s))
    pad_top = (img_size - h1) // 2
    pad_left = (img_size - w1) // 2
    return s, pad_left, pad_top


def _vflip_net(pts, H: int = IMG_SIZE):
    """Flip coordinates vertically within the network canvas."""
    return [(float(x), float(H - 1 - y)) for x, y in pts]


def net_poly_to_orig_poly(pts_net, h0: int, w0: int, img_size: int = IMG_SIZE):
    """Project polygon points from network canvas back to original image coordinates."""
    s, pad_left, pad_top = _letterbox_params(h0, w0, img_size)
    inv = 1.0 / s if s != 0 else 1.0
    out = []
    for x, y in pts_net:
        xo = (x - pad_left) * inv
        yo = (y - pad_top) * inv
        xo = max(0.0, min(xo, w0 - 1))
        yo = max(0.0, min(yo, h0 - 1))
        out.append((float(xo), float(yo)))
    return out


def convert_to_orig_img_size(
    df: pd.DataFrame,
    image_col: str = "image_path",
    coord_col: str = "poly_b",
    img_size: int = 2048,
    apply_vflip: bool = True,
) -> pd.Series:
    """Row-wise helper converting network-space polygons into original resolution."""
    def _row_op(r):
        pts = _to_xy_list(r.get(coord_col))
        img_path = r.get(image_col)
        if pts is None or img_path is None or pd.isna(img_path):
            return np.nan
        if apply_vflip:
            pts = _vflip_net(pts, H=img_size)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return np.nan
        h0, w0 = img.shape[:2]
        return net_poly_to_orig_poly(pts, h0, w0, img_size)

    return df.apply(_row_op, axis=1)
