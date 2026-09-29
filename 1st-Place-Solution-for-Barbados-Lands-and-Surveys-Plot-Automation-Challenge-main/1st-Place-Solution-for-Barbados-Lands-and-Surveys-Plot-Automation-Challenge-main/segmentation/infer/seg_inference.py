"""
Ensemble segmentation inference fusing model logits, bias priors, and OCR metadata.

Runs 8-way TTA (test-time augmentation) on trained Unet++ model, extracts polygons
from probability maps using Otsu thresholding and morphological ops, then selects
best geometry by comparing IoU with TTAs. Merges predictions with OCR
metadata to generate final submission.

Input:
    - outputs/seg_model_checkpoints/: trained Lightning checkpoint
    - outputs/bias_model_checkpoints/bias_model.pt: bias embeddings
    - data/sub_text_extraction.csv: OCR predictions from text pipeline
    - data/Test.csv: test set metadata

Output:
    - data/final_submission.csv: competition submission
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations import Compose
from matplotlib.patches import Polygon
from PIL import Image
from scipy.stats import entropy
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from segmentation.train.train_seg import PolyLit, val_transforms
from utils.base_utils import flip_poly_y, load_config, map_geo_to_pixel_column
from utils.seg_utils import (
    convert_to_orig_img_size,
    iou_from_polys,
    load_bias_model,
    predict_bias,
)

BASE_CONFIG = load_config("configs/base.yaml")
SEG_CONFIG = load_config("configs/segmentation.yaml")

ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
TEST_IMAGES_DIR = Path(BASE_CONFIG["test_images_dir"])
TEST_CSV_PATH = Path(BASE_CONFIG["test_csv_path"])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_MODEL_CKPT_PATH = (
    ROOT_DIR
    / "outputs"
    / "seg_model_checkpoints"
    / "model-epoch=78-val_loss=0.0151-val_iou=0.9894-train_loss=0.4716-train_iou=0.6694.ckpt"
)
BIAS_MODEL_CKPT_PATH = ROOT_DIR / "outputs" / "bias_model_checkpoints" / "bias_model.pt"
MIN_AREA_FRAC = 0.001
EPS_FRAC = 0.004
USE_TTA = False  # True = 4-way flip TTA


# Build evaluation dataframe: merge test images, metadata, OCR predictions
images_df = pd.Series(list(Path(TEST_IMAGES_DIR).glob("*"))).to_frame(name="image_path")
images_df["ID"] = images_df["image_path"].apply(lambda x: x.stem.split("_")[-1])
test_df = pd.read_csv(TEST_CSV_PATH)
test_df = test_df.merge(images_df, on="ID", how="left")
text_preds_df = pd.read_csv(DATA_DIR / "sub_text_extraction.csv")
test_df = test_df.merge(text_preds_df, on="ID", how="left")
test_df["Land Surveyor"] = " "


class InferPolyDataset(Dataset):
    """
    Inference dataset loading survey plan images with IDs and metadata.

    Returns transformed images with LT Num, surveyor name, and ID for inference.
    """

    def __init__(
        self,
        df,
        transforms: Compose = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        img = Image.open(image_path).convert("RGB")

        img_np = np.array(img)
        augmented = self.transforms(image=img_np)
        img_tensor = augmented["image"]

        out = {
            "image": img_tensor,
            "image_path": str(image_path),
            "ltnum": row["LT Num"],
            "surveyor": row["Land Surveyor"],
            "id": row["ID"],
        }
        return out


test_loader = DataLoader(
    InferPolyDataset(
        test_df,
        val_transforms,
    ),
    **SEG_CONFIG.data_loader.val,
)

# Load trained segmentation model and bias embeddings
seg_model = (
    PolyLit.load_from_checkpoint(SEG_MODEL_CKPT_PATH, map_location=DEVICE)
    .to(DEVICE)
    .eval()
)
bias_model = load_bias_model(BIAS_MODEL_CKPT_PATH)


def _safe_poly(p: Optional[Polygon]) -> Optional[Polygon]:
    """Return a valid polygon or None after buffering away degeneracies."""
    if p is None:
        return None
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty:
        return None
    return p


def _coords_to_poly(coords: List[Tuple[float, float]]) -> Optional[Polygon]:
    """Convert raw (x, y) sequences into a shapely polygon if possible."""
    if not coords or len(coords) < 3:
        return None
    try:
        return _safe_poly(Polygon(coords))
    except Exception:
        return None


def _union_polys(polys: List[Polygon]) -> Optional[Polygon]:
    """Merge multiple candidates into the dominant polygon component."""
    polys = [p for p in polys if p is not None]
    if not polys:
        return None
    u = unary_union(polys)
    if isinstance(u, Polygon):
        return _safe_poly(u)
    if isinstance(u, MultiPolygon):
        return _safe_poly(max(u.geoms, key=lambda g: g.area))
    return None


def coords_obj_to_polygon(obj):
    """Normalize various coordinate payloads (flat list, list of rings) to a polygon."""
    if not obj:
        return None
    if (
        isinstance(obj[0], (tuple, list))
        and len(obj[0]) == 2
        and not isinstance(obj[0][0], (tuple, list))
    ):
        return _coords_to_poly(obj)
    polys = [_coords_to_poly(poly) for poly in obj]
    polys = [p for p in polys if p is not None]
    return _union_polys(polys)


@torch.no_grad()
def predict_logits_with_tta(model, x: torch.Tensor) -> torch.Tensor:
    """Average logits across scale/flip variants when TTA is enabled."""
    if not USE_TTA:
        return model(x)

    H, W = x.shape[-2], x.shape[-1]
    scales = (0.75, 1.0, 1.25)
    acc = None
    n = 0

    def _resize(inp, size_hw):
        return F.interpolate(inp, size=size_hw, mode="bilinear", align_corners=False)

    for s in scales:
        if s == 1.0:
            xs = x
        else:
            hs, ws = max(1, int(round(H * s))), max(1, int(round(W * s)))
            xs = _resize(x, (hs, ws))

        li = model(xs)
        if s != 1.0:
            li = _resize(li, (H, W))
        acc = li if acc is None else acc + li
        n += 1

        li = model(torch.flip(xs, dims=[-1]))
        li = torch.flip(li, dims=[-1])
        if s != 1.0:
            li = _resize(li, (H, W))
        acc = acc + li
        n += 1

        li = model(torch.flip(xs, dims=[-2]))
        li = torch.flip(li, dims=[-2])
        if s != 1.0:
            li = _resize(li, (H, W))
        acc = acc + li
        n += 1

        li = model(torch.flip(xs, dims=[-1, -2]))
        li = torch.flip(li, dims=[-1, -2])
        if s != 1.0:
            li = _resize(li, (H, W))
        acc = acc + li
        n += 1

    return acc / float(n)


def otsu_binarize_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convert probability map to binary mask using Otsu thresholding."""
    p8 = np.clip((probs * 255.0).astype(np.uint8), 0, 255)
    p8 = cv2.GaussianBlur(p8, (3, 3), 0)
    _, th = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (th > 0).astype(np.uint8)


def adaptive_close(mask: np.ndarray) -> np.ndarray:
    """Close mask holes with kernel size adapted to predicted parcel area."""
    h, w = mask.shape
    frac = mask.sum() / float(h * w + 1e-6)
    k = 3 if frac < 0.02 else (5 if frac < 0.06 else 7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    return m


def biggest_contour(
    mask: np.ndarray, min_area_frac=MIN_AREA_FRAC
) -> Optional[np.ndarray]:
    """Extract largest contour above minimal area threshold."""
    h, w = mask.shape
    min_area = max(1, int(min_area_frac * h * w))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    return None if not cs else max(cs, key=cv2.contourArea)


def contour_to_polygon(c: np.ndarray, eps_frac=EPS_FRAC) -> Optional[Polygon]:
    """Simplify OpenCV contour to polygon via Douglas-Peucker approximation."""
    if c is None or len(c) < 3:
        return None
    perim = cv2.arcLength(c, True)
    eps = max(1.0, float(eps_frac * perim))
    approx = cv2.approxPolyDP(c, epsilon=eps, closed=True)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx.reshape(-1, 1, 2)]
    return _coords_to_poly(pts)


def _poly_to_exterior_coords(geom):
    """Extract exterior coordinate list from shapely Polygon/MultiPolygon."""
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return list(geom.exterior.coords)
    if isinstance(geom, MultiPolygon):
        part = max(geom.geoms, key=lambda g: g.area, default=None)
        return list(part.exterior.coords) if part is not None else None
    return None


def _resize(inp: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resize helper for TTA."""
    return F.interpolate(inp, size=size_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def _forward_single(
    model, x: torch.Tensor, scale: float, flip: str, out_hw: Tuple[int, int]
) -> torch.Tensor:
    """Run forward pass with specified scale and flip augmentation."""
    H, W = out_hw
    xs = (
        x
        if scale == 1.0
        else _resize(x, (max(1, int(round(H * scale))), max(1, int(round(W * scale)))))
    )
    if "h" in flip:
        xs = torch.flip(xs, dims=[-1])
    if "v" in flip:
        xs = torch.flip(xs, dims=[-2])
    li = model(xs)
    if "v" in flip:
        li = torch.flip(li, dims=[-2])
    if "h" in flip:
        li = torch.flip(li, dims=[-1])
    if scale != 1.0:
        li = _resize(li, (H, W))
    return li


@torch.no_grad()
def infer_polys_orig_ensemble_8tta(model, loader) -> pd.DataFrame:
    """Run 8-way TTA inference and attach bias geometry/name hints per sample."""
    TTA_SPECS = [
        (1.0, "h"),
        (1.0, "v"),
        (1.0, "hv"),
        (0.75, ""),
        (1.25, ""),
        (0.75, "h"),
        (1.25, "v"),
        (0.75, "hv"),
    ]
    TTA_COLS = [
        "poly_h",
        "poly_v",
        "poly_hv",
        "poly_s0.75",
        "poly_s1.25",
        "poly_s0.75_h",
        "poly_s1.25_v",
        "poly_s0.75_hv",
    ]

    rows = []
    for batch in tqdm(loader):
        x = batch["image"].to(DEVICE, non_blocking=True)
        paths = [str(p) for p in batch["image_path"]]
        ltnums = batch["ltnum"]
        surveyors = batch["surveyor"]
        ids = batch["id"]

        B, _, H, W = x.shape

        logits_orig = _forward_single(model, x, scale=1.0, flip="", out_hw=(H, W))

        per_tta_logits: Dict[str, torch.Tensor] = {}
        for (s, f), col in zip(TTA_SPECS, TTA_COLS):
            per_tta_logits[col] = _forward_single(
                model, x, scale=s, flip=f, out_hw=(H, W)
            )

        acc = None
        for col in TTA_COLS:
            li = per_tta_logits[col]
            acc = li if acc is None else acc + li
        logits_ens = acc / float(len(TTA_COLS))

        probs_orig = torch.sigmoid(logits_orig).detach().cpu().numpy()
        probs_ens = torch.sigmoid(logits_ens).detach().cpu().numpy()
        per_tta_probs = {
            col: torch.sigmoid(per_tta_logits[col]).detach().cpu().numpy()
            for col in TTA_COLS
        }

        for i in range(B):
            row = {
                "image_path": paths[i],
                "ltnum": str(ltnums[i]),
                "surveyor": str(surveyors[i]),
                "ID": str(ids[i]),
            }

            # --- main predictions ---
            m0 = otsu_binarize_from_probs(probs_orig[i, 0])
            m0 = adaptive_close(m0)
            c0 = biggest_contour(m0, min_area_frac=MIN_AREA_FRAC)
            p0 = contour_to_polygon(c0, eps_frac=EPS_FRAC)
            row["poly_o"] = p0

            me = otsu_binarize_from_probs(probs_ens[i, 0])
            me = adaptive_close(me)
            ce = biggest_contour(me, min_area_frac=MIN_AREA_FRAC)
            pe = contour_to_polygon(ce, eps_frac=EPS_FRAC)
            row["poly_e"] = pe

            for col in TTA_COLS:
                pm = per_tta_probs[col][i, 0]
                m = otsu_binarize_from_probs(pm)
                m = adaptive_close(m)
                c = biggest_contour(m, min_area_frac=MIN_AREA_FRAC)
                p = contour_to_polygon(c, eps_frac=EPS_FRAC)
                row[col] = p

            try:
                out = predict_bias(row["ltnum"], bias_model)
                row["poly_b"] = out.get("bias_geom")
                row["pref_sname"] = out.get("bias_names")
            except Exception as e:
                row["poly_b"] = None
                row["pref_sname"] = None

            rows.append(row)

    df = pd.DataFrame(rows)

    def _as_coords(x):
        return None if x is None else _poly_to_exterior_coords(x)

    df["poly_o"] = df["poly_o"].apply(_as_coords)
    df["poly_e"] = df["poly_e"].apply(_as_coords)
    for col in TTA_COLS:
        df[col] = df[col].apply(_as_coords)
    df["poly_b"] = map_geo_to_pixel_column(
        df,
        flip_fn=lambda pth, pts: flip_poly_y(pth, pts, H=2048),
        align_kwargs={"strict": False},
    )
    df["poly_b"] = convert_to_orig_img_size(
        df,
        image_col="image_path",
        coord_col="poly_b",
        img_size=2048,
        apply_vflip=False,
    )
    df["poly_e"] = convert_to_orig_img_size(
        df,
        image_col="image_path",
        coord_col="poly_e",
        img_size=2048,
        apply_vflip=False,
    )

    return df


# Run 8-way TTA inference, collect polygon predictions with bias hints
pred_df = infer_polys_orig_ensemble_8tta(seg_model, test_loader)

# Compute IoU between TTA predictions and bias geometry to select best candidate
POLY_COLS = ["poly_o", "poly_e", "poly_h", "poly_v"]
iou_results = []
for i, row in pred_df.iterrows():
    res = {"image_path": row["image_path"], "ltnum": row["ltnum"], "ID": row["ID"]}
    poly_b = row["poly_b"]
    for col in POLY_COLS:
        poly = row[col]
        poly = flip_poly_y(row["image_path"], poly, H=2048)
        res[f"iou_{col}"] = iou_from_polys(poly, poly_b)
    iou_results.append(res)
iou_df = pd.DataFrame(iou_results)

pred_df = pred_df.merge(iou_df, on=["image_path", "ltnum", "ID"], how="left")

geoms = {}
for i in range(pred_df.shape[0]):
    poly_e = pred_df.loc[i, "poly_e"]
    poly_b = pred_df.loc[i, "poly_b"]
    iou = pred_df.loc[i, "iou_poly_e"]
    id = pred_df.loc[i, "ID"]

    if iou is not None and iou > 0.9:
        geoms[id] = poly_b
    else:
        geoms[id] = flip_poly_y(pred_df.loc[i, "image_path"], poly_e)

geoms = pd.DataFrame(geoms.items(), columns=["ID", "geometry"])
text_preds_df = text_preds_df.drop(columns=["geometry"], errors="ignore")
final_sub = text_preds_df.merge(geoms, on="ID", how="left")
final_sub = final_sub.drop(columns=["Land Surveyor"])


if __name__ == "__main__":
    final_sub.to_csv(DATA_DIR / "final_submission.csv", index=False)
    print(final_sub.head())
