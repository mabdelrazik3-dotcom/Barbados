#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build bias embeddings to mimic different labelers' preferred names and geometry styles:
- LT Num -> (bias index, preferred-name distribution, geometry)
- Robust geometry handling (Polygon/MultiPolygon, 2D/3D, nested rings, flat lists)
- Float64 embeddings + fixed-point int64 copy for exact decimal strings
- Saves a single Torch checkpoint to outputs/bias_model_checkpoints/bias_model.pt
"""

import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.base_utils import load_config, wkt_polygon_to_xy_list

getcontext().prec = 45  # high precision for decimal parsing

# ---------- Utility: name lists by frequency ----------
from typing import List as _List


def build_name_lists_by_freq(
    df: pd.DataFrame, bias_col: str, name_cols: _List[str], min_count: int = 2
) -> pd.DataFrame:
    """Aggregate per-bias preferred names ordered by frequency and alpha fallback."""
    if bias_col not in df.columns:
        raise ValueError(f"{bias_col} not in dataframe")
    for col in name_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not in dataframe")

    result = pd.DataFrame({bias_col: pd.unique(df[bias_col])})
    for col in name_cols:
        tmp = df[[bias_col, col]].dropna(subset=[col]).copy()
        counts = tmp.groupby([bias_col, col]).size().rename("count").reset_index()
        counts = counts[counts["count"] >= min_count]
        if counts.empty:
            result[f"{col}_names_freq"] = [[] for _ in range(len(result))]
            continue
        counts = counts.sort_values(
            [bias_col, "count", col], ascending=[True, False, True]
        )
        lists = (
            counts.groupby(bias_col)[col]
            .apply(list)
            .reset_index(name=f"preferred_{col}_name")
        )
        result = result.merge(lists, on=bias_col, how="left")

    for col in name_cols:
        cand = f"preferred_{col}_name"
        out_col = f"preferred_{col}"
        if cand in result.columns:
            result.rename(columns={cand: out_col}, inplace=True)
        if out_col in result.columns:
            result[out_col] = result[out_col].apply(
                lambda x: x if isinstance(x, list) else []
            )
    return result.sort_values(bias_col).reset_index(drop=True)


def attach_name_lists(
    df: pd.DataFrame, mapping: pd.DataFrame, bias_col: str
) -> pd.DataFrame:
    """Join preferred-name lists back onto the working dataframe for downstream use."""
    if bias_col not in df.columns or bias_col not in mapping.columns:
        raise ValueError(f"{bias_col} must exist in both dataframes")
    return df.merge(mapping, on=bias_col, how="left")


def _canon(n: str) -> str:
    """Normalize name: lowercase, remove periods, collapse spaces."""
    n = str(n).strip().replace(".", " ")
    n = " ".join(n.split())
    return n.lower()


def build_preferred_names_with_variants(
    df: pd.DataFrame, bias_col: str, name_col: str
) -> Dict[str, List[str]]:
    """Group aliases by canonical form while preserving variant orderings per bias."""
    out: Dict[str, List[str]] = {}
    for b, g in df[[bias_col, name_col]].dropna().groupby(bias_col):
        by_canon: Dict[str, int] = {}
        var_counts: Dict[str, int] = {}
        var2canon: Dict[str, str] = {}
        for n in g[name_col].astype(str):
            c = _canon(n)
            by_canon[c] = by_canon.get(c, 0) + 1
            var_counts[n] = var_counts.get(n, 0) + 1
            var2canon[n] = c
        canons = sorted(by_canon.items(), key=lambda kv: (-kv[1], kv[0]))
        pref: List[str] = []
        for c, _ in canons:
            variants = [v for v, cc in var2canon.items() if cc == c]
            variants_sorted = sorted(variants, key=lambda v: (-var_counts[v], v))
            pref.extend(variants_sorted)
        out[str(b)] = pref
    return out


# Accepts shapely Polygon/MultiPolygon OR list-like coords (2D/3D, nested rings, flat lists)
CoordLike = Union[Tuple[float, float], Tuple[float, float, float], List, Tuple]


def _extract_coords_from_shapely(geom) -> List[List[Tuple[str, str]]]:
    """Return list of rings, each ring is a list of (x_str,y_str)."""
    rings: List[List[Tuple[str, str]]] = []
    if isinstance(geom, Polygon):
        pts = list(geom.exterior.coords)  # includes closing vertex
        rings.append([(str(x), str(y)) for x, y, *rest in pts])
    elif isinstance(geom, MultiPolygon):
        # take all exteriors; choose largest ring later
        for p in geom.geoms:
            pts = list(p.exterior.coords)
            rings.append([(str(x), str(y)) for x, y, *rest in pts])
    else:
        raise ValueError("Unsupported shapely geometry type")
    return rings


def _as_xy_pairs_from_list(obj: CoordLike) -> List[List[Tuple[str, str]]]:
    """
    Normalize list-like geometry into list-of-rings.
    - [(x,y), (x,y,z), ...] -> one ring
    - [[(x,y),...], [(x,y),...]] -> multiple rings
    - [x1, y1, x2, y2, ...] -> one ring (flat)
    Returns: List[rings], ring = List[(x_str, y_str)]
    """

    def pairify(vals: List[Union[str, int, float, Decimal]]) -> List[Tuple[str, str]]:
        if len(vals) % 2 != 0:
            raise ValueError("Flat coordinate list must have even length")
        out = []
        for i in range(0, len(vals), 2):
            out.append((str(vals[i]), str(vals[i + 1])))
        return out

    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        raise ValueError("Geometry list must be a non-empty list/tuple")

    # Case: ring list: [[(x,y),...], ...]
    if isinstance(obj[0], (list, tuple)):
        # If it's flat numeric: [x1,y1,...]
        if all(isinstance(v, (int, float, str, Decimal)) for v in obj):
            return [pairify([*map(lambda v: v, obj)])]
        # Otherwise nested points or nested rings
        rings: List[List[Tuple[str, str]]] = []
        # Detect if this is a list of points: [(x,y), (x,y,z)]
        if (
            len(obj) > 0
            and isinstance(obj[0], (list, tuple))
            and len(obj[0]) >= 2
            and isinstance(obj[0][0], (int, float, str, Decimal))
        ):
            rings.append([(str(p[0]), str(p[1])) for p in obj])  # one ring
            return rings
        # Otherwise assume list of rings
        for ring in obj:
            if not isinstance(ring, (list, tuple)):
                continue
            if len(ring) == 0:
                continue
            # ring may be list of points or flat numbers
            if all(isinstance(v, (int, float, str, Decimal)) for v in ring):
                rings.append(pairify([*ring]))
            else:
                rings.append(
                    [
                        (str(p[0]), str(p[1]))
                        for p in ring
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                    ]
                )
        return rings

    # Case: flat numeric list [x1, y1, ...] but obj[0] isn't list/tuple (already handled above)
    if all(isinstance(v, (int, float, str, Decimal)) for v in obj):
        return [pairify([*obj])]

    raise ValueError("Unrecognized geometry list format")


def _pick_largest_ring(rings: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    """Pick the largest ring by vertex count (proxy for area when geometry isn't shapely anymore)."""
    if not rings:
        raise ValueError("No rings found")
    return max(rings, key=lambda r: len(r))


def normalize_geometry_cell(cell) -> List[Tuple[str, str]]:
    """
    Normalize a geometry cell to a single sequence of (x_str, y_str) pairs (exterior-like).
    Handles shapely geometries and list-like structures.
    """
    # Fast path for shapely geometry
    if isinstance(cell, (Polygon, MultiPolygon)):
        rings = _extract_coords_from_shapely(cell)
        return _pick_largest_ring(rings)

    # If upstream converted geometry to python lists already (like your earlier pipeline)
    if isinstance(cell, (list, tuple)):
        rings = _as_xy_pairs_from_list(cell)
        return _pick_largest_ring(rings)

    # If None or unsupported
    raise ValueError(
        "Unsupported geometry cell (expect shapely geometry or list-like coords)"
    )


def _decimals_in_str_num(s: str) -> int:
    """Count decimal places in string representation of number."""
    s = s.strip()
    if "e" in s.lower():
        return max(0, -Decimal(s).as_tuple().exponent)
    if "." in s:
        return len(s.split(".")[1].rstrip("0"))
    return 0


class BiasEmbeds(nn.Module):
    def __init__(self, coords, mask, bias_idx_vec, name_dist, bias_vocab_size: int):
        super().__init__()
        # Store as frozen embeddings (dtype preserved)
        self.coords = nn.Embedding.from_pretrained(coords, freeze=True)  # float64
        self.mask = nn.Embedding.from_pretrained(mask, freeze=True)  # float64
        self.bias_idx = nn.Embedding.from_pretrained(
            bias_idx_vec.double().unsqueeze(1), freeze=True
        )  # [N,1] float64
        self.name_dist = nn.Embedding.from_pretrained(name_dist, freeze=True)  # float64
        self.bias_vocab_size = bias_vocab_size

    def forward(self, ids: torch.Tensor):
        coords = self.coords(ids)  # [B, 2*maxV], float64
        mask = self.mask(ids)  # [B, maxV],  float64
        b_idx = self.bias_idx(ids).squeeze(-1).round().long()  # [B]
        names = self.name_dist(ids)  # [B, |names|], float64
        return coords, mask, b_idx, names


def main():
    """Build and save bias embeddings checkpoint."""
    # --------------------------
    # Config / paths
    # --------------------------
    BASE_CONFIG = load_config("configs/base.yaml")
    ROOT_DIR = Path(BASE_CONFIG["root_dir"])
    DATA_DIR = ROOT_DIR / "data"
    CKPT_DIR = ROOT_DIR / "outputs" / "bias_model_checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH = CKPT_DIR / "bias_model.pt"

    # --------------------------
    # Load data
    train = pd.read_csv(DATA_DIR / "Train.csv")
    train["geometry"] = train["geometry"].map(wkt_polygon_to_xy_list)
    train.insert(
        1, "bias_mapper", train["LT Num"].apply(lambda x: "".join(x.split(".")[:2]))
    )
    bias_df = train.drop(columns=["ID"])

    name_map = build_name_lists_by_freq(
        df=bias_df, bias_col="bias_mapper", name_cols=["Land Surveyor"], min_count=2
    )
    bias_df = attach_name_lists(bias_df, name_map, bias_col="bias_mapper")

    # Convert all geometries to normalized (x_str,y_str) pairs, drop invalid rows
    geom_strs: List[List[Tuple[str, str]]] = []
    drop_idx: List[int] = []
    for i, g in enumerate(bias_df["geometry"].tolist()):
        try:
            geom_strs.append(normalize_geometry_cell(g))
        except Exception:
            drop_idx.append(i)

    if drop_idx:
        bias_df = bias_df.drop(index=bias_df.index[drop_idx]).reset_index(drop=True)
        geom_strs = [normalize_geometry_cell(g) for g in bias_df["geometry"].tolist()]

    max_decimals = 0
    for pts in geom_strs:
        for xs, ys in pts:
            max_decimals = max(
                max_decimals, _decimals_in_str_num(xs), _decimals_in_str_num(ys)
            )

    # Determine fixed-point scaling factor for int64 representation
    scale_pow = int(max_decimals)
    scale_dec = Decimal(10) ** scale_pow

    # Build both float64 and scaled int64 representations
    geoms_f64: List[np.ndarray] = []
    geoms_i64: List[np.ndarray] = []
    for pts in geom_strs:
        fpts = []
        ipts = []
        for xs, ys in pts:
            dx, dy = Decimal(xs), Decimal(ys)
            fpts.append((float(dx), float(dy)))
            ipts.append(
                (
                    int((dx * scale_dec).to_integral_value()),
                    int((dy * scale_dec).to_integral_value()),
                )
            )
        geoms_f64.append(np.asarray(fpts, dtype=np.float64))
        geoms_i64.append(np.asarray(ipts, dtype=np.int64))

    V_list = [p.shape[0] for p in geoms_f64]
    maxV = int(max(V_list))
    N = len(geoms_f64)

    # Pad all geometries to maxV vertices, create masks for valid vertices
    padded_f64 = np.zeros((N, maxV, 2), dtype=np.float64)
    padded_i64 = np.zeros((N, maxV, 2), dtype=np.int64)
    mask_np = np.zeros((N, maxV), dtype=np.float64)

    for i, (P, Pi) in enumerate(zip(geoms_f64, geoms_i64)):
        V = P.shape[0]
        padded_f64[i, :V] = P
        padded_i64[i, :V] = Pi
        mask_np[i, :V] = 1.0

    coords_table = torch.from_numpy(
        padded_f64.reshape(N, 2 * maxV)
    )  # [N, 2*maxV], float64
    coords_fixed_table = torch.from_numpy(
        padded_i64.reshape(N, 2 * maxV)
    )  # [N, 2*maxV], int64
    mask_table = torch.from_numpy(mask_np)  # [N, maxV],  float64

    # Build lookup tables: LT Num -> index, bias_mapper -> index, surveyor name -> index
    lt_ids = bias_df["LT Num"].astype(str).tolist()
    id2idx = {k: i for i, k in enumerate(lt_ids)}
    idx2id = lt_ids

    bias_vocab = sorted(pd.unique(bias_df["bias_mapper"].astype(str)))
    bias2idx = {b: i for i, b in enumerate(bias_vocab)}
    idx2bias = {i: b for b, i in zip(bias_vocab, range(len(bias_vocab)))}
    bias_idx_vec = torch.tensor(
        [bias2idx[str(b)] for b in bias_df["bias_mapper"].astype(str)], dtype=torch.long
    )  # [N]

    name_vocab = sorted(pd.unique(bias_df["Land Surveyor"].astype(str).fillna("")))
    name2idx = {n: i for i, n in enumerate(name_vocab)}

    # Build per-bias surveyor name distribution (probability vectors)
    counts: Dict[str, Dict[str, int]] = {}
    for b, n in zip(
        bias_df["bias_mapper"].astype(str),
        bias_df["Land Surveyor"].astype(str).fillna(""),
    ):
        counts.setdefault(b, {}).setdefault(n, 0)
        counts[b][n] += 1

    name_probs_by_bias = torch.zeros(
        len(bias_vocab), len(name_vocab), dtype=torch.float64
    )
    for b in bias_vocab:
        cdict = counts.get(b, {})
        total = float(sum(cdict.values())) if cdict else 1.0
        b_idx = bias2idx[b]
        for n, ct in cdict.items():
            name_probs_by_bias[b_idx, name2idx[n]] = ct / total
    eps = 1e-12
    name_probs_by_bias = name_probs_by_bias + eps
    name_probs_by_bias = name_probs_by_bias / name_probs_by_bias.sum(
        dim=1, keepdim=True
    )
    name_dist_table = name_probs_by_bias[bias_idx_vec]  # [N, |names|], float64

    model = BiasEmbeds(
        coords_table, mask_table, bias_idx_vec, name_dist_table, len(bias_vocab)
    )

    # Save checkpoint with model state
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": {
                "maxV": maxV,
                "lt_ids": lt_ids,
                "id2idx": id2idx,
                "idx2bias": idx2bias,
                "bias_vocab": bias_vocab,
                "name_vocab": name_vocab,
                "preferred_names_by_bias": build_preferred_names_with_variants(
                    bias_df, bias_col="bias_mapper", name_col="Land Surveyor"
                ),
                "coords_fixed_int": coords_fixed_table,  # int64 tensor [N, 2*maxV]
                "coords_scale_pow": int(scale_pow),  # scale = 10 ** scale_pow
            },
        },
        CKPT_PATH,
    )
    print(f"Saved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    main()
