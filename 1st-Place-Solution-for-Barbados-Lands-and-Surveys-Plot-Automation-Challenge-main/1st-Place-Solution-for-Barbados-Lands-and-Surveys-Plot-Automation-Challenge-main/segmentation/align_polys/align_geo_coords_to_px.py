"""
Align geographic polygon coordinates to pixel coordinates using Qwen VLM.

This script uses a locally hosted Qwen3-VL-30B model to convert geographic
survey plan coordinates to pixel-space coordinates. The VLM reasons about
spatial relationships to provide accurate alignment.

Input: df.csv (with ID, image_path, geometry columns)
Output: geom_px_df.csv (ID, geometry_px)

Note: Requires vLLM server running on localhost:8000
"""

import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

from utils.base_utils import load_config

# VLM service configuration
BASE_URL = "http://localhost:8000/v1"  # Local vLLM server
MODEL    = "qwen3-vl-30b-a3b-thinking"
API_KEY  = "dummy"                      # Placeholder for local server
SIZE     = 1000                         # Standard canvas size for VLM input

def letterbox_to_square(im, size=1000, bg="white"):
    """Resize an image into a square canvas while preserving aspect ratio."""
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    imr = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=bg)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(imr, (ox, oy))
    return canvas


def to_xy_list(g):
    """
    Normalize various geometry formats into (x, y) coordinate list.

    Handles: Shapely Polygon/MultiPolygon, JSON strings, lists, tuples.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
        if g is None:
            return []
        if isinstance(g, Polygon):
            return [(float(x), float(y)) for x, y in g.exterior.coords]
        if isinstance(g, MultiPolygon):
            poly = max(list(g.geoms), key=lambda p: p.area)
            return [(float(x), float(y)) for x, y in poly.exterior.coords]
    except Exception:
        pass
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except Exception:
            try:
                g = eval(g)
            except Exception:
                g = None
    out = []
    if isinstance(g, (list, tuple)):
        for p in g:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
    return out


def build_hint(size=1000, xy=None, margin=64, stroke=(0, 0, 0)):
    """
    Create hint image showing rough polygon outline for VLM guidance.

    Renders polygon on white canvas to help VLM locate the target parcel.
    """
    if not xy:
        return None
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w0, h0 = maxx - minx, maxy - miny
    if w0 <= 0 or h0 <= 0:
        return None
    S = size - 2 * margin
    if w0 >= h0:
        scale = S / w0
        W = S
        H = h0 * scale
    else:
        scale = S / h0
        H = S
        W = w0 * scale
    offx = (size - W) / 2 - minx * scale
    offy = (size - H) / 2 - miny * scale
    im = Image.new("RGB", (size, size), color="white")
    d = ImageDraw.Draw(im)
    pts = [(p[0] * scale + offx, p[1] * scale + offy) for p in xy]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    d.line(pts, width=10, fill=stroke)
    return im


def extract_json_str(text: str):
    """
    Extract JSON from VLM response text.

    Tries: direct JSON parse → markdown code blocks → manual bracket matching.
    """
    try:
        obj = json.loads(text.strip())
        return json.dumps(obj)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        return m.group(1)
    s, best, start = text, None, text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start : j + 1]
                    if '"vertices_px_clockwise"' in cand:
                        best = cand
                    break
        start = s.find("{", start + 1)
    return best


def chat(payload: dict) -> str:
    """Send chat completion request to local vLLM server, return response text."""
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def process_row(row) -> list:
    """
    Process one survey plan row through VLM to get pixel coordinates.

    Steps:
    1. Load and letterbox plan image to 1000x1000
    2. Create polygon hint image from geographic coordinates
    3. Send both images to VLM with extraction prompt
    4. Parse JSON response to get pixel vertices

    Returns: List of [x, y] pixel coordinate pairs
    """
    image_path = row["image_path"]
    geom = row["geometry"] if "geometry" in row else None

    # Prepare images for VLM
    plan = Image.open(image_path).convert("RGB")
    plan_1k = letterbox_to_square(plan, SIZE)  # Resize to standard size
    xy = to_xy_list(geom)
    hint_1k = build_hint(SIZE, xy) if xy else None  # Create polygon hint

    # Save temporary files for VLM server access
    tmp_dir = os.path.dirname(image_path)
    sid = uuid.uuid4().hex[:8]
    tmp_plan = os.path.join(tmp_dir, f".__tmp_plan_{SIZE}_{sid}.png")
    plan_1k.save(tmp_plan)
    tmp_hint = None
    if hint_1k is not None:
        tmp_hint = os.path.join(tmp_dir, f".__tmp_hint_{SIZE}_{sid}.png")
        hint_1k.save(tmp_hint)

    # Construct VLM prompt messages
    system_msg = (
        "You are a cadastral plan extraction agent.\n"
        "Two images follow: (1) a cadastral plan letterboxed to 1000x1000; "
        "(2) a shape hint on a 1000x1000 canvas.\n"
        "Return ONLY JSON with: parcel_label, confidence, image_width=1000, image_height=1000, "
        "vertices_px_clockwise=[[x1,y1],...]. No extra fields. Coordinates are in the same 1000x1000 frame."
    )

    schema_msg = (
        "Return ONLY this JSON (no backticks):\n"
        "{\n"
        '  "parcel_label": "<matched text or \\"unknown\\">",\n'
        '  "confidence": <float>,\n'
        '  "image_width": 1000,\n'
        '  "image_height": 1000,\n'
        '  "vertices_px_clockwise": [[x1,y1],[x2,y2],...]\n'
        "}"
    )

    user_msg = (
        "Use image #1 to extract only the parcel polygon corners matching the shape in image #2 (hint). "
        "Return strictly the schema above."
    )

    msg = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_msg + "\n" + schema_msg},
                {"type": "image_url", "image_url": {"url": f"file://{tmp_plan}"}},
            ]
            + ([{"type": "image_url", "image_url": {"url": f"file://{tmp_hint}"}}] if tmp_hint else []),
        },
    ]

    payload = {
        "model": MODEL,
        "messages": msg,
        "temperature": 0.0,  # Deterministic output
        "response_format": {"type": "json_object"},
    }

    # Call VLM and clean up temporary files
    try:
        txt = chat(payload)
    except Exception:
        # Retry without JSON format constraint if needed
        payload.pop("response_format", None)
        txt = chat(payload)
    finally:
        # Clean up temporary image files
        for p in [tmp_plan, tmp_hint]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # Parse VLM response to extract coordinates
    json_str = extract_json_str(txt)
    if not json_str:
        return []
    try:
        out = json.loads(json_str)
    except Exception:
        return []

    verts_raw = out.get("vertices_px_clockwise", []) or []

    # Convert vertices to numeric format
    try:
        V = np.array(verts_raw, dtype=float)
        return [[float(x), float(y)] for x, y in V.tolist()]
    except Exception:
        # Fallback: try direct conversion
        try:
            return [[float(p[0]), float(p[1])] for p in verts_raw]
        except Exception:
            return []

def main():
    """
    Main entry point: process all rows in df.csv and save pixel polygons.

    Iterates through each survey plan, calls VLM for alignment,
    and saves results to geom_px_df.csv.
    """
    BASE_CONFIG = load_config("configs/base.yaml")
    ROOT_DIR = Path(BASE_CONFIG["root_dir"])  # keep path as given
    DATA_DIR = ROOT_DIR / "data"

    df = pd.read_csv(DATA_DIR / "df.csv")

    if "ID" not in df.columns:
        raise KeyError("Expected an 'ID' column in df.csv")
    if "image_path" not in df.columns:
        raise KeyError("Expected an 'image_path' column in df.csv")

    rows = []
    for _, row in df.iterrows():
        try:
            geom_px = process_row(row)
        except Exception:
            geom_px = []
        rows.append({"ID": row["ID"], "geometry_px": json.dumps(geom_px)})

    out_df = pd.DataFrame(rows, columns=["ID", "geometry_px"])

    out_path = DATA_DIR / "geom_px_df.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
