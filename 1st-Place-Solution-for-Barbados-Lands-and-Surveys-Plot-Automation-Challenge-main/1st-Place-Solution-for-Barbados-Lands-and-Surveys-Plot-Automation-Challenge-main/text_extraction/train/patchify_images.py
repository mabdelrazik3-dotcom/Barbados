"""
Generate 7-crop tiled patches from survey plans for VLM training/inference.

Creates full, top, bottom, top-left, top-right, bottom-left, bottom-right crops,
then letterboxes each to square (1024px and 896px) with preserved aspect ratio.
Parallel processing for speed.

Output:
    - data/patched_1024/: 1024px crops
    - data/patched_896/: 896px crops
    - data/patched_1024.csv, data/patched_896.csv: CSV with paths to all crops
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))


from utils.base_utils import load_config

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

BASE_CONFIG = load_config("configs/base.yaml")
TEXT_CONFIG = load_config("configs/text_extraction.yaml")
ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"


def patchify_and_resize(
    df: pd.DataFrame, images_outp_dir, csv_path, image_size: int, image_path_col: str
) -> pd.DataFrame:
    """
    Create full-res patches, then resize with aspect ratio preserved via square letterboxing.
    Expects df to contain columns: 'ID' and image_path_col.
    Saves images under: images_outp_dir/<ID>/<side>_<size>x<size>.png
    Writes a wide CSV to csv_path and returns the resulting dataframe (indexed by ID only).
    If there are multiple rows per ID, the first occurrence is used.
    """
    required_cols = {"ID", image_path_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    images_outp_dir = Path(images_outp_dir)
    images_outp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)

    def letterbox_square(
        img: Image.Image, target: int, fill=(255, 255, 255)
    ) -> Image.Image:
        """Resize image to fit in target square, pad with white."""
        w, h = img.size
        s = target / max(w, h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (target, target), fill)
        canvas.paste(im, ((target - nw) // 2, (target - nh) // 2))
        return canvas

    side_to_col = {
        "full": "full_path",
        "top": "top_path",
        "bottom": "bottom_path",
        "top_left": "tl_path",
        "top_right": "tr_path",
        "bottom_left": "bl_path",
        "bottom_right": "br_path",
    }

    def process_one(rec):
        """Load image, create 7 crops, letterbox and save each."""
        out_rows = []
        ID = str(rec["ID"])
        img_path = Path(str(rec[image_path_col]))

        try:
            with Image.open(img_path) as im0:
                img = im0.convert("RGB")
        except Exception:
            return out_rows  # skip bad images

        w, h = img.size
        mw, mh = w // 2, h // 2

        tiles = {
            "full": img,
            "top": img.crop((0, 0, w, mh)),
            "bottom": img.crop((0, mh, w, h)),
            "top_left": img.crop((0, 0, mw, mh)),
            "top_right": img.crop((mw, 0, w, mh)),
            "bottom_left": img.crop((0, mh, mw, h)),
            "bottom_right": img.crop((mw, mh, w, h)),
        }

        out_dir = images_outp_dir / ID
        out_dir.mkdir(parents=True, exist_ok=True)

        for side, tile in tiles.items():
            save_path = out_dir / f"{side}_{image_size}x{image_size}.png"
            try:
                letterbox_square(tile, image_size).save(save_path)
                out_rows.append(
                    {"ID": ID, "col": side_to_col[side], "image_path": str(save_path)}
                )
            except Exception:
                continue

        return out_rows

    # Process one row per ID in parallel
    df_first = df.drop_duplicates(subset=["ID"]).copy()

    records = df_first[["ID", image_path_col]].to_dict("records")
    rows = []
    max_workers = min(32, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, r) for r in records]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=f"patching {image_size}"
        ):
            rows.extend(fut.result())

    if not rows:
        out_df = df_first.copy()
        out_df.to_csv(csv_path, index=False)
        print(f"wrote (no patches): {csv_path}")
        return out_df

    paths_df = pd.DataFrame(rows)
    wide = paths_df.pivot_table(
        index=["ID"], columns="col", values="image_path", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    base = df_first.copy()
    if "geometry" in base.columns:
        base = base.drop(columns=["geometry"])

    patched_df = base.merge(wide, on=["ID"], how="left")
    patched_df.to_csv(csv_path, index=False)
    print(f"wrote: {csv_path}")
    return patched_df


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "df.csv").drop_duplicates(
        subset=["ID"], ignore_index=True
    )
    df = df[~df.image_path.isna()].reset_index(drop=True)

    patched_df = patchify_and_resize(
        df=df,
        images_outp_dir=DATA_DIR / "patched_1024",
        csv_path=DATA_DIR / "patched_1024.csv",
        image_size=1024,
        image_path_col="image_path",
    )

    patched_df = patchify_and_resize(
        df=df,
        images_outp_dir=DATA_DIR / "patched_896",
        csv_path=DATA_DIR / "patched_896.csv",
        image_size=896,
        image_path_col="image_path",
    )
