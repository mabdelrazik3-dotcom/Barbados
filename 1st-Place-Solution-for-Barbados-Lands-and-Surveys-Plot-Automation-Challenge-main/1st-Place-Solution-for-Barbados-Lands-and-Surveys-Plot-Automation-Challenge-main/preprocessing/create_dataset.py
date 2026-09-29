"""
Assemble training/test metadata, K-fold splits, and image paths.

Creates unified dataframe (df.csv) merging:
- Train.csv and Test.csv (competition data)
- geom_px_df.csv (pixel-space polygon coordinates)
- Image file paths from survey_plans directory
- K-fold cross-validation splits for training

Output: data/df.csv - unified dataset for all downstream scripts
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.base_utils import load_config, wkt_polygon_to_xy_list

# Load configuration
BASE_CONFIG = load_config("configs/base.yaml")

ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "survey_plans"

FOLDS = BASE_CONFIG["folds"]  # Number of K-fold splits (default: 5)
SEED = BASE_CONFIG["seed"]    # Random seed for reproducibility

# Load data files
train = pd.read_csv(DATA_DIR / "Train.csv")
test = pd.read_csv(DATA_DIR / "Test.csv")
sample_submission = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
geom_px_df = pd.read_csv(DATA_DIR / "geom_px_df.csv")
geom_px_df["geometry_px"] = geom_px_df["geometry_px"].apply(eval)  # Parse string to list

# Convert WKT geometry to coordinate lists
train.geometry = train.geometry.apply(wkt_polygon_to_xy_list)
train = train.drop_duplicates(subset=["ID"]).reset_index(drop=True)

# Create K-fold splits for cross-validation
train["fold"] = -1
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X=train)):
    train.loc[val_idx, "fold"] = fold

# Map image IDs to file paths
imgs_df = pd.Series(list(IMAGES_DIR.glob("*.jpg"))).to_frame(name="image_path")
imgs_df["ID"] = imgs_df["image_path"].apply(lambda x: x.stem.split("_")[-1])

# Merge all data sources
df = pd.concat(
    [train.assign(split="train"), test.assign(split="test")], ignore_index=True
)
df = df.merge(imgs_df, on="ID", how="left")        # Add image paths
df = df.merge(geom_px_df, on="ID", how="left")     # Add pixel polygons

if __name__ == "__main__":
    df.to_csv(DATA_DIR / "df.csv", index=False)
