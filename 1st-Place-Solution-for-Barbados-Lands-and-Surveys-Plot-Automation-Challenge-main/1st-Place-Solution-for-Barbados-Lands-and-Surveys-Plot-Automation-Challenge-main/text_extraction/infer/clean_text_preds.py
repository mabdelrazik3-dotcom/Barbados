"""Clean Unsloth OCR outputs into competition-ready metadata columns."""

import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils.base_utils import load_config

BASE_CONFIG = load_config("configs/base.yaml")
ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"

df = pd.read_csv(DATA_DIR / "text_predictions_df.csv")

# --- helpers  ---
CANONICAL_PARISHES = [
    "Christ Church",
    "St. Andrew",
    "St. George",
    "St. James",
    "St. John",
    "St. Joseph",
    "St. Lucy",
    "St. Michael",
    "St. Peter",
    "St. Philip",
    "St. Thomas",
]

# Build robust regex patterns for each canonical parish
_st_names = [p.split(". ", 1)[1] for p in CANONICAL_PARISHES if p.startswith("St. ")]
_PATTERNS = {"Christ Church": re.compile(r"\bchrist\s*church\b", re.I)}
for name in _st_names:
    canon = f"St. {name}"
    _PATTERNS[canon] = re.compile(rf"\b(?:st\.?|saint)\s*{re.escape(name)}\b", re.I)

# Anchor-to-end variants
_PATTERNS_END = {
    canon: re.compile(p.pattern + r"(?:[\s,;:\-])*$", re.I)
    for canon, p in _PATTERNS.items()
}

_END_TRIM = re.compile(r'[\s"“”\',;:.-]+$')


def extract_parish(address: str, default: str = "St. Philip") -> str:
    """Infer the parish name from an address snippet with robust regex fallbacks."""
    if not isinstance(address, str) or not address.strip():
        return default
    s = re.sub(r'[“”"]', "", address.strip())
    s = re.sub(r"\s+", " ", s)
    s = _END_TRIM.sub("", s)

    for canon, p in _PATTERNS_END.items():
        if p.search(s):
            return canon

    for canon, p in _PATTERNS.items():
        if p.search(s):
            return canon

    return default


df["Parish"] = df["Address"].apply(extract_parish)

# Build regex variants for valid parishes
_ST_NAMES = [p.split(". ", 1)[1] for p in CANONICAL_PARISHES if p.startswith("St. ")]
_PATTERNS = [re.compile(r"\bchrist\s*church\b", re.I)]
_PATTERNS += [
    re.compile(rf"\b(?:st\.?|saint)\s*{re.escape(name)}\b", re.I) for name in _ST_NAMES
]

# Precompile anchored-to-end removal patterns
_SEP = r'[\s,;:\-/"“”\'()]*'
_PATTERNS_END = [re.compile(rf"{_SEP}(?:{p.pattern}){_SEP}$", re.I) for p in _PATTERNS]

_END_TRIM = re.compile(r'[\s,;:\-/"“”\'()]+$')


def strip_trailing_parish(s: str) -> str:
    """Remove appended parish names so the address field stops one column early."""
    if not isinstance(s, str):
        return s
    x = re.sub(r'[“”"]', "", s.strip())
    x = re.sub(r"\s+", " ", x)
    for p_end in _PATTERNS_END:
        if p_end.search(x):
            x = p_end.sub("", x)
            x = _END_TRIM.sub("", x)
            return x
    return x


df["Address"] = df["Address2"].apply(strip_trailing_parish)


def _strip_quotes(x):
    """Normalize quoted strings to bare tokens while preserving None."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    m = re.fullmatch(r"""['"]?(.*?)['"]?""", s)
    return m.group(1).strip() if m else s


df["Total Area"] = df["Total Area"].apply(_strip_quotes)
df["LT Num"] = df["LT Num"].apply(_strip_quotes)
df["Certified date"] = df["Certified date"].apply(_strip_quotes)


# helpers
def _to_str(x):
    return "" if x is None else str(x)


# patterns
_sq_m_pat = re.compile(
    r"""
    ^\s*
    (
      sq(?:uare)?\.?\s*m(?:eters?)?  # sq m / square meters
      |sqm                           # sqm
      |m²                            # m²
      |m\^?2                         # m^2 / m2
      |m\s*2                         # m 2
    )
    \s*$
""",
    re.IGNORECASE | re.VERBOSE,
)

_ha_pat = re.compile(r"^\s*(?:hectares?|hectare|ha)\s*$", re.IGNORECASE)


def _norm_unit(x):
    """Canonicalize unit strings to competition choices ('sq m' or 'ha')."""
    s = _to_str(x)
    s = s.replace("“", "").replace("”", "").strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    if _sq_m_pat.fullmatch(s):
        return "sq m"
    if _ha_pat.fullmatch(s):
        return "ha"
    return None  # enforce only {"sq m","ha"}


# apply to dataframe column
df["Unit of Measurement"] = df["Unit of Measurement"].apply(_norm_unit)
df["Land Surveyor"] = df["Land Surveyor2"]
df = df.drop(columns=["Land Surveyor2", "Address2"])

df.to_csv(DATA_DIR / "text_predictions_df_cleaned.csv", index=False)


def clean_land_surveyor_names(names_array):
    """
    Clean land surveyor names by removing middle initials while preserving:
    - Names that start with initials (like H.A. King)
    - Names with nicknames in quotes
    - Names with compound elements like St. Clair
    - Professional designations like JP
    """

    def clean_single_name(name):
        if pd.isna(name) or not name or name.strip() == "":
            return name
        name = str(name).strip()
        # Skip names that start with initials (like "H.A King" or "H.A. King")
        if re.match(r"^[A-Z]\.?\s*[A-Z]\.?\s+", name):
            return name
        # Skip names with quotes (nicknames like D.C "Vallan" Franklin JP)
        if '"' in name:
            return name
        # Skip single names (like "Simba")
        if len(name.split()) <= 1:
            return name
        # Protect "St." in compound names like "Michelle E. St. Clair"
        name_protected = name.replace(" St. ", " PROTECTED_ST ")
        # Remove middle initials patterns:
        # 1. Single initials: "Lennox J Reid" → "Lennox Reid"
        name_cleaned = re.sub(r"\s+[A-Z]\.?\s+", " ", name_protected)
        # 2. Multiple initials: "Jamal K.L. Gaskin" → "Jamal Gaskin"
        name_cleaned = re.sub(r"\s+[A-Z]\.[A-Z]\.?\s+", " ", name_cleaned)
        # 3. Space-separated initials: "Lee B S Brathwaite" → "Lee Brathwaite"
        name_cleaned = re.sub(r"\s+[A-Z]\s+[A-Z]\s+", " ", name_cleaned)
        # 4. Complex patterns like "Lee B.S Brathwaite" or "Sekani H.C Franklin"
        name_cleaned = re.sub(r"\s+[A-Z]\.[A-Z]\s+", " ", name_cleaned)
        # 5. Handle remaining single initials that might be left
        name_cleaned = re.sub(r"\s+[A-Z]\.?\s+", " ", name_cleaned)
        # Restore protected "St."
        name_cleaned = name_cleaned.replace(" PROTECTED_ST ", " St. ")
        # Clean up multiple spaces and trim
        name_cleaned = re.sub(r"\s+", " ", name_cleaned).strip()
        return name_cleaned

    # Apply cleaning to each name in the array
    if isinstance(names_array, np.ndarray):
        return np.array([clean_single_name(name) for name in names_array])
    elif isinstance(names_array, (list, pd.Series)):
        return [clean_single_name(name) for name in names_array]
    else:
        return clean_single_name(names_array)


def clean_target_survey(text: str) -> str:
    """Lowercase, remove periods and commas, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[.,]", " ", text)  # remove periods and commas
    text = re.sub(r"\s+", " ", text)  # normalize multiple spaces
    return text.strip()


def format_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TargetSurvey and keeps only required columns.
    Applies lowercase, removes ., , and normalizes spaces.
    """
    df["TargetSurvey"] = (
        df["Land Surveyor"].astype(str).str.strip()
        + " "
        + df["Surveyed For"].astype(str).str.strip()
        + " "
        + df["Address"].astype(str).str.strip()
    ).apply(clean_target_survey)

    columns_to_keep = [
        "ID",
        "TargetSurvey",
        "Certified date",
        "Total Area",
        "Unit of Measurement",
        "Parish",
        "LT Num",
        "geometry",
    ]
    return df[columns_to_keep]


df["Land Surveyor"] = df["Land Surveyor"].apply(clean_land_surveyor_names)
LS = df["Land Surveyor"].tolist()
df["geometry"] = " "
df = format_dataset(df)
df["Land Surveyor"] = LS  # Restore cleaned names

df.to_csv(DATA_DIR / "sub_text_extraction.csv", index=False)
print(df.head())
