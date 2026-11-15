"""DarPrice - data preparation.

Merges the two raw sources (houses_data_eng.csv + morocco_houses_dataset.csv),
forces ONE column contract, removes leaks/outliers, and saves a clean CSV to
data/processed/houses_combined.csv.

Run with:
    python -m src.prep_data
    # or, from the repo root:
    python src/prep_data.py
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCE_1 = ROOT / "data/raw/houses_data_eng.csv"
SOURCE_2 = ROOT / "data/raw/morocco_houses_dataset.csv"
OUT_FILE = ROOT / "data/processed/houses_combined.csv"

KEEP = [
    "city",
    "neighborhood",
    "property_type",
    "condition",
    "age",
    "price",
    "area",
    "rooms",
    "bathrooms",
    "floor",
    "amenities_count",
]

TEXT_COLS = ["city", "neighborhood", "property_type", "condition"]
NUMERIC_COLS = ["price", "area", "rooms", "bathrooms", "floor", "age"]

# Market sanity bounds used to remove scraped junk / typos.
MIN_PRICE = 50_000
MIN_AREA = 10
MIN_PRICE_PER_SQM = 2_000
MAX_PRICE_PER_SQM = 80_000
MAX_AGE = 200
MIN_FLOOR = -10
MAX_FLOOR = 200

# Columns that describe the listing, not the price. price_per_sqm is leakage.
LEAKY_OR_ID_COLUMNS = ["price_per_sqm", "reference", "description", "listing_date"]


def normalize_text_value(value: object) -> str:
    """Lowercase, trim, and strip accents from a single value.

    Without this, OneHotEncoder treats "fes" and "fès" as different cities and
    the model loses signal by splitting the same city across two categories.
    """
    text = "" if pd.isna(value) else str(value).strip().lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(series: pd.Series) -> pd.Series:
    """Vectorized version of :func:`normalize_text_value`."""
    return series.map(normalize_text_value)


def contract(df: pd.DataFrame) -> pd.DataFrame:
    """Force an input frame to the exact column contract (missing -> NaN)."""
    out = df.copy()
    for col in KEEP:
        if col not in out.columns:
            out[col] = np.nan
    return out[KEEP]


def _count_yes_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Count how many boolean amenity columns equal 'yes'."""
    total = pd.Series(0, index=df.index, dtype="float64")
    for col in columns:
        total += (df[col].astype(str).str.strip().str.lower() == "yes").astype(float)
    return total


def load_source_1(path: Path = SOURCE_1) -> pd.DataFrame:
    """Messy source: prefer explicit Type/City/Neighborhood columns.

    The scraper sometimes ships clean ``Type`` / ``City`` / ``Nighberd`` columns
    (yes, with that typo) and sometimes only an ``address`` string. We use the
    explicit columns when available and fall back to parsing the address.
    """
    df1 = pd.read_csv(path)
    df1.columns = [str(c).strip() for c in df1.columns]

    explicit = {"Type": "property_type", "City": "city", "Nighberd": "neighborhood"}
    for source_col, target_col in explicit.items():
        if source_col in df1.columns:
            df1[target_col] = df1[source_col].astype(str).str.strip().str.lower()

    if "address" in df1.columns:
        parsed = (
            df1["address"]
            .astype(str)
            .str.extract(r"^(?P<type>.+?) à vendre (?P<city>.+?) - (?P<quarter>.+)$")
        )
        fallbacks = {
            "property_type": parsed["type"].str.strip().str.lower(),
            "city": parsed["city"].str.strip().str.lower(),
            "neighborhood": parsed["quarter"].str.strip().str.lower(),
        }
        for target_col, fallback in fallbacks.items():
            if target_col not in df1.columns:
                df1[target_col] = fallback
            else:
                missing = df1[target_col].isna() | df1[target_col].eq("")
                df1.loc[missing, target_col] = fallback[missing]
        df1 = df1.dropna(subset=["city"])

    amenity_cols = [c for c in ["ascenseur", "terrasse", "parking"] if c in df1.columns]
    df1["amenities_count"] = _count_yes_columns(df1, amenity_cols) if amenity_cols else np.nan

    df1["floor"] = (
        pd.to_numeric(df1["floor"], errors="coerce") if "floor" in df1.columns else np.nan
    )

    return df1.rename(
        columns={
            "new_price": "price",
            "surface": "area",
            "chambres": "rooms",
            "salles de bains": "bathrooms",
        }
    )


def load_source_2(path: Path = SOURCE_2) -> pd.DataFrame:
    """Cleaner source: keep sales only and drop leakage/identifier columns."""
    df2 = pd.read_csv(path)
    if "transaction" in df2.columns:
        df2 = df2[df2["transaction"] == "vente"].copy()

    df2 = df2.drop(columns=LEAKY_OR_ID_COLUMNS, errors="ignore")

    if "amenities" in df2.columns:
        df2["amenities_count"] = (
            df2["amenities"]
            .fillna("")
            .astype(str)
            .map(lambda value: len([a for a in value.split(",") if a.strip()]))
        )
    else:
        df2["amenities_count"] = np.nan

    return df2.rename(columns={"type": "property_type", "surface": "area"})


def build_dataset(source_1: Path = SOURCE_1, source_2: Path = SOURCE_2) -> pd.DataFrame:
    """Load both sources and stack them under the shared contract."""
    return pd.concat(
        [contract(load_source_1(source_1)), contract(load_source_2(source_2))],
        ignore_index=True,
    )


def clean(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean an already-combined frame (pure function: no file loading).

    Steps: drop non-residential rows, coerce numerics, remove exact duplicates,
    remove missing price/area, apply market sanity bounds, normalize text, and
    drop empty property types.
    """
    rows = {"input": len(df)}
    df = df.copy()
    df = df[~df["property_type"].isin(["bureau", "nan", ""])]
    rows["non_residential"] = len(df)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    rows["duplicates"] = len(df)

    df = df.dropna(subset=["price", "area"])
    rows["missing_price_area"] = len(df)

    df = df[(df["price"] > MIN_PRICE) & (df["area"] >= MIN_AREA)]
    rows["price_area_bounds"] = len(df)

    price_per_sqm = df["price"] / df["area"]
    df = df[(price_per_sqm > MIN_PRICE_PER_SQM) & (price_per_sqm < MAX_PRICE_PER_SQM)]
    rows["price_per_sqm_bounds"] = len(df)

    df = df[df["age"].isna() | df["age"].between(0, MAX_AGE)]
    df = df[df["floor"].isna() | df["floor"].between(MIN_FLOOR, MAX_FLOOR)]
    df = df[
        (df["rooms"].isna() | (df["rooms"] >= 0))
        & (df["bathrooms"].isna() | (df["bathrooms"] >= 0))
    ]
    rows["numeric_sanity"] = len(df)

    for col in TEXT_COLS:
        df[col] = normalize_text(df[col])
    df = df[df["property_type"].ne("") & df["property_type"].ne("nan")]
    rows["normalized"] = len(df)

    if verbose:
        print("Cleaning funnel:")
        previous = rows["input"]
        for step, count in rows.items():
            if step == "input":
                print(f"  {step:<22} {count:>7,}")
            else:
                print(f"  {step:<22} {count:>7,}  (-{previous - count:,})")
            previous = count

    return df.reset_index(drop=True)


def main() -> None:
    raw1 = load_source_1()
    raw2 = load_source_2()
    print(f"Source 1 rows: {len(raw1):,} | Source 2 rows: {len(raw2):,}")

    df = clean(build_dataset(), verbose=True)
    print(f"\nClean shape: {df.shape}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
