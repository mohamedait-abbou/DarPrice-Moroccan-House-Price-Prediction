"""DarPrice - data preparation.

Merges the two raw sources (houses_data_eng.csv + morocco_houses_dataset.csv),
forces ONE column contract, removes leaks/outliers, and saves a clean CSV
to data/processed/houses_combined.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

KEEP = [
    "city", "neighborhood", "property_type", "condition", "age",
    "price", "area", "rooms", "bathrooms", "floor", "amenities_count",
]


def contract(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in KEEP if c in df.columns]].copy()
    for c in KEEP:
        if c not in out.columns:
            out[c] = np.nan
    return out[KEEP]


def load_source_1() -> pd.DataFrame:
    """Messy source: extract type/city/neighborhood from the address string."""
    df1 = pd.read_csv(ROOT / "data/raw/houses_data_eng.csv")
    df1.columns = [str(c).strip() for c in df1.columns]

    m = df1["address"].str.extract(
        r"^(?P<type>.+?) à vendre (?P<city>.+?) - (?P<quarter>.+)$"
    )
    df1["property_type"] = m["type"].str.strip().str.lower()
    df1["city"] = m["city"].str.strip().str.lower()
    df1["neighborhood"] = m["quarter"].str.strip().str.lower()
    df1 = df1.dropna(subset=["city"])

    if "ascenseur" in df1.columns:
        df1["amenities_count"] = (
            df1["ascenseur"].astype(str).str.strip().str.lower() == "yes"
        ).astype(float)

    df1["floor"] = pd.to_numeric(df1.get("floor"), errors="coerce")
    df1 = df1.rename(columns={
        "new_price": "price",
        "surface": "area",
        "chambres": "rooms",
        "salles de bains": "bathrooms",
    })
    return df1


def load_source_2() -> pd.DataFrame:
    """Cleaner source: keep sales only and drop leakage columns."""
    df2 = pd.read_csv(ROOT / "data/raw/morocco_houses_dataset.csv")
    df2 = df2[df2["transaction"] == "vente"].copy()
    # Trap 2 + 3: price_per_sqm leaks the answer, reference is pure noise
    df2 = df2.drop(
        columns=["price_per_sqm", "reference", "description", "listing_date"],
        errors="ignore",
    )
    df2["amenities_count"] = (
        df2["amenities"].fillna("").astype(str).str.split(",").str.len()
    )
    return df2.rename(columns={"type": "property_type", "surface": "area"})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([contract(load_source_1()), contract(load_source_2())],
                   ignore_index=True)
    df = df[~df["property_type"].isin(["bureau", "nan", ""])]

    for col in ["price", "area", "rooms", "bathrooms", "floor", "age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["price", "area"])
    df = df[(df["price"] > 50_000) & (df["area"] >= 10)]          # no 500 MAD houses

    ppm2 = df["price"] / df["area"]                               # Moroccan market reality
    df = df[(ppm2 > 2_000) & (ppm2 < 80_000)]

    for col in ["city", "neighborhood", "property_type", "condition"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df.reset_index(drop=True)


def main() -> None:
    raw1 = load_source_1().shape[0]
    raw2 = load_source_2().shape[0]
    print(f"Source 1 rows: {raw1:,} | Source 2 rows: {raw2:,}")

    df = clean(df=None)
    print(f"Clean shape: {df.shape}")

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "houses_combined.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
