import pandas as pd

from src.prep_data import load_source_1, load_source_2


def test_load_source_1_prefers_explicit_columns_and_counts_amenities(tmp_path):
    source = tmp_path / "source1.csv"
    pd.DataFrame(
        {
            "new_price": [1_000_000],
            "address": ["Appartement à vendre Casablanca - Maarif"],
            "chambres": [3],
            "salles de bains": [2],
            "surface": [100],
            "ascenseur": ["yes"],
            "terrasse": ["yes"],
            "parking": ["no"],
            "floor": [2],
            "Type": ["Appartement"],
            "City": ["Casablanca"],
            "Nighberd": ["Maarif"],
        }
    ).to_csv(source, index=False)

    df = load_source_1(source)

    assert df.loc[0, "city"] == "casablanca"
    assert df.loc[0, "neighborhood"] == "maarif"
    assert df.loc[0, "property_type"] == "appartement"
    assert df.loc[0, "amenities_count"] == 2.0


def test_load_source_1_falls_back_to_address_when_columns_are_missing(tmp_path):
    source = tmp_path / "source1.csv"
    pd.DataFrame(
        {
            "new_price": [1_000_000],
            "address": ["Villa à vendre Fès - Route Immouzer"],
            "chambres": [4],
            "salles de bains": [3],
            "surface": [250],
            "ascenseur": ["no"],
            "floor": [1],
        }
    ).to_csv(source, index=False)

    df = load_source_1(source)

    assert df.loc[0, "city"] == "fès"
    assert df.loc[0, "neighborhood"] == "route immouzer"
    assert df.loc[0, "property_type"] == "villa"
    assert df.loc[0, "amenities_count"] == 0.0


def test_load_source_2_keeps_only_sales_and_counts_amenities(tmp_path):
    source = tmp_path / "source2.csv"
    pd.DataFrame(
        {
            "reference": ["a", "b"],
            "type": ["appartement", "villa"],
            "transaction": ["vente", "location"],
            "price": [1_000_000, 2_000_000],
            "surface": [100, 200],
            "price_per_sqm": [10_000, 10_000],
            "rooms": [3, 4],
            "bathrooms": [2, 3],
            "floor": [2, 0],
            "city": ["casablanca", "rabat"],
            "neighborhood": ["maarif", "agdal"],
            "condition": ["bon etat", "neuf"],
            "age": [10, 0],
            "amenities": ["garage, parking", ""],
            "description": ["x", "y"],
            "listing_date": ["2025-01-01", "2025-01-02"],
        }
    ).to_csv(source, index=False)

    df = load_source_2(source)

    assert len(df) == 1
    assert df.loc[0, "property_type"] == "appartement"
    assert df.loc[0, "amenities_count"] == 2
    assert "price_per_sqm" not in df.columns
    assert "reference" not in df.columns
