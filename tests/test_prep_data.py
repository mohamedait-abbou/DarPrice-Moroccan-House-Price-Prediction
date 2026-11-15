import pandas as pd

from src.prep_data import clean, contract, normalize_text, normalize_text_value


def _combined_row(**overrides) -> dict:
    row = {
        "city": "casablanca",
        "neighborhood": "maarif",
        "property_type": "appartement",
        "condition": "bon etat",
        "age": 10.0,
        "price": 1_000_000,
        "area": 100.0,
        "rooms": 3.0,
        "bathrooms": 2.0,
        "floor": 2.0,
        "amenities_count": 3.0,
    }
    row.update(overrides)
    return row


def test_normalize_text_lowercases_trims_and_removes_accents():
    raw = pd.Series(["  Casablanca ", "Meknès", "FÈS", None])
    assert normalize_text(raw).tolist() == ["casablanca", "meknes", "fes", ""]


def test_normalize_text_value_handles_scalars():
    assert normalize_text_value("  Tétouan ") == "tetouan"
    assert normalize_text_value(None) == ""


def test_contract_forces_exact_columns():
    out = contract(pd.DataFrame({"city": ["rabat"], "unknown": [1]}))
    assert list(out.columns) == [
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
    assert out["neighborhood"].isna().all()


def test_clean_removes_outliers_and_duplicates():
    df = pd.DataFrame(
        [
            _combined_row(),
            _combined_row(),  # duplicate
            _combined_row(price=10_000),  # too cheap
            _combined_row(property_type="bureau"),  # non-residential
            _combined_row(area=100, price=20_000_000),  # price/m2 > 80k
        ]
    )

    cleaned = clean(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["price"] == 1_000_000


def test_clean_removes_invalid_numeric_ranges():
    df = pd.DataFrame(
        [
            _combined_row(),
            _combined_row(age=-5),
            _combined_row(floor=999),
            _combined_row(rooms=-1),
            _combined_row(bathrooms=-2),
        ]
    )

    cleaned = clean(df)

    assert len(cleaned) == 1


def test_clean_merges_accented_city_duplicates():
    df = pd.DataFrame(
        [
            _combined_row(city="fès"),
            _combined_row(city="fes", price=1_200_000),
            _combined_row(city="Meknès"),
            _combined_row(city="meknes", price=1_500_000),
        ]
    )

    cleaned = clean(df)

    assert set(cleaned["city"]) == {"fes", "meknes"}


def test_clean_normalizes_categories_to_lowercase():
    df = pd.DataFrame(
        [
            _combined_row(
                city="Casablanca",
                neighborhood="Maarif",
                property_type="Appartement",
                condition="Bon Etat",
            )
        ]
    )

    row = clean(df).iloc[0]

    assert row["city"] == "casablanca"
    assert row["neighborhood"] == "maarif"
    assert row["property_type"] == "appartement"
    assert row["condition"] == "bon etat"


def test_clean_verbose_reports_funnel(capsys):
    clean(pd.DataFrame([_combined_row()]), verbose=True)
    assert "Cleaning funnel" in capsys.readouterr().out
