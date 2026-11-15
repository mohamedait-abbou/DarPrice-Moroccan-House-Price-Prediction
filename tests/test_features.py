import pandas as pd

from src.features import DERIVED_FEATURES, FeatureEngineer


def _frame(**overrides) -> pd.DataFrame:
    row = {
        "city": "casablanca",
        "neighborhood": "maarif",
        "property_type": "appartement",
        "condition": "bon etat",
        "age": 10.0,
        "price": 1_000_000,
        "area": 100.0,
        "rooms": 4.0,
        "bathrooms": 2.0,
        "floor": 2.0,
        "amenities_count": 3.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_feature_engineer_adds_expected_columns():
    out = FeatureEngineer().fit_transform(_frame())
    for column in DERIVED_FEATURES:
        assert column in out.columns


def test_feature_engineer_values():
    out = FeatureEngineer().fit_transform(_frame()).iloc[0]
    assert out["area_per_room"] == 25.0
    assert out["rooms_plus_bath"] == 6.0
    assert out["amenities_per_100m2"] == 3.0
    assert out["is_new"] == 0


def test_feature_engineer_is_new_flag():
    out = FeatureEngineer().fit_transform(_frame(age=0)).iloc[0]
    assert out["is_new"] == 1


def test_feature_engineer_handles_zero_rooms_and_area():
    out = FeatureEngineer().fit_transform(_frame(rooms=0, area=0)).iloc[0]
    assert out["area_per_room"] == 0.0
    assert out["amenities_per_100m2"] == 300.0


def test_feature_engineer_does_not_mutate_input():
    frame = _frame()
    original_columns = list(frame.columns)
    FeatureEngineer().fit_transform(frame)
    assert list(frame.columns) == original_columns
