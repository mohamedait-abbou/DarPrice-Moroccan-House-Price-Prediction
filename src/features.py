"""DarPrice - feature engineering.

The raw listing features are useful, but a few ratios capture how a house is
used (space per room, total wet rooms, amenity density) better than the raw
columns alone. Keeping this logic in a small scikit-learn transformer means it
runs *inside* the saved Pipeline, so training and serving always agree.

The transformer is deliberately stateless: it only derives new columns from the
input row, which makes it safe to use inside cross-validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DERIVED_FEATURES = [
    "area_per_room",
    "rooms_plus_bath",
    "amenities_per_100m2",
    "is_new",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add interpretable ratio features to a listing DataFrame.

    All operations are row-wise, so the transformer can be applied to a single
    row (the API) or a full dataset (training) without fitting anything.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        area = pd.to_numeric(df.get("area"), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms"), errors="coerce")
        bathrooms = pd.to_numeric(df.get("bathrooms"), errors="coerce")
        amenities = pd.to_numeric(df.get("amenities_count"), errors="coerce")
        age = pd.to_numeric(df.get("age"), errors="coerce")

        safe_area = area.clip(lower=1)
        safe_rooms = rooms.clip(lower=1)

        df["area_per_room"] = area / safe_rooms
        df["rooms_plus_bath"] = rooms.fillna(0) + bathrooms.fillna(0)
        df["amenities_per_100m2"] = amenities.fillna(0) / (safe_area / 100)
        df["is_new"] = (age == 0).astype(int)

        return df

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        base = list(input_features) if input_features is not None else []
        return [*base, *DERIVED_FEATURES]
