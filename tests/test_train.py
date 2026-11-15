import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.train import (
    analyze_errors,
    build_metadata,
    build_preprocessor,
    evaluate,
    feature_importance,
    make_pipeline,
)


def _training_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "city": rng.choice(["casablanca", "rabat", "fes"], n),
            "neighborhood": rng.choice(["maarif", "agdal", "medina"], n),
            "property_type": rng.choice(["appartement", "villa"], n),
            "condition": rng.choice(["bon etat", "neuf"], n),
            "age": rng.integers(0, 40, n).astype(float),
            "area": rng.integers(40, 400, n).astype(float),
            "rooms": rng.integers(1, 8, n).astype(float),
            "bathrooms": rng.integers(1, 5, n).astype(float),
            "floor": rng.integers(0, 10, n).astype(float),
            "amenities_count": rng.integers(0, 6, n).astype(float),
        }
    )
    frame["price"] = (
        200_000 + frame["area"] * 8_000 + frame["rooms"] * 50_000 + rng.normal(0, 30_000, n)
    )
    return frame


def test_build_preprocessor_returns_numeric_and_categorical_branches():
    frame = _training_frame().drop(columns=["price"])
    pre = build_preprocessor(frame)
    names = [name for name, _, _ in pre.transformers]
    assert names == ["num", "cat"]


def test_make_pipeline_fits_and_predicts():
    frame = _training_frame()
    X = frame.drop(columns=["price"])
    y_log = np.log1p(frame["price"])

    pipeline = make_pipeline(X, LinearRegression())
    pipeline.fit(X, y_log)
    predictions = pipeline.predict(X.head(3))

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()


def test_evaluate_returns_raw_mad_metrics():
    frame = _training_frame()
    X = frame.drop(columns=["price"])
    y_log = np.log1p(frame["price"])

    pipeline = make_pipeline(X, LinearRegression())
    pipeline.fit(X, y_log)
    metrics = evaluate(pipeline, X, frame["price"])

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]


def test_feature_importance_returns_ranked_list():
    frame = _training_frame()
    X = frame.drop(columns=["price"])
    pipeline = make_pipeline(X, LinearRegression())
    pipeline.fit(X, np.log1p(frame["price"]))

    importances = feature_importance(pipeline)

    assert importances
    assert importances[0]["importance"] >= importances[-1]["importance"]


def test_build_metadata_contains_api_contract():
    frame = _training_frame()
    metadata = build_metadata(
        frame,
        model_name="TestModel",
        metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.5},
        best_params={},
        importances=[],
        n_train=24,
        n_test=6,
    )

    assert metadata["model_name"] == "TestModel"
    assert "city" in metadata["categorical_values"]
    assert "area" in metadata["numeric_ranges"]
    assert metadata["input_features"][-1] == "amenities_count"


def test_analyze_errors_writes_reports(tmp_path):
    frame = _training_frame()
    X = frame.drop(columns=["price"])
    pipeline = make_pipeline(X, LinearRegression())
    pipeline.fit(X, np.log1p(frame["price"]))

    report = analyze_errors(pipeline, X, frame["price"], out_dir=tmp_path)

    assert "overall" in report
    assert (tmp_path / "error_analysis.json").exists()
    assert (tmp_path / "per_city.csv").exists()
    assert (tmp_path / "worst_predictions.csv").exists()
