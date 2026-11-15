"""DarPrice - model training.

Pipeline:
- log-transform the right-skewed target (price) and invert with expm1 at serve time
- derive ratio features inside the sklearn Pipeline (no train/serve skew)
- compare LinearRegression / RandomForest / ExtraTrees / XGBoost with 5-fold CV
- tune the three tree models with RandomizedSearchCV (CV on TRAIN only)
- pick the winner by CV MAE, then evaluate it ONCE on the held-out test set
- save the full Pipeline + metrics.json + models/metadata.json + error reports

Run with:
    python -m src.train
    # or, from the repo root:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    from src.features import FeatureEngineer
    from src.prep_data import OUT_FILE, ROOT
except ImportError:  # allows `python src/train.py`
    from features import FeatureEngineer
    from prep_data import OUT_FILE, ROOT

MODEL_PATH = ROOT / "models/darprice_model.joblib"
METADATA_PATH = ROOT / "models/metadata.json"
METRICS_PATH = ROOT / "metrics.json"
REPORTS_DIR = ROOT / "reports"

TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5
TARGET = "price"
CURRENCY = "MAD"
N_JOBS = int(os.getenv("DARP_PRICE_N_JOBS", "2"))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the numeric/categorical preprocessing branch for engineered data."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def make_pipeline(X: pd.DataFrame, model: object) -> Pipeline:
    """Wrap feature engineering + preprocessing + regressor in one Pipeline."""
    engineer = FeatureEngineer()
    engineered = engineer.fit_transform(X)
    return Pipeline(
        [
            ("features", engineer),
            ("pre", build_preprocessor(engineered)),
            ("reg", model),
        ]
    )


def make_kfold() -> KFold:
    """Shuffled folds matter because rows can be grouped by source/city."""
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def cv_mae(pipeline: Pipeline, X: pd.DataFrame, y_log: pd.Series) -> float:
    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=make_kfold(),
        scoring="neg_mean_absolute_error",
        n_jobs=N_JOBS,
    )
    return float(-scores.mean())


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test_mad: pd.Series) -> dict:
    """Evaluate on the raw MAD scale (the scale a user actually cares about)."""
    y_pred_mad = np.expm1(pipeline.predict(X_test))
    return {
        "mae": float(mean_absolute_error(y_test_mad, y_pred_mad)),
        "rmse": float(root_mean_squared_error(y_test_mad, y_pred_mad)),
        "r2": float(r2_score(y_test_mad, y_pred_mad)),
    }


def feature_importance(pipeline: Pipeline) -> list[dict]:
    """Return the top transformed features for tree models or linear models."""
    pre = pipeline.named_steps["pre"]
    reg = pipeline.named_steps["reg"]
    names = pre.get_feature_names_out()

    if hasattr(reg, "feature_importances_"):
        values = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        values = np.abs(np.ravel(reg.coef_))
    else:
        return []

    ranked = sorted(zip(names, values, strict=True), key=lambda pair: pair[1], reverse=True)
    return [{"feature": str(name), "importance": float(value)} for name, value in ranked[:20]]


def build_metadata(
    df: pd.DataFrame,
    model_name: str,
    metrics: dict,
    best_params: dict,
    importances: list[dict],
    n_train: int,
    n_test: int,
) -> dict:
    """Describe the inputs and performance so the API/UI can validate requests."""
    X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_values = {
        col: sorted(
            {
                str(value)
                for value in X[col].dropna().tolist()
                if str(value).strip() and str(value) != "nan"
            }
        )
        for col in cat_cols
    }
    numeric_ranges = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in num_cols
        if X[col].notna().any()
    }

    return {
        "model_name": model_name,
        "target": "log1p(price)",
        "currency": CURRENCY,
        "input_features": list(X.columns),
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "categorical_values": categorical_values,
        "numeric_ranges": numeric_ranges,
        "derived_features": FeatureEngineer().get_feature_names_out(),
        "metrics": metrics,
        "best_params": best_params,
        "feature_importance_top20": importances,
        "dataset": str(OUT_FILE.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
    }


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _group_metrics(frame: pd.DataFrame, column: str) -> list[dict]:
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            n=("actual_price_MAD", "size"),
            mae_MAD=("abs_error_MAD", "mean"),
            median_abs_error_MAD=("abs_error_MAD", "median"),
            mape_pct=("pct_error", "mean"),
            mean_actual_MAD=("actual_price_MAD", "mean"),
            mean_predicted_MAD=("predicted_price_MAD", "mean"),
        )
        .reset_index()
        .sort_values("mae_MAD")
    )
    return grouped.to_dict(orient="records")


def analyze_errors(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_mad: pd.Series,
    out_dir: Path = REPORTS_DIR,
) -> dict:
    """Build per-city/per-type error tables, worst predictions, and plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    predicted = np.expm1(pipeline.predict(X_test))
    actual = np.asarray(y_test_mad, dtype=float)
    frame = X_test.reset_index(drop=True).copy()
    frame["actual_price_MAD"] = actual
    frame["predicted_price_MAD"] = predicted
    frame["abs_error_MAD"] = np.abs(actual - predicted)
    frame["pct_error"] = frame["abs_error_MAD"] / np.clip(actual, 1, None) * 100

    by_city = _group_metrics(frame, "city")
    by_type = _group_metrics(frame, "property_type")

    worst_columns = [
        "city",
        "neighborhood",
        "property_type",
        "area",
        "rooms",
        "actual_price_MAD",
        "predicted_price_MAD",
        "abs_error_MAD",
        "pct_error",
    ]
    worst = frame.nlargest(25, "abs_error_MAD")[worst_columns].round(2).to_dict(orient="records")

    pd.DataFrame(by_city).to_csv(out_dir / "per_city.csv", index=False)
    pd.DataFrame(by_type).to_csv(out_dir / "per_property_type.csv", index=False)
    frame.nlargest(25, "abs_error_MAD")[worst_columns].to_csv(
        out_dir / "worst_predictions.csv", index=False
    )

    report = {
        "overall": {
            "mae_MAD": float(frame["abs_error_MAD"].mean()),
            "median_abs_error_MAD": float(frame["abs_error_MAD"].median()),
            "rmse_MAD": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "mape_pct": float(frame["pct_error"].mean()),
            "within_20pct": float((frame["pct_error"] <= 20).mean()),
            "within_50pct": float((frame["pct_error"] <= 50).mean()),
        },
        "error_quantiles_MAD": {
            str(q): float(frame["abs_error_MAD"].quantile(q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]
        },
        "by_city": by_city,
        "by_property_type": by_type,
        "worst_predictions": worst,
    }
    (out_dir / "error_analysis.json").write_text(json.dumps(report, indent=2))

    _make_plots(frame, by_city, pipeline, out_dir)
    return report


def _make_plots(
    frame: pd.DataFrame,
    by_city: list[dict],
    pipeline: Pipeline,
    out_dir: Path,
) -> None:
    """Create portfolio-friendly charts; skip silently if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 1. Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["actual_price_MAD"], frame["predicted_price_MAD"], alpha=0.35, s=14)
    upper = float(max(frame["actual_price_MAD"].max(), frame["predicted_price_MAD"].max()))
    ax.plot([0, upper], [0, upper], "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual price (MAD)")
    ax.set_ylabel("Predicted price (MAD)")
    ax.set_title("Predicted vs actual house prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=140)
    plt.close(fig)

    # 2. Residual distribution
    residuals = frame["actual_price_MAD"] - frame["predicted_price_MAD"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, color="#2563eb", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted, MAD)")
    ax.set_ylabel("Number of listings")
    ax.set_title("Residual distribution on the held-out test set")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=140)
    plt.close(fig)

    # 3. MAE by city (top 12 by volume)
    city_frame = pd.DataFrame(by_city).sort_values("n", ascending=False).head(12)
    city_frame = city_frame.sort_values("mae_MAD")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(city_frame["city"], city_frame["mae_MAD"], color="#059669")
    ax.set_xlabel("MAE (MAD)")
    ax.set_title("Mean absolute error by city (top 12 by listings)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_city_mae.png", dpi=140)
    plt.close(fig)

    # 4. Feature importance
    importances = feature_importance(pipeline)
    if importances:
        names = [item["feature"] for item in importances[:15]][::-1]
        values = [item["importance"] for item in importances[:15]][::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(names, values, color="#7c3aed")
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 features (gain importance)")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=140)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DarPrice model.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip hyperparameter tuning and use small models (smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    global TEST_SIZE, SEED, CV_FOLDS
    args = parse_args()
    TEST_SIZE, SEED, CV_FOLDS = args.test_size, args.seed, args.cv_folds

    df = pd.read_csv(OUT_FILE)
    y = df[TARGET].astype(float)
    y_log = np.log1p(y)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=SEED
    )
    y_test_mad = np.expm1(y_test_log)

    # ---- 1. Baseline comparison with shuffled 5-fold CV on TRAIN only ----
    tree_estimators = 40 if args.quick else 200
    candidates = {
        "LinearRegression": make_pipeline(X_train, LinearRegression()),
        "RandomForest": make_pipeline(
            X_train,
            RandomForestRegressor(n_estimators=tree_estimators, random_state=SEED, n_jobs=N_JOBS),
        ),
        "ExtraTrees": make_pipeline(
            X_train,
            ExtraTreesRegressor(n_estimators=tree_estimators, random_state=SEED, n_jobs=N_JOBS),
        ),
        "XGBoost": make_pipeline(
            X_train,
            XGBRegressor(
                n_estimators=200 if not args.quick else 60,
                learning_rate=0.08,
                max_depth=5,
                random_state=SEED,
                n_jobs=N_JOBS,
                tree_method="hist",
            ),
        ),
    }

    print(f"5-fold CV MAE in log-space ({CV_FOLDS} shuffled folds)")
    print("-" * 44)
    cv_results = {}
    for name, pipeline in candidates.items():
        cv_results[name] = cv_mae(pipeline, X_train, y_train_log)
        print(f"{name:<20} {cv_results[name]:.4f}")

    tuned: dict[str, tuple[Pipeline, float, dict]] = {}
    if not args.quick:
        # ---- 2. Tune the tree models, still using CV on TRAIN only ----
        print("\nTuning RandomForest...")
        rf_pipeline = make_pipeline(
            X_train, RandomForestRegressor(random_state=SEED, n_jobs=N_JOBS)
        )
        rf_search = RandomizedSearchCV(
            rf_pipeline,
            {
                "reg__n_estimators": [150, 250, 400],
                "reg__max_depth": [None, 10, 20, 30],
                "reg__min_samples_split": [2, 5, 10],
                "reg__min_samples_leaf": [1, 2, 4],
            },
            n_iter=12,
            cv=make_kfold(),
            scoring="neg_mean_absolute_error",
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        rf_search.fit(X_train, y_train_log)
        print(f"  best CV MAE: {-rf_search.best_score_:.4f}")
        print(f"  best params: {rf_search.best_params_}")

        print("Tuning ExtraTrees...")
        et_pipeline = make_pipeline(X_train, ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS))
        et_search = RandomizedSearchCV(
            et_pipeline,
            {
                "reg__n_estimators": [150, 250, 400],
                "reg__max_depth": [None, 10, 20, 30],
                "reg__min_samples_split": [2, 5, 10],
                "reg__min_samples_leaf": [1, 2, 4],
            },
            n_iter=10,
            cv=make_kfold(),
            scoring="neg_mean_absolute_error",
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        et_search.fit(X_train, y_train_log)
        print(f"  best CV MAE: {-et_search.best_score_:.4f}")
        print(f"  best params: {et_search.best_params_}")

        print("Tuning XGBoost...")
        xgb_pipeline = make_pipeline(
            X_train, XGBRegressor(random_state=SEED, n_jobs=N_JOBS, tree_method="hist")
        )
        xgb_search = RandomizedSearchCV(
            xgb_pipeline,
            {
                "reg__n_estimators": [150, 250, 350],
                "reg__max_depth": [3, 5, 7],
                "reg__learning_rate": [0.03, 0.08, 0.15],
                "reg__subsample": [0.7, 0.9, 1.0],
                "reg__colsample_bytree": [0.7, 0.9, 1.0],
                "reg__min_child_weight": [1, 3, 5],
            },
            n_iter=20,
            cv=make_kfold(),
            scoring="neg_mean_absolute_error",
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        xgb_search.fit(X_train, y_train_log)
        print(f"  best CV MAE: {-xgb_search.best_score_:.4f}")
        print(f"  best params: {xgb_search.best_params_}")

        tuned = {
            "Tuned_RandomForest": (
                rf_search.best_estimator_,
                -rf_search.best_score_,
                rf_search.best_params_,
            ),
            "Tuned_ExtraTrees": (
                et_search.best_estimator_,
                -et_search.best_score_,
                et_search.best_params_,
            ),
            "Tuned_XGBoost": (
                xgb_search.best_estimator_,
                -xgb_search.best_score_,
                xgb_search.best_params_,
            ),
        }

    # ---- 3. Pick the winner by CV MAE (NEVER by the test set) ----
    all_cv = {
        **cv_results,
        **{name: score for name, (_, score, _) in tuned.items()},
    }
    best_name = min(all_cv, key=all_cv.get)
    if best_name in tuned:
        best_pipeline = tuned[best_name][0]
        best_params = tuned[best_name][2]
    else:
        # cross_val_score clones the estimator, so the candidate is still unfitted.
        best_pipeline = candidates[best_name].fit(X_train, y_train_log)
        best_params = {}

    # ---- 4. Evaluate the winner ONCE on the held-out test set ----
    metrics = evaluate(best_pipeline, X_test, y_test_mad)
    print("\nHeld-out test set (raw MAD scale)")
    print("-" * 60)
    print(f"{'model':<20} {'MAE':>13} {'RMSE':>13} {'R2':>8}")
    print(
        f"{best_name:<20} {metrics['mae']:>13,.0f} {metrics['rmse']:>13,.0f} {metrics['r2']:>8.3f}"
    )

    # ---- 5. Error analysis + plots ----
    report = analyze_errors(best_pipeline, X_test, y_test_mad)
    print("\nError analysis")
    print(f"  median abs error : {report['overall']['median_abs_error_MAD']:>12,.0f} MAD")
    print(f"  MAPE             : {report['overall']['mape_pct']:>12.1f} %")
    print(f"  within 20%       : {report['overall']['within_20pct']:>12.1%}")

    # ---- 6. Persist model + reproducible metadata ----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    importances = feature_importance(best_pipeline)
    metadata = build_metadata(
        df, best_name, metrics, best_params, importances, len(X_train), len(X_test)
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    metrics_report = {
        **metadata,
        "cv_mae_log": all_cv,
        "test": metrics,
        "error_analysis": report["overall"],
    }
    METRICS_PATH.write_text(json.dumps(metrics_report, indent=2))

    print(f"\nBest model (by CV): {best_name} (CV MAE {all_cv[best_name]:.4f})")
    print(f"Saved model    -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metadata -> {METADATA_PATH.relative_to(ROOT)}")
    print(f"Saved metrics  -> {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved reports  -> {REPORTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
