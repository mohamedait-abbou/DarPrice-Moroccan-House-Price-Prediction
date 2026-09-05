"""DarPrice - model training.

Improvements over the original version:
- LOG-TRANSFORMS the target (price) because it is heavily right-skewed.
  Tree models fitted on ~uniform log-price generalise far better to
  expensive houses. Predictions are exponentiated back to MAD with expm1.
- Reports RMSE as well as MAE / R2.
- Adds 5-fold cross-validation so the result does not depend on a single
  "lucky" train/test split.
- Adds XGBoost (gradient boosting) alongside LinearRegression and
  RandomForest.
- Hyperparameter-tunes RandomForest and XGBoost with RandomizedSearchCV
  and picks the best model by held-out MAE.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from prep_data import ROOT

MODEL_PATH = ROOT / "models/darprice_randomforest.joblib"
TEST_SIZE = 0.2
SEED = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/houses_combined.csv")

    y = df["price"].astype(float)
    y_log = np.log1p(y)
    X = df.drop(columns=["price"])

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=SEED
    )
    # keep raw prices for reporting
    _, _, y_train_mad, y_test_mad = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pre = build_preprocessor(X_train)

    # ---- Cross-validated estimate of every candidate on TRAIN ----
    def cv_pipeline(model):
        return Pipeline([("pre", pre), ("reg", model)])

    candidates = {
        "LinearRegression": cv_pipeline(LinearRegression()),
        "RandomForest": cv_pipeline(
            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
        "XGBoost": cv_pipeline(
            XGBRegressor(n_estimators=200, learning_rate=0.08,
                         max_depth=5, random_state=SEED, n_jobs=-1)),
    }

    print(f"5-fold CV MAE in log-space ({'model':<16})\n" + "-" * 40)
    cv_results = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train_log, cv=5,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
        log_mae = -scores.mean()
        cv_results[name] = log_mae
        print(f"{name:<20} {log_mae:.4f}")

    # ---- Hyperparameter tuning (RandomizedSearchCV) ----
    print("\nTuning RandomForest...")
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_pipe = Pipeline([("pre", pre), ("reg", rf)])
    rf_grid = {
        "reg__n_estimators": [100, 200, 400],
        "reg__max_depth": [None, 10, 20, 30],
        "reg__min_samples_split": [2, 5, 10],
        "reg__min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_grid, n_iter=15, cv=3, scoring="neg_mean_absolute_error",
        random_state=SEED, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train, y_train_log)
    print(f"  best RF: {rf_search.best_params_}")

    print("Tuning XGBoost...")
    xgb = XGBRegressor(random_state=SEED, n_jobs=-1)
    xgb_pipe = Pipeline([("pre", pre), ("reg", xgb)])
    xgb_grid = {
        "reg__n_estimators": [100, 200, 300],
        "reg__max_depth": [3, 5, 7],
        "reg__learning_rate": [0.03, 0.08, 0.15],
        "reg__subsample": [0.7, 0.9, 1.0],
        "reg__colsample_bytree": [0.7, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_grid, n_iter=20, cv=3, scoring="neg_mean_absolute_error",
        random_state=SEED, n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train, y_train_log)
    print(f"  best XGB: {xgb_search.best_params_}")

    # ---- Compare best tuned models on held-out test set ----
    teste_models = {
        "Tuned_RandomForest": rf_search.best_estimator_,
        "Tuned_XGBoost": xgb_search.best_estimator_,
    }

    print("\nHeld-out test set (raw MAD scale)\n" + "-" * 60)
    print(f"{'model':<18} {'MAE':>13} {'RMSE':>13} {'R2':>8}")
    results = {}
    for name, model in teste_models.items():
        y_pred_log = model.predict(X_test)
        y_pred_mad = np.expm1(y_pred_log)
        mae = mean_absolute_error(y_test_mad, y_pred_mad)
        rmse = root_mean_squared_error(y_test_mad, y_pred_mad)
        r2 = r2_score(y_test_mad, y_pred_mad)
        results[name] = (mae, rmse, r2)
        print(f"{name:<18} {mae:>13,.0f} {rmse:>13,.0f} {r2:>8.3f}")

    # ---- Pick the winner by MAE and save it ----
    best_name = min(results, key=lambda k: results[k][0])
    best_pipeline = teste_models[best_name]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} (MAE {results[best_name][0]:,.0f} MAD)")
    print(f"Saved -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
