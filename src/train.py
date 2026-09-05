"""DarPrice - model training.

Trains a LinearRegression baseline and a RandomForest on the cleaned data,
compares them, saves the best pipeline to models/darprice_randomforest.joblib
and prints the top features the model uses.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep_data import ROOT

MODEL_PATH = ROOT / "models/darprice_randomforest.joblib"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

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


def evaluate(name: str, model: Pipeline, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name:<18} MAE: {mae:,.0f} MAD | R2: {r2:.3f}")


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/houses_combined.csv")

    y = df["price"]
    X = df.drop(columns=["price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_mae = (y_test - y_train.mean()).abs().mean()
    print(f"{'Baseline (mean)':<18} MAE: {baseline_mae:,.0f} MAD\n")

    pre = build_preprocessor(X)

    linear = Pipeline([("pre", pre), ("reg", LinearRegression())])
    linear.fit(X_train, y_train)
    evaluate("LinearRegression", linear, X_test, y_test)

    forest = Pipeline([
        ("pre", pre),
        ("reg", RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    forest.fit(X_train, y_train)
    evaluate("RandomForest", forest, X_test, y_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(forest, MODEL_PATH)
    print(f"\nSaved -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
