# Build DarPrice Yourself — The Complete Teaching Guide

This document explains **everything** in the project: what each file does, why
each design decision was made, and how you can rebuild the whole system from
scratch. Read it top to bottom once, then use it as a reference.

If you only want to run the project, jump to
[Quick start](#quick-start). If you want to learn, read everything.

---

## Table of contents

1. [What you are building](#1-what-you-are-building)
2. [The big picture](#2-the-big-picture)
3. [Prerequisites](#3-prerequisites)
4. [Repository tour](#4-repository-tour)
5. [Step 1 — Understand the data](#5-step-1--understand-the-data)
6. [Step 2 — Clean the data (`src/prep_data.py`)](#6-step-2--clean-the-data-srcprep_datapy)
7. [Step 3 — Engineer features (`src/features.py`)](#7-step-3--engineer-features-srcfeaturespy)
8. [Step 4 — Train the model (`src/train.py`)](#8-step-4--train-the-model-srctrainpy)
9. [Step 5 — Analyze errors (`src/evaluate.py`)](#9-step-5--analyze-errors-srcevaluatepy)
10. [Step 6 — Serve predictions (`app/main.py`)](#10-step-6--serve-predictions-appmainpy)
11. [Step 7 — Build a UI (`app/streamlit_app.py`)](#11-step-7--build-a-ui-appstreamlit_apppy)
12. [Step 8 — Test everything](#12-step-8--test-everything)
13. [Step 9 — Containerize with Docker](#13-step-9--containerize-with-docker)
14. [Step 10 — Automate with CI/CD](#14-step-10--automate-with-cicd)
15. [Quick start](#15-quick-start)
16. [What was improved in this version](#16-what-was-improved-in-this-version)
17. [Troubleshooting](#17-troubleshooting)
18. [Exercises](#18-exercises)
19. [Deploy for free](#19-deploy-for-free)
20. [Glossary](#20-glossary)

---

## 1. What you are building

DarPrice predicts the sale price of a Moroccan house in **MAD** (Moroccan dirham)
from listing features such as city, neighborhood, property type, surface area,
rooms, bathrooms, floor, age, and number of amenities.

It is a **complete ML system**, not just a notebook:

```text
raw CSVs
   │
   ▼
data cleaning + feature engineering
   │
   ▼
model training + cross-validation + tuning
   │
   ▼
saved sklearn Pipeline + metrics + reports
   │
   ├──▶ FastAPI REST service
   ├──▶ Streamlit demo UI
   ├──▶ pytest test suite
   ├──▶ Docker image
   └──▶ GitHub Actions CI/CD
```

The goal is to practice the same workflow used by real ML teams:

1. Get and understand data.
2. Clean it and remove leakage.
3. Build a reproducible training pipeline.
4. Evaluate honestly.
5. Serve the model behind an API.
6. Test, containerize, and automate.

---

## 2. The big picture

The most important architectural idea in this project is the **sklearn
`Pipeline`**.

A pipeline bundles preprocessing and the model into one object:

```python
Pipeline(
    [
        ("features", FeatureEngineer()),
        ("pre", ColumnTransformer([...])),
        ("reg", XGBRegressor(...)),
    ]
)
```

Why this matters:

- During training, the pipeline learns imputation values, scaling statistics,
  and category vocabularies.
- The **same** object is saved with `joblib`.
- At serving time, the API calls `pipeline.predict(raw_features)`.
- Therefore preprocessing is **identical** in training and serving. This is
  called avoiding **train/serve skew**.

If you preprocess manually before saving, you must remember to repeat every
transformation in the API. That is a classic source of production bugs.

---

## 3. Prerequisites

You need:

- **Python 3.12+** (the pinned dependencies in this repo were tested on Python
  3.14).
- `pip` and `venv` (bundled with Python).
- Git.
- Optional: Docker Desktop / Docker Engine.
- Optional: a free GitHub account.

Check your Python:

```bash
python --version
```

You should see 3.12 or newer.

---

## 4. Repository tour

```text
DarPrice/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI service
│   └── streamlit_app.py        # Streamlit demo UI
├── src/
│   ├── __init__.py
│   ├── prep_data.py            # merge + clean raw sources
│   ├── features.py             # FeatureEngineer transformer
│   ├── train.py                # CV, tuning, evaluation, artifacts
│   └── evaluate.py             # standalone error analysis
├── tests/
│   ├── test_prep_data.py
│   ├── test_sources.py
│   ├── test_features.py
│   ├── test_train.py
│   └── test_api.py
├── data/
│   ├── raw/                    # original CSVs (gitignored)
│   ├── processed/
│   │   └── houses_combined.csv # committed, 7,014 × 11
│   └── README.md
├── models/
│   ├── darprice_model.joblib   # committed pipeline
│   └── metadata.json           # categories, ranges, metrics
├── reports/                    # error analysis + charts
├── metrics.json                # full training report
├── docs/
│   ├── TUTORIAL.md             # this file
│   └── MODEL_CARD.md
├── .github/workflows/
│   ├── ci.yml                  # lint + format + tests + docker build
│   └── cd.yml                  # publish image to GHCR
├── Dockerfile                  # API image
├── Dockerfile.streamlit        # Streamlit UI image
├── docker-compose.yml          # API + UI stack
├── render.yaml                 # Render Blueprint (one-click deploy)
├── .env.example                # env var template
├── Makefile
├── requirements.txt            # runtime + training
├── requirements-train.txt      # + plotting
├── requirements-dev.txt        # + tests, lint, pre-commit
├── requirements-streamlit.txt  # Streamlit UI client (streamlit + requests)
├── pyproject.toml              # ruff + pytest + coverage config
└── README.md
```

---

## 5. Step 1 — Understand the data

### 5.1 The two sources

| File | Rows | What it is |
|---|---|---|
| `data/raw/houses_data_eng.csv` | 4,675 | Scraped listings. The address string looks like `Appartement à vendre Casablanca - Maarif`. Some exports also include clean `Type`, `City`, `Nighberd` columns. |
| `data/raw/morocco_houses_dataset.csv` | 10,000 | Tabular sales and rental listings with `transaction`, `price`, `surface`, `amenities`, etc. |

The raw files are **not committed** because they are large and may have license
restrictions. The cleaned dataset is committed so the project works from a
fresh clone.

### 5.2 The target

The target is `price`, measured in MAD. Raw prices are heavily right-skewed:
a few very expensive villas pull the mean far above the median. Training a model
directly on raw price makes it focus too much on expensive houses.

The fix is to train on `log1p(price)`:

```python
y_log = np.log1p(price)
```

`log1p(x) = log(1 + x)` is safe when price is 0. To get back to MAD:

```python
price = np.expm1(log_price)
```

`expm1` is the inverse of `log1p` and is numerically more accurate than
`np.exp(x) - 1` for small values.

After the transform, the skew drops from about **1.66** to about **−0.14**, which
is close to symmetric.

### 5.3 Leakage

Leakage means a feature contains information about the answer that you would not
have at prediction time.

In `morocco_houses_dataset.csv`, the column `price_per_sqm` is computed from the
price itself. If you keep it, the model can "cheat". We drop it:

```python
LEAKY_OR_ID_COLUMNS = ["price_per_sqm", "reference", "description", "listing_date"]
```

`reference` is an identifier, `description` is free text, and `listing_date` is
metadata. None of them should be model inputs in this version.

### 5.4 Data quality

A surprising fact: source 1 has 4,675 rows but only **24 unique feature
combinations**. The same listings repeat many times. `drop_duplicates()` in
`clean()` removes the copies. This matters because duplicates could otherwise
appear in both train and test and inflate the score.

Source 2 is cleaner and supplies most of the final 7,014 rows.

### 5.5 The final schema

| Column | Type | Meaning |
|---|---|---|
| `city` | text | Normalized city |
| `neighborhood` | text | Normalized quarter |
| `property_type` | text | appartement, villa, studio, maison, duplex, riad |
| `condition` | text | neuf, bon etat, tres bon etat, excellent etat, a renover |
| `age` | number | Building age in years (NaN when unknown) |
| `price` | number | Sale price in MAD (**target**) |
| `area` | number | Surface in m² |
| `rooms` | number | Bedrooms |
| `bathrooms` | number | Bathrooms |
| `floor` | number | Floor number |
| `amenities_count` | number | Number of amenities |

---

## 6. Step 2 — Clean the data (`src/prep_data.py`)

Run it:

```bash
python -m src.prep_data
```

You will see a **cleaning funnel**:

```text
Source 1 rows: 4,675 | Source 2 rows: 6,990
Cleaning funnel:
  input                   11,665
  non_residential         11,478  (-187)
  duplicates               7,014  (-4,464)
  ...
```

This tells you exactly how many rows each rule removed. Always print a funnel
when cleaning data: silent filtering is how datasets mysteriously shrink.

### 6.1 The column contract

```python
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
```

`contract(df)` forces any source into exactly these columns. Missing columns
become `NaN`. This lets us stack two messy sources without `KeyError`s:

```python
def contract(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in KEEP:
        if col not in out.columns:
            out[col] = np.nan
    return out[KEEP]
```

### 6.2 Text normalization

```python
def normalize_text_value(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
```

What it does:

- `"  Casablanca "` → `"casablanca"`
- `"Meknès"` → `"meknes"`
- `"FÈS"` → `"fes"`

Without this, `OneHotEncoder` treats `fes` and `fès` as two different cities and
splits the signal. The API reuses the **same** function, so `"Casablanca"` from a
user produces the same prediction as `"casablanca"`.

### 6.3 Source 1

Source 1 is messy. `load_source_1()`:

1. Reads the CSV and strips column names.
2. Prefers explicit `Type`, `City`, `Nighberd` columns if they exist.
3. Falls back to parsing `address` with a regex:
   `^(?P<type>.+?) à vendre (?P<city>.+?) - (?P<quarter>.+)$`
4. Counts amenities: `ascenseur`, `terrasse`, `parking` equal to `"yes"`.
5. Renames columns (`new_price` → `price`, `surface` → `area`, etc.).

### 6.4 Source 2

`load_source_2()`:

1. Keeps only `transaction == "vente"` (sales, not rentals).
2. Drops leakage/identifier columns.
3. Counts amenities by splitting the comma-separated `amenities` string.

### 6.5 Cleaning rules

`clean(df)` is a **pure function**: it does not load files, so it is easy to
test. It applies:

| Rule | Why |
|---|---|
| Drop `bureau` / empty property types | We model residential houses |
| Coerce numeric columns | Bad strings become `NaN` |
| Drop exact duplicates | Prevent leakage and double-counting |
| Drop missing price/area | Can't train without them |
| `price > 50,000` and `area >= 10` | Remove junk listings |
| `2,000 < price/area < 80,000` | Remove absurd price per m² |
| Age 0–200, floor −10–200, rooms/bathrooms ≥ 0 | Numeric sanity |
| Normalize text, drop empty types | One category per real value |

The sanity bounds are constants at the top of the file. If the Moroccan market
changes, you can tune them in one place.

---

## 7. Step 3 — Engineer features (`src/features.py`)

Raw columns are useful, but ratios often carry extra signal:

```python
DERIVED_FEATURES = [
    "area_per_room",
    "rooms_plus_bath",
    "amenities_per_100m2",
    "is_new",
]
```

`FeatureEngineer` is a scikit-learn transformer:

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        area = pd.to_numeric(df.get("area"), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms"), errors="coerce")
        ...
        df["area_per_room"] = area / rooms.clip(lower=1)
        df["rooms_plus_bath"] = rooms.fillna(0) + bathrooms.fillna(0)
        df["amenities_per_100m2"] = amenities.fillna(0) / (area.clip(lower=1) / 100)
        df["is_new"] = (age == 0).astype(int)
        return df
```

Why `BaseEstimator` and `TransformerMixin`?

- `BaseEstimator` gives `get_params` / `set_params`, so the transformer works
  inside `Pipeline`, `GridSearchCV`, and `RandomizedSearchCV`.
- `TransformerMixin` gives `fit_transform`.

Why is it stateless? It only derives columns from the current row. There is
nothing to learn, so it cannot leak information across folds.

**Important pickling note:** the saved pipeline contains a `FeatureEngineer`
object. When you load the model later, Python must be able to import
`src.features`. That is why the Docker image copies the `src/` folder too.

---

## 8. Step 4 — Train the model (`src/train.py`)

Run it:

```bash
python -m src.train
```

For a fast smoke test without tuning:

```bash
python -m src.train --quick
```

### 8.1 Load and split

```python
df = pd.read_csv(OUT_FILE)
y = df["price"].astype(float)
y_log = np.log1p(y)
X = df.drop(columns=["price"])

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
y_test_mad = np.expm1(y_test_log)
```

`random_state=42` makes the split reproducible. The test set is **only** used
for the final evaluation.

### 8.2 Preprocessing

```python
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
pre = ColumnTransformer(
    [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)
```

- **Median imputation** for numbers: robust to outliers.
- **StandardScaler**: mean 0, standard deviation 1. Helps linear models; trees
  do not need it, but it does not hurt.
- **Most-frequent imputation** for categories.
- **OneHotEncoder**: one binary column per category.
- `handle_unknown="ignore"`: if the API receives a new city, the encoder outputs
  all zeros instead of crashing.

### 8.3 Candidate models

| Model | Why include it |
|---|---|
| `LinearRegression` | Simple baseline; shows how much non-linearity matters |
| `RandomForestRegressor` | Strong tabular baseline, robust to outliers |
| `ExtraTreesRegressor` | Like Random Forest but more randomized; often fast and strong |
| `XGBRegressor` | Gradient boosting; usually the best on tabular data |

Each candidate is wrapped in the full pipeline:

```python
candidates = {
    "XGBoost": make_pipeline(X_train, XGBRegressor(...)),
    ...
}
```

### 8.4 Cross-validation

```python
def make_kfold():
    return KFold(n_splits=5, shuffle=True, random_state=42)


scores = cross_val_score(
    pipeline,
    X_train,
    y_train_log,
    cv=make_kfold(),
    scoring="neg_mean_absolute_error",
    n_jobs=2,
)
```

Why `shuffle=True`? Rows can be grouped by source or city. Shuffling makes each
fold more representative.

Why `neg_mean_absolute_error`? scikit-learn maximizes scores, so it returns the
negative MAE. We flip the sign when printing.

We cross-validate in **log space** because that is the space the model is trained
in. The final test metrics are reported in **MAD** because that is what users
care about.

### 8.5 Hyperparameter tuning

```python
search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=20,
    cv=make_kfold(),
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=2,
)
search.fit(X_train, y_train_log)
```

`RandomizedSearchCV` samples combinations instead of trying all of them. It is
much cheaper than `GridSearchCV` and usually finds a good configuration.

We tune Random Forest, Extra Trees, and XGBoost. The search runs **only on the
training set**, never on the test set.

### 8.6 Model selection

```python
all_cv = {**cv_results, **{name: score for name, (_, score, _) in tuned.items()}}
best_name = min(all_cv, key=all_cv.get)
```

The winner is the model with the lowest CV MAE. This is the honest way to select
a model: use validation/CV for decisions, use the test set once at the end.

### 8.7 Final evaluation

```python
def evaluate(pipeline, X_test, y_test_mad):
    y_pred_mad = np.expm1(pipeline.predict(X_test))
    return {
        "mae": float(mean_absolute_error(y_test_mad, y_pred_mad)),
        "rmse": float(root_mean_squared_error(y_test_mad, y_pred_mad)),
        "r2": float(r2_score(y_test_mad, y_pred_mad)),
    }
```

- **MAE** (Mean Absolute Error): average absolute error in MAD.
- **RMSE** (Root Mean Squared Error): penalizes large errors more.
- **R²**: 1.0 is perfect, 0.0 is as good as predicting the mean, negative is
  worse than the mean.

### 8.8 Error analysis

`analyze_errors()` writes:

- `reports/error_analysis.json` — overall metrics, error quantiles, per-city,
  per-property-type, worst predictions.
- `reports/per_city.csv` and `reports/per_property_type.csv`.
- `reports/worst_predictions.csv`.
- `reports/predicted_vs_actual.png`, `residuals.png`, `per_city_mae.png`,
  `feature_importance.png`.

This is how you move from "my model has R² 0.9" to "my model is worse in
Marrakech villas because …".

### 8.9 Artifacts

| File | Contents |
|---|---|
| `models/darprice_model.joblib` | Full fitted pipeline |
| `models/metadata.json` | Model name, categories, numeric ranges, metrics, versions |
| `metrics.json` | CV scores, test metrics, best params, feature importance, error analysis |
| `reports/` | Human-readable analysis |

`models/metadata.json` is what powers the `/metadata` endpoint and the Streamlit
form. It tells the UI which cities, neighborhoods, and property types are valid.

---

## 9. Step 5 — Analyze errors (`src/evaluate.py`)

```bash
python -m src.evaluate
```

This reloads the saved model, reproduces the exact same train/test split, and
regenerates `reports/`. It does **not** retrain, so it is fast.

Use it when you want to ask questions like:

- Which city has the highest MAE?
- Are errors larger for villas or apartments?
- What do the worst 25 predictions look like?
- Are residuals centered around zero, or is the model biased?

---

## 10. Step 6 — Serve predictions (`app/main.py`)

Run:

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

### 10.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Liveness + whether model/metadata loaded |
| `GET` | `/metadata` | Valid categories and numeric ranges |
| `GET` | `/model-info` | Metrics, params, feature importance |
| `POST` | `/predict` | Predict one house |
| `POST` | `/predict/batch` | Predict up to 100 houses |

### 10.2 Request validation

Pydantic validates every field:

```python
class House(BaseModel):
    city: str = Field(..., min_length=1)
    area: float = Field(..., gt=0, le=10_000)
    rooms: int = Field(..., ge=0, le=50)
    ...
```

Invalid input returns **HTTP 422** automatically. This protects the model from
negative areas, missing fields, and absurd values.

### 10.3 Normalization at serve time

```python
@field_validator("city", "neighborhood", "property_type", "condition", mode="before")
@classmethod
def normalize_categories(cls, value: object) -> str:
    return normalize_text_value(value)
```

This is critical. The model was trained on lowercase, accent-free text. Without
this validator, a user sending `"Casablanca"` would hit an unknown category and
get a worse prediction. The validator makes serving consistent with training.

### 10.4 Prediction

```python
features = pd.DataFrame([house.model_dump()])
log_price = float(model.predict(features)[0])
price = float(np.expm1(log_price))
```

The pipeline handles feature engineering, imputation, encoding, and regression.
The API only inverts the log transform.

### 10.5 Batch prediction

```python
@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(payload: BatchRequest) -> BatchPrediction:
    predictions = [_predict_one(house) for house in payload.houses]
    return BatchPrediction(predictions=predictions, count=len(predictions))
```

Batch is useful for CSV uploads or internal tools. The limit (100) protects the
server from huge requests.

### 10.6 Configuration

| Environment variable | Default |
|---|---|
| `DARP_PRICE_MODEL_PATH` | `models/darprice_model.joblib` |
| `DARP_PRICE_METADATA_PATH` | `models/metadata.json` |
| `DARP_PRICE_METRICS_PATH` | `metrics.json` |
| `DARP_PRICE_CORS_ORIGINS` | `*` |
| `DARP_PRICE_LOG_LEVEL` | `INFO` |

Example:

```bash
DARP_PRICE_CORS_ORIGINS="https://myapp.com" uvicorn app.main:app
```

### 10.7 Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "casablanca",
    "neighborhood": "maarif",
    "property_type": "appartement",
    "condition": "bon etat",
    "area": 100,
    "rooms": 3,
    "bathrooms": 2,
    "floor": 2,
    "age": 10,
    "amenities_count": 3
  }'
```

Response:

```json
{
  "predicted_price_MAD": 1392269,
  "predicted_price_per_sqm_MAD": 13923,
  "currency": "MAD",
  "model": "XGBoost"
}
```

---

## 11. Step 7 — Build a UI (`app/streamlit_app.py`)

Install the API and UI dependencies (the UI is a thin HTTP client):

```bash
pip install -r requirements-dev.txt -r requirements-streamlit.txt
```

Start the API in one terminal:

```bash
uvicorn app.main:app --reload
```

Start Streamlit in another:

```bash
streamlit run app/streamlit_app.py
```

The UI:

1. Fetches `/metadata` to populate dropdowns with real categories.
2. Sends the form to `/predict`.
3. Shows the estimated price, price per m², and the model name.
4. Shows model metrics and top features in an expander.

Set a custom API URL with:

```bash
DARP_PRICE_API_URL="https://my-api.example.com" streamlit run app/streamlit_app.py
```

Why a separate UI? Recruiters and non-technical users will not use `curl`.
A small Streamlit app makes the project tangible in 30 seconds.

---

## 12. Step 8 — Test everything

```bash
make test
# or
pytest --cov=app --cov=src --cov-report=term-missing
```

What the tests cover:

| File | What it checks |
|---|---|
| `test_prep_data.py` | Normalization, column contract, outlier removal, duplicate removal, numeric sanity |
| `test_sources.py` | Source 1 column fallback + amenity counting; source 2 sales-only filtering |
| `test_features.py` | Derived columns, zero-division safety, no input mutation |
| `test_train.py` | Pipeline fitting, metrics, feature importance, metadata, error reports |
| `test_api.py` | Health, metadata, model-info, prediction, normalization, batch, validation, timing header |

Why test data cleaning? Because cleaning bugs are silent. A test that a
`bureau` row is removed is worth more than a test that `1 + 1 == 2`.

Why test the API? Because the API is the contract with users. If the response
schema changes unexpectedly, tests catch it.

---

## 13. Step 9 — Containerize with Docker

Build and run:

```bash
make docker-build
make docker-run
# or run the full stack (API + Streamlit UI)
docker compose up --build
```

`docker compose up --build` starts two services: the API on `:8000` and the
Streamlit UI on `:8501` (it waits for the API healthcheck before starting).

The API `Dockerfile`:

1. Starts from `python:3.14-slim`.
2. Installs only runtime dependencies (`xgboost-cpu`, so no CUDA/NVIDIA bloat).
3. Copies `app/`, `src/`, `models/`, and `metrics.json`.
4. Creates a non-root `appuser`.
5. Adds a `HEALTHCHECK` that hits `/health`.
6. Runs `uvicorn` on `0.0.0.0:8000`.

`Dockerfile.streamlit` is a separate, smaller image that installs only
`streamlit` + `requests` and serves the UI on `:8501`.

Why copy `src/`? Because the pickled pipeline imports `src.features`. Without
it, loading the model fails.

Why non-root? Security best practice. If the app is compromised, the attacker
does not get root inside the container.

Why `HEALTHCHECK`? Orchestrators (Docker, Kubernetes, Render) can restart an
unhealthy container automatically.

To deploy the API, the repo ships [`render.yaml`](../render.yaml), a Render
Blueprint with `runtime: docker` and `healthCheckPath: /health`. Create a new
Blueprint on Render, point it at the repo, and it builds the `Dockerfile`.

---

## 14. Step 10 — Automate with CI/CD

### CI (`.github/workflows/ci.yml`)

Runs on every push and pull request:

1. Install Python 3.14.
2. Install `requirements-dev.txt`.
3. `pip check`.
4. `ruff check .`
5. `ruff format --check .`
6. `pytest --cov=app --cov=src`.
7. Build the Docker image.

If any step fails, the pull request is blocked. This is how you keep `main`
green.

### CD (`.github/workflows/cd.yml`)

On every push to `main` (and on version tags), it builds the Docker image and
pushes it to **GitHub Container Registry**:

```text
ghcr.io/mohamedait-abbou/darprice:latest
ghcr.io/mohamedait-abbou/darprice:sha-<commit>
```

Pull and run it anywhere Docker is available:

```bash
docker pull ghcr.io/mohamedait-abbou/darprice:latest
docker run -p 8000:8000 ghcr.io/mohamedait-abbou/darprice:latest
```

### Dependabot (`.github/dependabot.yml`)

Opens monthly pull requests for Python and GitHub Actions updates. Review them,
run CI, and merge.

---

## 15. Quick start

```bash
git clone https://github.com/mohamedait-abbou/DarPrice-Moroccan-House-Price-Prediction.git
cd DarPrice-Moroccan-House-Price-Prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# 1. Serve the committed model
uvicorn app.main:app --reload
# open http://localhost:8000/docs

# 2. Optional: retrain from the committed cleaned dataset
python -m src.train

# 3. Optional: rebuild the dataset from raw CSVs
#    (place the two CSVs in data/raw/ first)
python -m src.prep_data

# 4. Optional: Streamlit demo
pip install -r requirements-streamlit.txt
streamlit run app/streamlit_app.py

# 5. Tests + lint
make check
```

---

## 16. What was improved in this version

This version upgrades the original project in every layer.

### Data

- Source 1 now uses the explicit `Type` / `City` / `Nighberd` columns when they
  exist and falls back to address parsing.
- Amenity counting now includes `ascenseur`, `terrasse`, and `parking` for
  source 1.
- Numeric sanity rules for age, floor, rooms, and bathrooms.
- A printed **cleaning funnel** so no row disappears silently.
- Clear documentation of the duplicate-heavy source 1.

### Features and model

- New `FeatureEngineer` transformer (`area_per_room`, `rooms_plus_bath`,
  `amenities_per_100m2`, `is_new`) inside the pipeline.
- Shuffled 5-fold CV instead of the default ordered folds.
- Added `ExtraTreesRegressor` as a candidate.
- More XGBoost hyperparameters, including `min_child_weight`.
- `tree_method="hist"` for faster XGBoost training.
- Configurable parallelism via `DARP_PRICE_N_JOBS`.
- `--quick` mode for fast smoke tests.
- Full error analysis: per-city, per-property-type, worst predictions, MAPE,
  within-20%, error quantiles, and charts.
- `models/metadata.json` so the API/UI can validate inputs.

### API

- Input normalization at serve time (case + accents).
- `/metadata` and `/model-info` endpoints.
- `/predict/batch` endpoint.
- `predicted_price_per_sqm_MAD`, `currency`, and `model` in the response.
- `X-Process-Time-Ms` header.
- Configurable model/metadata/metrics paths and CORS origins.
- Better health response (model + metadata status).
- Version bumped to 2.0.0.

### UI and tooling

- Streamlit demo UI.
- `Makefile` with one command per task.
- Split requirements: runtime, training, dev, Streamlit.
- `pyproject.toml` now configures pytest `pythonpath`, coverage, and ruff.
- `pre-commit` hooks.
- Tests expanded from 12 to 33.
- CI now runs coverage and builds the Docker image.
- New CD workflow publishes the image to GHCR.
- Dependabot configuration.
- Docker image copies `src/` (needed for unpickling) and runs as non-root.
- New `docs/MODEL_CARD.md`, `CONTRIBUTING.md`, and this tutorial.

---

## 17. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run modules with `python -m src.train` (not `python src/train.py`), or run
pytest from the repository root. `pyproject.toml` sets `pythonpath = ["."]` so
plain `pytest` also works.

### `FileNotFoundError: models/darprice_model.joblib`

The model is committed, but if you deleted it, retrain:

```bash
python -m src.train
```

### `FileNotFoundError: data/processed/houses_combined.csv`

The processed dataset is committed. If it is missing, rebuild it from raw data:

```bash
python -m src.prep_data
```

### `Address already in use` (port 8000)

Another process is using the port:

```bash
uvicorn app.main:app --port 8001
```

### The API returns 503

The model failed to load. Check `/health`; it returns the exact error. Common
causes: missing model file, incompatible scikit-learn version, or a missing
`src/` folder when loading a pickled pipeline.

### Streamlit says it cannot reach the API

Start the API first, or set the URL:

```bash
DARP_PRICE_API_URL="http://localhost:8000" streamlit run app/streamlit_app.py
```

### XGBoost is slow or runs out of memory

Lower parallelism:

```bash
DARP_PRICE_N_JOBS=1 python -m src.train
```

### `ruff format --check` fails

Run:

```bash
ruff format .
```

### Docker build fails on `pip install`

Check that you are using a Python version compatible with the pinned wheels.
The project targets Python 3.12+ and was tested on 3.14.

---

## 18. Exercises

Try these to deepen your understanding:

1. **Add a feature.** Add `bathrooms_per_room` to `FeatureEngineer`, retrain,
   and compare CV MAE. Does it help?
2. **Add a city.** Append a new city to the raw data, rebuild, retrain, and
   check `/metadata` to see it appear.
3. **Try log(area).** Add `log_area` and see whether the tree models change.
4. **Add a new model.** Try `HistGradientBoostingRegressor` or
   `CatBoostRegressor`. Compare CV MAE.
5. **Add prediction intervals.** Train XGBoost with `objective="reg:quantileerror"`
   at quantiles 0.1 and 0.9, then return a range from the API.
6. **Add a `/predict/csv` endpoint.** Accept a CSV upload and return a CSV.
7. **Add temporal validation.** If you have `listing_date`, split by date instead
   of randomly.
8. **Add monitoring.** Log every request and prediction; compare the live input
   distribution to the training distribution.
9. **Add a model registry.** Store model versions and metrics, and let
   `/model-info` show the active version.
10. **Deploy it.** Put the Docker image on Render/Railway/Fly.io and add the live
    link to the README.

---

## 19. Deploy for free

### Option A — Render (Docker)

1. Push the repo to GitHub.
2. In Render, create a **Web Service** and connect the repo.
3. Choose **Docker** as the environment.
4. Render uses the `Dockerfile` automatically.
5. Add environment variables if needed (`DARP_PRICE_CORS_ORIGINS`).
6. Deploy. Render gives you a public URL.

### Option B — Railway

1. Create a new project from the GitHub repo.
2. Railway detects the Dockerfile.
3. Deploy and copy the public URL.

### Option C — Hugging Face Spaces (Docker)

1. Create a new Space with the **Docker** SDK.
2. Push the repo.
3. Make sure the app listens on port `7860` or set the port in the Space
   settings. You can override the Docker `CMD` with
   `uvicorn app.main:app --host 0.0.0.0 --port 7860`.

### Option D — Streamlit Community Cloud

Deploy `app/streamlit_app.py` and set `DARP_PRICE_API_URL` to your deployed API
URL in the Streamlit secrets/environment.

After deploying, add the live link to the README and your CV.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **Feature** | An input column used by the model (e.g., `area`). |
| **Target** | The value being predicted (`price`). |
| **Leakage** | A feature that contains information about the target that would not be available at prediction time. |
| **Pipeline** | A scikit-learn object that chains preprocessing and a model. |
| **ColumnTransformer** | Applies different preprocessing to different column groups. |
| **OneHotEncoder** | Turns a categorical column into binary indicator columns. |
| **Imputer** | Fills missing values (median for numbers, most frequent for categories). |
| **StandardScaler** | Centers and scales numeric features. |
| **Cross-validation (CV)** | Splits training data into folds to estimate generalization. |
| **MAE** | Mean absolute error, in the target's units (MAD). |
| **RMSE** | Root mean squared error; penalizes large errors more. |
| **R²** | Proportion of variance explained; 1.0 is perfect. |
| **MAPE** | Mean absolute percentage error. |
| **Log transform** | `log1p(price)` compresses the long right tail of prices. |
| **Train/serve skew** | When preprocessing differs between training and serving. |
| **Artifact** | A saved file produced by training (model, metrics, metadata). |
| **CI/CD** | Continuous Integration / Continuous Deployment. |
| **GHCR** | GitHub Container Registry. |

---

## Final checklist: build it from scratch

- [ ] Create a virtual environment and install dependencies.
- [ ] Put the raw CSVs in `data/raw/`.
- [ ] Run `python -m src.prep_data` and inspect the cleaning funnel.
- [ ] Open the cleaned CSV and check the schema.
- [ ] Run `python -m src.train` and read the CV table.
- [ ] Look at `metrics.json` and `reports/`.
- [ ] Run `uvicorn app.main:app --reload` and test `/docs`.
- [ ] Run the Streamlit UI.
- [ ] Run `make check`.
- [ ] Build the Docker image.
- [ ] Push to GitHub and watch CI pass.
- [ ] Deploy and share the live link.

You now have a complete, reproducible, end-to-end ML project. The best next step
is to change something — add a feature, a model, or a deployment target — and
see how the whole system reacts.
