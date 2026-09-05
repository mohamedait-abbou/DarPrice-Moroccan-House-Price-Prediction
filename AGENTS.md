# DarPrice — Project Memory

This file gives an AI assistant the full project context at session start.
Read this before doing anything; it explains what has been done and what is next.

## Overview

End-to-end Machine Learning project that **predicts Moroccan house prices (MAD)**
from house features (city, neighborhood, property type, condition, area, rooms,
bathrooms, floor, age, amenities_count). Served as a **FastAPI** REST API.

- Data: 2 raw scraped CSVs (~120k rows) merged + cleaned → `data/processed/houses_combined.csv` (7,014 rows × 11 cols)
- Model: trained via scikit-learn `Pipeline` (ColumnTransformer: median imputer + StandardScaler for numeric, most_frequent imputer + OneHotEncoder for categorical)
- Saved model: `models/darprice_randomforest.joblib` (gitignored, generated locally)

## Tech Stack

Python · pandas · numpy · scikit-learn · XGBoost · joblib · FastAPI · pydantic · uvicorn

## IMPORTANT — target is log-transformed

We train on `np.log1p(price)` (price is heavily right-skewed: skew 1.66 → −0.14).
- Training target: `y_log = np.log1p(price)`
- To return real money: `price = float(np.expm1(log_price))` (already done in `app/main.py`)

## Done so far

- ✅ Git initialized, branch `main`, pushed to GitHub (SSH):
  `https://github.com/mohamedait-abbou/DarPrice-Moroccan-House-Price-Prediction`
- ✅ Professional README with metrics, structure, quick start
- ✅ Phase A model improvements in `src/train.py`:
  - Log-transform target
  - RMSE metric + 5-fold cross-validation
  - Added XGBoost (XGBRegressor) next to LinearRegression + RandomForest
  - Hyperparameter tuning via `RandomizedSearchCV` (RF + XGB)
- ✅ Final model = **Tuned XGBoost**: MAE 567,307 MAD, RMSE 896,779, R² 0.901
- ✅ `lessons/` folder created (teacher-style explanations of every change)

## Files

- `app/main.py` — FastAPI: GET `/` health check, POST `/predict` returns `{"predicted_price_MAD": N}`
- `src/prep_data.py` — merge 2 sources, clean, remove outliers/leakage (`price_per_sqm` leaks the answer), save clean CSV
- `src/train.py` — CV model comparison, tuning, held-out eval, saves best model
- `test_code.ipynb` — exploratory notebook
- `lessons/00..05_*.md` — personal study notes (see warning below)

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/prep_data.py        # raw CSVs -> clean CSV (data files are gitignored)
python src/train.py            # trains, prints CV + held-out MAE/RMSE/R2, saves model
uvicorn app.main:app --reload  # serve API at http://localhost:8000/docs
```

## Git rules

- `.gitignore` excludes: `venv/`, `data/raw/*.csv`, `data/raw/*.zip`, `data/processed/`,
  `models/*.joblib`, `lessons/`, `expliction1.txt`, `__pycache__/`
- **NEVER push** `lessons/` or `expliction1.txt` — they are personal learning material.
- Model + data files are generated locally; do not commit them.

## Next steps (not started)

### Phase B — code quality
- Unit tests for `prep_data.py`, `train.py`, and the API (`app/main.py`)
- `prep_data.py` bug: `clean(df)` takes a `df` arg but ignores it (always reloads sources)
- Pin exact versions in `requirements.txt` (currently loose `>=`)

### Phase C — visibility / portfolio
- Streamlit web UI (friendly demo)
- Dockerfile + containerization
- Deploy to Render / Railway (live link for CV/LinkedIn)

### Bonus
- Feature engineering (e.g., area/rooms ratio, price/m² bands)
- Error analysis — which listings does the model miss most?

## Reference — current model metrics

| Model | MAE (MAD) | RMSE (MAD) | R² |
|---|---|---|---|
| Linear Regression | 766,054 | — | 0.866 |
| Tuned Random Forest | 581,773 | 945,604 | 0.890 |
| **Tuned XGBoost (final)** | **567,307** | **896,779** | **0.901** |

Cross-val (log-space MAE): Linear 0.237 · RF 0.189 · XGB 0.184