# 🏠 DarPrice — Moroccan House Price Prediction

DarPrice is an end-to-end Machine Learning project that **predicts the price of a house in Morocco (MAD)** from its features: city, neighborhood, property type, condition, area, rooms, bathrooms, floor, age, and amenities.

It ships as a **FastAPI web service**: send house details, get a predicted price.

## 📊 Results

Trained on ~100k+ real listings merged and cleaned from 2 raw sources:

| Model | MAE | R² |
|---|---|---|
| Baseline (always predicts the mean) | ~1.7M MAD | 0.00 |
| Linear Regression | 766,054 MAD | 0.866 |
| **Random Forest (final model)** | **~585,105 MAD** | **0.902** ✅ |

**R² = 0.902** → the model explains 90% of what drives house prices.
Top feature the model learned: **area (m²) ≈ 86% of importance**, followed by property type (villa), age, and city.

## 🏭 How it works

The project uses a single scikit-learn `Pipeline` so training and prediction always apply the exact same cleaning:

1. **ColumnTransformer** — splits numeric vs. categorical columns
2. **Imputer + Scaler + OneHotEncoder** — fills missing values, scales numbers, encodes text
3. **RandomForestRegressor** — learns price patterns from 100 decision trees

## 📁 Project structure

```
DarPrice/
├── app/
│   └── main.py          # FastAPI serving the trained model at /predict
├── data/
│   ├── raw/             # original scraped datasets
│   └── processed/       # clean combined dataset (generated)
├── models/
│   └── darprice_randomforest.joblib   # saved pipeline (generated)
├── src/
│   ├── prep_data.py     # merge 2 sources + cleaning + outlier removal
│   └── train.py         # train, evaluate, save the model
├── test_code.ipynb      # exploratory notebook
├── requirements.txt
└── README.md
```

## 🚀 Quick start

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare the data (raw -> clean CSV)
python src/prep_data.py

# 3. Train the model (prints MAE / R² for both models)
python src/train.py

# 4. Serve predictions
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs and try `POST /predict`:

```json
{
  "city": "casablanca",
  "neighborhood": "maarif",
  "property_type": "appartement",
  "condition": "bon etat",
  "area": 100,
  "rooms": 3,
  "bathrooms": 2,
  "floor": 2,
  "age": 10,
  "amenities_count": 3,
  "transaction": "vente",
  "amenities": ""
}
```

Response:

```json
{ "predicted_price_MAD": 1513162 }
```

## 🛠️ Tech stack

Python · pandas · scikit-learn · FastAPI · joblib
