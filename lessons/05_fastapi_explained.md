# Lesson 5: FastAPI — Your Model Goes Online 🌍

Decoding `app/main.py` — the smallest file (35 lines!) but it makes your model
usable by ANYONE on the internet.

---

## 1. What is an API?

A restaurant 🍽️:
- You (the customer) don't enter the kitchen
- You give a **request** to the waiter: "pizza, no olives"
- The kitchen cooks, the waiter brings back a **response**

Your API:
- Customer sends: house details (JSON)
- Kitchen = your trained model predicts
- Response: `{"predicted_price_MAD": 1261077}`

## 2. The file, piece by piece

### Setup
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
model = joblib.load(ROOT / "models/darprice_randomforest.joblib")
```
⚠️ Notice: model loads **ONCE at startup**, not per request. Loading takes seconds; predicting takes milliseconds. Load slow things once.

```python
app = FastAPI(title="DarPrice API 🏠")
```
Creates the "restaurant". This `app` object is what uvicorn serves.

### The order form (pydantic)
```python
class House(BaseModel):
    city: str
    area: float
    rooms: float
    ...
```
This is a **class** — a mold/shape. Pydantic uses it as a BOUNCER 🚪:
- Missing field? → rejected automatically with a clear error
- `"area": "big"` instead of a number? → rejected automatically

You write ZERO validation code. That's pydantic's whole job.
(`transaction: str = "vente"` means: optional field, defaults to "vente" if not given.)

### Route 1 — home page
```python
@app.get("/")
def home():
    return {"message": "DarPrice API is alive 🏠🇲🇦"}
```
- `@app.get("/")` = decorator: "when someone VISITS `/`, run the function below and send back its result"
- GET = asking to read something

### Route 2 — prediction
```python
@app.post("/predict")
def predict(house: House):
    df = pd.DataFrame([house.model_dump()])
    price = model.predict(df)[0]
    return {"predicted_price_MAD": round(price)}
```

Line by line:
1. `house.model_dump()` → turn the House object into a plain dict
   `{"city": "casablanca", "area": 100, ...}`
2. `[dict]` → wrap in a list = ONE-row table. sklearn wants a TABLE, not one line.
3. `model.predict(df)` → returns an array of predictions like `[1261077.3]`
4. `[0]` → take the first (only) element
5. Return JSON. Python dict → JSON happens automatically ✨

POST = sending data IN (vs GET = just reading).

## 3. Running the restaurant

```bash
uvicorn app.main:app --reload
```
Read as: `file:object` → "serve the `app` object from `app/main.py`".
- `--reload` = auto-restart when you edit the file (development only!)

Then open **http://localhost:8000/docs** — FastAPI gives you a FREE interactive test page. Click POST /predict → "Try it out" → fill the form → Execute. No frontend needed!

## 4. Test without a browser too

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"casablanca","neighborhood":"maarif","property_type":"appartement","condition":"bon etat","area":100,"rooms":3,"bathrooms":2,"floor":2,"age":10,"amenities_count":3}'
```

## 5. Why this matters for jobs 💼

"ML model that lives in a notebook" = student project.
"ML model served behind an API anyone can call" = engineer project.
This little file is what makes DarPrice portfolio-worthy.

---

## ✅ QUIZ

1. Why load the model once at startup?
2. Client forgets to send `area`. What happens and WHO handles it?
3. What does `[0]` do in `model.predict(df)[0]`?
4. GET vs POST — which one for /predict and why?
5. What URL shows the free interactive testing page?

<details>
<summary>👉 Solutions</summary>

1. It's slow (seconds); doing it per request would make every prediction wait. Load once, predict fast forever.
2. Request rejected with automatic 422 error listing the missing field — pydantic's bouncer handles it; you wrote zero validation code.
3. predict() always returns a LIST (one entry per row). [0] grabs our single prediction.
4. POST: we are SENDING new data for processing, not fetching stored info.
5. http://localhost:8000/docs

</details>

---

**Next: Lesson 6 — practice EDITING your own project ✏️**
