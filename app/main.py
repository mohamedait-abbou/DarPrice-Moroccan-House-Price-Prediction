from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
model = joblib.load(ROOT / "models/darprice_randomforest.joblib")

app = FastAPI(title="DarPrice API 🏠")

# the order ticket: what the customer must tell us
class House(BaseModel):
    city: str
    neighborhood: str
    property_type: str
    condition: str
    area: float
    rooms: float
    bathrooms: float
    floor: float
    age: float
    amenities_count: float
    transaction: str = "vente"
    amenities: str = ""

@app.get("/")
def home():
    return {"message": "DarPrice API is alive 🏠🇲🇦"}

@app.post("/predict")
def predict(house: House):
    df = pd.DataFrame([house.model_dump()])
    price = model.predict(df)[0]
    return {"predicted_price_MAD": round(price)}