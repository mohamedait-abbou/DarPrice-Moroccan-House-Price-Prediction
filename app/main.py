"""DarPrice FastAPI service.

Loads the trained sklearn Pipeline and predicts a house price in MAD.

Run with:
    uvicorn app.main:app --reload

Environment variables:
    DARP_PRICE_MODEL_PATH       path to the joblib pipeline
    DARP_PRICE_METADATA_PATH    path to models/metadata.json
    DARP_PRICE_METRICS_PATH     path to metrics.json
    DARP_PRICE_CORS_ORIGINS     comma-separated origins (default: *)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from src.prep_data import normalize_text_value
except ImportError:  # allows running the module from unusual working directories

    def normalize_text_value(value: object) -> str:
        text = "" if value is None else str(value).strip().lower()
        return text


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("DARP_PRICE_MODEL_PATH", ROOT / "models/darprice_model.joblib"))
METADATA_PATH = Path(os.getenv("DARP_PRICE_METADATA_PATH", ROOT / "models/metadata.json"))
METRICS_PATH = Path(os.getenv("DARP_PRICE_METRICS_PATH", ROOT / "metrics.json"))

VERSION = "2.0.0"

logger = logging.getLogger("darprice")
logging.basicConfig(
    level=os.getenv("DARP_PRICE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(
    title="DarPrice API",
    description=(
        "Predicts Moroccan house prices (MAD) from listing features. "
        "The model is a full scikit-learn Pipeline, so preprocessing is identical "
        "in training and serving."
    ),
    version=VERSION,
)

_origins = [origin.strip() for origin in os.getenv("DARP_PRICE_CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        logger.warning("Artifact not found: %s", path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read %s: %s", path, exc)
    return None


model = None
model_error: str | None = None
metadata: dict | None = _load_json(METADATA_PATH)
metrics: dict | None = _load_json(METRICS_PATH)

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
except Exception as exc:  # keep the API up so /health can explain the failure
    model_error = f"{type(exc).__name__}: {exc}"
    logger.error("Could not load model: %s", model_error)


class House(BaseModel):
    city: str = Field(..., min_length=1, examples=["casablanca"])
    neighborhood: str = Field(..., min_length=1, examples=["maarif"])
    property_type: str = Field(..., min_length=1, examples=["appartement"])
    condition: str = Field(..., min_length=1, examples=["bon etat"])
    area: float = Field(..., gt=0, le=10_000, examples=[100])
    rooms: int = Field(..., ge=0, le=50, examples=[3])
    bathrooms: int = Field(..., ge=0, le=50, examples=[2])
    floor: int = Field(..., ge=-10, le=200, examples=[2])
    age: int = Field(..., ge=0, le=200, examples=[10])
    amenities_count: int = Field(..., ge=0, le=50, examples=[3])

    @field_validator("city", "neighborhood", "property_type", "condition", mode="before")
    @classmethod
    def normalize_categories(cls, value: object) -> str:
        """Match the training normalization: lowercase + no accents."""
        return normalize_text_value(value)


class Prediction(BaseModel):
    predicted_price_MAD: int
    predicted_price_per_sqm_MAD: int | None = None
    currency: str = "MAD"
    model: str | None = None


class BatchRequest(BaseModel):
    houses: list[House] = Field(..., min_length=1, max_length=100)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    count: int


class Health(BaseModel):
    status: str
    model_loaded: bool
    metadata_loaded: bool
    model_path: str
    model_error: str | None = None
    version: str


def _model_name() -> str | None:
    if metadata:
        return metadata.get("model_name")
    return None


def _predict_one(house: House) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {model_error}")

    features = pd.DataFrame([house.model_dump()])
    log_price = float(model.predict(features)[0])
    price = float(np.expm1(log_price))
    price_per_sqm = price / house.area if house.area > 0 else None

    return Prediction(
        predicted_price_MAD=round(price),
        predicted_price_per_sqm_MAD=round(price_per_sqm) if price_per_sqm else None,
        currency="MAD",
        model=_model_name(),
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/")
def home() -> dict:
    return {
        "service": "DarPrice API",
        "version": VERSION,
        "message": "DarPrice API is alive",
        "docs": "/docs",
        "endpoints": ["/health", "/metadata", "/model-info", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        metadata_loaded=metadata is not None,
        model_path=str(MODEL_PATH),
        model_error=model_error,
        version=VERSION,
    )


@app.get("/metadata")
def get_metadata() -> dict:
    """Return valid categories and numeric ranges for building a UI form."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata unavailable")
    return metadata


@app.get("/model-info")
def get_model_info() -> dict:
    """Return training metrics, best params, and feature importance."""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics unavailable")
    return metrics


@app.post("/predict", response_model=Prediction)
def predict(house: House) -> Prediction:
    return _predict_one(house)


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(payload: BatchRequest) -> BatchPrediction:
    predictions = [_predict_one(house) for house in payload.houses]
    return BatchPrediction(predictions=predictions, count=len(predictions))
