"""DarPrice Streamlit demo.

A friendly UI on top of the FastAPI service.

Run the API first:
    uvicorn app.main:app --reload

Then run this app:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("DARP_PRICE_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="DarPrice — Moroccan House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data(ttl=60)
def fetch_json(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def human_mad(value: float | int) -> str:
    return f"{value:,.0f} MAD"


st.title("🏠 DarPrice")
st.caption("Estimate a Moroccan house price from its listing features.")

try:
    metadata = fetch_json("/metadata")
except Exception as exc:
    st.error(
        "Could not reach the DarPrice API at "
        f"`{API_URL}`.\n\nStart it with `uvicorn app.main:app --reload`.\n\n"
        f"Details: {exc}"
    )
    st.stop()

categories = metadata.get("categorical_values", {})
ranges = metadata.get("numeric_ranges", {})
model_name = metadata.get("model_name", "model")
metrics = metadata.get("metrics", {})

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** `{model_name}`")
    st.write(f"**Target:** `{metadata.get('target', 'log1p(price)')}`")
    st.metric("Test MAE", human_mad(metrics.get("mae", 0)))
    st.metric("Test R²", f"{metrics.get('r2', 0):.3f}")
    st.divider()
    st.write(f"**API:** `{API_URL}`")
    st.write(f"**Trained:** {metadata.get('trained_at_utc', 'unknown')}")
    st.caption("Predictions are estimates from historical listings, not a formal valuation.")


def _options(field: str, fallback: list[str]) -> list[str]:
    values = categories.get(field) or fallback
    return values


with st.form("prediction_form"):
    st.subheader("Property details")
    left, middle, right = st.columns(3)

    with left:
        city = st.selectbox("City", _options("city", ["casablanca"]))
        neighborhood = st.selectbox("Neighborhood", _options("neighborhood", ["maarif"]))
        property_type = st.selectbox("Property type", _options("property_type", ["appartement"]))

    with middle:
        condition = st.selectbox("Condition", _options("condition", ["bon etat"]))
        area = st.number_input(
            "Area (m²)",
            min_value=10.0,
            max_value=10_000.0,
            value=float(ranges.get("area", {}).get("median", 100.0)),
            step=5.0,
        )
        rooms = st.number_input("Rooms", min_value=0, max_value=50, value=3, step=1)

    with right:
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=50, value=2, step=1)
        floor = st.number_input("Floor", min_value=-10, max_value=200, value=2, step=1)
        age = st.number_input("Age (years)", min_value=0, max_value=200, value=10, step=1)

    amenities_count = st.slider("Number of amenities", min_value=0, max_value=50, value=3)
    submitted = st.form_submit_button("Estimate price", use_container_width=True)

if submitted:
    payload = {
        "city": city,
        "neighborhood": neighborhood,
        "property_type": property_type,
        "condition": condition,
        "area": float(area),
        "rooms": int(rooms),
        "bathrooms": int(bathrooms),
        "floor": int(floor),
        "age": int(age),
        "amenities_count": int(amenities_count),
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        price = result["predicted_price_MAD"]
        st.success(f"### Estimated price: {human_mad(price)}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total price", human_mad(price))
        col_b.metric(
            "Price per m²",
            human_mad(result.get("predicted_price_per_sqm_MAD") or price / area),
        )
        col_c.metric("Model", result.get("model") or model_name)
        st.caption("This is a statistical estimate. Verify with a local real-estate professional.")

with st.expander("Model performance and top features"):
    st.json(metrics, expanded=False)
    importances = metadata.get("feature_importance_top20", [])
    if importances:
        st.bar_chart({item["feature"]: item["importance"] for item in importances[:15]})

st.divider()
st.caption(
    "DarPrice · scikit-learn + XGBoost + FastAPI · "
    "Source code: github.com/mohamedait-abbou/DarPrice-Moroccan-House-Price-Prediction"
)
