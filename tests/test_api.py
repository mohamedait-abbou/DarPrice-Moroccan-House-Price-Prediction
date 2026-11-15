from fastapi.testclient import TestClient

from app.main import app, metadata, model

client = TestClient(app)

VALID_PAYLOAD = {
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
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "alive" in response.json()["message"]


def test_health_reports_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == ("ok" if model is not None else "degraded")
    assert body["model_loaded"] is (model is not None)
    assert body["metadata_loaded"] is (metadata is not None)


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "categorical_values" in body
    assert "numeric_ranges" in body
    assert "model_name" in body


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "test" in body or "metrics" in body


def test_predict_returns_positive_price():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    price = body["predicted_price_MAD"]
    assert isinstance(price, int)
    assert price > 0
    assert body["currency"] == "MAD"
    assert body["predicted_price_per_sqm_MAD"] == round(price / VALID_PAYLOAD["area"])


def test_predict_normalizes_city_case_and_accents():
    lower = client.post("/predict", json=VALID_PAYLOAD)
    mixed = client.post(
        "/predict",
        json={
            **VALID_PAYLOAD,
            "city": "Casablanca",
            "neighborhood": "Maarif",
            "property_type": "Appartement",
            "condition": "Bon Etat",
        },
    )
    assert lower.status_code == mixed.status_code == 200
    assert lower.json()["predicted_price_MAD"] == mixed.json()["predicted_price_MAD"]


def test_batch_predict():
    response = client.post(
        "/predict/batch",
        json={"houses": [VALID_PAYLOAD, {**VALID_PAYLOAD, "area": 200, "rooms": 5}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert all(item["predicted_price_MAD"] > 0 for item in body["predictions"])


def test_batch_rejects_empty_list():
    response = client.post("/predict/batch", json={"houses": []})
    assert response.status_code == 422


def test_predict_rejects_negative_area():
    response = client.post("/predict", json={**VALID_PAYLOAD, "area": -5})
    assert response.status_code == 422


def test_predict_rejects_missing_feature():
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "rooms"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_process_time_header():
    response = client.get("/")
    assert "X-Process-Time-Ms" in response.headers
