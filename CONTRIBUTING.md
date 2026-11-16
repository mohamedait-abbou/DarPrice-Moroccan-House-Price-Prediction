# Contributing to DarPrice

Thanks for taking a look! This project is meant to be a readable, end-to-end
example of a production-style ML system, so contributions that improve clarity,
correctness, or reproducibility are very welcome.

## Development setup

```bash
git clone https://github.com/mohamedait-abbou/DarPrice-Moroccan-House-Price-Prediction.git
cd DarPrice-Moroccan-House-Price-Prediction
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

The committed `data/processed/houses_combined.csv` and
`models/darprice_model.joblib` let you run the API and the tests without
downloading the raw data.

## Useful commands

```bash
make data       # rebuild the cleaned dataset from data/raw/
make train      # retrain + tune + evaluate + regenerate reports
make evaluate   # re-run error analysis for the saved model
make serve      # FastAPI at http://localhost:8000/docs
make docker-up  # API (:8000) + Streamlit UI (:8501) via docker compose
make test       # pytest + coverage
make lint       # ruff check
make format     # ruff format
make check      # exactly what CI runs
```

## Pull request checklist

- [ ] `make check` passes (lint, format, tests).
- [ ] New behavior has a test.
- [ ] No raw data, secrets, or personal learning notes are committed.
- [ ] If the model changes, `metrics.json`, `models/metadata.json`, and
      `reports/` are regenerated and committed.
- [ ] If the API contract changes, the README is updated.

## Commit style

Use short, imperative messages, for example:

```text
feat: add batch prediction endpoint
fix: normalize accented city names at serve time
docs: explain the log-target trick
```

## Data and privacy

Only commit the processed dataset and the trained artifacts. Do not commit
`data/raw/`, personal notes (`lessons/`, `expliction1.txt`), or `.env` files.
