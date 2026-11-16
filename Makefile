.PHONY: help install install-dev install-streamlit data train evaluate serve streamlit test lint format check docker-build docker-run docker-build-ui docker-up docker-down clean

PYTHON ?= python
UVICORN ?= uvicorn

help:  ## Show this help
	@echo "DarPrice - common commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install training + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

install-streamlit:  ## Install the optional Streamlit demo
	$(PYTHON) -m pip install -r requirements-streamlit.txt

data:  ## Rebuild data/processed/houses_combined.csv from raw CSVs
	$(PYTHON) -m src.prep_data

train:  ## Train, tune, evaluate and save the model + reports
	$(PYTHON) -m src.train

evaluate:  ## Re-run error analysis for the saved model
	$(PYTHON) -m src.evaluate

serve:  ## Run the FastAPI service with auto-reload
	$(UVICORN) app.main:app --reload

streamlit:  ## Run the Streamlit demo (API must be running)
	streamlit run app/streamlit_app.py

test:  ## Run tests with coverage
	$(PYTHON) -m pytest --cov=app --cov=src --cov-report=term-missing

lint:  ## Check code style
	ruff check .

format:  ## Auto-format code
	ruff format .

check:  ## Run everything CI runs
	ruff check .
	ruff format --check .
	$(PYTHON) -m pytest --cov=app --cov=src --cov-report=term-missing

docker-build:  ## Build the production API Docker image
	docker build -t darprice:local .

docker-run:  ## Run the production API Docker image
	docker run --rm -p 8000:8000 darprice:local

docker-build-ui:  ## Build the Streamlit UI Docker image
	docker build -f Dockerfile.streamlit -t darprice-ui:local .

docker-up:  ## Run API + Streamlit UI with docker compose
	docker compose up --build

docker-down:  ## Stop and remove the compose stack
	docker compose down

clean:  ## Remove caches and generated charts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -f reports/*.png
