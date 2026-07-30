.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON := .venv/bin/python
UV     := uv

.PHONY: help setup lint format typecheck test test-unit test-integration test-e2e \
        data dataset train evaluate api dashboard mlflow pipeline up down logs clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv (Python 3.12) and install the project with dev extras
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev,telemetry,kaggle]"

lock: ## Freeze the current environment into requirements.lock (used by the Docker build)
	$(UV) pip freeze --python $(PYTHON) --exclude-editable > requirements.lock

lint: ## Ruff + Black (check only) + mypy
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts
	$(PYTHON) -m mypy

format: ## Auto-fix with Ruff and format with Black
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m black src tests scripts

typecheck: ## mypy only
	$(PYTHON) -m mypy

test: ## Run the full test suite (excluding tests that need the network)
	$(PYTHON) -m pytest -m "not network"

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest -m unit

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest -m integration

test-e2e: ## Run end-to-end pipeline tests
	$(PYTHON) -m pytest -m e2e

coverage: ## Run tests with a coverage report
	$(PYTHON) -m pytest -m "not network" --cov --cov-report=term-missing --cov-report=xml

data: ## Bootstrap the database and apply incremental updates
	$(PYTHON) scripts/update_data.py

dataset: ## Build the modelling dataset
	$(PYTHON) scripts/build_dataset.py

train: ## Train the model zoo and register the best model
	$(PYTHON) scripts/train.py

evaluate: ## Evaluate the active model and generate explainability artefacts
	$(PYTHON) scripts/evaluate.py

pipeline: data dataset train evaluate ## Run the whole pipeline end to end

api: ## Run the FastAPI service with auto-reload
	$(PYTHON) scripts/serve_api.py --reload

dashboard: ## Run the Streamlit dashboard
	$(PYTHON) scripts/run_dashboard.py

mlflow: ## Run a local MLflow UI against the file store
	$(PYTHON) -m mlflow ui --backend-store-uri file:./data/mlruns --port 5000

up: ## Start the full stack (postgres, mlflow, api, dashboard)
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

logs: ## Tail the stack logs
	docker compose logs -f

docker-pipeline: ## Bootstrap, build the dataset, train and evaluate inside the container stack
	docker compose --profile pipeline run --rm pipeline python scripts/update_data.py --bootstrap
	docker compose --profile pipeline run --rm pipeline python scripts/build_dataset.py
	docker compose --profile pipeline run --rm pipeline python scripts/train.py
	docker compose --profile pipeline run --rm pipeline python scripts/evaluate.py

clean: ## Remove caches and build artefacts (keeps ingested data)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
