.PHONY: setup ingest dbt-build dbt-docs train serve-api dashboard test lint fmt docker-build docker-run

setup:  ## Install the package + dev tools (editable install)
	pip install -e ".[dev]"

ingest:  ## Load data/raw/*.csv into DuckDB's raw schema
	python -m home_credit.ingest.load_raw_data

dbt-build:  ## Load raw data, then run dbt build (staging -> intermediate -> marts) + tests
	./scripts/build_warehouse.sh

dbt-docs:  ## Serve dbt docs locally
	cd warehouse && DBT_PROFILES_DIR=. dbt docs serve

train:  ## Run the full modelling pipeline (CV, calibration, MLflow logging)
	python -m home_credit.modeling.run_pipeline

serve-api:  ## Run the FastAPI scoring endpoint
	uvicorn home_credit.serving.api:app --reload --port 8000

dashboard:  ## Run the Streamlit dashboard
	streamlit run dashboard/app.py

test:  ## Run the test suite
	pytest tests/ -v

lint:  ## Static checks (ruff + black --check)
	ruff check src tests
	black --check src tests

fmt:  ## Auto-format
	ruff check --fix src tests
	black src tests

docker-build:  ## Build the serving image
	docker build -t home-credit-default-risk .

docker-run:  ## Run the serving image, mounting a pre-built model dir
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models home-credit-default-risk
