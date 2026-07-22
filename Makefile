.PHONY: setup ingest dbt-build dbt-docs validate eda train explain serve-api dashboard test lint fmt docker-build docker-run

setup:  ## Install the package + dev tools (editable install)
	pip install -e ".[dev]"

ingest:  ## Load data/raw/*.csv into DuckDB's raw schema
	python -m home_credit.ingest.load_raw_data

dbt-build:  ## Load raw data, then run dbt build (staging -> intermediate -> marts) + tests + mart validation
	./scripts/build_warehouse.sh

dbt-docs:  ## Serve dbt docs locally
	cd warehouse && DBT_PROFILES_DIR=. dbt docs serve

validate:  ## Validate the built mart against its pandera schema (also runs at the start of `train`)
	python -m home_credit.validation.validate_mart

eda:  ## Regenerate docs/eda_summary.md from the built mart
	python -m home_credit.eda

train:  ## Run the full modelling pipeline (CV, calibration, MLflow logging)
	python -m home_credit.modeling.run_pipeline

explain:  ## Regenerate reports/shap_summary.png + shap_feature_importance.csv from the trained champion
	python -m home_credit.explain.shap_explainer

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

docker-run:  ## Run the serving image, mounting the pre-built model dir + DuckDB warehouse
	docker run --rm -p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/data:/app/data \
		home-credit-default-risk
