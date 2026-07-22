# Serving image: the FastAPI scoring app (home_credit.serving.api) only -
# not dbt/training/the dashboard, which are dev-time / one-off tooling, not
# something a deployed scoring service needs at runtime.
#
# Installs the package's full dependency set from pyproject.toml (not the
# "dev" extras group - pytest/ruff/black aren't needed at runtime). A real
# production deployment would likely trim this further with a dedicated
# "serving" extras group excluding dbt/mlflow/streamlit; out of scope here.
#
# Models and the DuckDB warehouse are NOT baked into the image - both are
# gitignored, regenerated artifacts (`make train`, `make dbt-build`) that
# differ per environment/run - they're bind-mounted in at `docker run` time
# instead (see the `docker-run` Makefile target below).
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

ENV MODEL_DIR=/app/models \
    DUCKDB_PATH=/app/data/home_credit.duckdb \
    REPORTS_DIR=/app/reports

EXPOSE 8000

CMD ["uvicorn", "home_credit.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
