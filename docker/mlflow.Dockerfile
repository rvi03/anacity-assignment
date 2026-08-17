# The published MLflow image ships without a PostgreSQL driver, so a server
# started against a postgresql:// backend store fails at connect time with a
# missing-module error rather than anything that names the real cause.
# See https://github.com/mlflow/mlflow/issues/9513.
#
# MLflow's own SQLAlchemy usage is on the psycopg2 dialect; application code
# uses psycopg 3. The service's dependency graph is not ours to modernise.
FROM ghcr.io/mlflow/mlflow:v3.15.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.10
