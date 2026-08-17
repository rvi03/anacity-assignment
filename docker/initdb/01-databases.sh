#!/usr/bin/env bash
# Creates the databases the entrypoint does not.
#
# The Postgres entrypoint creates POSTGRES_DB only. Two more are needed:
#   mlflow         backend store for the tracking server
#   facility_test  target for the storage tests, so a test run can create,
#                  truncate, and drop tables without touching real data
#
# Runs once, on an empty data directory. Re-creating these later means
# `make down-v` first.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
	--username "${POSTGRES_USER}" \
	--dbname "${POSTGRES_DB}" <<-SQL
	CREATE DATABASE mlflow OWNER "${POSTGRES_USER}";
	CREATE DATABASE facility_test OWNER "${POSTGRES_USER}";
SQL
