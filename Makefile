UV ?= uv
COMPOSE ?= docker compose -f docker/compose.yaml

.PHONY: all check lint typecheck test test-llm llm-reproduce verify-rigour test-rigour fmt up down down-v logs ps

## all: the whole pipeline, end to end, into the artifacts directory
all: up
	$(UV) run python -m facility_prediction.cli pipeline

## check: everything a step must pass before it is done
check: lint typecheck test

## lint: format check + lint, no writes
lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

## typecheck: static types, blocking
typecheck:
	$(UV) run mypy

## test: the full suite
test: up
	$(UV) run pytest

## test-llm: the LLM track's own tests; no services needed
test-llm:
	$(UV) run pytest tests/llm

## llm-reproduce: replay the LLM result from its committed answers.
## No GPU, no adapter load, no API key — the cache is the mechanism.
llm-reproduce: up
	$(UV) run python -m facility_prediction.cli llm-final --from-cache
	$(UV) run python -m facility_prediction.cli llm-review
	$(UV) run pytest tests/llm/test_replay.py

## verify: recompute every committed value and compare
verify: up
	$(UV) run python -m facility_prediction.cli verify

## register: register the frozen heads and query every track at once
register: up
	$(UV) run python -m facility_prediction.cli register

## tune: post-freeze stretch variants, validation only, budget-bounded
tune: up
	$(UV) run python -m facility_prediction.cli tune

## verify-rigour: extended distributions, plots, and error slices
verify-rigour: up
	$(UV) run python -m facility_prediction.cli profile
	$(UV) run python -m facility_prediction.cli slices
	$(UV) run pytest tests/test_profiles.py tests/test_errors.py

## test-rigour: the future-perturbation and shuffled-label controls
test-rigour: up
	$(UV) run pytest tests/test_leakage_evidence.py

## fmt: apply formatting and safe lint fixes
fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

## up: start postgres + mlflow, wait until healthy, apply migrations; idempotent
up: .env
	$(COMPOSE) up -d --wait
	$(UV) run alembic upgrade head

# The service stack and the application read the same file, so a first run
# seeds it from the committed example rather than failing on a missing value.
.env: .env.example
	cp $< $@

## down: stop the services, keeping the data
down:
	$(COMPOSE) down

## down-v: stop the services and delete the data volumes
down-v:
	$(COMPOSE) down --volumes

## logs: follow service logs
logs:
	$(COMPOSE) logs -f

## ps: show service status
ps:
	$(COMPOSE) ps
