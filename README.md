# Facility usage prediction

Given a resident and a moment in time, predict four things about their
next facility booking: **which facility**, **which weekday**, **which
hour**, and **how long after that moment they will book**.

Everything runs on a seeded synthetic dataset that ships with the repo
and regenerates bit-for-bit from configuration.

## Run it

Docker is a prerequisite: Postgres is the store of record.

```bash
git clone <repo> && cd <repo>
uv sync            # exact dependency graph from uv.lock
make up            # postgres + mlflow, waits for health, applies migrations
make all           # the ML pipeline             (~4m30s, CPU only)
make verify        # recompute every committed value and compare
```

Then open **`artifacts/predictions_review.html`** — the predicted-versus-
actual results, filterable, no server needed. The same rows are also
written as `.xlsx` and `.csv`.

```bash
make test          # the full suite
make review-ui     # rebuild just the results page
make llm-reproduce # replay the LLM answers; no GPU, no API key
make verify-rigour # generator profile, plots, error slices
make test-rigour   # leakage property test + shuffled-label control
make down          # stop; `make down-v` also drops the volumes
```

`make all` covers the ML track only — it does not train the language
model. That took 2h14m on Apple Silicon and is a separate command;
`make llm-reproduce` replays its recorded answers instead, which is why
no GPU is needed to check its result.

`uv sync` is the dependency claim. `pip install -e .` re-resolves against
looser constraints and will drift.

## Where to read about it

**[`docs/REVIEW.md`](docs/REVIEW.md)** — one document covering the
problem, the data, the feature choices, the modelling approach,
alternatives considered with measured outcomes, the leakage controls,
testing, results, limitations, and a map of the repository.

Results appear there and on the results page, and nowhere else, so there
is one place to correct if a number changes.
