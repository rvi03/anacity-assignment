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
make all           # generate → sample → split → features → baseline →
                   # train → evaluate → review        (~4m30s, CPU only)
```

`make all` writes `artifacts/predictions_review.xlsx` and the matching
CSV — the predicted-versus-actual table, one row per model per record.

```bash
make verify        # recompute every committed value and compare
make test          # the full suite
make llm-reproduce # replay the LLM answers; no GPU, no API key
make verify-rigour # generator profile, plots, error slices
make test-rigour   # leakage property test + shuffled-label control
make down          # stop; `make down-v` also drops the volumes
```

`uv sync` is the dependency claim. `pip install -e .` re-resolves against
looser constraints and will drift.

## Layout

```
data/          the synthetic dataset
src/           the package
  cli/         subcommands, and every output path declared once
  data/        generator, sampling, split, Postgres access
  features/    the modelling table and its leakage contract
  models/      the baseline rule and the four CatBoost heads
  evaluation/  metrics, freeze and seal, error slices, verifier
  llm/         the language-model track
tests/         mirrors src/
configs/       one seed drives the generator, the splits, and every fit
artifacts/     committed results: metrics, models, the review workbook
runs/          gitignored: adapter weights, rendered prompts, logs
```

Only `data/storage.py` opens a database connection, and only `cli/` may
import a single track's code. Both rules are asserted by a test that
reads the source, so they cannot quietly erode.

## Reproducibility

One `seed` in `configs/default.yaml` drives the generator, the splits,
and every fit. Identity is a canonical row digest, not a file hash,
because Postgres never promised stable page bytes. `make verify`
recomputes the digests, both splits' metrics, the workbook hash, and the
experiment tracker's headline metrics, and fails on any difference.

The holdout is scored once, ever. A freeze written before any holdout
row is read and a seal written after make that enforced rather than
intended; a second scoring attempt is refused.

The `mlflow` database is excluded from every comparison, and dropping it
still reproduces every deliverable.

## Results and documentation

Results are **not** repeated here, so there is one place to correct if a
number changes.

- [`docs/FINAL-REPORT.md`](docs/FINAL-REPORT.md) — what was built, what
  it scored, and what that means. Start here.
- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — data, features, leakage
  controls, modelling decisions, alternatives considered with measured
  outcomes, testing, results, and limitations.
