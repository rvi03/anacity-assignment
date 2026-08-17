# Facility usage prediction — review document

Everything a reviewer needs, in one place. The only other thing to open
is the results page, `artifacts/predictions_review.html`.

---

## 1. Start here

```bash
uv sync            # install the exact dependencies from uv.lock
make up            # start Postgres and MLflow, apply migrations
make all           # generate → features → fit the four CatBoost heads →
                   # score → write results          (~4.5 min, CPU only)
make verify        # recompute every number in this document
```

Then open **`artifacts/predictions_review.html`** — a filterable page
with four tabs: the headline scores, every record beside what actually
happened, what each model is, and how to read the numbers.

Two more commands, both optional:

```bash
make test          # 829 tests
make llm-reproduce # replay the language-model result; no GPU, no API key
```

**`make all` does not train the language model.** It runs the ML
pipeline only, and 4.5 minutes is what that costs on a CPU. The language
model took **2 h 14 m** to train on Apple Silicon and is a separate
command; `make llm-reproduce` replays its recorded answers in seconds
instead, which is why no GPU is needed to check its result.

| | Cost | Command |
|---|---|---|
| Whole ML pipeline, from scratch | ~4.5 min, any CPU | `make all` |
| Replay the language model's result | seconds | `make llm-reproduce` |
| Retrain the language model | ~2 h 14 m, Apple Silicon | `python -m facility_prediction.cli llm-tier-b` |
| Re-score it after retraining | ~40 min | `python -m facility_prediction.cli llm-final` |

`make verify` is the one that matters. It recomputes the input digests,
the configuration hash, both splits' metrics and the results hash, and
**fails on any difference** rather than repairing it.

---

## 2. The problem, and how it is scored

For a resident at a moment in time, predict their next booking:

| Prediction | Counts as correct when |
|---|---|
| Facility — which of 8 | it names the exact facility |
| Weekday | it names the exact day |
| Hour | it names the exact hour |
| Notification — how long until they book | it lands within **±25%** of the real delay |

The first three are exact-match, because there is no such thing as
nearly the gym. The fourth allows a margin because it is a question
about magnitude, not about a minute.

**The score is the mean of those four rates.** The same number chose the
model and leads this document, so no model can be selected on one
measure and reported with a more flattering one.

---

## 3. Results

**Full holdout — 3,596 chronologically later records, scored once.**

| | Best of both | CatBoost alone | Habit rule |
|---|---:|---:|---:|
| Facility | 0.3757 | 0.3757 | **0.3765** |
| Weekday | 0.2845 | 0.2764 | 0.2845 |
| Hour | 0.2036 | 0.2036 | **0.2041** |
| Notification | **0.1204** | 0.1204 | 0.1190 |
| **Overall** | **0.2460** | 0.2440 | **0.2460** |

**The machine-learning result ties the simple rule. It does not beat
it.** The tie is exact and real: the rule wins 3 facilities and 2 hours,
the model wins 5 notifications, and they cancel. CatBoost on its own
*loses*, 0.2440 to 0.2460.

**A second finding, and the more useful one.** On validation — data we
could re-check — the model led 0.2401 to 0.2320. That lead did not
survive to data it had never seen. This is what tuning against a split
looks like when you then measure honestly.

**All four models on the same 500 records** (the shared comparison set,
which is the only set the language model answered):

| Model | Overall | Facility | Weekday | Hour | Notification |
|---|---:|---:|---:|---:|---:|
| Best of both | **0.2545** | 0.3760 | 0.3080 | 0.2280 | 0.1060 |
| CatBoost alone | 0.2540 | 0.3760 | 0.3060 | 0.2280 | 0.1060 |
| Habit rule | 0.2510 | 0.3800 | 0.3080 | 0.2040 | 0.1120 |
| Fine-tuned language model | 0.2185 | 0.3120 | 0.2800 | 0.2100 | 0.0720 |

The language model's deficit is −0.0325 with a paired 95% interval of
[−0.0525, −0.0130]. The interval excludes zero, so it is a real loss and
not sampling noise.

### Why the ceiling is this low

Most of what is predictable here is that residents repeat themselves,
and the habit rule already captures it. A deliberately unfair
benchmark — a lookup allowed to see each resident's entire history
*including the future* — reaches only 0.3791 / 0.2732 / 0.2040 on
facility / weekday / hour. The trained models match or exceed that, so
they already sit at the ceiling of what habit alone can tell you.

Booking delay is close to a dead end for any model. Within a resident,
the gap to the next booking carries almost no memory of the previous gap
(coefficient of variation 1.09):

```
  answering the same delay every time    0.111
  knowing each resident's true rate      0.163
                                         └────┘  ~5 points, the whole band
```

### Where it fails

| Slice | Score | Compared with |
|---|---:|---|
| Less popular half of the catalog | 0.2877 | 0.4038 popular half |
| Weekday bookings | 0.2197 | 0.2692 weekends |
| Residents seen in training | 0.2396 | 0.2593 never seen |

Cold start is **not** the weakness — unseen residents score slightly
better. The largest single error mode is a drift event: a Yoga Room
opens partway through the two years and draws demand from the Gym, and
the model keeps answering Gym. That one confusion is 164 records.

---

## 4. The data

500 residents, 24 months, 8 facilities, **21,442 bookings**, from a
seeded generator. It hides structure the models must rediscover:
per-resident archetypes and preferences, facility popularity, capacity
limits, and three dated drift events — a facility that opens late and
takes another's demand, a group of residents who change habits, and one
facility with a seasonal swing. **18 acceptance checks** assert each
property is actually present, and they run inside `make all` rather than
only in tests.

Each modelling row is a **rolling origin**: the moment a resident made a
booking, predicting their next one. That gives **20,948 rows** from
21,442 bookings; residents with no prior history are excluded and
counted, not dropped silently. The split is chronological, 70/15/15, so
the holdout is strictly later than everything fitted on.

---

## 5. Feature choices

**161 columns** — 7 categorical, 154 numeric — in three families.

| Family | What it carries | Why |
|---|---|---|
| Origin | calendar position of the moment itself | the same resident behaves differently on a Monday morning and a Saturday evening |
| Resident history | counts and rates over 7/30/90/180-day windows; rolling favourite over the last 3/5/10 bookings; time-band shares; inter-booking intervals; preference decayed at a 30-day half-life; entropy; transition counts | this is where nearly all the signal is, so habit is described at several timescales rather than one — a short window tracks a change, a long one resists noise |
| Community history | what everyone else did over 30 and 90-day windows ending strictly before the origin | drift is a community effect: the Yoga Room opening changes behaviour no resident's own history predicts |

Two deliberate choices worth naming:

- **Multiple window lengths rather than one.** The right lookback is not
  knowable in advance, and a drift event makes short and long windows
  disagree in a way the model can use.
- **Missing categoricals carry an explicit `__MISSING__` token**, not a
  blank. "No history yet" is then a value the model can learn from
  instead of a hole.

The definitions are in `src/facility_prediction/features/features.py`.

---

## 6. Modelling approach

Four models, plus a rule to beat and one composition of them.

**The habit rule (baseline).** Predicts what the resident does most
often and most recently, with an order-1 transition on their last
facility, falling back to community behaviour where a resident has no
history. Not a token comparison — it is hard to beat, and it is the bar.

**Four CatBoost heads**, one per component, over the same feature table:

| Output | Estimator | Rounds kept |
|---|---|---:|
| Facility | 8-class classifier | 144 |
| Weekday | 7-class classifier | 100 |
| Hour | 24-class classifier | 18 |
| Notification | delay-**range** classifier | 129 |

Two decisions carry the modelling weight:

- **Notification classifies, it does not regress.** As a regression it
  was the worst part of the system: 0.095, below the 0.111 a single
  global constant achieves. Minimising minute-space error optimises a
  different quantity from a ±25% window, and on a right-skewed delay
  distribution its optimum sits where almost no window is hit. It now
  predicts a range narrow enough to fit inside the tolerance, and
  decodes by heaviest *window* rather than heaviest range — 0.131 on
  validation, above the baseline's 0.127.
- **Training length is searched, not fixed.** One round count shared by
  heads separating 7, 8 and 24 classes is a guess. Each head cuts the
  latest 15% of its *training* rows as an inner fold, takes the best
  count there, then refits on all training rows. No validation or
  holdout row takes part.

**"Best of both."** Per component, whichever source scored better on
validation — chosen before the holdout was opened, then frozen. Nothing
is blended; each answer comes whole from one source, and both sources
stay separately reported.

```
  facility ──> CatBoost      weekday ──> habit rule
  hour     ──> CatBoost      notify  ──> CatBoost
```

**The fine-tuned language model.** A 4-bit Qwen3-4B with a LoRA adapter,
run locally, answering all four components in one schema-constrained
JSON object. It cannot invent a facility or a label outside the allowed
set. Delay is discretised into 17 ranges sized so that naming the right
range always counts as a match; that costs a measured ceiling of 0.9545
rather than 1.0. Trained once — 1,000 rows, 2 h 14 m on a laptop — at a
size fixed on timing alone, before any quality number existed.

It lost, and the reason is visible: across 500 records it used **2 of
its 17 ranges**, so it had effectively learned one answer for everyone.
Training it harder made it worse (0.1875 against 0.2188). Two candidate
explanations — the label form and untuned learning rate — are recorded
as untested guesses, not findings.

---

## 7. Alternatives considered

| Considered | Outcome |
|---|---|
| Notification as MAE/RMSE regression, raw and log target | **Measured and rejected**: 0.054–0.072, below a global constant |
| `CatBoost SurvivalAft` on the delay | **Measured and rejected**, same range |
| Facility as learning-to-rank (`YetiRank`) | Built and configurable; 0.3561–0.3614 against 0.3526 — **within noise** (SE ≈ 0.008), ships off |
| Notification framed around each resident's own cadence | Built and configurable; ships off, reported as a declared alternative |
| Exponential recency weights on preference | Measured and rejected — worse at every half-life tried |
| Reconstructing usage time from gap plus lead | Measured and rejected: weekday 0.1365, hour 0.0361 |
| Order-1 Markov as the primary facility rule | Measured and rejected: 0.3228 against 0.3524 for a plain personal mode |
| `has_time=True` on CatBoost | Measured and rejected: facility 0.3518 → 0.3486 |
| Sequential recommenders (SASRec, BERT4Rec) | **Deliberately skipped and recorded as skipped** — defensible on paper, both break the "a reviewer runs this in minutes" constraint |
| Neural temporal point processes | Same |

---

## 8. Leakage controls

The part a reviewer should be most suspicious of, so it carries the most
machinery. Six numbered controls, each with tests that fail if the
control is removed:

| | Control | How it is enforced |
|---|---|---|
| L1 | Resident history ≤ origin; target > origin | asserted **per row, every run** — 20,948 rows over 903,547 events |
| L2 | No target or target-derived column is an input | manifest diffed against a denylist, 0 violations over 161 columns |
| L3 | Community windows stop strictly before the origin | `closed='left'` asserted on every rolling call, over 220,232,404 events |
| L4 | The split is chronological and cannot overlap | boundary assertions |
| L7 | Split membership is frozen and cannot reach a fit | fit-call audit: 13,915 rows, all training |
| L8 | Generator hidden state never reaches the model table | column-set assertion |

L6 governs chained models, which do not exist here. It is absent and
nothing is claimed about it.

Two further checks look for the **effect** of a leak rather than
asserting a rule:

- **Future perturbation** — change or delete bookings after a row's
  origin; every feature of that row must be byte-identical. Cases are
  generated by Hypothesis over timezone-aware instants.
- **Shuffled-label control** — break the link between features and
  target, fit, and score on unseen rows. No head may beat its own
  majority-class prior by more than sampling noise.

Both carry a **sensitivity check**, because a blind instrument passes
everything: the perturbation test proves the same edit *does* move a row
it is allowed to move, and the negative control proves it fires when a
target is planted in the feature matrix.

**The holdout is scored once, ever.** A freeze written before any
holdout row is read, and a seal written after, make that enforced rather
than intended. A second scoring attempt is refused.

---

## 9. Testing

**829 tests.** The categories matter more than the count: a unit test
for every metric with hand-checked fixtures including boundary and
zero-denominator cases; the leakage controls; storage and layering; the
two effect-based verifications; determinism; and tests that deliberately
break each consistency gate to prove it fires. 22 of them break one
committed value each and confirm `make verify` catches it.

Two architectural rules are asserted by reading the source, not by
running it: only `data/storage.py` may open a database connection, and
only the `cli/` package may import a single track's code.

---

## 10. Limitations

- **The machine-learning result ties the baseline on unseen data.**
  Everything above is reported against that, not around it.
- **The dataset is synthetic.** Every result is a statement about a
  simulation whose structure was chosen, not about a real community.
- **The notification component is quantised**, so part of its ceiling is
  structural rather than modelling.
- **One holdout scoring** means no confidence interval from repeated
  evaluation. The intervals quoted are bootstraps over rows, not runs.
- **The language model is one configuration, trained once.** No sweep
  over learning rate, rank, or label form. This is a result about that
  setup, not about what a fine-tuned 4B model could do here.
- **Leakage is proven for the feature path, asserted for the prompt
  path.** The property test and negative control cover the shared
  features; the language-model branch enforces ordering and carries no
  target, which is an enforced procedure rather than a proof.
- **The generator has a measured bias**, reported rather than tuned
  away: Clubhouse realises 0.1196 against 0.09 configured, and the
  bootstrap interval excludes the configured value.

---

## 11. Where things are

```
data/synthetic_bookings.csv          the dataset
artifacts/predictions_review.html    the results page — open this
artifacts/predictions_review.xlsx    same rows, workbook form
artifacts/predictions_review.csv     same rows again
artifacts/models/*.cbm               the four trained heads
artifacts/metrics.json               every recorded number
artifacts/freeze.json + holdout_seal.json   scored-once enforcement

src/facility_prediction/
  cli/          subcommands; every output path declared once
  data/         generator, sampling, split, the only Postgres access
  features/     the modelling table and its leakage contract
  models/       the habit rule and the four heads
  evaluation/   metrics, freeze and seal, error slices, verifier
  llm/          the language-model track
tests/          mirrors src/
```

Reproducibility rests on one `seed` in `configs/default.yaml`, which
drives the generator, the splits and every fit. Identity is a canonical
row digest rather than a file hash, because Postgres never promised
stable page bytes.
