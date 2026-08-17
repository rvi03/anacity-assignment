# Final report: predicting the next facility booking

## The short version

We evaluated three approaches to predict a resident's next facility
booking:

- A simple rule based on a resident's past behaviour.
- Four CatBoost machine-learning models, one for each prediction.
- A fine-tuned, locally run language model.

We also report one fixed combination of the baseline and CatBoost outputs.
It was chosen on validation data before the final holdout was scored.

| Comparison | Records | Result |
|---|---:|---|
| Selected combination vs baseline | 3,596 | Both scored **0.2460** overall. The selected combination did not beat the baseline. |
| Language model vs baseline | 500 | The language model scored **0.2185**; the baseline scored **0.2510**. The language model came last. |

The useful result is not a narrow model win. It is the honest finding that
resident habits are predictable only to a limited extent, and that a simple
frequency-and-recency rule captures most of the available signal.

This submission contains the source code, the synthetic dataset, the
four trained CatBoost model files, and the predicted-versus-actual
workbook. A fixed seed regenerates every reported result from scratch.
The fine-tuned language model's weights are 208 MB and are not included;
its recorded answers are, and they reproduce its every reported number
without a GPU.

---

## 1. What the system predicts

At any point in time, the system predicts a resident's next booking:

| Prediction | A correct prediction means |
|---|---|
| Facility | The exact facility is correct. |
| Weekday | The exact day of the week is correct. |
| Hour | The exact hour is correct. |
| Notification time | The predicted delay is within 25% of the actual delay. |

The overall score is the average of these four match rates. We used that
same score to choose the model before looking at the final holdout data.
This prevents choosing a model by one measure and presenting it with a
more flattering one.

## 2. Approaches and training

Each approach receives the same kind of booking history and produces the
same four predictions.

```text
┌─────────────────┐    ┌───────────────┐    ┌──────────────────────────────┐
│ booking history │───>│ baseline rule │───>│ facility · weekday · hour ·   │
└─────────────────┘    └───────────────┘    │ notification-time prediction │
                                             └──────────────────────────────┘

┌─────────────────┐    ┌────────────────┐    ┌──────────────────────────────┐
│ booking history │───>│ CatBoost heads │───>│ facility · weekday · hour ·   │
└─────────────────┘    └────────────────┘    │ notification-time prediction │
                                              └──────────────────────────────┘

┌─────────────────┐    ┌────────────────┐    ┌──────────────────────────────┐
│ booking history │───>│ language model │───>│ facility · weekday · hour ·   │
└─────────────────┘    └────────────────┘    │ notification-time prediction │
                                              └──────────────────────────────┘
```

The diagram compares inputs and outputs; training and evaluation happen
separately for each approach.

### 2.1 Baseline: a simple habit rule

The baseline predicts what a resident does most often and most recently.
For a resident with no history, it falls back to community-wide behaviour.
It also uses the resident's last facility to estimate the next one.

This is a serious baseline, not a token comparison. People often repeat
their routines, so this rule captures much of the signal in the data.

| Training detail | Baseline rule |
|---|---|
| Learns from | Each resident's earlier bookings only. |
| Rule | Uses the most frequent and most recent behaviour, plus the facility that usually follows the last facility. |
| New residents | Falls back to community-wide behaviour. |
| Model fitting | None. It is a deterministic rule. |

### 2.2 Machine learning: four CatBoost models

We trained four CatBoost models on the same 161 leakage-safe features.

| Model | Prediction | Model type | Training rounds |
|---|---|---|---:|
| 1 | Facility | 8-class classifier | 144 |
| 2 | Weekday | 7-class classifier | 100 |
| 3 | Hour | 24-class classifier | 18 |
| 4 | Notification time | Delay-range classifier | 129 |

The notification model predicts a delay range rather than an exact number.
An earlier version predicted minutes directly and scored 0.095, below the
0.111 achieved by always predicting one constant value. That approach was
optimising average error, while the project score checks whether a value is
within a 25% window. Five direct-number variants were tested and rejected.
The range-based version reached 0.131 on validation.

Each model's training length was selected using the latest portion of its
training data, then the model was retrained on all training rows at that
length. The final holdout data was never used for this choice.

| Training detail | CatBoost models |
|---|---|
| Training rows | 13,915 chronologically earlier records. |
| Inputs | 161 features: 7 categorical and 154 numeric. |
| Inner validation | The latest 15% of training data selected the number of rounds. |
| Final fit | Retrained on all training rows with the selected round count. |
| Runtime | CPU only, 10 threads, fixed seed. |

#### Selected combination: the best source for each prediction

Before the holdout was opened, we used validation results to choose either
the baseline or CatBoost for each prediction type.

| Prediction | Source used |
|---|---|
| Facility | CatBoost |
| Weekday | Baseline |
| Hour | CatBoost |
| Notification time | CatBoost |

```text
Source used in the selected combination

                       facility   weekday   hour   notification time
baseline rule             ○          ●       ○             ○
CatBoost heads            ●          ○       ●             ●
```

This is not an average or blend. Each output comes entirely from one
source, and the standalone results remain reported for comparison.

### 2.3 Fine-tuned language model

We also fine-tuned Qwen3-4B, a small open-weight language model that runs
locally on a laptop. It receives a written booking-history summary and
returns all four predictions at once.

| Safeguard | What it means |
|---|---|
| Fixed response shape | The model must return all four required fields. |
| Fixed answer lists | It cannot invent a facility or another invalid label. |
| 17 delay ranges | Each range is narrow enough that the correct range counts as a match. |
| Frozen base model | Only a small set of added parameters is trained. |

Using delay ranges has a known cost: even a perfect range classifier could
score at most 0.9545 on the delay metric.

| Training detail | Language model |
|---|---|
| Base model | 4-bit `Qwen3-4B-Instruct-2507`, run locally on Apple Silicon. |
| What trained | A LoRA adapter; the quantised base weights stayed frozen. |
| Adapter | Rank 8 across the top 16 transformer blocks, on attention and feed-forward projections. |
| Training run | 1,000 rows, 1 pass, 1,000 iterations, and 250 optimizer updates. |
| Optimisation | Adam, learning rate 0.0001, batch size 1, four-step gradient accumulation. |
| Cost | 2 hours 14 minutes; peak training memory 11.99 GB. |
| Answering | Greedy, schema-constrained JSON; only allowed labels can be returned. |

---

## 3. Results

### Results coverage

```text
                                   full 3,596-record holdout   shared 500-record comparison
selected baseline + ML heads                    ●                          ●
CatBoost heads only                             ●                          ●
baseline rule                                   ●                          ●
language model                                  ○                          ●
```

The language model was scored once on the frozen 500-record comparison,
not on all 3,596 records. The baseline and CatBoost results beside it use
the exact same 500 records.

### Full chronological holdout: 3,596 later records

| Prediction | Selected baseline + ML heads | CatBoost heads only | Baseline rule |
|---|---:|---:|---:|
| Facility | 0.3757 | 0.3757 | **0.3765** |
| Weekday | 0.2845 | 0.2764 | 0.2845 |
| Hour | 0.2036 | 0.2036 | **0.2041** |
| Notification time | **0.1204** | 0.1204 | 0.1190 |
| **Overall** | **0.2460** | 0.2440 | **0.2460** |

```text
Overall score on the full 3,596-record holdout

Selected baseline + ML heads  ████████████████████  0.2460
Baseline rule                 ████████████████████  0.2460
CatBoost heads only           ███████████████████▌  0.2440
```

The bars show only the full-holdout overall score. CatBoost alone is 0.0020
below the other two results. On individual answers, the baseline gets 3
more facilities and 2 more hours right; the selected combination gets 5
more notification-time predictions right.

### Shared comparison: 500 held-out records, including the language model

| Model | Overall | Facility | Weekday | Hour | Notification |
|---|---:|---:|---:|---:|---:|
| **Selected baseline + ML heads** | **0.2545** | 0.3760 | 0.3080 | 0.2280 | 0.1060 |
| CatBoost heads only | 0.2540 | 0.3760 | 0.3060 | 0.2280 | 0.1060 |
| Baseline rule | 0.2510 | 0.3800 | 0.3080 | 0.2040 | 0.1120 |
| Language model | 0.2185 | 0.3120 | 0.2800 | 0.2100 | 0.0720 |

The language model is 0.0325 below the baseline on this shared comparison.
Its paired 95% interval is −0.0525 to −0.0130, so the difference is
unlikely to be a sampling fluctuation.

```text
                         ┌───────────────────────┐
                         │ generate booking data │
                         └───────────┬───────────┘
                         ┌───────────▽───────────┐
                         │ train and choose on   │
                         │ earlier records only  │
                         └───────────┬───────────┘
                         ┌───────────▽───────────┐
                         │ freeze the choices    │
                         └───────────┬───────────┘
                         ┌───────────▽───────────┐
                         │ score later records   │
                         │ once                  │
                         └───────────────────────┘
```

The diagram shows the evaluation order; it does not show the individual
data-quality and leakage checks.

---

## 4. What the results tell us

### There is a low ceiling for this problem

We also measured an intentionally unfair reference: a lookup that can see
each resident's entire history, including future bookings. It reaches
0.3791 for facility, 0.2732 for weekday, and 0.2040 for hour. The learned
models already match or exceed that static personalised reference.

This does not make future information acceptable in a real model. It shows
that a resident's past habits contain only so much information about their
next booking.

### Validation success did not carry over

On validation data, the selected result scored 0.2401 and the baseline
scored 0.2320. That apparent lead disappeared on the chronologically later
holdout. This is an important reminder that a tuned model can look better
on development data without being better on unseen data.

### Booking delay is difficult to predict

| Reference | Notification match rate |
|---|---:|
| Predict the same delay every time | 0.111 |
| Know each resident's true average rate | 0.163 |

There is only about five percentage points of room between a constant
answer and an unrealistic per-resident reference. The time between
bookings has little usable memory from one booking to the next.

### Where the machine-learning models struggle

| Finding | Measure | Comparison |
|---|---:|---|
| Less-popular facilities | Facility accuracy: 0.2877 | 0.4038 for popular facilities |
| Weekday bookings | Weekday accuracy: 0.2197 | 0.2692 for weekend bookings |
| Residents seen in training | Overall score: 0.2396 | 0.2593 for unseen residents |

```text
Facility accuracy

Popular facilities       ████████████████████  0.4038
Less-popular facilities  ██████████████▌       0.2877

Weekday accuracy

Weekend bookings         █████████████▌        0.2692
Weekday bookings         ██████████▌           0.2197
```

The bars compare each pair within its own measure; bar lengths are not
intended to compare facility accuracy with weekday accuracy.

Cold start is not the main weakness: unseen residents are predicted
slightly better than residents already seen in training.

The largest single error pattern follows a change in the simulated
community. A Yoga Room opens during the two-year period and attracts
people who previously used the Gym. The model keeps predicting Gym,
creating 164 errors. It cannot learn a behavioural shift that had not yet
occurred in its training data.

### Why the language model performed poorly

Across 500 records, it used only 2 of its 17 delay ranges, behaving almost
like a single constant predictor. Training it for longer made its overall
score worse: 0.1875 rather than 0.2188. Two untested explanations remain:
the delay labels may be awkward to generate, and the training settings were
not tuned.

---

## 5. Why these results are credible

```text
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ only earlier history │───>│ choices fixed before │───>│ later records scored │
│ reaches a prediction │    │ final scoring        │    │ once                 │
└─────────────────────┘    └─────────────────────┘    └──────────────────────┘
```

The diagram summarises the safeguards; the checks below verify each stage.

- Six checks prevent the machine-learning models from seeing information
  that would not have been available at prediction time. They run for every
  row in every execution.
- The checks are tested by deliberately corrupting the data and confirming
  that the protections fail as expected.
- The holdout data can be scored once only. A second scoring attempt is
  refused.
- Language-model answers were hashed before the correct answers were read,
  preventing changes after the score was known.
- One fixed seed controls data generation, splitting, and model fitting.
  `make verify` recalculates the reported figures and fails if they differ.

## 6. Conclusion

The language model is not worth continuing for this problem. It is beaten
by a local rule that needs no specialised hardware and runs in 25 seconds,
while the language model takes 4.4 seconds per record and still performs
worse.

Keep the CatBoost pipeline as a measured comparator and diagnostic tool,
not as a claimed improvement on the baseline. On unseen data, it matches
the baseline rule rather than beating it. Its value is the disciplined
measurement and the ability to inspect where predictions fail.

The language-model experiment used about 9 hours of local compute, above
its planned budget. One of its 500 answers paired a facility with an hour
when that facility is closed. That result is reported as generated; it was
not silently corrected.

## 7. Limits of this work

- The data is synthetic. The models can only learn patterns that the data
  generator created.
- Two more complex model families were deliberately not used because they
  would make the project too slow and difficult to reproduce.
- The language-model result reflects one configuration trained once. It is
  not a general claim about all fine-tuned language models.
- Future-information leakage is formally tested for the machine-learning
  pipeline. The language-model pipeline enforces correct ordering and
  structure, but does not have the same formal proof.

---

## 8. Reproducing the results

No trained model files are included. The following commands rebuild the
machine-learning pipeline from source on an ordinary CPU in about five
minutes:

```bash
uv sync     # install the locked dependencies
make up     # start the database and experiment tracker
make all    # generate data, train, score, and write the review table
make verify # recompute and check the reported figures
```

`make all` regenerates the dataset, retrains the four CatBoost models,
builds the selected result, and writes the predicted-versus-actual review
table. `make verify` recomputes the report figures and fails if any have
drifted.

Retraining the language model requires Apple Silicon and about two and a
half hours:

```bash
python -m facility_prediction.cli llm-tier-b    # training, about 2 hours
python -m facility_prediction.cli llm-final     # scoring, about 40 minutes
```

When the recorded answer file is present, `make llm-reproduce` recomputes
the language-model figures in seconds without specialised hardware or a
paid service. It also verifies that no answer was altered, removed, or
swapped.
