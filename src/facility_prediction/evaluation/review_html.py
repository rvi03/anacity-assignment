"""The review table as a page a reviewer can open and filter.

The brief accepts "a spreadsheet or readable UI table". The workbook is
the spreadsheet; this is the table. Both are rendered from the *same*
scored frame, so they cannot drift apart and disagree about a record.

    scored predictions (one row per model per record)
              │
              ├─> build_predictions_sheet ──> .xlsx and .csv
              │
              └─> render_page ─────────────> .html

The page is a single self-contained file. No CDN, no fetch, no server:
a reviewer double-clicks it. That constraint is why the rows are
dictionary-encoded below rather than written out as objects — the
facility names, weekdays, hours and model names repeat across 11,288
rows, and spelling each one out inflates the payload roughly threefold
for no gain a reader can see.

What the page shows, in the order a reviewer needs it:

    headline rates per model ─> the four components side by side
                             ─> filters
                             ─> the record-level table
                             ─> what the numbers do and do not mean

Leakage contract: this module reads saved predictions and renders them.
It fits nothing, scores nothing, and computes no feature, so it has no
origin to respect. It shows whichever split it is handed; the caller
owns that choice, and `review_split` makes it from the seal.
"""

from __future__ import annotations

import html
import json
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction.evaluation import review

_LOGGER = logging.getLogger(__name__)

PAGE_FILENAME = "predictions_review.html"

# How many of a resident's most recent bookings the page carries per
# row. The brief's own illustration shows two, and every extra one is
# paid for 11,288 times.
HISTORY_SHOWN = 2

# Rows the table paints at once. The filter runs over every row; only
# the painting is capped, and the count above the table always states
# the unpainted total.
PAGE_SIZE = 100

COMPONENTS = (
    ("facility", "Facility", "match_facility"),
    ("weekday", "Weekday", "match_usage_weekday"),
    ("hour", "Hour", "match_usage_hour"),
    ("notification", "Notification", "match_notification"),
)

# Categorical slots 1-4 of the validated default palette, in fixed
# order. Assigned to models by identity, never by rank, so filtering
# one out never repaints the others.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500")

MATCH_MARK = "✓"

# What each model is called on screen. The keys are the names the
# predictions table records, and those stay as they are — they are in
# the metrics file, the database and the frozen digests, so renaming
# them would rewrite recorded history to fix a caption. This maps them
# to something a reviewer can read without knowing the project's
# vocabulary: "tier_b_adapter" was the sealed training size, which
# means nothing to anyone outside the build.
DISPLAY_NAMES = {
    "frequency_recency": "Habit rule (baseline)",
    "catboost": "CatBoost models",
    "selected_heads": "Best of both",
    "tier_b_adapter": "Fine-tuned language model",
}


def display_name(model: str) -> str:
    """Return the on-screen name for a recorded model name.

    Args:
        model: The name the predictions table records.

    Returns:
        The reviewer-facing label, or the recorded name tidied up if
        this model has no entry.
    """
    return DISPLAY_NAMES.get(model, model.replace("_", " "))


# What each model is, in one sentence a non-specialist can read. Keyed
# by the model name the predictions table records, so a model that
# appears without an entry still renders — it just says less.
MODEL_NOTES = {
    "frequency_recency": (
        "A rule, not a learned model. It answers what this resident "
        "books most often and most recently, falls back to what the "
        "community does when a resident has no history, and uses the "
        "facility that usually follows their last one. This is the bar "
        "every other model has to clear."
    ),
    "catboost": (
        "Four gradient-boosted classifiers, one per component, over the "
        "same 161 features. The notification head predicts a range of "
        "delays rather than a number of minutes, because the score is a "
        "±25% window and minimising average error optimises something "
        "else."
    ),
    "selected_heads": (
        "Per component, whichever of the habit rule and the CatBoost "
        "models scored better on validation — decided before the "
        "holdout was opened, then frozen. Nothing is blended: each "
        "answer comes whole from one source."
    ),
    "tier_b_adapter": (
        "A 4-bit Qwen3-4B with a LoRA adapter, run locally, answering "
        "all four components in one schema-constrained JSON object. It "
        "cannot invent a facility or a label outside the allowed set."
    ),
}


def _match(value: Any) -> bool:
    """Read one match cell, which the sheet writes as a tick or a cross."""
    return str(value).strip() == MATCH_MARK


def _short_history(value: Any) -> str:
    """Keep the most recent `HISTORY_SHOWN` bookings from a history cell.

    The sheet separates bookings with a return glyph, which suits a
    spreadsheet cell and reads as line noise in a table row. Here each
    booking becomes its own line instead.

    Args:
        value: The rendered history string, oldest booking first.

    Returns:
        The same string trimmed to its last few bookings, one per line.
    """
    text = "" if pd.isna(value) else str(value)
    parts = [
        part.strip()
        for part in text.replace(review.HISTORY_SEPARATOR, "\n").split("\n")
        if part.strip()
    ]
    return "\n".join(parts[-HISTORY_SHOWN:])


def model_rates(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Recompute each model's match rates from the rendered rows.

    Read off the same cells the reviewer can see rather than imported
    from the metrics file, so a number on this page is always the one
    its own table supports.

    Args:
        predictions: The rendered predictions sheet.

    Returns:
        One record per model: its label, its four component rates, its
        overall, and how many rows it was scored on.
    """
    rates = []
    for (track, model), group in predictions.groupby(
        ["track", "model"], sort=True
    ):
        components = {
            key: float(group[column].map(_match).mean())
            for key, _, column in COMPONENTS
        }
        rates.append(
            {
                "track": str(track),
                "model": str(model),
                "label": display_name(str(model)),
                "recorded": str(model),
                "note": MODEL_NOTES.get(str(model), ""),
                "rows": len(group),
                "components": components,
                "overall": sum(components.values()) / len(components),
            }
        )
    return sorted(rates, key=lambda item: -item["overall"])


def _encode(predictions: pd.DataFrame) -> dict[str, Any]:
    """Dictionary-encode the table so the page stays a sane size.

    Args:
        predictions: The rendered predictions sheet.

    Returns:
        The lookup tables and the row matrix the page's script reads.
    """
    frame = predictions.copy()
    frame["model_label"] = frame["model"].astype(str).map(display_name)

    models = sorted(frame["model_label"].unique())
    facilities = sorted(
        set(frame["predicted_facility"].astype(str))
        | set(frame["actual_facility"].astype(str))
    )
    weekdays = sorted(
        set(frame["predicted_usage_weekday"].astype(str))
        | set(frame["actual_usage_weekday"].astype(str))
    )
    hours = sorted(
        set(frame["predicted_usage_hour"].astype(str))
        | set(frame["actual_usage_hour"].astype(str))
    )

    index = {
        "model": {name: position for position, name in enumerate(models)},
        "facility": {
            name: position for position, name in enumerate(facilities)
        },
        "weekday": {name: position for position, name in enumerate(weekdays)},
        "hour": {name: position for position, name in enumerate(hours)},
    }

    rows = [
        [
            index["model"][str(row.model_label)],
            str(row.record),
            str(row.resident),
            _short_history(row.past_bookings),
            index["facility"][str(row.predicted_facility)],
            index["weekday"][str(row.predicted_usage_weekday)],
            index["hour"][str(row.predicted_usage_hour)],
            str(row.suggested_send),
            index["facility"][str(row.actual_facility)],
            index["weekday"][str(row.actual_usage_weekday)],
            index["hour"][str(row.actual_usage_hour)],
            str(row.actual_booked),
            int(_match(row.match_facility)),
            int(_match(row.match_usage_weekday)),
            int(_match(row.match_usage_hour)),
            int(_match(row.match_notification)),
            int(bool(row.cold_start)),
        ]
        for row in frame.itertuples(index=False)
    ]

    return {
        "models": models,
        "facilities": facilities,
        "weekdays": weekdays,
        "hours": hours,
        "rows": rows,
    }


def _series_variables() -> tuple[str, str]:
    """Emit the categorical slots as CSS custom properties, both modes.

    Returns:
        The light-mode block and the dark-mode block. Dark is a set of
        steps chosen for the dark surface, not an automatic flip of the
        light values.
    """
    light = "\n".join(
        f"      --series-{position + 1}: {value};"
        for position, value in enumerate(SERIES_LIGHT)
    )
    dark = "\n".join(
        f"      --series-{position + 1}: {value};"
        for position, value in enumerate(SERIES_DARK)
    )
    return light, dark


def render_page(
    predictions: pd.DataFrame,
    split: str,
    reason: str,
) -> str:
    """Render the whole page, data and all, as one HTML string.

    Args:
        predictions: The rendered predictions sheet.
        split: The split these rows come from.
        reason: The sentence stating what that split may be read as.

    Returns:
        A self-contained HTML document.
    """
    payload = _encode(predictions)
    rates = model_rates(predictions)
    light, dark = _series_variables()

    data = json.dumps(
        {
            "split": split,
            "reason": reason,
            "rates": rates,
            "components": [
                {"key": key, "label": label} for key, label, _ in COMPONENTS
            ],
            "pageSize": PAGE_SIZE,
            **payload,
        },
        separators=(",", ":"),
    )

    return _TEMPLATE.format(
        light=light,
        dark=dark,
        split=html.escape(split),
        reason=html.escape(reason),
        rows=f"{len(predictions):,}",
        models=len(payload["models"]),
        data=data.replace("</", "<\\/"),
    )


def write_page(
    predictions: pd.DataFrame,
    split: str,
    reason: str,
    path: pathlib.Path,
) -> int:
    """Write the page and report its size.

    Args:
        predictions: The rendered predictions sheet.
        split: The split these rows come from.
        reason: The sentence stating what that split may be read as.
        path: Destination file.

    Returns:
        The number of bytes written.
    """
    page = render_page(predictions, split, reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    _LOGGER.info(
        "wrote %s: %d rows, %.1f MB",
        path,
        len(predictions),
        len(page.encode("utf-8")) / 1e6,
    )
    return len(page.encode("utf-8"))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Facility booking predictions — review</title>
<style>
  :root {{
    color-scheme: light;
    --surface-0: #f4f4f2;
    --surface-1: #fcfcfb;
    --surface-2: #eeeeec;
    --border:    #d9d9d4;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #78776f;
    --good: #007a4d;
    --bad:  #b4341f;
{light}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-0: #121211;
      --surface-1: #1a1a19;
      --surface-2: #232322;
      --border:    #383836;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #918f85;
      --good: #4bb17e;
      --bad:  #e66767;
{dark}
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --surface-2: #232322;
    --border:    #383836;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #918f85;
    --good: #4bb17e;
    --bad:  #e66767;
{dark}
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface-0);
    color: var(--text-primary);
    font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto,
          sans-serif;
  }}
  .wrap {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 64px; }}
  header h1 {{ font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  header p {{ margin: 0; color: var(--text-secondary); max-width: 70ch; }}
  .note {{
    margin: 16px 0 0; padding: 12px 14px; border-radius: 8px;
    background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text-secondary); font-size: 13.5px; max-width: 90ch;
  }}
  h2 {{ font-size: 15px; margin: 36px 0 12px; letter-spacing: 0.02em;
        text-transform: uppercase; color: var(--text-muted); }}

  .tiles {{ display: grid; gap: 12px; margin-top: 12px;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 10px; padding: 14px 16px; }}
  .tile .key {{ display: flex; align-items: center; gap: 8px;
                font-size: 13px; color: var(--text-secondary); }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; flex: none; }}
  .tile .num {{ font-size: 30px; font-variant-numeric: tabular-nums;
                letter-spacing: -0.02em; margin: 6px 0 2px; }}
  .tile .sub {{ font-size: 12.5px; color: var(--text-muted); }}

  .chart {{ background: var(--surface-1); border: 1px solid var(--border);
            border-radius: 10px; padding: 18px 18px 8px; overflow-x: auto; }}
  .grp {{ margin-bottom: 18px; }}
  .grp h3 {{ font-size: 13px; font-weight: 600; margin: 0 0 8px;
             color: var(--text-secondary); }}
  .bar-row {{ display: grid; grid-template-columns: 130px 1fr 52px;
              align-items: center; gap: 10px; margin-bottom: 2px; }}
  .bar-name {{ font-size: 12.5px; color: var(--text-secondary);
               white-space: nowrap; overflow: hidden;
               text-overflow: ellipsis; }}
  .bar-track {{ background: var(--surface-2); border-radius: 4px;
                height: 14px; }}
  .bar-fill {{ height: 14px; border-radius: 4px; }}
  .bar-val {{ font-size: 12.5px; font-variant-numeric: tabular-nums;
              text-align: right; color: var(--text-primary); }}

  .filters {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end;
              margin: 12px 0 14px; }}
  .field {{ display: flex; flex-direction: column; gap: 4px; }}
  .field label {{ font-size: 11.5px; text-transform: uppercase;
                  letter-spacing: 0.04em; color: var(--text-muted); }}
  select, input {{
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 7px;
    padding: 7px 9px; font: inherit; font-size: 13.5px; min-width: 150px;
  }}
  button {{
    background: var(--surface-2); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 7px;
    padding: 7px 12px; font: inherit; font-size: 13.5px; cursor: pointer;
  }}
  button:hover {{ border-color: var(--text-muted); }}

  .count {{ font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }}
  .scroll {{ overflow-x: auto; border: 1px solid var(--border);
             border-radius: 10px; background: var(--surface-1); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 9px 11px;
            border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ position: sticky; top: 0; background: var(--surface-2);
        font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.03em;
        color: var(--text-muted); font-weight: 600; white-space: nowrap; }}
  tbody tr:hover {{ background: var(--surface-2); }}
  td.hist {{ color: var(--text-secondary); font-size: 12px;
             max-width: 300px; white-space: pre-line; }}
  .scale-note {{ margin: 0 0 16px; font-size: 12.5px;
                 color: var(--text-muted); }}
  .mono {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .yes {{ color: var(--good); font-weight: 600; }}
  .no  {{ color: var(--bad); }}
  .tag {{ display: inline-block; padding: 1px 6px; border-radius: 4px;
          background: var(--surface-2); border: 1px solid var(--border);
          font-size: 11.5px; color: var(--text-secondary); }}
  .pager {{ display: flex; gap: 10px; align-items: center; margin-top: 12px; }}

  details {{ background: var(--surface-1); border: 1px solid var(--border);
             border-radius: 10px; padding: 14px 16px; margin-top: 10px; }}
  details + details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; font-weight: 600; font-size: 14px; }}
  details p, details li {{ color: var(--text-secondary); max-width: 85ch; }}
  details code {{ background: var(--surface-2); padding: 1px 5px;
                  border-radius: 4px; font-size: 12.5px; }}
  .tabs {{ display: flex; gap: 4px; margin: 26px 0 0;
            border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
  .tab {{ appearance: none; background: none; border: none;
          border-bottom: 2px solid transparent; border-radius: 0;
          padding: 10px 14px; font: inherit; font-size: 14px;
          color: var(--text-secondary); cursor: pointer; }}
  .tab:hover {{ color: var(--text-primary); }}
  .tab[aria-selected="true"] {{ color: var(--text-primary);
                                font-weight: 600;
                                border-bottom-color: var(--series-1); }}
  .tab:focus-visible {{ outline: 2px solid var(--series-1);
                        outline-offset: -2px; }}
  .panel[hidden] {{ display: none; }}
  .panel {{ padding-top: 4px; }}
  .panel > h2:first-of-type {{ margin-top: 22px; }}

  .model-card {{ background: var(--surface-1); border: 1px solid var(--border);
                 border-radius: 10px; padding: 16px 18px; margin-top: 12px; }}
  .model-card h3 {{ margin: 0 0 4px; font-size: 16px;
                    display: flex; align-items: center; gap: 9px; }}
  .model-card .meta {{ font-size: 12.5px; color: var(--text-muted);
                       margin: 0 0 10px; }}
  .model-card p.note {{ margin: 0 0 12px; background: none;
                        border: 0; padding: 0; font-size: 14px; }}
  .rates {{ display: grid; gap: 10px;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }}
  .rate {{ background: var(--surface-2); border-radius: 8px;
           padding: 9px 11px; }}
  .rate .lbl {{ font-size: 11.5px; text-transform: uppercase;
                letter-spacing: 0.04em; color: var(--text-muted); }}
  .rate .val {{ font-size: 18px; font-variant-numeric: tabular-nums; }}

  .theme {{ position: absolute; top: 26px; right: 20px; }}
</style>
</head>
<body>
<div class="wrap">
  <button class="theme" id="theme" type="button">Dark / light</button>

  <header>
    <h1>Facility booking predictions</h1>
    <p>
      For a resident at a moment in time, every model predicts four
      things: which facility, which weekday, which hour, and how long
      until they book. The <strong>Records</strong> tab holds one row per
      model per record, beside what actually happened.
    </p>
    <p class="note">
      Showing the <strong>{split}</strong> split — {rows} rows across
      {models} models. {reason}
    </p>
  </header>

  <nav class="tabs" role="tablist" aria-label="Sections">
    <button class="tab" role="tab" id="tab-overview"
            aria-controls="panel-overview"
            aria-selected="true">Overview</button>
    <button class="tab" role="tab" id="tab-records"
            aria-controls="panel-records" aria-selected="false">Records</button>
    <button class="tab" role="tab" id="tab-models"
            aria-controls="panel-models" aria-selected="false">Models</button>
    <button class="tab" role="tab" id="tab-method"
            aria-controls="panel-method" aria-selected="false">Method</button>
  </nav>

  <section class="panel" id="panel-overview" role="tabpanel"
           aria-labelledby="tab-overview" tabindex="0">
    <h2>Overall score by model</h2>
    <div class="tiles" id="tiles"></div>

    <h2>The four components</h2>
    <div class="chart" id="chart"></div>
  </section>

  <section class="panel" id="panel-records" role="tabpanel"
           aria-labelledby="tab-records" tabindex="0" hidden>
    <h2>Every prediction, beside what happened</h2>
    <div class="filters" id="filters">
      <div class="field">
        <label for="f-model">Model</label>
        <select id="f-model"><option value="">All models</option></select>
      </div>
      <div class="field">
        <label for="f-facility">Actual facility</label>
        <select id="f-facility">
          <option value="">All facilities</option>
        </select>
      </div>
      <div class="field">
        <label for="f-match">Match</label>
        <select id="f-match">
          <option value="">Any outcome</option>
          <option value="4">All four correct</option>
          <option value="3">Three or more correct</option>
          <option value="1">At least one correct</option>
          <option value="0">None correct</option>
          <option value="fac">Facility correct</option>
          <option value="!fac">Facility wrong</option>
        </select>
      </div>
      <div class="field">
        <label for="f-cold">Resident</label>
        <select id="f-cold">
          <option value="">Everyone</option>
          <option value="1">Unseen in training</option>
          <option value="0">Seen in training</option>
        </select>
      </div>
      <div class="field">
        <label for="f-text">Search</label>
        <input id="f-text" type="search" placeholder="record or resident">
      </div>
      <button id="reset" type="button">Reset</button>
    </div>

    <p class="count" id="count"></p>
    <div class="scroll">
      <table>
        <thead><tr>
          <th>Model</th><th>Record</th><th>Resident</th>
          <th>Recent bookings</th>
          <th>Predicted</th><th>Actual</th>
          <th>Facility</th><th>Day</th><th>Hour</th><th>Notify</th>
        </tr></thead>
        <tbody id="body"></tbody>
      </table>
    </div>
    <div class="pager">
      <button id="prev" type="button">Previous</button>
      <span class="count" id="page" style="margin:0"></span>
      <button id="next" type="button">Next</button>
    </div>
  </section>

  <section class="panel" id="panel-models" role="tabpanel"
           aria-labelledby="tab-models" tabindex="0" hidden>
    <h2>What each model is</h2>
    <div id="models"></div>
  </section>

  <section class="panel" id="panel-method" role="tabpanel"
           aria-labelledby="tab-method" tabindex="0" hidden>
    <h2>How to read these numbers</h2>

    <details open>
      <summary>How a prediction is scored</summary>
      <p>
        Facility, weekday and hour are exact-match: the answer counts only
        if it names the right one. Notification is scored inside a
        <strong>&plusmn;25% window</strong> of the real delay, because it is
        a question about magnitude rather than about a minute.
      </p>
      <p>
        The overall score is the mean of those four rates. The same number
        chose the model and leads the report, so no model can be picked on
        one measure and presented with a more flattering one.
      </p>
    </details>

    <details>
      <summary>Why the scores sit near a quarter</summary>
      <p>
        Most of what is predictable here is that residents repeat
        themselves, and the frequency/recency rule already captures it. A
        deliberately unfair reference — a lookup allowed to see each
        resident's whole history including the future — reaches only
        0.3791 / 0.2732 / 0.2040 on facility / weekday / hour. The trained
        models match or exceed that, so they are at the ceiling of what
        habit alone can tell you.
      </p>
      <p>
        Booking delay is the hardest of the four for every model: the gap
        to a resident's next booking carries almost no memory of the
        previous gap, and the whole winnable band is about five points wide.
      </p>
    </details>

    <details>
      <summary>Leakage controls</summary>
      <p>
        Every feature is computed from events at or before its row's
        prediction origin. Six numbered controls enforce that, asserted
        <em>per row on every run</em> rather than only in tests: resident
        history bounded by the origin, no target-derived column as an
        input, community windows closed strictly before the origin, a
        chronological split that cannot overlap, frozen split membership
        audited at every fit, and generator hidden state kept out of the
        modelling table.
      </p>
      <p>
        Two further checks look for the <em>effect</em> of a leak: a
        future-perturbation property test, and a shuffled-label control.
        Both carry a sensitivity check, because an instrument that has gone
        blind passes everything.
      </p>
    </details>

    <details>
      <summary>Scored once, and reproducible</summary>
      <p>
        The holdout is scored once, ever. A freeze written before any
        holdout row is read and a seal written after make that enforced
        rather than intended; a second attempt is refused.
      </p>
      <p>
        One seed drives the generator, the splits and every fit.
        <code>make verify</code> recomputes every committed value —
        digests, both splits' metrics, the workbook hash — and fails on any
        difference. This page is rendered from the same scored frame as the
        workbook, so the two cannot disagree.
      </p>
    </details>

    <details>
      <summary>What this page is not</summary>
      <p>
        It is a record-level view, not the result. Sampling the table until
        a model looks good is exactly the mistake the single-scoring rule
        exists to prevent. The headline figures are computed over every row
        on the selected split, and the full write-up lives in
        <code>docs/REVIEW.md</code>.
      </p>
    </details>
  </section>
</div>

<script id="payload" type="application/json">{data}</script>
<script>
(function () {{
  var D = JSON.parse(document.getElementById('payload').textContent);
  var SERIES = 4;

  function pct(value) {{ return (value * 100).toFixed(1) + '%'; }}
  function slot(index) {{
    return 'var(--series-' + ((index % SERIES) + 1) + ')';
  }}
  function colorOf(label) {{ return slot(D.models.indexOf(label)); }}

  // ---- tabs -----------------------------------------------------------
  // Panels are hidden, not unmounted: the table keeps its filter state
  // and its scroll position when a reader looks at the method and
  // comes back.
  var tabs = [].slice.call(document.querySelectorAll('[role="tab"]'));

  function select(tab, focus) {{
    tabs.forEach(function (other) {{
      var chosen = other === tab;
      other.setAttribute('aria-selected', chosen ? 'true' : 'false');
      other.tabIndex = chosen ? 0 : -1;
      document.getElementById(other.getAttribute('aria-controls'))
        .hidden = !chosen;
    }});
    if (focus) tab.focus();
  }}

  tabs.forEach(function (tab, index) {{
    tab.tabIndex = tab.getAttribute('aria-selected') === 'true' ? 0 : -1;
    tab.addEventListener('click', function () {{ select(tab, false); }});
    tab.addEventListener('keydown', function (event) {{
      var step = event.key === 'ArrowRight' ? 1
               : event.key === 'ArrowLeft' ? -1 : 0;
      if (step) {{
        event.preventDefault();
        select(tabs[(index + step + tabs.length) % tabs.length], true);
      }} else if (event.key === 'Home') {{
        event.preventDefault(); select(tabs[0], true);
      }} else if (event.key === 'End') {{
        event.preventDefault(); select(tabs[tabs.length - 1], true);
      }}
    }});
  }});

  // ---- headline tiles ------------------------------------------------
  var tiles = document.getElementById('tiles');
  D.rates.forEach(function (rate) {{
    var card = document.createElement('div');
    card.className = 'tile';

    var key = document.createElement('div');
    key.className = 'key';
    var dot = document.createElement('span');
    dot.className = 'swatch';
    dot.style.background = colorOf(rate.label);
    key.appendChild(dot);
    key.appendChild(document.createTextNode(rate.label));

    var num = document.createElement('div');
    num.className = 'num';
    num.textContent = rate.overall.toFixed(4);

    var sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = rate.rows.toLocaleString() + ' rows · ' + rate.track;

    card.appendChild(key); card.appendChild(num); card.appendChild(sub);
    tiles.appendChild(card);
  }});

  // ---- component bars, every bar directly labelled --------------------
  var chart = document.getElementById('chart');

  // One scale across all four groups, from zero. Renormalising each
  // group to its own maximum would draw a 0.072 bar and a 0.376 bar
  // the same length, and turn a rounding difference into a visible gap.
  var top = 0;
  D.rates.forEach(function (rate) {{
    D.components.forEach(function (component) {{
      top = Math.max(top, rate.components[component.key]);
    }});
  }});
  top = Math.ceil(top * 10 + 0.5) / 10;

  var scale = document.createElement('p');
  scale.className = 'scale-note';
  scale.textContent = 'One shared scale, 0% to ' + (top * 100).toFixed(0) +
    '%, so the four components are comparable with each other.';
  chart.appendChild(scale);
  D.components.forEach(function (component) {{
    var group = document.createElement('div');
    group.className = 'grp';
    var heading = document.createElement('h3');
    heading.textContent = component.label;
    group.appendChild(heading);

    D.rates.forEach(function (rate) {{
      var value = rate.components[component.key];
      var row = document.createElement('div');
      row.className = 'bar-row';

      var name = document.createElement('div');
      name.className = 'bar-name';
      name.textContent = rate.label;

      var track = document.createElement('div');
      track.className = 'bar-track';
      var fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = Math.max(2, (value / top) * 100) + '%';
      fill.style.background = colorOf(rate.label);
      fill.title = rate.label + ' — ' + component.label + ' ' + pct(value);
      track.appendChild(fill);

      var val = document.createElement('div');
      val.className = 'bar-val';
      val.textContent = pct(value);

      row.appendChild(name); row.appendChild(track); row.appendChild(val);
      group.appendChild(row);
    }});
    chart.appendChild(group);
  }});

  // ---- what each model is ---------------------------------------------
  var models = document.getElementById('models');
  D.rates.forEach(function (rate) {{
    var card = document.createElement('div');
    card.className = 'model-card';

    var heading = document.createElement('h3');
    var dot = document.createElement('span');
    dot.className = 'swatch';
    dot.style.background = colorOf(rate.label);
    heading.appendChild(dot);
    heading.appendChild(document.createTextNode(rate.label));

    var meta = document.createElement('p');
    meta.className = 'meta';
    meta.textContent = rate.track + ' track · scored on ' +
      rate.rows.toLocaleString() + ' rows · recorded in the data as "' +
      rate.recorded + '"';

    card.appendChild(heading);
    card.appendChild(meta);

    if (rate.note) {{
      var note = document.createElement('p');
      note.className = 'note';
      note.textContent = rate.note;
      card.appendChild(note);
    }}

    var grid = document.createElement('div');
    grid.className = 'rates';
    var cells = D.components.concat([{{key: null, label: 'Overall'}}]);
    cells.forEach(function (component) {{
      var box = document.createElement('div');
      box.className = 'rate';
      var label = document.createElement('div');
      label.className = 'lbl';
      label.textContent = component.label;
      var value = document.createElement('div');
      value.className = 'val';
      value.textContent = component.key
        ? pct(rate.components[component.key])
        : rate.overall.toFixed(4);
      box.appendChild(label); box.appendChild(value);
      grid.appendChild(box);
    }});
    card.appendChild(grid);
    models.appendChild(card);
  }});

  // ---- filters --------------------------------------------------------
  var M = 0, REC = 1, RES = 2, HIST = 3,
      PF = 4, PW = 5, PH = 6, SEND = 7,
      AF = 8, AW = 9, AH = 10, BOOKED = 11,
      MF = 12, MW = 13, MH = 14, MN = 15, COLD = 16;

  function fill(select, values) {{
    values.forEach(function (value, index) {{
      var option = document.createElement('option');
      option.value = String(index);
      option.textContent = value;
      select.appendChild(option);
    }});
  }}
  fill(document.getElementById('f-model'), D.models);
  fill(document.getElementById('f-facility'), D.facilities);

  var state = {{ model: '', facility: '', match: '', cold: '', text: '' }};
  var page = 0;

  function keep(row) {{
    if (state.model !== '' && row[M] !== +state.model) return false;
    if (state.facility !== '' && row[AF] !== +state.facility) return false;
    if (state.cold !== '' && row[COLD] !== +state.cold) return false;
    if (state.text) {{
      var needle = state.text.toLowerCase();
      if (row[REC].toLowerCase().indexOf(needle) < 0 &&
          row[RES].toLowerCase().indexOf(needle) < 0) return false;
    }}
    if (state.match !== '') {{
      var hits = row[MF] + row[MW] + row[MH] + row[MN];
      if (state.match === 'fac') return row[MF] === 1;
      if (state.match === '!fac') return row[MF] === 0;
      if (state.match === '4') return hits === 4;
      if (state.match === '3') return hits >= 3;
      if (state.match === '1') return hits >= 1;
      if (state.match === '0') return hits === 0;
    }}
    return true;
  }}

  function cell(text, className) {{
    var td = document.createElement('td');
    td.textContent = text;
    if (className) td.className = className;
    return td;
  }}
  function markCell(hit) {{
    var td = document.createElement('td');
    td.textContent = hit ? '✓' : '✗';
    td.className = hit ? 'yes' : 'no';
    return td;
  }}

  function render() {{
    var matched = D.rows.filter(keep);
    var pages = Math.max(1, Math.ceil(matched.length / D.pageSize));
    if (page >= pages) page = pages - 1;
    if (page < 0) page = 0;

    document.getElementById('count').textContent =
      matched.length.toLocaleString() + ' of ' +
      D.rows.length.toLocaleString() + ' rows match';
    document.getElementById('page').textContent =
      'Page ' + (page + 1) + ' of ' + pages;

    var body = document.getElementById('body');
    body.textContent = '';
    matched.slice(page * D.pageSize, (page + 1) * D.pageSize)
      .forEach(function (row) {{
        var tr = document.createElement('tr');

        var model = document.createElement('td');
        var tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = D.models[row[M]];
        model.appendChild(tag);
        tr.appendChild(model);

        tr.appendChild(cell(row[REC], 'mono'));
        tr.appendChild(cell(row[RES] + (row[COLD] ? ' · new' : ''), 'mono'));
        tr.appendChild(cell(row[HIST], 'hist'));
        tr.appendChild(cell(
          D.facilities[row[PF]] + ' / ' + D.weekdays[row[PW]] + ' / ' +
          D.hours[row[PH]] + ' / notify ' + row[SEND], 'hist'));
        tr.appendChild(cell(
          D.facilities[row[AF]] + ' / ' + D.weekdays[row[AW]] + ' / ' +
          D.hours[row[AH]] + ' / booked ' + row[BOOKED], 'hist'));
        tr.appendChild(markCell(row[MF]));
        tr.appendChild(markCell(row[MW]));
        tr.appendChild(markCell(row[MH]));
        tr.appendChild(markCell(row[MN]));
        body.appendChild(tr);
      }});
  }}

  function bind(id, key) {{
    document.getElementById(id).addEventListener('input', function (event) {{
      state[key] = event.target.value;
      page = 0;
      render();
    }});
  }}
  bind('f-model', 'model');
  bind('f-facility', 'facility');
  bind('f-match', 'match');
  bind('f-cold', 'cold');
  bind('f-text', 'text');

  document.getElementById('reset').addEventListener('click', function () {{
    state = {{ model: '', facility: '', match: '', cold: '', text: '' }};
    ['f-model', 'f-facility', 'f-match', 'f-cold', 'f-text']
      .forEach(function (id) {{ document.getElementById(id).value = ''; }});
    page = 0;
    render();
  }});
  document.getElementById('prev').addEventListener('click', function () {{
    page -= 1; render();
  }});
  document.getElementById('next').addEventListener('click', function () {{
    page += 1; render();
  }});
  document.getElementById('theme').addEventListener('click', function () {{
    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute(
      'data-theme', dark ? 'light' : 'dark');
  }});

  render();
}})();
</script>
</body>
</html>
"""
