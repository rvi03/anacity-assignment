"""Renders one prediction sample into the prompt the model reads.

One template, one field order, one numeric precision. The same sample
renders to the same bytes every time, and any edit to the template
changes its hash — which is what makes a cached generation safe to
replay and an uncached one safe to compare.

    sample summaries ─┐
    recent events ────┼──> user prompt ──> hash ──> prompt manifest
    config catalogs ──┘

The facility catalog and the delay labels are rendered from
configuration at run time. Neither is ever typed into a second file: a
hardcoded list that drifts from the generator or from the bucket merge
would reject valid answers at decode time.

Leakage contract: this module renders what it is handed. The caller
bounds the history at the origin; nothing here reads a booking table,
a split, or a target.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import pandas as pd

from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import prompt_features

SYSTEM_INSTRUCTION = (
    "You predict a resident's next facility booking from history "
    "available at the prediction origin.\n"
    "Return one JSON object matching the supplied schema. Use only the "
    "allowed facility and weekday values.\n"
    "notification_delay_bucket is one of the supplied bucket labels "
    "measured from the prediction origin."
)

# Weekday names in the order pandas numbers them, 0 is Monday. This is
# the output contract's enum, not a configurable catalog.
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

HOURS_IN_DAY = 24

FACILITY_FIELD = "facility"
USAGE_DAY_FIELD = "usage_day"
USAGE_HOUR_FIELD = "usage_hour"
BUCKET_FIELD = "notification_delay_bucket"
TARGET_FIELDS = (
    FACILITY_FIELD,
    USAGE_DAY_FIELD,
    USAGE_HOUR_FIELD,
    BUCKET_FIELD,
)


@dataclasses.dataclass(frozen=True)
class PromptContext:
    """Everything one prompt renders from.

    Attributes:
        origin: The prediction origin, timezone-aware.
        facilities: The facility catalog, in configured order.
        ladder: The frozen delay buckets, ascending.
        summary: Summary name to value, in render order.
        events: Prior bookings, oldest first.
        decimals: Decimal places every rendered float is fixed to.
    """

    origin: pd.Timestamp
    facilities: tuple[str, ...]
    ladder: list[buckets_module.Bucket]
    summary: dict[str, Any]
    events: list[prompt_features.Event]
    decimals: int


def _number(value: float, decimals: int) -> str:
    """Renders a float at the fixed precision.

    Args:
        value: The number to render.
        decimals: Decimal places.

    Returns:
        The number as text, always with ``decimals`` places.
    """
    return f"{value:.{decimals}f}"


def _value(value: Any, decimals: int) -> str:
    """Renders one summary value.

    Args:
        value: The value to render.
        decimals: Decimal places for floats.

    Returns:
        The value as text; an absent value renders as ``none``, never
        as a substitute number.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return _number(value, decimals)
    if isinstance(value, dict):
        inner = ", ".join(
            f"{key}:{_value(item, decimals)}" for key, item in value.items()
        )
        return "{" + inner + "}"
    return str(value)


def render_facilities(facilities: tuple[str, ...]) -> str:
    """Renders the allowed facility list.

    Args:
        facilities: The catalog, in configured order.

    Returns:
        The catalog as one bracketed line.
    """
    return "[" + ", ".join(facilities) + "]"


def render_buckets(ladder: list[buckets_module.Bucket], decimals: int) -> str:
    """Renders the delay label legend.

    An ordinal label means nothing on its own, so every prompt carries
    the same legend of what each one covers.

    Args:
        ladder: The frozen buckets, ascending.
        decimals: Decimal places for the bounds.

    Returns:
        One line per label, in ladder order.

    Raises:
        ValueError: If a bucket has no frozen representative.
    """
    lines = []
    for bucket in ladder:
        if bucket.representative is None:
            msg = f"bucket {bucket.label} has no representative"
            raise ValueError(msg)
        lower = bucket.lower - buckets_module.TRANSFORM_OFFSET
        if bucket.upper is None:
            span = f">= {_number(lower, decimals)} min"
        else:
            upper = bucket.upper - buckets_module.TRANSFORM_OFFSET
            span = f"{_number(lower, decimals)}-{_number(upper, decimals)} min"
        lines.append(f"  {bucket.label} {span}")
    return "\n".join(lines)


def render_summary(summary: dict[str, Any], decimals: int) -> str:
    """Renders the history summary block.

    Args:
        summary: Summary name to value, in render order.
        decimals: Decimal places for floats.

    Returns:
        One ``name=value`` line per entry.
    """
    return "\n".join(
        f"{name}={_value(value, decimals)}" for name, value in summary.items()
    )


def render_events(events: list[prompt_features.Event], decimals: int) -> str:
    """Renders the recent-booking block, oldest first.

    Args:
        events: Prior bookings, oldest first.
        decimals: Decimal places for floats.

    Returns:
        One numbered line per event, or ``none`` when there are none.
    """
    if not events:
        return "none"
    lines = []
    for position, event in enumerate(events, start=1):
        booked = (
            f"{WEEKDAY_NAMES[event.booked_weekday][:3]} "
            f"{event.booked_hour:02d}:{event.booked_minute:02d}"
        )
        used = (
            f"{WEEKDAY_NAMES[event.used_weekday][:3]} {event.used_hour:02d}:00"
        )
        gap = (
            "none"
            if event.gap_days is None
            else _number(event.gap_days, decimals)
        )
        lines.append(
            f"{position}|facility={event.facility}|booked={booked}"
            f"|used={used}|lead_hours={_number(event.lead_hours, decimals)}"
            f"|gap_days={gap}"
        )
    return "\n".join(lines)


def render_prompt(context: PromptContext) -> str:
    """Renders the user prompt for one sample.

    Args:
        context: Everything the prompt renders from.

    Returns:
        The prompt text, identical for identical input.
    """
    return "\n".join(
        [
            f"PREDICTION_ORIGIN: {context.origin.isoformat()}",
            f"ALLOWED_FACILITIES: {render_facilities(context.facilities)}",
            "ALLOWED_DELAY_BUCKETS (delay from the origin; contiguous, "
            "no gaps):",
            render_buckets(context.ladder, context.decimals),
            "",
            "HISTORY_SUMMARY:",
            render_summary(context.summary, context.decimals),
            "",
            "RECENT_BOOKINGS_OLDEST_TO_NEWEST:",
            render_events(context.events, context.decimals),
            "",
            "Predict the next booking.",
        ]
    )


def render_target(
    facility: str, usage_weekday: int, usage_hour: int, bucket_label: str
) -> str:
    """Renders the answer a training row teaches.

    Args:
        facility: The facility booked.
        usage_weekday: Weekday of usage, 0 is Monday.
        usage_hour: Hour of usage.
        bucket_label: The delay bucket the notification falls in.

    Returns:
        The answer as compact JSON, fields in contract order.

    Raises:
        ValueError: If the weekday or hour is outside its range.
    """
    if not 0 <= usage_weekday < len(WEEKDAY_NAMES):
        msg = f"usage weekday {usage_weekday} is outside the week"
        raise ValueError(msg)
    if not 0 <= usage_hour < HOURS_IN_DAY:
        msg = f"usage hour {usage_hour} is outside the day"
        raise ValueError(msg)
    return json.dumps(
        {
            FACILITY_FIELD: facility,
            USAGE_DAY_FIELD: WEEKDAY_NAMES[usage_weekday],
            USAGE_HOUR_FIELD: usage_hour,
            BUCKET_FIELD: bucket_label,
        },
        separators=(", ", ": "),
    )


def output_schema(
    facilities: tuple[str, ...], labels: list[str]
) -> dict[str, Any]:
    """Builds the JSON schema a generation is constrained to.

    Args:
        facilities: The facility catalog, in configured order.
        labels: The delay labels, in ladder order.

    Returns:
        The schema, with every enum taken from what was passed in.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TARGET_FIELDS),
        "properties": {
            FACILITY_FIELD: {"enum": list(facilities)},
            USAGE_DAY_FIELD: {"enum": list(WEEKDAY_NAMES)},
            USAGE_HOUR_FIELD: {"enum": list(range(HOURS_IN_DAY))},
            BUCKET_FIELD: {"enum": list(labels)},
        },
    }


def template_hash(version: int, prompt: str) -> str:
    """Hashes the rendered template so a cache can key on it.

    Args:
        version: The configured template version.
        prompt: One rendered prompt, standing for the template's shape.

    Returns:
        The hex SHA-256 over the system instruction, the version, and
        the prompt.
    """
    hasher = hashlib.sha256()
    hasher.update(SYSTEM_INSTRUCTION.encode("utf-8"))
    hasher.update(str(version).encode("utf-8"))
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def schema_hash(schema: dict[str, Any]) -> str:
    """Hashes the output schema exactly as it will be sent.

    Args:
        schema: The schema returned by :func:`output_schema`.

    Returns:
        The hex SHA-256 over its canonical JSON.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
