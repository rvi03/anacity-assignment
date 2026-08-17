"""The HTML review page must agree with the sheet it is rendered from.

The page carries its own copy of the data, so the failure mode worth
testing is drift: a page that shows different rows, different rates, or
a different split from the workbook beside it. Every test here compares
the page against the frame it was built from rather than against a
recorded value.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from facility_prediction.evaluation import review_html


def _sheet() -> pd.DataFrame:
    """Two models over two records, with a known set of match marks."""
    return pd.DataFrame(
        [
            {
                "track": "traditional",
                "model": "selected_heads",
                "record": "B0001",
                "resident": "R001",
                "origin": "Mon 2024-01-01 09:00",
                "past_bookings": "Gym / Mon / 07:00 / booked Sun 08:00 ⏎ "
                "Pool / Wed / 18:00 / booked Tue 09:00 ⏎ "
                "Gym / Fri / 07:00 / booked Thu 08:00",
                "predicted_facility": "Gym",
                "predicted_usage_weekday": "Mon",
                "predicted_usage_hour": "07:00",
                "suggested_send": "Sun 2024-01-07 08:00",
                "actual_facility": "Gym",
                "actual_usage_weekday": "Tue",
                "actual_usage_hour": "07:00",
                "actual_booked": "Sun 2024-01-07 08:30",
                "match_facility": "✓",
                "match_usage_weekday": "✗",
                "match_usage_hour": "✓",
                "match_notification": "✓",
                "cold_start": False,
            },
            {
                "track": "baseline",
                "model": "frequency_recency",
                "record": "B0002",
                "resident": "R002",
                "origin": "Tue 2024-01-02 10:00",
                "past_bookings": "Pool / Sat / 18:00 / booked Fri 09:00",
                "predicted_facility": "Pool",
                "predicted_usage_weekday": "Sat",
                "predicted_usage_hour": "18:00",
                "suggested_send": "Fri 2024-01-12 09:00",
                "actual_facility": "Tennis",
                "actual_usage_weekday": "Sat",
                "actual_usage_hour": "19:00",
                "actual_booked": "Fri 2024-01-12 09:10",
                "match_facility": "✗",
                "match_usage_weekday": "✓",
                "match_usage_hour": "✗",
                "match_notification": "✗",
                "cold_start": True,
            },
        ]
    )


def _payload(page: str) -> dict:
    """Pull the embedded data back out of a rendered page."""
    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>',
        page,
        re.S,
    )
    assert match, "the page carries no payload"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_every_row_reaches_the_page():
    payload = _payload(review_html.render_page(_sheet(), "test", "because"))

    assert len(payload["rows"]) == 2


def test_rates_are_recomputed_from_the_rendered_marks():
    rates = {rate["model"]: rate for rate in review_html.model_rates(_sheet())}

    assert rates["selected_heads"]["components"] == {
        "facility": 1.0,
        "weekday": 0.0,
        "hour": 1.0,
        "notification": 1.0,
    }
    assert rates["selected_heads"]["overall"] == pytest.approx(0.75)
    assert rates["frequency_recency"]["overall"] == pytest.approx(0.25)


def test_models_are_ordered_by_score_so_the_tiles_read_top_down():
    rates = review_html.model_rates(_sheet())

    assert [rate["model"] for rate in rates] == [
        "selected_heads",
        "frequency_recency",
    ]


def test_history_is_trimmed_to_the_most_recent_bookings():
    payload = _payload(review_html.render_page(_sheet(), "test", "because"))
    history = payload["rows"][0][3]

    assert history.count("\n") == review_html.HISTORY_SHOWN - 1
    # the trim keeps the newest and drops the oldest
    assert "Fri / 07:00" in history
    assert "Mon / 07:00" not in history


def test_the_split_and_its_reason_are_stated_on_the_page():
    page = review_html.render_page(_sheet(), "validation", "still sealed")

    assert "validation" in page
    assert "still sealed" in page


def test_the_page_needs_no_network():
    page = review_html.render_page(_sheet(), "test", "because")

    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "https://" not in page
    assert "fetch(" not in page
    assert "<script src=" not in page


def test_labels_never_reach_the_dom_as_markup():
    """Row values are data; they must be inserted as text, not HTML."""
    page = review_html.render_page(_sheet(), "test", "because")
    script = page[page.index("<script>") :]

    assert "innerHTML" not in script


def test_every_tab_has_the_panel_it_controls():
    """A tab pointing at a missing panel renders an empty section."""
    page = review_html.render_page(_sheet(), "test", "because")
    controlled = set(re.findall(r'aria-controls="([^"]+)"', page))
    panels = set(re.findall(r'<section class="panel" id="([^"]+)"', page))

    assert controlled
    assert controlled == panels


def test_exactly_one_tab_starts_selected():
    """Two selected tabs would paint two panels on top of each other."""
    page = review_html.render_page(_sheet(), "test", "because")
    tablist = page[page.index('<nav class="tabs"') : page.index("</nav>")]

    assert tablist.count('aria-selected="true"') == 1
    assert tablist.count('aria-selected="false"') == 3

    # and every panel but that one starts hidden
    panels = page.count('role="tabpanel"')
    assert page.count('tabindex="0" hidden') == panels - 1


def test_internal_model_names_never_reach_the_screen():
    """`tier_b_adapter` names a training size nobody outside the build knows."""
    assert review_html.display_name("tier_b_adapter") == (
        "Fine-tuned language model"
    )
    assert review_html.display_name("frequency_recency") == (
        "Habit rule (baseline)"
    )


def test_the_recorded_name_is_still_shown_beside_the_label():
    """A reviewer matching the page to the CSV needs the recorded id."""
    page = review_html.render_page(_sheet(), "test", "because")
    payload = _payload(page)

    assert {rate["recorded"] for rate in payload["rates"]} == {
        "selected_heads",
        "frequency_recency",
    }


def test_each_model_is_described_on_the_models_tab():
    rates = review_html.model_rates(_sheet())

    assert all(rate["note"] for rate in rates)


def test_an_unknown_model_still_renders():
    """A new model must not break the page just for lacking a blurb."""
    sheet = _sheet()
    sheet.loc[0, "model"] = "something_new"

    rates = {rate["model"]: rate for rate in review_html.model_rates(sheet)}

    assert rates["something_new"]["note"] == ""
    assert rates["something_new"]["overall"] == pytest.approx(0.75)


def test_the_page_is_written_where_it_is_asked_for(tmp_path):
    destination = tmp_path / "nested" / review_html.PAGE_FILENAME

    size = review_html.write_page(_sheet(), "test", "because", destination)

    assert destination.is_file()
    assert size == len(destination.read_text(encoding="utf-8").encode("utf-8"))
