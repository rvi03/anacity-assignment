"""Hand-checked metric fixtures — every number here is worked by hand.

The interesting cases are the ones a scoring rule gets wrong quietly:
the exact edge of the notification tolerance from both sides, a delay
of zero, a clamped negative prediction, an hour pair that wraps around
midnight, a class the model never predicts, and a baseline with nothing
to divide by.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from facility_prediction.evaluation import evaluate

TARGET = evaluate.TARGET_COLUMNS
PREDICTED = evaluate.PREDICTED_COLUMNS
MATCH_RATIO = 1.25
SUPPORT_MINUTES = (30, 60, 180)


def joined(rows=1, **overrides):
    """One or more rows that match on every component, before overrides."""
    data = {
        TARGET[evaluate.FACILITY]: ["Gym"] * rows,
        PREDICTED[evaluate.FACILITY]: ["Gym"] * rows,
        TARGET[evaluate.USAGE_WEEKDAY]: [2] * rows,
        PREDICTED[evaluate.USAGE_WEEKDAY]: [2] * rows,
        TARGET[evaluate.USAGE_HOUR]: [7] * rows,
        PREDICTED[evaluate.USAGE_HOUR]: [7] * rows,
        TARGET[evaluate.NOTIFICATION]: [99.0] * rows,
        PREDICTED[evaluate.NOTIFICATION]: [99.0] * rows,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def matches_of(frame):
    return evaluate.component_matches(frame, MATCH_RATIO)


def test_an_exact_component_matches_only_on_equality():
    frame = joined(
        2,
        **{
            TARGET[evaluate.FACILITY]: ["Gym", "Pool"],
            PREDICTED[evaluate.FACILITY]: ["Gym", "Gym"],
        },
    )

    matched = matches_of(frame)[evaluate.FACILITY]

    assert list(matched) == [True, False]


def test_the_notification_matches_at_exactly_plus_25_percent():
    # (124 + 1) / (99 + 1) = 1.25 exactly
    frame = joined(**{PREDICTED[evaluate.NOTIFICATION]: [124.0]})

    matched = matches_of(frame)[evaluate.NOTIFICATION]

    assert list(matched) == [True]


def test_the_notification_matches_at_exactly_minus_25_percent():
    # (79 + 1) / (99 + 1) = 0.8, the reciprocal of 1.25
    frame = joined(**{PREDICTED[evaluate.NOTIFICATION]: [79.0]})

    matched = matches_of(frame)[evaluate.NOTIFICATION]

    assert list(matched) == [True]


def test_just_outside_the_notification_tolerance_does_not_match():
    frame = joined(2, **{PREDICTED[evaluate.NOTIFICATION]: [124.5, 78.5]})

    matched = matches_of(frame)[evaluate.NOTIFICATION]

    assert list(matched) == [False, False]


def test_a_zero_delay_is_matched_within_a_quarter_of_a_minute():
    frame = joined(
        2,
        **{
            TARGET[evaluate.NOTIFICATION]: [0.0, 0.0],
            PREDICTED[evaluate.NOTIFICATION]: [0.25, 0.3],
        },
    )

    matched = matches_of(frame)[evaluate.NOTIFICATION]

    assert list(matched) == [True, False]


def test_a_negative_prediction_is_clamped_and_counted():
    predicted = pd.Series([-30.0, 0.0, 60.0, -1.0])

    clamped, rate = evaluate.clamp_delays(predicted)

    assert list(clamped) == [0.0, 0.0, 60.0, 0.0]
    assert rate == 0.5


def test_the_match_uses_the_clamped_delay_not_the_raw_one():
    frame = joined(
        **{
            TARGET[evaluate.NOTIFICATION]: [0.0],
            PREDICTED[evaluate.NOTIFICATION]: [-0.1],
        }
    )

    matched = matches_of(frame)[evaluate.NOTIFICATION]

    assert list(matched) == [True]


def test_score_counts_the_components_that_matched():
    frame = joined(
        4,
        **{
            PREDICTED[evaluate.FACILITY]: ["Gym", "Pool", "Pool", "Gym"],
            PREDICTED[evaluate.USAGE_WEEKDAY]: [2, 2, 3, 3],
            PREDICTED[evaluate.USAGE_HOUR]: [7, 7, 8, 7],
            PREDICTED[evaluate.NOTIFICATION]: [99.0, 99.0, 400.0, 400.0],
        },
    )

    scores = evaluate.score_rows(matches_of(frame))

    assert list(scores) == [4, 3, 0, 2]


def test_the_roll_ups_are_the_shares_they_claim_to_be():
    frame = joined(
        4,
        **{
            PREDICTED[evaluate.FACILITY]: ["Gym", "Pool", "Pool", "Gym"],
            PREDICTED[evaluate.USAGE_WEEKDAY]: [2, 2, 3, 3],
            PREDICTED[evaluate.USAGE_HOUR]: [7, 7, 8, 7],
            PREDICTED[evaluate.NOTIFICATION]: [99.0, 99.0, 400.0, 400.0],
        },
    )

    summary = evaluate.match_summary(matches_of(frame))

    assert summary["score_mean"] == 2.25
    assert summary["overall"] == 0.5625
    assert summary["strict"] == 0.25
    assert summary["at_least_three"] == 0.5
    assert summary[evaluate.FACILITY] == 0.5


def test_circular_hour_error_wraps_around_midnight():
    actual = pd.Series([23, 0, 6, 12])
    predicted = pd.Series([0, 23, 7, 0])

    error = evaluate.circular_hour_error(actual, predicted)

    assert list(error) == [1, 1, 1, 12]


def test_within_one_hour_uses_the_circular_distance():
    actual = pd.Series([23, 12])
    predicted = pd.Series([0, 0])

    metrics = evaluate.hour_metrics(actual, predicted)

    assert metrics["within_1_hour"] == 0.5
    assert metrics["circular_mae_hours"] == 6.5


def test_macro_f1_penalises_a_class_that_is_never_predicted():
    actual = pd.Series(["Gym", "Gym", "Pool"])
    predicted = pd.Series(["Gym", "Gym", "Gym"])

    # Gym: tp 2, predicted 3, actual 2 -> f1 0.8. Pool: f1 0.0.
    assert evaluate.macro_f1(actual, predicted) == 0.4
    assert evaluate.accuracy(actual, predicted) == pytest.approx(2 / 3)


def test_per_class_recall_omits_a_class_that_never_occurs():
    actual = pd.Series(["Gym", "Gym", "Pool"])
    predicted = pd.Series(["Gym", "Tennis", "Gym"])

    recalls = evaluate.per_class_recall(actual, predicted)

    assert recalls == {"Gym": 0.5, "Pool": 0.0}


def test_the_confusion_matrix_covers_a_label_only_ever_predicted():
    actual = pd.Series(["Gym", "Gym"])
    predicted = pd.Series(["Gym", "Tennis"])

    matrix = evaluate.confusion_matrix(actual, predicted)

    assert list(matrix.index) == ["Gym", "Tennis"]
    assert matrix.loc["Gym", "Tennis"] == 1
    assert matrix.loc["Tennis", "Gym"] == 0


def test_top_3_accuracy_counts_the_third_likeliest_class():
    actual = pd.Series(["Pool", "Tennis"])
    probabilities = pd.DataFrame(
        {
            "Gym": [0.5, 0.5],
            "Pool": [0.2, 0.3],
            "Tennis": [0.2, 0.1],
            "Yoga Room": [0.1, 0.1],
        }
    )

    assert evaluate.top_k_accuracy(actual, probabilities) == 1.0
    assert evaluate.top_k_accuracy(actual, probabilities, k=1) == 0.0


def test_top_k_refuses_a_label_with_no_probability_column():
    actual = pd.Series(["Sauna"])
    probabilities = pd.DataFrame({"Gym": [1.0]})

    with pytest.raises(ValueError, match="Sauna"):
        evaluate.top_k_accuracy(actual, probabilities)


def test_notification_metrics_are_the_hand_worked_values():
    actual = pd.Series([0.0] * 10)
    predicted = pd.Series([float(step * 10) for step in range(10)])

    metrics = evaluate.notification_metrics(actual, predicted, SUPPORT_MINUTES)

    assert metrics["mae_minutes"] == 45.0
    assert metrics["median_ae_minutes"] == 45.0
    assert metrics["p90_ae_minutes"] == pytest.approx(81.0)
    assert metrics["signed_bias_minutes"] == 45.0
    assert metrics["within_30_minutes"] == 0.4
    assert metrics["within_60_minutes"] == 0.7
    assert metrics["within_180_minutes"] == 1.0


def test_the_median_absolute_log_ratio_is_symmetric():
    actual = pd.Series([99.0, 99.0])
    predicted = pd.Series([124.0, 79.0])

    metrics = evaluate.notification_metrics(actual, predicted, SUPPORT_MINUTES)

    assert metrics["median_abs_log_ratio"] == pytest.approx(math.log(1.25))


def test_a_negative_delay_has_no_log_ratio():
    actual = pd.Series([10.0])
    predicted = pd.Series([-1.0])

    with pytest.raises(ValueError, match="non-negative"):
        evaluate.notification_match(actual, predicted, MATCH_RATIO)


def test_match_lift_is_positive_when_the_model_matches_more_often():
    lift = evaluate.match_lift(model=0.6, baseline=0.4)

    assert lift.absolute == pytest.approx(0.2)
    assert lift.relative == pytest.approx(0.5)


def test_match_lift_is_undefined_when_the_baseline_never_matches():
    lift = evaluate.match_lift(model=0.3, baseline=0.0)

    assert lift.relative is None
    assert lift.absolute == 0.3


def test_mae_reduction_is_positive_when_the_model_errs_less():
    reduction = evaluate.mae_reduction(model=80.0, baseline=100.0)

    assert reduction.absolute == 20.0
    assert reduction.relative == pytest.approx(0.2)


def test_mae_reduction_is_undefined_when_the_baseline_error_is_zero():
    reduction = evaluate.mae_reduction(model=5.0, baseline=0.0)

    assert reduction.relative is None
    assert reduction.absolute == -5.0


def test_evaluating_reports_rows_heads_and_matches():
    frame = joined(2, **{PREDICTED[evaluate.NOTIFICATION]: [99.0, -20.0]})

    report = evaluate.evaluate_predictions(frame, MATCH_RATIO, SUPPORT_MINUTES)

    assert report["rows"] == 2
    assert report["heads"][evaluate.NOTIFICATION]["clamp_rate"] == 0.5
    assert report["heads"][evaluate.USAGE_HOUR]["circular_mae_hours"] == 0.0
    assert report["matches"]["strict"] == 0.5
    assert "top_3_accuracy" not in report["heads"][evaluate.FACILITY]


def test_top_k_is_reported_for_every_classification_head_or_none():
    frame = joined(1)
    probabilities = {evaluate.FACILITY: pd.DataFrame({"Gym": [1.0]})}

    with pytest.raises(ValueError, match="every classification head"):
        evaluate.evaluate_predictions(
            frame, MATCH_RATIO, SUPPORT_MINUTES, probabilities
        )


def test_an_empty_prediction_set_is_refused():
    frame = joined(0)

    with pytest.raises(ValueError, match="empty"):
        evaluate.evaluate_predictions(frame, MATCH_RATIO, SUPPORT_MINUTES)


def test_a_missing_prediction_column_is_named():
    frame = joined(1).drop(columns=[PREDICTED[evaluate.USAGE_HOUR]])

    with pytest.raises(ValueError, match="predicted_usage_hour"):
        evaluate.component_matches(frame, MATCH_RATIO)


def test_a_tolerance_that_is_not_a_tolerance_is_refused():
    frame = joined(1)

    with pytest.raises(ValueError, match=r"above 1\.0"):
        evaluate.component_matches(frame, 1.0)


# --- selection and the verdict ----------------------------------------


def perfect_and_wrong(rows: int, wrong: int):
    """A report pair: the model right on ``rows``, the baseline on fewer."""
    model = evaluate.evaluate_predictions(
        joined(rows), MATCH_RATIO, SUPPORT_MINUTES
    )
    baseline = evaluate.evaluate_predictions(
        joined(
            rows,
            **{
                PREDICTED[evaluate.FACILITY]: ["Pool"] * wrong
                + ["Gym"] * (rows - wrong)
            },
        ),
        MATCH_RATIO,
        SUPPORT_MINUTES,
    )
    return model, baseline


def test_the_selection_score_is_the_share_of_matched_components():
    report = evaluate.evaluate_predictions(
        joined(2, **{PREDICTED[evaluate.FACILITY]: ["Pool", "Gym"]}),
        MATCH_RATIO,
        SUPPORT_MINUTES,
    )

    assert evaluate.selection_score(report) == pytest.approx(7 / 8)


def test_a_report_without_a_match_summary_cannot_be_selected_on():
    with pytest.raises(ValueError, match="no match summary"):
        evaluate.selection_score({"rows": 1})


def test_a_better_model_beats_the_baseline():
    model, baseline = perfect_and_wrong(4, 2)

    comparison = evaluate.compare_reports(model, baseline)

    assert comparison["selection_score"]["verdict"] == "beats"
    assert comparison["selection_score"]["model"] == 1.0


def test_an_equal_model_ties_with_the_baseline():
    model, baseline = perfect_and_wrong(4, 0)

    comparison = evaluate.compare_reports(model, baseline)

    assert comparison["selection_score"]["verdict"] == "ties"


def test_a_worse_model_is_reported_as_losing():
    better, worse = perfect_and_wrong(4, 2)

    comparison = evaluate.compare_reports(worse, better)

    assert comparison["selection_score"]["verdict"] == "loses"
    assert comparison["match_lift"]["facility"]["absolute"] < 0


def test_the_comparison_carries_a_lift_for_every_component():
    model, baseline = perfect_and_wrong(4, 2)

    lifts = evaluate.compare_reports(model, baseline)["match_lift"]

    assert set(lifts) == {*evaluate.COMPONENTS, "overall", "strict"}


def test_the_comparison_reports_the_notification_error_reduction():
    model = evaluate.evaluate_predictions(
        joined(2), MATCH_RATIO, SUPPORT_MINUTES
    )
    baseline = evaluate.evaluate_predictions(
        joined(2, **{PREDICTED[evaluate.NOTIFICATION]: [199.0, 199.0]}),
        MATCH_RATIO,
        SUPPORT_MINUTES,
    )

    reduction = evaluate.compare_reports(model, baseline)["mae_reduction"]

    assert reduction["model"] == pytest.approx(0.0)
    assert reduction["absolute"] == pytest.approx(100.0)
    assert reduction["relative"] == pytest.approx(1.0)


def test_a_zero_baseline_leaves_the_relative_lift_undefined():
    model, baseline = perfect_and_wrong(4, 4)

    comparison = evaluate.compare_reports(model, baseline)

    assert comparison["match_lift"]["facility"]["relative"] is None
    assert comparison["match_lift"]["facility"]["absolute"] == 1.0
