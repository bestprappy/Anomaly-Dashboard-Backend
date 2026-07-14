from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.severity_duration import (
    DURATION_BANDS,
    EVENT_COLUMNS,
    SEVERITY_BANDS,
    build_severity_duration_events,
    duration_intuition_report,
    duration_type_crosstab,
    recompute_quantile_severity,
    severity_band,
    severity_duration_matrix,
)


def _history(site_id: str, monthly_kwh: dict[str, float]) -> pd.DataFrame:
    """Build the subset of DataBillContainer.master_df used by the analysis."""

    months = pd.PeriodIndex(monthly_kwh, freq="M")
    return pd.DataFrame(
        {
            "Site_ID": site_id,
            "date": months.to_timestamp(),
            "kwh": list(monthly_kwh.values()),
        }
    )


def _classified(
    site_id: str,
    months: list[str],
    severities: list[float],
    anomaly_types: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": site_id,
            "anom_m": pd.PeriodIndex(months, freq="M"),
            "quantile_severity": severities,
            "anom_type": anomaly_types,
        }
    )


def test_recompute_quantile_severity_uses_band_width_and_clips_below_band() -> None:
    flagged = pd.DataFrame(
        {
            "target_kwh_next": [90.0, 110.0, 120.0, 160.0],
            "q05": [80.0, 80.0, 80.0, 80.0],
            "q95": [100.0, 100.0, 100.0, 100.0],
        },
        index=[10, 20, 30, 40],
    )

    result = recompute_quantile_severity(flagged)

    pd.testing.assert_series_equal(
        result,
        pd.Series([0.0, 0.5, 1.0, 3.0], index=flagged.index),
    )


def test_severity_band_boundaries_are_left_inclusive() -> None:
    scores = pd.Series([0.0, 0.999, 1.0, 2.999, 3.0, np.nan])

    result = severity_band(scores)

    assert result.iloc[:5].astype(str).tolist() == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
    ]
    assert pd.isna(result.iloc[5])
    assert list(result.cat.categories) == list(SEVERITY_BANDS)
    assert result.cat.ordered


@pytest.mark.parametrize(
    ("monthly_kwh", "expected_duration", "expected_band", "expected_type"),
    [
        (
            {
                "2024-01": 100,
                "2024-02": 100,
                "2024-03": 100,
                "2024-04": 100,
                "2024-05": 200,
                "2024-06": 100,
            },
            1,
            "Single month",
            "spike_up",
        ),
        (
            {
                "2024-01": 100,
                "2024-02": 100,
                "2024-03": 100,
                "2024-04": 100,
                "2024-05": 200,
                "2024-06": 150,
                "2024-07": 100,
            },
            2,
            "2-3 months",
            "step_up",
        ),
        (
            {
                "2024-01": 100,
                "2024-02": 100,
                "2024-03": 100,
                "2024-04": 100,
                "2024-05": 200,
                "2024-06": 150,
                "2024-07": 140,
                "2024-08": 100,
            },
            3,
            "2-3 months",
            "step_up",
        ),
        (
            {
                "2024-01": 100,
                "2024-02": 100,
                "2024-03": 100,
                "2024-04": 100,
                "2024-05": 200,
                "2024-06": 150,
                "2024-07": 140,
                "2024-08": 130,
                "2024-09": 100,
            },
            4,
            ">=4 months",
            "step_up",
        ),
    ],
)
def test_duration_buckets_count_consecutive_elevated_months(
    monthly_kwh: dict[str, float],
    expected_duration: int,
    expected_band: str,
    expected_type: str,
) -> None:
    history = _history("site-a", monthly_kwh)
    classified = _classified("site-a", ["2024-05"], [1.25], [expected_type])

    events = build_severity_duration_events(classified, history)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["duration_months"] == expected_duration
    assert event["duration_band"] == expected_band
    assert bool(event["duration_confirmed"])
    assert not bool(event["right_censored"])
    assert event["duration_status"] == "returned_below_threshold"
    assert event["expected_type_from_duration"] == expected_type
    assert bool(event["intuition_match"])


def test_open_four_month_run_is_confirmed_as_long_despite_right_censoring() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
            "2024-06": 150,
            "2024-07": 140,
            "2024-08": 130,
        },
    )
    classified = _classified("SITE-A", ["2024-05"], [3.0], ["step_up"])

    event = build_severity_duration_events(classified, history).iloc[0]

    assert event["duration_months"] == 4
    assert event["duration_band"] == ">=4 months"
    assert bool(event["duration_confirmed"])
    assert bool(event["right_censored"])
    assert event["duration_status"] == "end_of_history"


def test_short_end_of_history_run_is_right_censored_and_excluded_from_matrix() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
        },
    )
    classified = _classified("SITE-A", ["2024-05"], [0.5], ["spike_up"])

    events = build_severity_duration_events(classified, history)
    event = events.iloc[0]

    assert event["duration_months"] == 1
    assert pd.isna(event["duration_band"])
    assert not bool(event["duration_confirmed"])
    assert bool(event["right_censored"])
    assert event["duration_status"] == "end_of_history"
    assert severity_duration_matrix(events).to_numpy().sum() == 0


def test_missing_calendar_month_censors_duration_instead_of_bridging_gap() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
            # June is absent; July proves this is a gap rather than history end.
            "2024-07": 100,
        },
    )
    classified = _classified("SITE-A", ["2024-05"], [0.5], ["spike_up"])

    event = build_severity_duration_events(classified, history).iloc[0]

    assert event["duration_months"] == 1
    assert event["end_month"] == pd.Period("2024-05", freq="M")
    assert pd.isna(event["duration_band"])
    assert not bool(event["duration_confirmed"])
    assert bool(event["right_censored"])
    assert event["duration_status"] == "missing_month"


def test_zero_kwh_is_a_missing_read_that_censors_duration() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
            # A zero bill is preprocessed as a missing read, not a return.
            "2024-06": 0,
            "2024-07": 100,
        },
    )
    classified = _classified("SITE-A", ["2024-05"], [0.5], ["spike_up"])

    event = build_severity_duration_events(classified, history).iloc[0]

    assert event["duration_months"] == 1
    assert event["end_month"] == pd.Period("2024-05", freq="M")
    assert pd.isna(event["duration_band"])
    assert not bool(event["duration_confirmed"])
    assert bool(event["right_censored"])
    assert event["duration_status"] == "missing_month"


def test_detection_walks_back_to_earlier_qualifying_event_start() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
            "2024-06": 190,
            "2024-07": 180,
            "2024-08": 100,
        },
    )
    # The model did not surface the event until July, two months after onset.
    classified = _classified("SITE-A", ["2024-07"], [2.0], ["step_up"])

    event = build_severity_duration_events(classified, history).iloc[0]

    assert event["detection_month"] == pd.Period("2024-07", freq="M")
    assert event["start_month"] == pd.Period("2024-05", freq="M")
    assert event["months_before_detection"] == 2
    assert event["duration_months"] == 3
    assert event["duration_band"] == "2-3 months"
    assert event["detection_quantile_severity"] == pytest.approx(2.0)


def test_overlapping_flags_collapse_to_one_event_using_onset_not_peak_severity() -> None:
    history = _history(
        "SITE-A",
        {
            "2024-01": 100,
            "2024-02": 100,
            "2024-03": 100,
            "2024-04": 100,
            "2024-05": 200,
            "2024-06": 220,
            "2024-07": 210,
            "2024-08": 100,
        },
    )
    classified = _classified(
        "site-a",
        ["2024-05", "2024-06", "2024-07"],
        [0.5, 4.2, 2.0],
        ["step_up", "step_up", "step_up"],
    )

    events = build_severity_duration_events(classified, history)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["event_id"] == "SITE-A:2024-05"
    assert event["duration_months"] == 3
    assert event["n_flagged_months"] == 3
    assert event["detection_quantile_severity"] == pytest.approx(0.5)
    assert event["peak_quantile_severity"] == pytest.approx(4.2)
    assert event["severity_band"] == "Low"

    matrix = severity_duration_matrix(events)
    assert matrix.loc["2-3 months", "Low"] == 1
    assert matrix.loc["2-3 months", "High"] == 0


def test_matrix_is_always_three_by_three_including_empty_categories() -> None:
    one_event = pd.DataFrame(
        {
            "duration_band": ["Single month"],
            "severity_band": ["Medium"],
            "duration_confirmed": [True],
        }
    )

    counts = severity_duration_matrix(one_event)
    percentages = severity_duration_matrix(one_event, row_percent=True)
    empty = severity_duration_matrix(one_event.iloc[0:0])

    for matrix in (counts, percentages, empty):
        assert matrix.shape == (3, 3)
        assert matrix.index.tolist() == list(DURATION_BANDS)
        assert matrix.columns.tolist() == list(SEVERITY_BANDS)
        assert matrix.index.name == "Duration"
        assert matrix.columns.name == "Severity"

    assert counts.loc["Single month", "Medium"] == 1
    assert counts.to_numpy().sum() == 1
    assert percentages.loc["Single month", "Medium"] == pytest.approx(100.0)
    assert percentages.loc["2-3 months"].eq(0).all()
    assert empty.to_numpy().sum() == 0


def test_intuition_report_counts_matches_and_excludes_unconfirmed_events() -> None:
    events = pd.DataFrame(
        {
            "duration_band": ["Single month", "2-3 months", ">=4 months", pd.NA],
            "duration_confirmed": [True, True, True, False],
            "detection_anom_type": ["spike_up", "step_up", "spike_up", "spike_up"],
            "intuition_match": [True, True, False, pd.NA],
        }
    )

    report = duration_intuition_report(events)

    assert report["n_events_total"] == 4
    assert report["n_events_in_matrix"] == 3
    assert report["n_events_excluded_unconfirmed_duration"] == 1
    assert report["overall_agreement"]["successes"] == 2
    assert report["overall_agreement"]["n"] == 3
    assert report["overall_agreement"]["rate"] == pytest.approx(0.6667)
    assert report["overall_agreement"]["ci95"][0] is not None
    assert report["overall_agreement"]["ci95"][1] is not None

    assert report["by_duration"]["Single month"] == {
        "expected_type": "spike_up",
        "successes": 1,
        "n": 1,
        "rate": 1.0,
        "ci95": [pytest.approx(0.2065), 1.0],
    }
    assert report["by_duration"]["2-3 months"]["expected_type"] == "step_up"
    assert report["by_duration"]["2-3 months"]["rate"] == 1.0
    assert report["by_duration"][">=4 months"]["expected_type"] == "step_up"
    assert report["by_duration"][">=4 months"]["rate"] == 0.0

    crosstab = duration_type_crosstab(events)
    assert crosstab.shape == (3, 2)
    assert crosstab.loc["Single month", "spike_up"] == 1
    assert crosstab.loc["2-3 months", "step_up"] == 1
    assert crosstab.loc[">=4 months", "spike_up"] == 1
    assert crosstab.loc[">=4 months", "step_up"] == 0


def test_empty_classified_input_returns_stable_event_schema() -> None:
    classified = pd.DataFrame(
        columns=["site_id", "anom_m", "quantile_severity", "anom_type"]
    )
    history = pd.DataFrame(columns=["Site_ID", "date", "kwh"])

    events = build_severity_duration_events(classified, history)

    assert events.empty
    assert events.columns.tolist() == EVENT_COLUMNS
