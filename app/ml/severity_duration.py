"""Event-level severity/duration analysis for upper-tail anomalies.

The quantile model emits flagged *site-months*.  A sustained level shift can
therefore produce several flags for what is operationally one event.  This
module converts the classified rows plus their raw monthly histories into one
row per event and builds the requested 3 x 3 matrix:

    columns: Low, Medium, High quantile severity
    rows:    Single month, 2-3 months, >=4 months

Duration is deliberately observed from the raw series rather than copied from
``anom_type``.  It is the consecutive run, starting at the anomaly month, for
which kWh stays above ``elevated_ratio * pre-event baseline``.  This gives us a
useful consistency check of the existing four-month-median spike/step
classifier instead of simply restating its label.

Short open runs at the end of history (or before a missing calendar month) are
right-censored.  They are retained in the event audit table but excluded from
the matrix because they may later move from 1 to 2-3 or >=4 months.  Once four
elevated months have been observed, the >=4 bucket is known even if the event
has not ended.

Each confirmed (duration_band, severity_band) cell also maps to a fixed
business ``action`` — Ignore / Review / Investigate — via ``action_for`` /
``ACTION_MATRIX_BY_BAND``.  This is what downstream consumers (the event
table, the bulk plot export in ``app/ml/plotting.py``) use to turn the matrix
into a worklist rather than just a report.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from app.ml.classify import SURFACED_TYPES


SEVERITY_BANDS = ("Low", "Medium", "High")
DURATION_BANDS = ("Single month", "2-3 months", ">=4 months")

# Business action per confirmed (duration_band, severity_band) cell.
# Ordered least -> most urgent; order matters for _ACTION_PRIORITY below,
# which resolves "this site has multiple events with different actions"
# down to the single most urgent one.
ACTION_LABELS = ("Ignore", "Review", "Investigate")

_ACTION_MATRIX: dict[tuple[str, str], str] = {
    ("Single month", "Low"): "Ignore",
    ("Single month", "Medium"): "Review",
    ("Single month", "High"): "Investigate",
    ("2-3 months", "Low"): "Review",
    ("2-3 months", "Medium"): "Review",
    ("2-3 months", "High"): "Investigate",
    (">=4 months", "Low"): "Investigate",
    (">=4 months", "Medium"): "Investigate",
    (">=4 months", "High"): "Investigate",
}

# Priority order for picking one action when something (e.g. a site's export
# folder) has to resolve multiple events down to a single action — always
# surface the most urgent one, never let "Investigate" hide behind "Ignore".
ACTION_PRIORITY = {label: i for i, label in enumerate(ACTION_LABELS)}

EVENT_COLUMNS = [
    "site_id",
    "event_id",
    "detection_month",
    "start_month",
    "months_before_detection",
    "end_month",
    "first_return_month",
    "baseline_kwh",
    "elevated_threshold_kwh",
    "onset_kwh",
    "onset_ratio",
    "duration_months",
    "duration_band",
    "duration_confirmed",
    "right_censored",
    "duration_status",
    "detection_quantile_severity",
    "peak_quantile_severity",
    "severity_band",
    "action",
    "detection_anom_type",
    "expected_type_from_duration",
    "intuition_match",
    "n_flagged_months",
]


@dataclass(frozen=True)
class SeverityDurationConfig:
    """Thresholds used to turn classified site-months into events.

    ``severity_medium_min`` and ``severity_high_min`` preserve the dashboard's
    existing score cut points (<1, 1-<3, >=3) while relabelling the tiers as
    Low/Medium/High for the matrix.
    """

    severity_medium_min: float = 1.0
    severity_high_min: float = 3.0
    baseline_months: int = 4
    min_baseline_months: int = 4
    up_ratio: float = 1.5
    elevated_ratio: float = 1.3
    long_duration_months: int = 4

    def __post_init__(self) -> None:
        if not np.isfinite(self.severity_medium_min):
            raise ValueError("severity_medium_min must be finite")
        if not (
            np.isfinite(self.severity_high_min)
            and self.severity_high_min > self.severity_medium_min
        ):
            raise ValueError("severity_high_min must be greater than severity_medium_min")
        if self.baseline_months < 1:
            raise ValueError("baseline_months must be >= 1")
        if not (1 <= self.min_baseline_months <= self.baseline_months):
            raise ValueError("min_baseline_months must be between 1 and baseline_months")
        if not (np.isfinite(self.up_ratio) and self.up_ratio > 1):
            raise ValueError("up_ratio must be > 1")
        if not (np.isfinite(self.elevated_ratio) and self.elevated_ratio > 1):
            raise ValueError("elevated_ratio must be > 1")
        if self.long_duration_months != 4:
            raise ValueError("long_duration_months must be 4 for the fixed >=4 matrix bucket")


def recompute_quantile_severity(flagged: pd.DataFrame) -> pd.Series:
    """Recompute the backend's number-of-band-widths severity definition."""

    required = {"target_kwh_next", "q05", "q95"}
    missing = required - set(flagged.columns)
    if missing:
        raise ValueError(f"flagged is missing required columns: {sorted(missing)}")
    width = (flagged["q95"] - flagged["q05"]).clip(lower=1e-9)
    over = (flagged["target_kwh_next"] - flagged["q95"]).clip(lower=0)
    return (over / width).round(3)


def severity_band(
    scores: pd.Series | Iterable[float],
    config: SeverityDurationConfig = SeverityDurationConfig(),
) -> pd.Series:
    """Assign ordered Low/Medium/High categories to continuous scores."""

    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores)
    numeric = pd.to_numeric(scores, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, config.severity_medium_min, config.severity_high_min, np.inf],
        labels=SEVERITY_BANDS,
        right=False,
        ordered=True,
    )


def action_for(duration_band, severity_band_value) -> str | None:
    """Business action for a confirmed (duration_band, severity_band) cell.

    Returns None when either band is unset — e.g. an event whose duration
    hasn't been confirmed yet (still open, under 4 elevated months, so it
    could still move rows). Callers should treat None as "not enough data
    to act on yet", not default it to any particular action.
    """
    if pd.isna(duration_band) or pd.isna(severity_band_value):
        return None
    return _ACTION_MATRIX.get((str(duration_band), str(severity_band_value)))


def action_matrix() -> pd.DataFrame:
    """Static 3x3 grid of the business action per (duration, severity) cell.

    Same axes as severity_duration_matrix() — lets the frontend render the
    action legend without hardcoding the 9 labels client-side.
    """
    return pd.DataFrame(
        [[_ACTION_MATRIX[(d, s)] for s in SEVERITY_BANDS] for d in DURATION_BANDS],
        index=pd.Index(DURATION_BANDS, name="Duration"),
        columns=pd.Index(SEVERITY_BANDS, name="Severity"),
    )


def _normalise_classified(classified: pd.DataFrame) -> pd.DataFrame:
    required = {"site_id", "quantile_severity", "anom_type"}
    missing = required - set(classified.columns)
    if missing:
        raise ValueError(f"classified is missing required columns: {sorted(missing)}")

    d = classified.copy()
    if "anom_m" not in d.columns:
        if "bill_month" not in d.columns:
            raise ValueError("classified needs anom_m or bill_month")
        d["anom_m"] = pd.to_datetime(d["bill_month"]).dt.to_period("M") + 1
    else:
        d["anom_m"] = pd.PeriodIndex(d["anom_m"], freq="M")

    d["site_id"] = d["site_id"].astype(str).str.upper().str.strip()
    d["quantile_severity"] = pd.to_numeric(d["quantile_severity"], errors="coerce")
    return d.sort_values(["site_id", "anom_m"], kind="stable").reset_index(drop=True)


def _build_series_map(full_history: pd.DataFrame) -> dict[str, pd.Series]:
    required = {"Site_ID", "date", "kwh"}
    missing = required - set(full_history.columns)
    if missing:
        raise ValueError(f"full_history is missing required columns: {sorted(missing)}")

    d = full_history[list(required)].copy()
    d["site_id"] = d["Site_ID"].astype(str).str.upper().str.strip()
    d["month"] = pd.to_datetime(d["date"], errors="coerce").dt.to_period("M")
    # Match feature preprocessing: zero is a missing read, not a genuine
    # return to baseline. It must censor a short run rather than confirm it.
    d["kwh"] = pd.to_numeric(d["kwh"], errors="coerce").replace(0, np.nan)
    d = d.dropna(subset=["site_id", "month"])

    duplicate_mask = d.duplicated(["site_id", "month"], keep=False)
    if duplicate_mask.any():
        examples = (
            d.loc[duplicate_mask, ["site_id", "month"]]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .to_dict(orient="records")
        )
        raise ValueError(
            "full_history has duplicate site-month rows; enable the duplicate/common "
            f"site drop options before duration analysis. Examples: {examples}"
        )

    return {
        site_id: group.set_index("month")["kwh"].sort_index()
        for site_id, group in d.groupby("site_id", sort=False)
    }


def _duration_band(duration: int, confirmed: bool, config: SeverityDurationConfig):
    if not confirmed or duration < 1:
        return pd.NA
    if duration == 1:
        return DURATION_BANDS[0]
    if duration < config.long_duration_months:
        return DURATION_BANDS[1]
    return DURATION_BANDS[2]


def _baseline_before(
    series: pd.Series,
    onset: pd.Period,
    config: SeverityDurationConfig,
) -> float | None:
    baseline_index = pd.period_range(
        onset - config.baseline_months,
        onset - 1,
        freq="M",
    )
    baseline_values = series.reindex(baseline_index)
    finite_baseline = baseline_values[np.isfinite(baseline_values)]
    if len(finite_baseline) < config.min_baseline_months:
        return None
    baseline = float(finite_baseline.median())
    return baseline if np.isfinite(baseline) and baseline > 0 else None


def _is_qualifying_start(
    series: pd.Series,
    candidate: pd.Period,
    detection: pd.Period,
    config: SeverityDurationConfig,
) -> bool:
    """Whether ``candidate`` can be the same event's earlier true start."""

    if candidate not in series.index or detection < candidate:
        return False
    baseline = _baseline_before(series, candidate, config)
    if baseline is None:
        return False
    candidate_kwh = series.loc[candidate]
    if not np.isfinite(candidate_kwh) or candidate_kwh < config.up_ratio * baseline:
        return False

    # Do not bridge a return-to-baseline month or a missing calendar month.
    run = series.reindex(pd.period_range(candidate, detection, freq="M"))
    return bool(
        run.notna().all()
        and np.isfinite(run).all()
        and (run >= config.elevated_ratio * baseline).all()
    )


def _find_event_start(
    series: pd.Series | None,
    detection: pd.Period,
    config: SeverityDurationConfig,
) -> pd.Period:
    """Walk back from the first surfaced flag to an earlier qualifying jump.

    The ML model can miss the first elevated month and flag the next one. A
    prior month is adopted only when it independently clears the UP threshold
    against its own four-month baseline and the series remains continuously
    elevated through the detection month.
    """

    if series is None or series.empty:
        return detection
    start = detection
    while _is_qualifying_start(series, start - 1, detection, config):
        start -= 1
    return start


def _measure_duration(
    series: pd.Series | None,
    onset: pd.Period,
    config: SeverityDurationConfig,
) -> dict:
    result = {
        "start_month": onset,
        "end_month": pd.NaT,
        "first_return_month": pd.NaT,
        "baseline_kwh": np.nan,
        "elevated_threshold_kwh": np.nan,
        "onset_kwh": np.nan,
        "onset_ratio": np.nan,
        "duration_months": 0,
        "duration_band": pd.NA,
        "duration_confirmed": False,
        "right_censored": False,
        "duration_status": "unknown",
    }
    if series is None or series.empty or onset not in series.index:
        result["duration_status"] = "missing_onset"
        return result

    baseline = _baseline_before(series, onset, config)
    if baseline is None:
        result["duration_status"] = "insufficient_baseline"
        return result
    onset_kwh = series.loc[onset]
    if isinstance(onset_kwh, pd.Series):
        raise ValueError(f"duplicate history values remain for onset {onset}")
    onset_kwh = float(onset_kwh) if np.isfinite(onset_kwh) else np.nan
    result["baseline_kwh"] = baseline
    result["onset_kwh"] = onset_kwh

    result["onset_ratio"] = onset_kwh / baseline if np.isfinite(onset_kwh) else np.nan
    if not np.isfinite(onset_kwh):
        result["duration_status"] = "missing_onset"
        return result
    if onset_kwh < config.up_ratio * baseline:
        result["duration_status"] = "below_up_threshold"
        return result

    elevated_threshold = config.elevated_ratio * baseline
    result["elevated_threshold_kwh"] = elevated_threshold
    last_history_month = series.index.max()
    duration = 0
    last_elevated = pd.NaT
    cursor = onset

    while cursor <= last_history_month:
        if cursor not in series.index or not np.isfinite(series.loc[cursor]):
            result["right_censored"] = True
            result["duration_status"] = "missing_month"
            break
        value = float(series.loc[cursor])
        if value < elevated_threshold:
            result["first_return_month"] = cursor
            result["duration_status"] = "returned_below_threshold"
            break
        duration += 1
        last_elevated = cursor
        cursor += 1
    else:
        result["right_censored"] = True
        result["duration_status"] = "end_of_history"

    confirmed = duration >= config.long_duration_months or not result["right_censored"]
    result["duration_months"] = duration
    result["end_month"] = last_elevated
    result["duration_confirmed"] = bool(confirmed and duration >= 1)
    result["duration_band"] = _duration_band(duration, result["duration_confirmed"], config)
    return result


def build_severity_duration_events(
    classified: pd.DataFrame,
    full_history: pd.DataFrame,
    config: SeverityDurationConfig = SeverityDurationConfig(),
) -> pd.DataFrame:
    """Collapse classified site-month flags into an event-level audit table.

    The first surfaced flag is the detection month. The raw series is walked
    backward to an earlier qualifying jump when necessary, so duration starts
    at the observed event onset. ``detection_quantile_severity`` is used for
    the matrix because an unflagged earlier onset has no model severity score;
    peak severity is kept only as a descriptive field.

    Each event also gets an ``action`` (Ignore / Review / Investigate / None)
    from ``action_for(duration_band, severity_band)`` — None when duration
    isn't confirmed yet, since there isn't enough data to act on.
    """

    flags = _normalise_classified(classified)
    if flags.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    series_map = _build_series_map(full_history)
    seeds = flags[flags["anom_type"].isin(SURFACED_TYPES)]
    records: list[dict] = []

    for site_id, site_seeds in seeds.groupby("site_id", sort=True):
        site_flags = flags[flags["site_id"] == site_id]
        covered_through: pd.Period | None = None
        for seed in site_seeds.itertuples(index=False):
            detection = seed.anom_m
            if covered_through is not None and detection <= covered_through:
                continue

            series = series_map.get(site_id)
            onset = _find_event_start(series, detection, config)
            measured = _measure_duration(series_map.get(site_id), onset, config)
            end_month = measured["end_month"]
            if isinstance(end_month, pd.Period):
                covered_through = end_month
            else:
                covered_through = onset

            in_event = site_flags[
                (site_flags["anom_m"] >= onset)
                & (site_flags["anom_m"] <= covered_through)
            ]
            detection_severity = float(seed.quantile_severity)
            peak_severity = float(in_event["quantile_severity"].max())
            band = severity_band(pd.Series([detection_severity]), config).iloc[0]
            duration_band = measured["duration_band"]
            action = action_for(duration_band, band)
            if pd.isna(duration_band):
                expected_type = pd.NA
            elif duration_band == DURATION_BANDS[0]:
                expected_type = "spike_up"
            elif duration_band in DURATION_BANDS[1:]:
                expected_type = "step_up"
            else:
                expected_type = pd.NA
            match = (
                bool(seed.anom_type == expected_type)
                if not pd.isna(expected_type)
                else pd.NA
            )

            records.append(
                {
                    "site_id": site_id,
                    "event_id": f"{site_id}:{onset}",
                    "detection_month": detection,
                    "months_before_detection": int(detection.ordinal - onset.ordinal),
                    **measured,
                    "detection_quantile_severity": detection_severity,
                    "peak_quantile_severity": peak_severity,
                    "severity_band": band,
                    "action": action,
                    "detection_anom_type": seed.anom_type,
                    "expected_type_from_duration": expected_type,
                    "intuition_match": match,
                    "n_flagged_months": int(len(in_event)),
                }
            )

    events = pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)
    if events.empty:
        return events
    events["duration_band"] = pd.Categorical(
        events["duration_band"], categories=DURATION_BANDS, ordered=True
    )
    events["severity_band"] = pd.Categorical(
        events["severity_band"], categories=SEVERITY_BANDS, ordered=True
    )
    events["action"] = pd.Categorical(
        events["action"], categories=ACTION_LABELS, ordered=True
    )
    return events.sort_values(["site_id", "start_month"], kind="stable").reset_index(drop=True)


def severity_duration_matrix(events: pd.DataFrame, *, row_percent: bool = False) -> pd.DataFrame:
    """Return an always-3-by-3 count matrix (or within-duration row %)."""

    required = {"duration_band", "severity_band", "duration_confirmed"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    eligible = events[
        events["duration_confirmed"].fillna(False)
        & events["duration_band"].notna()
        & events["severity_band"].notna()
    ]
    # Start from fixed axes instead of relying on crosstab to expand empty
    # categoricals. Some pandas 2.x releases create duplicate category labels
    # when ``eligible`` is empty, which then makes ``reindex`` fail.
    counts = pd.DataFrame(
        0,
        index=pd.Index(DURATION_BANDS, name="Duration"),
        columns=pd.Index(SEVERITY_BANDS, name="Severity"),
        dtype=int,
    )
    observed = eligible.groupby(
        ["duration_band", "severity_band"], observed=True
    ).size()
    for (duration, severity), count in observed.items():
        if duration in DURATION_BANDS and severity in SEVERITY_BANDS:
            counts.loc[duration, severity] = int(count)
    if not row_percent:
        return counts
    denominator = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(denominator, axis=0).mul(100).fillna(0).round(1)


def _rate_with_wilson_interval(successes: int, total: int) -> dict:
    if total == 0:
        return {"successes": 0, "n": 0, "rate": None, "ci95": [None, None]}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return {
        "successes": int(successes),
        "n": int(total),
        "rate": round(p, 4),
        "ci95": [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)],
    }


def duration_intuition_report(events: pd.DataFrame) -> dict:
    """Compare observed duration with the existing spike/step label.

    This is a consistency check because both approaches use the same
    pre-event baseline and sustain threshold; it is not independent proof of
    the business cause of an event.
    """

    required = {
        "duration_band",
        "duration_confirmed",
        "detection_anom_type",
        "intuition_match",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    eligible = events[
        events["duration_confirmed"].fillna(False) & events["duration_band"].notna()
    ].copy()

    by_duration = {}
    for band in DURATION_BANDS:
        subset = eligible[eligible["duration_band"] == band]
        expected = "spike_up" if band == DURATION_BANDS[0] else "step_up"
        successes = int((subset["detection_anom_type"] == expected).sum())
        by_duration[band] = {
            "expected_type": expected,
            **_rate_with_wilson_interval(successes, len(subset)),
        }

    matches = int(eligible["intuition_match"].fillna(False).sum())
    excluded = int(len(events) - len(eligible))
    return {
        "n_events_total": int(len(events)),
        "n_events_in_matrix": int(len(eligible)),
        "n_events_excluded_unconfirmed_duration": excluded,
        "overall_agreement": _rate_with_wilson_interval(matches, len(eligible)),
        "by_duration": by_duration,
    }


def duration_type_crosstab(events: pd.DataFrame) -> pd.DataFrame:
    """Counts behind the spike/step intuition report, including empty rows."""

    eligible = events[
        events["duration_confirmed"].fillna(False) & events["duration_band"].notna()
    ]
    table = pd.crosstab(
        eligible["duration_band"], eligible["detection_anom_type"], dropna=False
    )
    return table.reindex(
        index=DURATION_BANDS,
        columns=list(SURFACED_TYPES),
        fill_value=0,
    ).astype(int)