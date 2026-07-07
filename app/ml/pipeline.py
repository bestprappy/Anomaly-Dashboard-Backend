"""
Orchestrates the two Process/Result actions the ML tab needs:

  build_pipeline()      Process tab "Build model" button — drop options ->
                         date-range filter -> features -> quantile band ->
                         anomaly flags. This is the expensive step (fits 3
                         HistGradientBoostingRegressor models).

  classify_pipeline()   Result tab — reclassify flagged anomalies into
                         spike_up / step_up / etc using whatever UP/DOWN/
                         SUSTAIN the user just typed in. Cheap, so it's a
                         separate step the user can re-run freely without
                         ever refitting the model.

Both read/write a single MLRunState instance so the router stays thin.
"""
from __future__ import annotations

import pandas as pd

from app.data_container import DataBillContainer
from app.ml.classify import classify_anomalies
from app.ml.config import ClassifyThresholds, DateRange, DropOptions, QuantileConfig, TARGET_COL
from app.ml.features import build_model_frame, select_model_ready
from app.ml.missing_rate import range_missing_summary
from app.ml.quantile_model import run_quantile_stage
from app.ml.site_filters import apply_drop_options
from app.ml.state import MLRunState


def preview_missing_rate(
    container: DataBillContainer, options: DropOptions, start_month: int, end_month: int
) -> dict:
    """Process steps 1+2 combined: what the drop options would remove, and
    how much missing data is left in the candidate date window — so the user
    can adjust either before committing to a build.
    """
    filtered, drop_report = apply_drop_options(container.master_df, container, options)
    missing = range_missing_summary(filtered, start_month, end_month)
    return {"drop_report": drop_report, "missing": missing}


def _month_col(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.year * 100 + dt_series.dt.month


def build_pipeline(
    container: DataBillContainer,
    options: DropOptions,
    train_range: DateRange,
    test_range: DateRange,
    qcfg: QuantileConfig,
    state: MLRunState,
) -> dict:
    filtered, drop_report = apply_drop_options(container.master_df, container, options)

    # Kept unfiltered-by-window so the classify step can look at months
    # outside the train/test span (e.g. the 4 months before the earliest
    # test-set anomaly).
    full_history = filtered[["Site_ID", "date", "kwh"]].copy()

    window = filtered[
        (filtered["month"] >= train_range.start_month) & (filtered["month"] <= test_range.end_month)
    ]
    featured = build_model_frame(window[["Site_ID", "date", "kwh"]])
    ready = select_model_ready(featured)

    if ready.empty:
        raise ValueError(
            "No model-ready rows in the selected window. Try a wider date range or "
            "fewer drop options — each site needs at least 7 consecutive clean months."
        )

    ready_month = _month_col(ready["bill_month"])
    train = ready[ready_month.between(train_range.start_month, train_range.end_month)]
    test = ready[ready_month.between(test_range.start_month, test_range.end_month)]

    if train.empty:
        raise ValueError("Train range has no model-ready rows — pick a wider train range.")
    if test.empty:
        raise ValueError("Test range has no model-ready rows — pick a wider test range.")

    result = run_quantile_stage(train, test, qcfg)

    state.drop_report = drop_report
    state.train_range = (train_range.start_month, train_range.end_month)
    state.test_range = (test_range.start_month, test_range.end_month)
    state.metrics = result["metrics"]
    state.test_flagged = result["test"]
    state.full_history = full_history
    state.classified = None  # invalidate any previous classification from an older model

    return {
        "drop_report": drop_report,
        "train_range": state.train_range,
        "test_range": state.test_range,
        "metrics": result["metrics"],
        "n_model_ready_rows": int(len(ready)),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
    }


def abnormal_dataframe(state: MLRunState) -> pd.DataFrame:
    """The plain (site_id, month/year, kwh) view of flagged anomalies asked
    for in the spec — the classification-agnostic output of the Process step.
    """
    if not state.is_built():
        raise RuntimeError("No model has been built yet.")
    flagged = state.test_flagged
    out = flagged[flagged["flag_quantile"]].copy()
    out["anom_month"] = (pd.to_datetime(out["bill_month"]).dt.to_period("M") + 1).astype(str)
    out["kwh"] = out[TARGET_COL]
    return (
        out[["site_id", "anom_month", "kwh", "q05", "q50", "q95", "quantile_severity"]]
        .sort_values("quantile_severity", ascending=False)
        .reset_index(drop=True)
    )


def classify_pipeline(state: MLRunState, thresholds: ClassifyThresholds) -> pd.DataFrame:
    if not state.is_built():
        raise RuntimeError("No model has been built yet.")
    state.thresholds = thresholds
    classified = classify_anomalies(state.test_flagged, state.full_history, thresholds, TARGET_COL)
    state.classified = classified
    return classified