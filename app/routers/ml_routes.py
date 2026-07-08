"""
/api/ml/* — the ML tab's backend.

Process:
  GET  /api/ml/drop-options   the 4 checkbox definitions (label + value)
  POST /api/ml/preview        drop-report + per-month missing-rate for a
                               candidate date window, so the user can see
                               data quality before committing
  POST /api/ml/build          fit the quantile band model on the chosen
                               drop options + train/test ranges; returns
                               metrics (band coverage, flagged rate)

  GET  /api/ml/abnormal        the flagged anomalies as (site_id, month, kwh)
                               — classification-agnostic, matches the spec's
                               "result it as a data frame of abnormal value"

Result:
  POST /api/ml/classify        (re)classify flagged anomalies with user-input
                                UP/DOWN/SUSTAIN thresholds; cheap, no refit
  GET  /api/ml/examples         up to N example plots (base64 PNG) for one
                                 of spike_up / step_up, over the configured
                                 date range (defaults to the model's own
                                 train->test window, override with
                                 plot_start/plot_end)
  GET  /api/ml/plots/download   zip of every plot for the requested types,
                                 same date-range rules as /examples

Kept as its own router rather than growing main.py further, so the ML
pipeline stays a self-contained, independently testable slice of the app.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.data_container import DataBillContainer
from app.ml.classify import SURFACED_TYPES, build_site_series
from app.ml.config import ClassifyThresholds, DateRange, DROP_OPTIONS, DROP_OPTION_LABELS, DropOptions, QuantileConfig
from app.ml.pipeline import abnormal_dataframe, build_pipeline, classify_pipeline, preview_missing_rate
from app.ml.plotting import render_all_zip, render_examples, yyyymm_to_period
from app.ml.schemas import BuildRequest, ClassifyRequest, PreviewRequest
from app.ml.state import ML_STATE
from app.ml.impact import monthly_impact_summary, residual_impact

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ml", tags=["ml"])


def _to_drop_options(drop_options_in) -> DropOptions:
    return DropOptions(**drop_options_in.dict())


def _get_ready_container() -> DataBillContainer:
    # Local import to avoid a circular import at module load time (main.py
    # imports this router; this router only needs main.STATE inside a call).
    from app.main import STATE

    container: DataBillContainer = STATE["container"]
    if not container.is_ready():
        raise HTTPException(status_code=409, detail="No billing data loaded yet. Upload the 5 files first.")
    return container


def _require_built() -> None:
    if not ML_STATE.is_built():
        raise HTTPException(status_code=409, detail="No model built yet. POST /api/ml/build first.")


def _require_classified() -> None:
    _require_built()
    if not ML_STATE.is_classified():
        raise HTTPException(status_code=409, detail="No classification yet. POST /api/ml/classify first.")


def _resolve_plot_range(
    plot_start: Optional[int], plot_end: Optional[int]
) -> tuple[pd.Period, pd.Period]:
    """Default to the exact window the model was built on. The end is
    extended by one month past test_end so the last flagged anomaly (which
    lands on test_end + 1, since the flag is on *next*-month kWh) is still
    visible — unless the caller explicitly passes plot_end, which is then
    taken literally.
    """
    default_start, _ = ML_STATE.train_range
    _, default_end = ML_STATE.test_range

    start_period = yyyymm_to_period(plot_start if plot_start is not None else default_start)
    if plot_end is not None:
        end_period = yyyymm_to_period(plot_end)
    else:
        end_period = yyyymm_to_period(default_end) + 1

    return start_period, end_period


# ---------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------

@router.get("/drop-options")
def drop_options():
    return {"options": [{"value": k, "label": DROP_OPTION_LABELS[k]} for k in DROP_OPTIONS]}


@router.post("/preview")
async def preview(body: PreviewRequest):
    container = _get_ready_container()
    options = _to_drop_options(body.drop_options)
    try:
        return await run_in_threadpool(
            preview_missing_rate, container, options, body.start_month, body.end_month
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/build")
async def build(body: BuildRequest):
    container = _get_ready_container()
    options = _to_drop_options(body.drop_options)
    try:
        train_range = DateRange(body.train_start, body.train_end)
        test_range = DateRange(body.test_start, body.test_end)
        qcfg = QuantileConfig(q_low=body.q_low, q_mid=body.q_mid, q_high=body.q_high)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        return await run_in_threadpool(
            build_pipeline, container, options, train_range, test_range, qcfg, ML_STATE
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("ML build failed")
        raise HTTPException(status_code=500, detail=f"Model build failed: {str(e)[:200]}") from e


@router.get("/abnormal")
def abnormal():
    _require_built()
    df = abnormal_dataframe(ML_STATE)
    return {"count": int(len(df)), "rows": df.to_dict(orient="records")}


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@router.post("/classify")
async def classify(body: ClassifyRequest):
    _require_built()
    try:
        thresholds = ClassifyThresholds(up=body.up, down=body.down, sustain=body.sustain)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    flag = await run_in_threadpool(classify_pipeline, ML_STATE, thresholds)
    counts = flag["anom_type"].value_counts().to_dict() if len(flag) else {}
    surfaced = flag[flag["anom_type"].isin(SURFACED_TYPES)].copy()
    if not surfaced.empty:
        surfaced["anom_month"] = surfaced["anom_m"].astype(str)
        rows = (
            surfaced[["site_id", "anom_month", "anom_val", "anom_type", "quantile_severity"]]
            .sort_values(["anom_type", "quantile_severity"], ascending=[True, False])
            .to_dict(orient="records")
        )
    else:
        rows = []
    return {"type_counts": counts, "surfaced_types": list(SURFACED_TYPES), "rows": rows}


@router.get("/examples")
def examples(
    anom_type: str = Query(..., pattern="^(spike_up|step_up)$"),
    limit: int = Query(5, ge=1, le=20),
    plot_start: Optional[int] = Query(None, description="YYYYMM, overrides the model's train_start"),
    plot_end: Optional[int] = Query(None, description="YYYYMM, overrides the model's test_end"),
):
    _require_classified()
    flag = ML_STATE.classified
    series_map = build_site_series(ML_STATE.full_history, set(flag["site_id"].unique()))
    start_period, end_period = _resolve_plot_range(plot_start, plot_end)
    pngs = render_examples(flag, series_map, anom_type, limit=limit,
                            start_period=start_period, end_period=end_period)
    return {
        "anom_type": anom_type,
        "count": len(pngs),
        "plot_range": {"start": str(start_period), "end": str(end_period)},
        "images": [
            {"site_id": sid, "png_base64": base64.b64encode(png).decode("ascii")}
            for sid, png in pngs.items()
        ],
    }


@router.get("/plots/download")
def download_plots(
    types: str = Query("spike_up,step_up", description="comma-separated: spike_up,step_up"),
    plot_start: Optional[int] = Query(None, description="YYYYMM, overrides the model's train_start"),
    plot_end: Optional[int] = Query(None, description="YYYYMM, overrides the model's test_end"),
):
    _require_classified()
    requested = tuple(t.strip() for t in types.split(",") if t.strip())
    invalid = set(requested) - set(SURFACED_TYPES)
    if invalid:
        raise HTTPException(
            status_code=422, detail=f"Unsupported types: {sorted(invalid)}. Allowed: {SURFACED_TYPES}"
        )

    flag = ML_STATE.classified
    series_map = build_site_series(ML_STATE.full_history, set(flag["site_id"].unique()))
    start_period, end_period = _resolve_plot_range(plot_start, plot_end)
    zip_bytes = render_all_zip(flag, series_map, requested,
                                start_period=start_period, end_period=end_period)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=anomaly_plots.zip"},
    )

@router.get("/impact")
def impact():
    """Total excess kWh + estimated baht cost of flagged spike_up anomalies
    (step_up is excluded — see impact.py docstring), broken down by provider
    (PEA/MEA) and by calendar month across the test range. Each site's own
    average baht/kWh (derived from its clean billing history) is used to
    price its excess — not a flat rate, since PEA/MEA and rate-category
    tariffs differ.
    """
    _require_classified()
    container = _get_ready_container()
    flag = ML_STATE.classified
    if flag.empty:
        return {"summary_by_provider": [], "summary_by_month": [], "unpriced_row_count": 0, "rows": []}
 
    detail, summary = residual_impact(flag, container.master_df)
    monthly = monthly_impact_summary(detail)
    unpriced = int(detail["avg_price_per_kwh"].isna().sum()) if not detail.empty else 0
 
    rows = []
    if not detail.empty:
        rows = detail[
            ["site_id", "provider", "company", "anom_type", "anom_val", "q50",
             "excess_kwh", "avg_price_per_kwh", "estimated_excess_baht"]
        ].sort_values("estimated_excess_baht", ascending=False).to_dict(orient="records")
 
    return {
        "summary_by_provider": summary.to_dict(orient="records"),
        "summary_by_month": monthly.to_dict(orient="records"),
    }