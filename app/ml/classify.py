"""
Stage 2: classify *what kind* of jump a flagged anomaly is.

The flag is on next-month kWh, so the unusual month is `bill_month + 1`. We
look at the site's full monthly series around that month and compare the
~4-month median before vs after:

  step_up    value jumps and *stays* high  -> sustained level shift (e.g. a
             merge/consolidation of load onto one site)
  spike_up   a one-month jump that reverts -> transient
  other      doesn't clear the jump threshold
  unknown    site has no usable history around that month

Only over-band values reach this stage at all (see quantile_model.py's
flag_and_slim, which only flags y > q_high), so there's no downward case to
classify here — a genuine drop would never have been flagged in the first
place. If that upstream filter ever changes to flag both tails again, the
down-side branch would need to come back here too.

Thresholds (up/sustain) are user input from the Result tab and are
re-appliable to an already-built model without refitting anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.config import ClassifyThresholds

SURFACED_TYPES = ("spike_up", "step_up")
ALL_TYPES = ("step_up", "spike_up", "other", "unknown")


def build_site_series(full_df: pd.DataFrame, site_ids: set[str]) -> dict[str, pd.Series]:
    """`full_df` is the *unfiltered-by-window* per-site monthly kwh history
    (Site_ID, date, kwh) captured at build time, so classification can see
    months outside the train/test model window (e.g. the 4 months before the
    very first test-set target).
    """
    d = full_df[full_df["Site_ID"].isin(site_ids)].copy()
    d["m"] = pd.to_datetime(d["date"]).dt.to_period("M")
    return {sid: g.set_index("m")["kwh"].sort_index() for sid, g in d.groupby("Site_ID")}


def classify_one(series: pd.Series | None, month, thresholds: ClassifyThresholds) -> str:
    if series is None or month not in series.index:
        return "unknown"
    before = series.loc[(series.index >= month - 4) & (series.index <= month - 1)].median()
    after = series.loc[(series.index >= month + 1) & (series.index <= month + 4)].median()
    v = series.loc[month]
    if not np.isfinite(before) or before <= 0:
        return "unknown"
    if v >= thresholds.up * before:
        return "step_up" if (np.isfinite(after) and after >= thresholds.sustain * before) else "spike_up"
    return "other"


def classify_anomalies(
    test_flagged: pd.DataFrame, full_df: pd.DataFrame, thresholds: ClassifyThresholds, target_col: str
) -> pd.DataFrame:
    """`test_flagged` = flagged-rows output of run_quantile_stage (slim
    columns, flag_quantile all True, all over-band). Adds anom_m (Period,
    the actual unusual month), anom_val, anom_type to the flagged subset.
    """
    flag = test_flagged[test_flagged["flag_quantile"]].copy()
    if flag.empty:
        flag["anom_m"] = pd.Series(dtype="object")
        flag["anom_val"] = pd.Series(dtype="float64")
        flag["anom_type"] = pd.Series(dtype="object")
        return flag

    flag["anom_m"] = pd.to_datetime(flag["bill_month"]).dt.to_period("M") + 1
    flag["anom_val"] = flag[target_col]

    series_map = build_site_series(full_df, set(flag["site_id"].unique()))
    flag["anom_type"] = [
        classify_one(series_map.get(sid), m, thresholds)
        for sid, m in zip(flag["site_id"], flag["anom_m"])
    ]
    return flag