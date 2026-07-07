"""
Helper for the date-range picker on the Process tab.

Old months tend to have more zero/missing kWh reads (meters not yet on the
billing system, backfilled exports, etc.), which quietly makes the model
worse if they're included in train/test. This module computes a per-month
missing rate so the frontend can show it right next to the date pickers
before the user commits to a train/test window.
"""
from __future__ import annotations

import pandas as pd


def monthly_missing_rate(df: pd.DataFrame) -> list[dict]:
    """Per month: how many site-months have zero/NaN kwh, out of the sites
    present that month.
    """
    if df.empty:
        return []
    d = df.copy()
    d["_missing"] = d["kwh"].isna() | (d["kwh"].fillna(0) == 0)
    g = d.groupby("month").agg(total_sites=("Site_ID", "nunique"), missing=("_missing", "sum"))
    g["missing_rate"] = (g["missing"] / g["total_sites"]).round(4)
    return [
        {
            "month": int(m),
            "total_sites": int(r.total_sites),
            "missing_count": int(r.missing),
            "missing_rate": float(r.missing_rate),
        }
        for m, r in g.iterrows()
    ]


def range_missing_summary(df: pd.DataFrame, start_month: int, end_month: int) -> dict:
    sub = df[(df["month"] >= start_month) & (df["month"] <= end_month)]
    per_month = monthly_missing_rate(sub)
    avg = (sum(m["missing_rate"] for m in per_month) / len(per_month)) if per_month else None
    return {
        "start_month": start_month,
        "end_month": end_month,
        "n_months": len(per_month),
        "avg_missing_rate": round(avg, 4) if avg is not None else None,
        "per_month": per_month,
    }