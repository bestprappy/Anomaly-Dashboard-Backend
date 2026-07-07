"""
Feature engineering — ported 1:1 from `00_prepare_model_input.ipynb` /
`site_jump_quantile.ipynb`, generalised to run inside the API against
whatever slice of the master table the Process step hands it, instead of a
fixed `model_input_active.csv` built ahead of time in a notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.config import FEATURE_COLS, MIN_HISTORY_MONTHS, TARGET_COL


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """`df` needs one row per site per month with columns Site_ID, date
    (datetime64), kwh. Returns a fully-featured frame; callers still need
    `select_model_ready` to slice the rows that are actually usable.
    """
    d = df.rename(columns={"Site_ID": "site_id", "date": "bill_month"}).copy()
    d["site_id"] = d["site_id"].astype(str).str.upper().str.strip()
    d["province"] = d["site_id"].str[:3]
    d = d.dropna(subset=["site_id", "bill_month"])
    d = d.drop_duplicates(subset=["site_id", "bill_month"], keep="first")

    # 0 kwh is a missing read, not a genuine zero-usage month
    d["kwh"] = d["kwh"].replace(0, np.nan)
    complete_sites = d.groupby("site_id")["kwh"].apply(lambda x: x.isna().sum())
    d = d[d["site_id"].isin(complete_sites[complete_sites == 0].index)].copy()

    d = d.sort_values(["site_id", "bill_month"]).reset_index(drop=True)
    d["year"] = d["bill_month"].dt.year
    d["month"] = d["bill_month"].dt.month
    d["quarter"] = d["bill_month"].dt.quarter
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d[TARGET_COL] = d.groupby("site_id")["kwh"].shift(-1)
    for lag in (1, 2, 3, 6):
        d[f"kwh_lag_{lag}"] = d.groupby("site_id")["kwh"].shift(lag)
    g = d.groupby("site_id")["kwh"]
    d["kwh_roll_3_mean"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    d["kwh_roll_6_mean"] = g.transform(lambda x: x.shift(1).rolling(6).mean())
    d["kwh_roll_3_std"] = g.transform(lambda x: x.shift(1).rolling(3).std())
    d["province_freq"] = d["province"].map(d["province"].value_counts(normalize=True))
    d["site_month_no"] = d.groupby("site_id").cumcount()
    return d


def select_model_ready(d: pd.DataFrame) -> pd.DataFrame:
    """Rows need >= MIN_HISTORY_MONTHS of prior history (for kwh_lag_6) and a
    known next-month target.
    """
    ready = d[(d["site_month_no"] >= MIN_HISTORY_MONTHS) & (d[TARGET_COL].notna())]
    return ready.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)