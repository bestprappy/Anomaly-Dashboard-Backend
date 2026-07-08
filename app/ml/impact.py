"""
Cost-impact estimate for flagged spike_up anomalies.

step_up is excluded here on purpose: a step_up is a *sustained* level shift
(new baseline), not a one-off loss, so pricing it the same way as a
transient spike overstates "money lost" — the site may simply be consuming
more permanently (a merge, a new load, etc.), which isn't a loss to recover.
spike_up is the transient, recoverable-looking case this estimate is for.

Residual kWh = actual kWh - predicted q50 kWh, summed over flagged spike_up
rows. To turn that into an estimated baht amount, each site's own average
baht/kWh — computed from its own clean "active" billing history — is used,
rather than a flat rate. PEA and MEA tariffs differ (provincial vs
metropolitan), and rates vary further by Rate_CAT/TOU-TOD within each
provider, so a single flat multiplier (e.g. "x5") would misprice sites on
different tariff tiers. A site's own bill_amount/kwh ratio already encodes
whatever tariff it's actually on — no external rate table needed.

If a site has no usable "active" billing history to derive a rate from
(rare — e.g. it was all-maintenance or all-anomalous), its estimated_baht
comes back as NaN rather than silently guessing a number for it.
"""
from __future__ import annotations

import pandas as pd

SPIKE_UP = "spike_up"


def site_avg_price_per_kwh(master_df: pd.DataFrame) -> pd.Series:
    """Average baht/kWh per site, computed only from 'active' billed months
    (bill_class == 'active', kwh > 0) so a maintenance-rate month or a zero
    read doesn't distort the site's real per-unit price.
    """
    active = master_df[(master_df["bill_class"] == "active") & (master_df["kwh"] > 0)]
    price = active.groupby("Site_ID").apply(lambda g: (g["bill_amount"] / g["kwh"]).mean())
    return price.rename("avg_price_per_kwh")


def residual_impact(flag: pd.DataFrame, master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    `flag` = classified anomalies (needs site_id, anom_type, anom_m, anom_val
    [actual kwh], q50 [predicted kwh]) — typically ML_STATE.classified,
    unfiltered; this function filters to spike_up itself.
    `master_df` = the container's (drop-option-filtered) master table, used
    both to derive each site's rate and to attach provider/company for
    grouping.

    Returns (per_row_detail, summary_by_provider). Both cover spike_up only.
    """
    d = flag[flag["anom_type"] == SPIKE_UP].copy()
    if d.empty:
        empty_summary = pd.DataFrame(
            columns=["provider", "n_anomalies", "n_priced", "total_excess_kwh", "total_estimated_baht"]
        )
        return d, empty_summary

    d["excess_kwh"] = (d["anom_val"] - d["q50"]).clip(lower=0)

    price = site_avg_price_per_kwh(master_df)
    d = d.merge(price, left_on="site_id", right_index=True, how="left")
    d["estimated_excess_baht"] = d["excess_kwh"] * d["avg_price_per_kwh"]

    site_meta = master_df[["Site_ID", "provider", "company"]].drop_duplicates("Site_ID")
    d = d.merge(site_meta, left_on="site_id", right_on="Site_ID", how="left").drop(columns=["Site_ID"])

    summary = (
        d.groupby("provider", dropna=False)
        .agg(
            n_anomalies=("site_id", "size"),
            n_priced=("avg_price_per_kwh", "count"),   # rows that actually got a rate
            total_excess_kwh=("excess_kwh", "sum"),
            total_estimated_baht=("estimated_excess_baht", "sum"),
        )
        .reset_index()
    )
    summary["total_estimated_baht"] = summary["total_estimated_baht"].round(2)
    summary["total_excess_kwh"] = summary["total_excess_kwh"].round(2)
    return d, summary


def monthly_impact_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month totals of excess kWh / estimated baht across the
    test range, so "how much did this cost us in March vs April" is a
    straight lookup rather than something the frontend has to re-aggregate.
    Requires `detail` to have `anom_m` (Period) and `excess_kwh` /
    `estimated_excess_baht` — i.e. the first return value of residual_impact.
    """
    if detail.empty:
        return pd.DataFrame(columns=["month", "n_anomalies", "total_excess_kwh", "total_estimated_baht"])

    g = (
        detail.groupby("anom_m")
        .agg(
            n_anomalies=("site_id", "size"),
            total_excess_kwh=("excess_kwh", "sum"),
            total_estimated_baht=("estimated_excess_baht", "sum"),
        )
        .reset_index()
        .sort_values("anom_m")
    )
    g["month"] = g["anom_m"].astype(str)
    g["total_excess_kwh"] = g["total_excess_kwh"].round(2)
    g["total_estimated_baht"] = g["total_estimated_baht"].round(2)
    return g[["month", "n_anomalies", "total_excess_kwh", "total_estimated_baht"]]