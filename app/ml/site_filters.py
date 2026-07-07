"""
Site-level exclusion logic for the ML "Process" step — the 4 checkboxes.

Each option removes an entire site's history, not just the offending rows.
Once a site is known to be a duplicate meter, shared across companies,
shut down, or maintenance-only, its whole time series is unreliable for
jump detection, so we drop it outright rather than patch individual months.
This mirrors what `00_prepare_model_input.ipynb` used to do by hand before
writing `model_input_active.csv` — it's now a reusable, inspectable step
instead of a one-off notebook cell.

All four options are derived straight from DataBillContainer's existing EDA
methods, so "what counts as a duplicate/common/shutdown/maintenance site" is
defined in exactly one place (data_container.py) and never re-implemented.
"""
from __future__ import annotations

import pandas as pd

from app.data_container import DataBillContainer
from app.ml.config import DropOptions


def duplicate_site_ids(container: DataBillContainer) -> set[str]:
    dup = container.eda_duplicates()
    return {row["Site_ID"] for row in dup["duplicate_site_ids"]}


def common_site_ids(container: DataBillContainer) -> set[str]:
    common = container.eda_common_sites()
    ids: set[str] = set(common.get("pea_mea_cross_common", {}).get("site_ids", []))
    for bucket in ("within_pea", "within_mea"):
        for pair in common.get(bucket, {}).values():
            ids.update(pair.get("site_ids", []))
    return ids


def shutdown_site_ids(master_df: pd.DataFrame) -> set[str]:
    if "is_shutdown" not in master_df.columns:
        return set()
    return set(master_df.loc[master_df["is_shutdown"], "Site_ID"].unique())


def maintenance_site_ids(master_df: pd.DataFrame) -> set[str]:
    if "bill_class" not in master_df.columns:
        return set()
    return set(master_df.loc[master_df["bill_class"] == "maintenance", "Site_ID"].unique())


def resolve_dropped_site_ids(container: DataBillContainer, options: DropOptions) -> dict[str, set[str]]:
    """{option_name: {site_ids}}, only for options that were selected — lets
    the API report *why* each site left the training set instead of just how
    many.
    """
    master = container.master_df
    out: dict[str, set[str]] = {}
    if options.duplicate_site:
        out["duplicate_site"] = duplicate_site_ids(container)
    if options.common_site:
        out["common_site"] = common_site_ids(container)
    if options.shutdown_site:
        out["shutdown_site"] = shutdown_site_ids(master)
    if options.maintenance_site:
        out["maintenance_site"] = maintenance_site_ids(master)
    return out


def apply_drop_options(
    master_df: pd.DataFrame, container: DataBillContainer, options: DropOptions
) -> tuple[pd.DataFrame, dict]:
    """Filter master_df by the selected drop options.

    Returns (filtered_df, report). A site can match more than one option, so
    `total_sites_dropped` is the size of the union, not the sum of the
    per-option counts.
    """
    dropped_by_option = resolve_dropped_site_ids(container, options)
    all_dropped: set[str] = set()
    for ids in dropped_by_option.values():
        all_dropped |= ids

    filtered = (
        master_df[~master_df["Site_ID"].isin(all_dropped)].copy()
        if all_dropped else master_df.copy()
    )

    report = {
        "options_applied": options.as_list(),
        "dropped_sites_by_option": {k: len(v) for k, v in dropped_by_option.items()},
        "total_sites_dropped": len(all_dropped),
        "sites_remaining": int(filtered["Site_ID"].nunique()) if len(filtered) else 0,
        "rows_remaining": int(len(filtered)),
    }
    return filtered, report