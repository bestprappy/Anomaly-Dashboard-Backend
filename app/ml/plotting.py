"""
Matplotlib rendering for the Result tab: line chart per site over a
*configured* date range (not full history) with the flagged anomaly
month(s) marked in red — plus a bulk PNG export packaged as a zip for the
download button.

The date range defaults to the train/test window the model was actually
built on (so what you see is what the model saw), but callers can override
it — see app/routers/ml_routes.py's /examples and /plots/download.

The bulk export is organised by business action first — Investigate /
Review / Ignore / Unclassified — then by anomaly type, so opening the zip
puts the highest-priority sites in front of the user immediately instead of
requiring them to hunt through a flat per-type folder. The action for each
site/type comes from the event table built by
``app.ml.severity_duration.build_severity_duration_events``.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from app.ml.severity_duration import ACTION_LABELS, ACTION_PRIORITY

UNCLASSIFIED_ACTION = "Unclassified"

# Folder ordering for the zip: most urgent first, unclassified last.
ZIP_ACTION_ORDER = (*ACTION_LABELS[::-1], UNCLASSIFIED_ACTION)


def _get_pyplot():
    """Import matplotlib lazily: it costs ~40+ MB of RSS, and on the 512 MB
    instance the upload/EDA endpoints must never pay for it. Only the two
    plot endpoints trigger this.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless server, never touches a display
    import matplotlib.pyplot as plt

    return plt


def yyyymm_to_period(month: int) -> pd.Period:
    """202305 -> Period('2023-05', 'M')"""
    return pd.Period(f"{month // 100:04d}-{month % 100:02d}", freq="M")


def _slice_series(series: pd.Series, start: pd.Period | None, end: pd.Period | None) -> pd.Series:
    if start is not None:
        series = series[series.index >= start]
    if end is not None:
        series = series[series.index <= end]
    return series


def _slice_anomalies(anomalies: pd.DataFrame, start: pd.Period | None, end: pd.Period | None) -> pd.DataFrame:
    if anomalies.empty:
        return anomalies
    mask = pd.Series(True, index=anomalies.index)
    if start is not None:
        mask &= anomalies["anom_m"] >= start
    if end is not None:
        mask &= anomalies["anom_m"] <= end
    return anomalies[mask]


def render_site_plot(
    site_id: str,
    series: pd.Series,
    anomalies: pd.DataFrame,
    anom_type: str,
    start_period: pd.Period | None = None,
    end_period: pd.Period | None = None,
) -> bytes:
    series = _slice_series(series, start_period, end_period)
    anomalies = _slice_anomalies(anomalies, start_period, end_period)

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 3))
    x = series.index.to_timestamp()
    ax.plot(x, series.values, "-o", ms=3, lw=1.5, color="steelblue")
    if not anomalies.empty:
        ax.scatter(anomalies["anom_m"].dt.to_timestamp(), anomalies["anom_val"],
                    color="red", s=60, zorder=5)
    ax.set_title(f"{site_id}  [{anom_type}]", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_examples(
    flag: pd.DataFrame,
    series_map: dict[str, pd.Series],
    anom_type: str,
    limit: int = 5,
    start_period: pd.Period | None = None,
    end_period: pd.Period | None = None,
) -> dict[str, bytes]:
    """Up to `limit` example plots for one anomaly type (Result step 3: "show
    example 5 plots from spike_up and step_up"), restricted to the given
    date range.
    """
    sids = flag.loc[flag["anom_type"] == anom_type, "site_id"].unique().tolist()[:limit]
    out = {}
    for sid in sids:
        series = series_map.get(sid)
        if series is None:
            continue
        # Only mark THIS type's anomalies — a site can have other flagged
        # months (e.g. "other") that shouldn't show up red on a spike_up
        # or step_up example plot.
        anoms = flag[(flag["site_id"] == sid) & (flag["anom_type"] == anom_type)]
        out[sid] = render_site_plot(sid, series, anoms, anom_type, start_period, end_period)
    return out


def _resolve_site_action(
    events: pd.DataFrame | None,
    site_id: str,
    anom_type: str,
) -> str:
    """Highest-priority action across this site's events of this type.

    A site can have several events of the same anom_type — e.g. a resolved
    spike_up from a year ago and a still-open one now — with different
    durations and therefore different actions. We always surface the most
    urgent one so an "Investigate" event never gets buried in an "Ignore"
    folder. Sites with no confirmed-action event for this type (excluded /
    unconfirmed duration, or events missing entirely) fall into
    UNCLASSIFIED_ACTION rather than being dropped or mis-filed.
    """
    if events is None or events.empty:
        return UNCLASSIFIED_ACTION
    matches = events[
        (events["site_id"] == site_id)
        & (events["detection_anom_type"] == anom_type)
        & events["action"].notna()
    ]
    if matches.empty:
        return UNCLASSIFIED_ACTION
    return max(matches["action"], key=lambda a: ACTION_PRIORITY.get(a, -1))


def render_all_zip(
    flag: pd.DataFrame,
    series_map: dict[str, pd.Series],
    types: tuple[str, ...],
    events: pd.DataFrame | None = None,
    start_period: pd.Period | None = None,
    end_period: pd.Period | None = None,
) -> bytes:
    """Every site plot for the requested types, zipped as
    `{action}/{type}/{site_id}.png` (Result step 4: "download all
    spike_up, step_up plots"), restricted to the given date range.

    ``events`` is the severity/duration event table from
    ``app.ml.severity_duration.build_severity_duration_events``. Its
    ``action`` column drives the top-level folder so the zip opens straight
    into a worklist — Investigate/ first, Unclassified/ last. Pass ``None``
    (or omit) to fall back to the old flat `{type}/{site_id}.png` layout
    under a single Unclassified/ folder, e.g. for callers that haven't
    computed events yet.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in types:
            sids = flag.loc[flag["anom_type"] == t, "site_id"].unique().tolist()
            for sid in sids:
                series = series_map.get(sid)
                if series is None:
                    continue
                anoms = flag[(flag["site_id"] == sid) & (flag["anom_type"] == t)]
                png = render_site_plot(sid, series, anoms, t, start_period, end_period)
                action = _resolve_site_action(events, sid, t)
                zf.writestr(f"{action}/{t}/{sid}.png", png)
    return buf.getvalue()