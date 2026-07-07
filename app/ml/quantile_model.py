"""
Stage 1 of the pipeline: quantile-regression prediction band.

Anomaly flag = actual next-month kWh falls outside [q_low, q_high].
Isolation Forest severity is intentionally not implemented — `quantile_severity`
(how many band-widths outside the band a point falls) is the only ranking
signal the product needs.

Memory notes (the API lives on a 512 MB instance):
  - sklearn is imported lazily inside the fit function so the upload/EDA
    endpoints never pay its ~50 MB import cost.
  - Feature matrices are cast to float32 before fitting.
  - The train set is never copied — its coverage/pinball metrics are computed
    straight from the predicted bands.
  - Only the *flagged* test rows (with the handful of columns the Result tab
    needs) are returned, not a full copy of the test frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.config import FEATURE_COLS, QuantileConfig, TARGET_COL

FLAGGED_KEEP_COLS = ["site_id", "bill_month", TARGET_COL]


def fit_quantile_models(X_train: pd.DataFrame, y_train: pd.Series, qcfg: QuantileConfig) -> dict:
    # Local import: keeps sklearn out of the baseline RSS for non-ML requests.
    from sklearn.ensemble import HistGradientBoostingRegressor

    models = {}
    for q in (qcfg.q_low, qcfg.q_mid, qcfg.q_high):
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            max_depth=8, min_samples_leaf=40, random_state=42,
        )
        m.fit(X_train, y_train)
        models[q] = m
    return models


def predict_band(models: dict, X: pd.DataFrame, qcfg: QuantileConfig) -> pd.DataFrame:
    preds = {q: models[q].predict(X) for q in models}
    # clamp any quantile-crossing (rare, but the low/mid/high models are fit
    # independently so it can happen)
    lo = np.minimum.reduce([preds[qcfg.q_low], preds[qcfg.q_mid], preds[qcfg.q_high]])
    hi = np.maximum.reduce([preds[qcfg.q_low], preds[qcfg.q_mid], preds[qcfg.q_high]])
    return pd.DataFrame({"q_low": lo, "q_mid": preds[qcfg.q_mid], "q_high": hi}, index=X.index)


def coverage(y: pd.Series, band: pd.DataFrame) -> float:
    return float(((y >= band["q_low"]) & (y <= band["q_high"])).mean())


def pinball_p50(y: pd.Series, band: pd.DataFrame) -> float:
    from sklearn.metrics import mean_pinball_loss

    return float(mean_pinball_loss(y, band["q_mid"], alpha=0.5))


def flag_and_slim(part: pd.DataFrame, band: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, int]:
    """Flag band-escapes and return (flagged-rows-only slim frame, n_flagged).

    Column names q05/q50/q95 are kept for backward compatibility with the
    /abnormal contract even though the quantiles are configurable.
    """
    y = part[target_col]
    band_width = (band["q_high"] - band["q_low"]).clip(lower=1e-9)
    flag_mask = (y < band["q_low"]) | (y > band["q_high"])
    over = (y - band["q_high"]).clip(lower=0)
    under = (band["q_low"] - y).clip(lower=0)
    severity = ((over + under) / band_width).round(3)

    flagged = part.loc[flag_mask, FLAGGED_KEEP_COLS].copy()
    flagged["q05"] = band.loc[flag_mask, "q_low"]
    flagged["q50"] = band.loc[flag_mask, "q_mid"]
    flagged["q95"] = band.loc[flag_mask, "q_high"]
    flagged["quantile_severity"] = severity[flag_mask]
    flagged["flag_quantile"] = True
    return flagged, int(flag_mask.sum())


def run_quantile_stage(train: pd.DataFrame, test: pd.DataFrame, qcfg: QuantileConfig) -> dict:
    X_train = train[FEATURE_COLS].astype(np.float32)
    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    models = fit_quantile_models(X_train, y_train, qcfg)

    train_band = predict_band(models, X_train, qcfg)
    train_metrics = {
        "coverage": round(coverage(y_train, train_band), 4),
        "pinball_p50": round(pinball_p50(y_train, train_band), 3),
    }
    del X_train, train_band

    X_test = test[FEATURE_COLS].astype(np.float32)
    test_band = predict_band(models, X_test, qcfg)
    del X_test, models

    flagged, n_flagged = flag_and_slim(test, test_band)
    metrics = {
        "train": train_metrics,
        "test": {
            "coverage": round(coverage(y_test, test_band), 4),
            "pinball_p50": round(pinball_p50(y_test, test_band), 3),
        },
        "n_flagged_test": n_flagged,
        "flagged_rate_test": round(n_flagged / len(test), 4) if len(test) else 0.0,
    }
    return {"flagged": flagged, "metrics": metrics}
