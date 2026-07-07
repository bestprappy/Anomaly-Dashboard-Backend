"""
Stage 1 of the pipeline: quantile-regression prediction band.

Anomaly flag = actual next-month kWh falls outside [q_low, q_high].
Isolation Forest severity is intentionally not implemented — `quantile_severity`
(how many band-widths outside the band a point falls) is the only ranking
signal the product needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from app.ml.config import FEATURE_COLS, QuantileConfig, TARGET_COL


def fit_quantile_models(
    X_train: pd.DataFrame, y_train: pd.Series, qcfg: QuantileConfig
) -> dict[float, HistGradientBoostingRegressor]:
    models = {}
    for q in (qcfg.q_low, qcfg.q_mid, qcfg.q_high):
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            max_depth=8, min_samples_leaf=40, random_state=42,
        )
        m.fit(X_train, y_train)
        models[q] = m
    return models


def predict_band(
    models: dict[float, HistGradientBoostingRegressor], X: pd.DataFrame, qcfg: QuantileConfig
) -> pd.DataFrame:
    preds = {q: models[q].predict(X) for q in models}
    # clamp any quantile-crossing (rare, but the low/mid/high models are fit
    # independently so it can happen)
    lo = np.minimum.reduce([preds[qcfg.q_low], preds[qcfg.q_mid], preds[qcfg.q_high]])
    hi = np.maximum.reduce([preds[qcfg.q_low], preds[qcfg.q_mid], preds[qcfg.q_high]])
    return pd.DataFrame({"q_low": lo, "q_mid": preds[qcfg.q_mid], "q_high": hi}, index=X.index)


def coverage(y: pd.Series, band: pd.DataFrame) -> float:
    return float(((y >= band["q_low"]) & (y <= band["q_high"])).mean())


def pinball_p50(y: pd.Series, band: pd.DataFrame) -> float:
    return float(mean_pinball_loss(y, band["q_mid"], alpha=0.5))


def flag_anomalies(part: pd.DataFrame, band: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    part = part.copy()
    part["q05"], part["q50"], part["q95"] = band["q_low"], band["q_mid"], band["q_high"]
    y = part[target_col]
    part["resid"] = y - part["q50"]
    part["band_width"] = (part["q95"] - part["q05"]).clip(lower=1e-9)
    part["flag_quantile"] = (y < part["q05"]) | (y > part["q95"])
    over = (y - part["q95"]).clip(lower=0)
    under = (part["q05"] - y).clip(lower=0)
    part["quantile_severity"] = ((over + under) / part["band_width"]).round(3)
    return part


def run_quantile_stage(train: pd.DataFrame, test: pd.DataFrame, qcfg: QuantileConfig) -> dict:
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    models = fit_quantile_models(X_train, y_train, qcfg)

    train_band = predict_band(models, X_train, qcfg)
    test_band = predict_band(models, X_test, qcfg)

    train_flagged = flag_anomalies(train, train_band)
    test_flagged = flag_anomalies(test, test_band)

    metrics = {
        "train": {
            "coverage": round(coverage(y_train, train_band), 4),
            "pinball_p50": round(pinball_p50(y_train, train_band), 3),
        },
        "test": {
            "coverage": round(coverage(y_test, test_band), 4),
            "pinball_p50": round(pinball_p50(y_test, test_band), 3),
        },
        "n_flagged_test": int(test_flagged["flag_quantile"].sum()),
        "flagged_rate_test": round(float(test_flagged["flag_quantile"].mean()), 4) if len(test_flagged) else 0.0,
    }
    return {"models": models, "train": train_flagged, "test": test_flagged, "metrics": metrics}