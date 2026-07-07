"""
ML pipeline configuration objects.

This is the single source of truth for everything the frontend can pick on
the "Process" tab before a model run: which site categories to drop, the
date windows for train/test, and the quantile-regression settings. The
"Result" tab's classification thresholds (UP/DOWN/SUSTAIN) live here too
since they're just as much a tunable config, but they're re-appliable to an
already-built model (see app/ml/pipeline.py::classify_pipeline) so changing
them never requires refitting.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------
# Drop options (Process step 1 — the 4 checkboxes)
# ---------------------------------------------------------------------

DROP_OPTIONS = ("duplicate_site", "common_site", "shutdown_site", "maintenance_site")

DROP_OPTION_LABELS = {
    "duplicate_site": "Duplicate site (same Site_ID repeated across raw files)",
    "common_site": "Common site (shared between PEA/MEA or between companies)",
    "shutdown_site": "Shutdown site (MEA is_shutdown flag)",
    "maintenance_site": "Maintenance site (bill_class == 'maintenance')",
}


@dataclass
class DropOptions:
    duplicate_site: bool = False
    common_site: bool = False
    shutdown_site: bool = False
    maintenance_site: bool = False

    def any_selected(self) -> bool:
        return any([self.duplicate_site, self.common_site, self.shutdown_site, self.maintenance_site])

    def as_list(self) -> list[str]:
        return [name for name in DROP_OPTIONS if getattr(self, name)]


# ---------------------------------------------------------------------
# Date ranges (Process step 2/3)
# ---------------------------------------------------------------------

@dataclass
class DateRange:
    start_month: int  # YYYYMM, inclusive
    end_month: int     # YYYYMM, inclusive

    def __post_init__(self):
        if self.start_month > self.end_month:
            raise ValueError(f"start_month {self.start_month} is after end_month {self.end_month}")

    def contains(self, month: int) -> bool:
        return self.start_month <= month <= self.end_month


# ---------------------------------------------------------------------
# Stage 1 — quantile regression band
# ---------------------------------------------------------------------

@dataclass
class QuantileConfig:
    q_low: float = 0.01
    q_mid: float = 0.50
    q_high: float = 0.99

    def __post_init__(self):
        if not (0 < self.q_low < self.q_mid < self.q_high < 1):
            raise ValueError("quantiles must satisfy 0 < q_low < q_mid < q_high < 1")


MIN_HISTORY_MONTHS = 6  # site_month_no >= 6 needed before a row is model-ready (kwh_lag_6)

FEATURE_COLS = [
    "province_freq", "year", "month", "quarter", "month_sin", "month_cos",
    "kwh_lag_1", "kwh_lag_2", "kwh_lag_3", "kwh_lag_6",
    "kwh_roll_3_mean", "kwh_roll_6_mean", "kwh_roll_3_std",
]
TARGET_COL = "target_kwh_next"


# ---------------------------------------------------------------------
# Stage 2 — anomaly type classification (Result step 2, user-tunable)
# ---------------------------------------------------------------------

@dataclass
class ClassifyThresholds:
    """Jump-ratio thresholds the user sets on the Result tab.

    up / down   how large the single-month jump needs to be, as a multiplier
                of the pre-jump ~4-month median (value >= up * before, or
                value <= down * before).
    sustain     how close the *post*-jump median has to stay to the peak for
                the jump to be called a sustained "step" rather than a
                transient "spike" (after >= sustain * before, for the up case).
    """
    up: float = 1.5
    down: float = 1 / 1.5
    sustain: float = 1.3

    def __post_init__(self):
        if self.up <= 1:
            raise ValueError("up ratio must be > 1")
        if not (0 < self.down < 1):
            raise ValueError("down ratio must be between 0 and 1")
        if self.sustain <= 1:
            raise ValueError("sustain ratio must be > 1")