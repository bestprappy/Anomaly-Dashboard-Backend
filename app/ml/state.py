"""
In-memory state for the ML pipeline, mirroring the single-instance STATE
pattern already used for the raw-data container in main.py. One run lives at
a time; a new POST /api/ml/build call replaces it. Same caveat as the data
container applies here (see main.py's deploy notes) — this resets on
restart and isn't shared across multiple workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.ml.config import ClassifyThresholds


@dataclass
class MLRunState:
    drop_report: Optional[dict] = None
    train_range: Optional[tuple[int, int]] = None
    test_range: Optional[tuple[int, int]] = None
    metrics: Optional[dict] = None
    test_flagged: Optional[pd.DataFrame] = None    # flagged test rows only (slim cols, flag_quantile all True)
    full_history: Optional[pd.DataFrame] = None    # per-site series of the *flagged* sites, for classification/plots
    classified: Optional[pd.DataFrame] = None      # set once /classify is called; cleared on rebuild
    thresholds: ClassifyThresholds = field(default_factory=ClassifyThresholds)

    def is_built(self) -> bool:
        return self.test_flagged is not None

    def is_classified(self) -> bool:
        return self.classified is not None


ML_STATE = MLRunState()