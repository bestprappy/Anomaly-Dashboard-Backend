---
title: Anomaly Dashboard API
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# ML Module — Site-Jump Anomaly Detection

This adds a self-contained `app/ml/` package plus `app/routers/ml_routes.py`
implementing the pipeline from `site_jump_quantile.ipynb` (quantile-regression
band + step/spike classification) as an API, split into **Process** and
**Result** steps as requested. **No Isolation Forest** — `quantile_severity`
(band-widths outside P5/P95) is the only ranking signal.

It replaces the manual `00_prepare_model_input.ipynb` step: instead of a
notebook writing `model_input_active.csv` ahead of time, `app/ml/site_filters.py`
+ `app/ml/features.py` do that filtering/feature-building live, straight off
`DataBillContainer.master_df`, driven by whatever the user picks in the
Process tab.

## File layout

```
app/
  ml/
    __init__.py
    config.py          # DropOptions, DateRange, QuantileConfig, ClassifyThresholds — all tunables in one place
    site_filters.py     # the 4 drop-option checkboxes, built on DataBillContainer's existing eda_* methods
    missing_rate.py      # per-month missing-rate for the date-range picker
    features.py           # feature engineering, ported from the notebooks
    quantile_model.py      # Stage 1: HistGradientBoostingRegressor quantile band + flagging
    classify.py              # Stage 2: spike_up / step_up / step_down / spike_down / other
    plotting.py                # matplotlib PNGs for examples + bulk zip download
    state.py                     # MLRunState — single in-memory run, mirrors main.py's STATE pattern
    pipeline.py                   # orchestrates the above; only file the router calls into
    schemas.py                     # pydantic request bodies
  routers/
    __init__.py
    ml_routes.py                    # FastAPI router, prefix /api/ml
```

Each module has one job, so a bug is easy to localize: wrong drop counts ->
`site_filters.py`; wrong features -> `features.py`; band looks miscalibrated
-> `quantile_model.py`; a jump classified wrong -> `classify.py` (and only
`classify.py` — it never refits anything).

## Integration into your existing repo

1. Copy `app/ml/` and `app/routers/` into your backend's `app/` package.
2. In `main.py`, add:

   ```python
   from app.routers.ml_routes import router as ml_router
   app.include_router(ml_router)
   ```

   (Anywhere after `app = FastAPI(...)` is fine — router registration order
   doesn't matter.)

3. Add to `requirements.txt` if not already present:
   ```
   scikit-learn
   matplotlib
   ```
4. No changes needed to `data_container.py` — the ML module only reads
   `container.master_df` and calls the existing `eda_duplicates()` /
   `eda_common_sites()` methods, it never mutates the container.

## API

### Process

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/ml/drop-options` | the 4 checkbox definitions `{value, label}` |
| POST | `/api/ml/preview` | drop report + per-month missing-rate for a candidate window |
| POST | `/api/ml/build` | fit the model, return coverage/flagged-rate metrics |
| GET | `/api/ml/abnormal` | flagged anomalies as `(site_id, month, kwh)` — the plain result the spec asked for |

**POST /api/ml/preview**
```json
{
  "drop_options": {"duplicate_site": true, "shutdown_site": true},
  "start_month": 202301,
  "end_month": 202612
}
```
Returns `drop_report` (how many sites each checkbox would remove) and
`missing.per_month` (a small series the frontend can chart next to the date
pickers — this is the "show missing rate" ask, so the user can see old,
low-quality months before picking a train start).

**POST /api/ml/build**
```json
{
  "drop_options": {"duplicate_site": true, "shutdown_site": true, "maintenance_site": true},
  "train_start": 202301, "train_end": 202512,
  "test_start": 202601, "test_end": 202612,
  "q_low": 0.05, "q_mid": 0.5, "q_high": 0.95
}
```
Fits 3 quantile models on `train`, evaluates the band on `test`, and stores
the result in `ML_STATE` for the later `/abnormal`, `/classify`, `/examples`,
`/plots/download` calls. Returns coverage (should sit near `q_high - q_low`,
e.g. ~90% for 0.05/0.95) and the flagged rate — the frontend's "did this run
well" readout before the user moves to Result.

**GET /api/ml/abnormal** → `{"count": N, "rows": [{site_id, anom_month, kwh, q05, q50, q95, quantile_severity}, ...]}`

### Result

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ml/classify` | user-input UP/DOWN/SUSTAIN → type counts + surfaced (spike_up/step_up) rows |
| GET | `/api/ml/severity-duration` | nine-cell severity x duration matrix for the latest classification |
| GET | `/api/ml/examples?anom_type=spike_up&limit=5` | up to 5 base64 PNGs |
| GET | `/api/ml/plots/download?types=spike_up,step_up` | zip of every plot for the requested types |

**POST /api/ml/classify**
```json
{"up": 1.5, "down": 0.667, "sustain": 1.3}
```
Cheap — reuses the already-built model's flagged rows and the site's full
kWh history captured at build time, so the user can retune these 3 numbers
as many times as they like without rebuilding.

**GET /api/ml/examples?anom_type=step_up&limit=5** → base64 PNGs the frontend
can drop straight into `<img src="data:image/png;base64,...">`.

**GET /api/ml/plots/download** → binary zip response
(`Content-Disposition: attachment`), structured `{type}/{site_id}.png`.

## Design notes / assumptions made explicit

- **Drop options are site-level, not row-level.** Once a Site_ID is flagged
  duplicate/common/shutdown/maintenance, its *entire* history is removed
  before feature-building — a half-clean series produces broken lag/rolling
  features and a misleading band. If you'd rather drop only the offending
  months, that's a one-line change in `site_filters.apply_drop_options`.
- **Train/test are explicit date ranges you choose**, not the notebook's 80/20
  chronological split — matches "select train range and test range" in the
  spec. The two windows are not required to be non-overlapping; validate that
  in the frontend if you want to enforce it.
- **Classification thresholds are separate from the build step** by design —
  section "Result step 2" says the user inputs UP/DOWN/SUSTAIN, so
  `/api/ml/classify` never touches the fitted models, only re-labels the
  already-flagged rows.
- **Only spike_up/step_up are surfaced** in `/classify`'s `rows`,
  `/examples`, and `/plots/download` (`step_down`/`spike_down`/`other` are
  still computed internally for correctness, just not returned), per "just 2
  types we concern about."

## Severity-duration 3 x 3 analysis notebook

`notebooks/severity_duration_matrix.ipynb` runs the production preprocessing,
quantile model, and classifier, then converts the flagged site-months into one
row per event. It produces a matrix with severity on the columns (Low < 1,
Medium 1 to < 3, High >= 3 band widths beyond q95) and observed duration on
the rows (single month, 2-3 months, >=4 months).

With `DATA_MODE = "auto"`, the notebook uses all five exports under
`app/ml/data/` when they are available and otherwise falls back to a
deterministic synthetic demo. The raw-data directory is ignored by Git so
confidential billing exports are not committed. Raw mode calls
`DataBillContainer.load_files()` and therefore uses the same PEA/MEA cleaning
as the API. Run Jupyter from this backend directory with the backend
environment selected as the notebook kernel, for example on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel
.\.venv\Scripts\python.exe -m jupyter lab
```

The reusable event logic lives in `app/ml/severity_duration.py`. It merges
consecutive flags from the same elevated run, assigns severity from the first
surfaced detection, and keeps short right-censored events out of the matrix
rather than silently calling them spikes. Set `EXPORT_RESULTS = True` in the
notebook to write the event audit, both matrices, and run metadata under
`notebooks/outputs/`.
