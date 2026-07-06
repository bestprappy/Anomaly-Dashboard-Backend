"""
main.py — FastAPI backend for the Billing EDA Dashboard.

Endpoints
---------
POST /api/upload                 upload the 5 raw bill files, build the master table
GET  /api/upload/status          check what's loaded
GET  /api/eda/summary            everything the EDA tab needs, in one call
GET  /api/eda/bill-range
GET  /api/eda/duplicates
GET  /api/eda/common-sites
GET  /api/eda/site-types
GET  /api/eda/missing-consequence
GET  /api/eda/maintenance-sites
GET  /api/eda/error-rates
GET  /api/sites                  list all Site_IDs (for a search box / dropdown)
GET  /api/site/{site_id}/trend   monthly kwh / bill_amount series for one site

Deploy notes (Render)
---------------------
This demo keeps the processed data in a single in-memory DataBillContainer
(`STATE["container"]`). That's fine for a small internal tool on a single
Render instance, but it means data resets on restart/redeploy and isn't
shared across multiple instances/workers. If you need persistence or
multi-worker support, swap STATE for a cache (e.g. Redis) or persist
`master_df` to disk/S3 after build_master() and load it back in on startup.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.data_container import DataBillContainer
from app.schemas import UploadStatus, SiteTrendResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Billing EDA Dashboard API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow the GitHub Pages static frontend (and local dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your GitHub Pages origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global container — see deploy notes above.
STATE: dict = {"container": DataBillContainer()}

def get_container() -> DataBillContainer:
    container: DataBillContainer = STATE["container"]
    if not container.is_ready():
        raise HTTPException(
            status_code=409,
            detail="No data loaded yet. POST the 5 files to /api/upload first."
        )
    return container


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadStatus)
async def upload_files(
    pea_bfkt: Optional[UploadFile] = File(None),
    pea_tuc: Optional[UploadFile] = File(None),
    mea_bfkt: Optional[UploadFile] = File(None),
    mea_tuc: Optional[UploadFile] = File(None),
    mea_tmv: Optional[UploadFile] = File(None),
):
    """
    Upload some or all of the 5 raw bill files. Can be called multiple times
    to add files incrementally; each call rebuilds the master table from
    whatever has been loaded so far.
    """
    incoming = {
        "pea_bfkt": pea_bfkt, "pea_tuc": pea_tuc,
        "mea_bfkt": mea_bfkt, "mea_tuc": mea_tuc, "mea_tmv": mea_tmv,
    }
    files = {k: (await v.read()) for k, v in incoming.items() if v is not None}
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    container: DataBillContainer = STATE["container"]
    try:
        t0=time.time(); container.load_files(files); t1=time.time()
        container.build_master(); t2=time.time()
        logger.info(f"load={t1-t0:.1f}s build_master={t2-t1:.1f}s")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process uploaded file(s): {e}")

    missing = container.missing_files()
    return UploadStatus(
        loaded_files=sorted(container._loaded_keys),
        missing_files=missing,
        ready=container.is_ready(),
        rows_total=len(container.master_df) if container.master_df is not None else 0,
        message="All 5 files loaded." if not missing else
                f"Loaded {len(container._loaded_keys)}/5 files. Still missing: {missing}",
    )


@app.get("/api/upload/status", response_model=UploadStatus)
def upload_status():
    container: DataBillContainer = STATE["container"]
    missing = container.missing_files()
    return UploadStatus(
        loaded_files=sorted(container._loaded_keys),
        missing_files=missing,
        ready=container.is_ready(),
        rows_total=len(container.master_df) if container.master_df is not None else 0,
        message="Ready." if container.is_ready() else "Waiting for uploads.",
    )


# ---------------------------------------------------------------------
# EDA endpoints
# ---------------------------------------------------------------------

@app.get("/api/eda/summary")
def eda_summary():
    return get_container().eda_summary()


@app.get("/api/eda/bill-range")
def eda_bill_range():
    return get_container().eda_bill_range()


@app.get("/api/eda/duplicates")
def eda_duplicates():
    return get_container().eda_duplicates()


@app.get("/api/eda/common-sites")
def eda_common_sites():
    return get_container().eda_common_sites()


@app.get("/api/eda/site-types")
def eda_site_types():
    return get_container().eda_site_types()


@app.get("/api/eda/missing-consequence")
def eda_missing_consequence(
    windows: str = Query("3,6,9", description="comma-separated month windows, e.g. 3,6,9")
):
    win = tuple(int(w) for w in windows.split(",") if w.strip())
    return get_container().eda_last_month_missing(windows=win)


@app.get("/api/eda/maintenance-sites")
def eda_maintenance_sites():
    return get_container().eda_maintenance_sites()


@app.get("/api/eda/error-rates")
def eda_error_rates():
    return get_container().eda_error_rates()


# ---------------------------------------------------------------------
# Site lookup / trend
# ---------------------------------------------------------------------

@app.get("/api/sites")
def list_sites(provider: Optional[str] = Query(None, pattern="^(PEA|MEA)$")):
    return {"site_ids": get_container().list_site_ids(provider=provider)}


@app.get("/api/site/{site_id}/trend", response_model=SiteTrendResponse)
def site_trend(
    site_id: str,
    metric: str = Query("kwh", pattern="^(kwh|bill_amount)$"),
    start_month: Optional[int] = Query(None, description="YYYYMM"),
    end_month: Optional[int] = Query(None, description="YYYYMM"),
):
    result = get_container().get_site_trend(
        site_id=site_id, metric=metric, start_month=start_month, end_month=end_month
    )
    return result