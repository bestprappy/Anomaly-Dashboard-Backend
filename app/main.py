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

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.data_container import DataBillContainer
from app.schemas import UploadStatus, SiteTrendResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to the repo root so the app works no matter
# which directory uvicorn is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
CHUNKS_DIR = Path(tempfile.gettempdir()) / "anomaly_chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Billing EDA Dashboard API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Allow the GitHub Pages static frontend (and local dev) to call this API.
# The API is cookie-less, so credentials stay disabled — a wildcard origin
# combined with credentials would defeat the browser's CORS protection.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your GitHub Pages origin in production
    allow_credentials=False,
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
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


VALID_FILE_KEYS = {"pea_bfkt", "pea_tuc", "mea_bfkt", "mea_tuc", "mea_tmv"}


@app.post("/api/upload/chunk")
async def upload_chunk(
    file_id: str = Query(..., description="Unique file identifier"),
    chunk_number: int = Query(..., ge=0, description="Chunk sequence number starting from 0"),
    total_chunks: int = Query(..., ge=1, description="Total number of chunks for this file"),
    file_key: str = Query(..., description="Which file this chunk belongs to (pea_bfkt, etc)"),
    chunk: UploadFile = File(...),
):
    """
    Upload a single chunk of a large file. Chunks are stored on disk to save memory.
    Call multiple times with different chunk_number values until total_chunks are received.
    """
    if file_key not in VALID_FILE_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file_key '{file_key}'. Must be one of: {sorted(VALID_FILE_KEYS)}"
        )
    # file_id becomes a directory name — reject anything that could escape CHUNKS_DIR
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(status_code=422, detail="Invalid file_id.")

    upload_dir = CHUNKS_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Persist metadata so finalize doesn't have to guess the file_key from the id
    meta_path = upload_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"file_key": file_key, "total_chunks": total_chunks}))

    chunk_data = await chunk.read()
    (upload_dir / f"{chunk_number}.chunk").write_bytes(chunk_data)

    logger.info(
        f"Received chunk {chunk_number + 1}/{total_chunks} for {file_key} "
        f"(file_id={file_id}, size={len(chunk_data)/1_000_000:.1f}MB)"
    )

    return {
        "file_id": file_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "status": "chunk_received",
    }


@app.post("/api/upload/finalize", response_model=UploadStatus)
async def finalize_chunks(
    file_id: str = Query(..., description="File ID from chunk uploads"),
):
    """
    Finalize a chunked upload: verify all chunks arrived, reassemble them into a
    single file on disk (never in RAM), then load and process it.
    """
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(status_code=422, detail="Invalid file_id.")

    upload_dir = CHUNKS_DIR / file_id
    if not upload_dir.exists():
        raise HTTPException(status_code=400, detail=f"No chunks found for file_id: {file_id}")

    meta_path = upload_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="Upload metadata missing; please re-upload.")
    meta = json.loads(meta_path.read_text())
    file_key: str = meta["file_key"]
    total_chunks: int = meta["total_chunks"]

    # Verify every chunk 0..N-1 is present before assembling
    missing_chunks = [
        i for i in range(total_chunks)
        if not (upload_dir / f"{i}.chunk").exists()
    ]
    if missing_chunks:
        raise HTTPException(
            status_code=400,
            detail=f"Missing chunks {missing_chunks[:10]} of {total_chunks}. Please re-upload."
        )

    # Reassemble on disk (streaming) so a 100MB file never sits in RAM as bytes
    assembled_path = upload_dir / "assembled.bin"
    total_size = 0
    with open(assembled_path, "wb") as out:
        for i in range(total_chunks):
            chunk_path = upload_dir / f"{i}.chunk"
            with open(chunk_path, "rb") as f:
                shutil.copyfileobj(f, out)
            total_size += chunk_path.stat().st_size
            chunk_path.unlink()  # free disk as we go

    logger.info(f"Reassembled {total_chunks} chunks for {file_key}: {total_size/1_000_000:.1f}MB")

    container: DataBillContainer = STATE["container"]
    try:
        t0 = time.time()
        with open(assembled_path, "rb") as fh:
            container.load_files({file_key: fh})
        container.build_master()
        logger.info(f"✓ Finalized {file_key}: {time.time()-t0:.1f}s")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to process file: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error while processing file")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:100]}") from e
    finally:
        try:
            shutil.rmtree(upload_dir)
            logger.info(f"Cleaned up chunk directory: {upload_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up chunk directory {upload_dir}: {e}")

    loaded = container.loaded_files()
    missing = container.missing_files()
    return UploadStatus(
        loaded_files=loaded,
        missing_files=missing,
        ready=container.is_ready(),
        rows_total=container.rows_total(),
        message="All 5 files loaded." if not missing else
                f"Loaded {len(loaded)}/5 files. Still missing: {missing}",
    )


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

    Recommended: upload files one-at-a-time (sequential) for better reliability
    on resource-constrained servers (e.g. Render free tier).
    """
    incoming = {
        "pea_bfkt": pea_bfkt, "pea_tuc": pea_tuc,
        "mea_bfkt": mea_bfkt, "mea_tuc": mea_tuc, "mea_tmv": mea_tmv,
    }
    files = {k: (await v.read()) for k, v in incoming.items() if v is not None}
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    file_summary = {k: f"{len(v) / 1_000_000:.1f}MB" for k, v in files.items()}
    logger.info(f"Uploading {len(files)} file(s): {file_summary}")

    container: DataBillContainer = STATE["container"]
    try:
        t0 = time.time(); container.load_files(files); t1 = time.time()
        container.build_master(); t2 = time.time()
        logger.info(f"✓ load={t1-t0:.1f}s build_master={t2-t1:.1f}s total={t2-t0:.1f}s")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to process uploaded file(s): {e}") from e
    except Exception as e:
        logger.exception("Unexpected error while processing uploaded file(s)")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:100]}") from e

    loaded = container.loaded_files()
    missing = container.missing_files()
    return UploadStatus(
        loaded_files=loaded,
        missing_files=missing,
        ready=container.is_ready(),
        rows_total=container.rows_total(),
        message="All 5 files loaded." if not missing else
                f"Loaded {len(loaded)}/5 files. Still missing: {missing}",
    )


@app.get("/api/upload/status", response_model=UploadStatus)
def upload_status():
    container: DataBillContainer = STATE["container"]
    return UploadStatus(
        loaded_files=container.loaded_files(),
        missing_files=container.missing_files(),
        ready=container.is_ready(),
        rows_total=container.rows_total(),
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
    try:
        win = tuple(int(w) for w in windows.split(",") if w.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="'windows' must be comma-separated integers, e.g. 3,6,9")
    if not win or any(w <= 0 for w in win):
        raise HTTPException(status_code=400, detail="'windows' must contain at least one positive integer.")
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