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

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from app import auth
from app.data_container import DataBillContainer
from app.schemas import UploadStatus, SiteTrendResponse
from app.routers.ml_routes import router as ml_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to the repo root so the app works no matter
# which directory uvicorn is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
CHUNKS_DIR = Path(tempfile.gettempdir()) / "anomaly_chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
STREAM_BLOCK_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
FILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")

app = FastAPI(title="Billing EDA Dashboard API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(ml_router)
auth.warn_if_open()


class RequireAuthMiddleware(BaseHTTPMiddleware):
    """Default-deny password gate for every endpoint (docs included).

    Only /api/health (keep-warm pings), /api/auth/login, / and /static stay
    public — see app/auth.py. OPTIONS passes through so CORS preflights
    keep working. Denied requests get a bare 401; they still pass back
    through CORSMiddleware (added after, so it wraps this one) and carry
    the CORS headers the browser needs to read the error.
    """

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS" or auth.is_public_path(request.url.path):
            return await call_next(request)
        if auth.request_is_authorized(request.headers.get("authorization")):
            return await call_next(request)
        return JSONResponse(
            {"detail": "Authentication required. Sign in with the dashboard password."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


app.add_middleware(RequireAuthMiddleware)

# Allow the GitHub Pages static frontend (and local dev) to call this API.
# The API is cookie-less, so credentials stay disabled — a wildcard origin
# combined with credentials would defeat the browser's CORS protection.
# Auth rides in the Authorization header, which allow_headers covers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your GitHub Pages origin in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request):
    """Exchange the shared password for an expiring bearer token. The
    password itself is never stored client-side and never logged here."""
    password = auth.app_password()
    if password is None:
        raise HTTPException(
            status_code=503,
            detail="Password auth is not configured on the server (APP_PASSWORD is unset).",
        )

    client_ip = request.client.host if request.client else "unknown"
    if not auth.LOGIN_RATE_LIMITER.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Wait a few minutes and try again.",
        )

    if not auth.password_matches(body.password, password):
        auth.LOGIN_RATE_LIMITER.record_failure(client_ip)
        raise HTTPException(status_code=401, detail="Incorrect password.")

    auth.LOGIN_RATE_LIMITER.clear(client_ip)
    token, expires_at = auth.issue_token(password)
    return {"token": token, "expires_at": expires_at}

# Single global container — see deploy notes above.
STATE: dict = {"container": DataBillContainer(), "lock": asyncio.Lock()}
UPLOAD_LOCKS: dict[str, asyncio.Lock] = {}
UPLOAD_LOCKS_GUARD = asyncio.Lock()

def get_container() -> DataBillContainer:
    container: DataBillContainer = STATE["container"]
    if not container.has_loaded_data():
        raise HTTPException(
            status_code=409,
            detail="No data loaded yet. Upload at least one billing file first."
        )
    container.ensure_master()
    return container


def upload_status_for(container: DataBillContainer) -> UploadStatus:
    loaded = container.loaded_files()
    missing = container.missing_files()
    return UploadStatus(
        loaded_files=loaded,
        missing_files=missing,
        ready=container.is_ready(),
        rows_total=container.rows_total(),
        message="All 5 files loaded." if not missing else
                f"Loaded {len(loaded)}/5 files. Still missing: {missing}",
        dropped_latest_month=container.dropped_latest_month,
    )


def validate_file_id(file_id: str) -> None:
    if not FILE_ID_RE.match(file_id):
        raise HTTPException(status_code=422, detail="Invalid file_id.")


async def get_upload_lock(file_id: str) -> asyncio.Lock:
    async with UPLOAD_LOCKS_GUARD:
        lock = UPLOAD_LOCKS.get(file_id)
        if lock is None:
            lock = asyncio.Lock()
            UPLOAD_LOCKS[file_id] = lock
        return lock


def ensure_metadata_consistent(
    meta: dict,
    *,
    file_key: str,
    total_chunks: int,
    file_name: Optional[str],
    file_size: Optional[int],
    chunk_size: Optional[int],
) -> None:
    mismatches: list[str] = []
    expected = {
        "file_key": file_key,
        "total_chunks": total_chunks,
        "file_name": file_name,
        "file_size": file_size,
        "chunk_size": chunk_size,
    }

    for key, value in expected.items():
        if value is not None and meta.get(key) not in (None, value):
            mismatches.append(key)

    if mismatches:
        raise HTTPException(
            status_code=409,
            detail=f"Upload metadata mismatch for: {', '.join(mismatches)}",
        )


def write_metadata(
    meta_path: Path,
    *,
    file_key: str,
    total_chunks: int,
    file_name: Optional[str],
    file_size: Optional[int],
    chunk_size: Optional[int],
) -> dict:
    meta = {
        "file_key": file_key,
        "total_chunks": total_chunks,
        "file_name": file_name,
        "file_size": file_size,
        "chunk_size": chunk_size,
        "created_at": time.time(),
    }
    tmp_path = meta_path.with_suffix(".json.part")
    tmp_path.write_text(json.dumps(meta))
    tmp_path.replace(meta_path)
    return meta


async def stream_upload_to_path(upload: UploadFile, destination: Path) -> int:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    bytes_written = 0

    try:
        with open(tmp_path, "wb") as out:
            while True:
                block = await upload.read(STREAM_BLOCK_BYTES)
                if not block:
                    break

                bytes_written += len(block)
                if bytes_written > MAX_CHUNK_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Chunk exceeds {MAX_CHUNK_BYTES // 1024 // 1024}MB limit.",
                    )

                out.write(block)

        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Empty chunk uploaded.")

        tmp_path.replace(destination)
        return bytes_written
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove partial chunk file: %s", tmp_path)
        raise


def validate_chunk_size(
    *,
    chunk_number: int,
    total_chunks: int,
    bytes_written: int,
    file_size: Optional[int],
    chunk_size: Optional[int],
) -> None:
    if file_size is None or chunk_size is None:
        return

    expected = (
        chunk_size
        if chunk_number < total_chunks - 1
        else file_size - (chunk_size * (total_chunks - 1))
    )

    if expected <= 0 or bytes_written != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Chunk {chunk_number} size mismatch: expected {expected} bytes, "
                f"received {bytes_written} bytes."
            ),
        )


def load_and_build(container: DataBillContainer, files: dict[str, object]) -> None:
    container.load_files(files)


async def finalize_chunk_upload(file_id: str) -> UploadStatus:
    validate_file_id(file_id)
    upload_lock = await get_upload_lock(file_id)

    async with upload_lock:
        upload_dir = CHUNKS_DIR / file_id
        if not upload_dir.exists():
            raise HTTPException(status_code=400, detail=f"No chunks found for file_id: {file_id}")

        meta_path = upload_dir / "meta.json"
        if not meta_path.exists():
            raise HTTPException(status_code=400, detail="Upload metadata missing; please re-upload.")

        meta = json.loads(meta_path.read_text())
        file_key: str = meta["file_key"]
        total_chunks: int = meta["total_chunks"]
        if file_key not in VALID_FILE_KEYS:
            raise HTTPException(status_code=422, detail="Invalid file_key in upload metadata.")

        missing_chunks = [
            i for i in range(total_chunks)
            if not (upload_dir / f"{i}.chunk").exists()
        ]
        if missing_chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Missing chunks {missing_chunks[:10]} of {total_chunks}. Please re-upload."
            )

        assembled_path = upload_dir / "assembled.bin"
        total_size = 0
        with open(assembled_path, "wb") as out:
            for i in range(total_chunks):
                chunk_path = upload_dir / f"{i}.chunk"
                with open(chunk_path, "rb") as f:
                    shutil.copyfileobj(f, out, STREAM_BLOCK_BYTES)
                total_size += chunk_path.stat().st_size
                chunk_path.unlink()

        expected_size = meta.get("file_size")
        if expected_size is not None and total_size != expected_size:
            raise HTTPException(
                status_code=400,
                detail=f"Assembled file size mismatch: expected {expected_size}, received {total_size}.",
            )

        logger.info(
            "Reassembled %s chunks for %s: %.1fMB",
            total_chunks,
            file_key,
            total_size / 1_000_000,
        )

        try:
            t0 = time.time()
            with open(assembled_path, "rb") as fh:
                async with STATE["lock"]:
                    container: DataBillContainer = STATE["container"]
                    await run_in_threadpool(load_and_build, container, {file_key: fh})
            logger.info("Finalized %s: %.1fs", file_key, time.time() - t0)
        except ValueError as e:
            logger.error("Validation error: %s", e)
            raise HTTPException(status_code=422, detail=f"Failed to process file: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error while processing file")
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:100]}") from e
        finally:
            try:
                shutil.rmtree(upload_dir)
                logger.info("Cleaned up chunk directory: %s", upload_dir)
            except Exception as e:
                logger.warning("Failed to clean up chunk directory %s: %s", upload_dir, e)

            async with UPLOAD_LOCKS_GUARD:
                UPLOAD_LOCKS.pop(file_id, None)

        return upload_status_for(STATE["container"])


async def process_direct_uploads(incoming: dict[str, Optional[UploadFile]]) -> UploadStatus:
    uploads = {key: upload for key, upload in incoming.items() if upload is not None}
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were provided.")

    for upload in uploads.values():
        await upload.seek(0)

    file_summary = {
        key: upload.filename or "unnamed"
        for key, upload in uploads.items()
    }
    logger.info("Uploading %s direct file(s): %s", len(uploads), file_summary)

    try:
        t0 = time.time()
        async with STATE["lock"]:
            container: DataBillContainer = STATE["container"]
            await run_in_threadpool(
                load_and_build,
                container,
                {key: upload.file for key, upload in uploads.items()},
            )
        logger.info("Direct upload processed in %.1fs", time.time() - t0)
    except ValueError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=422, detail=f"Failed to process uploaded file(s): {e}") from e
    except Exception as e:
        logger.exception("Unexpected error while processing uploaded file(s)")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)[:100]}") from e

    return upload_status_for(STATE["container"])


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
    file_name: Optional[str] = Query(None, description="Original client file name"),
    file_size: Optional[int] = Query(None, ge=0, description="Original file size in bytes"),
    chunk_size: Optional[int] = Query(None, gt=0, description="Configured client chunk size in bytes"),
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
    validate_file_id(file_id)
    if chunk_number >= total_chunks:
        raise HTTPException(status_code=422, detail="chunk_number must be less than total_chunks.")
    if chunk_size is not None and chunk_size > MAX_CHUNK_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"chunk_size must be at most {MAX_CHUNK_BYTES} bytes.",
        )

    upload_dir = CHUNKS_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_lock = await get_upload_lock(file_id)

    async with upload_lock:
        meta_path = upload_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            ensure_metadata_consistent(
                meta,
                file_key=file_key,
                total_chunks=total_chunks,
                file_name=file_name,
                file_size=file_size,
                chunk_size=chunk_size,
            )
        else:
            meta = write_metadata(
                meta_path,
                file_key=file_key,
                total_chunks=total_chunks,
                file_name=file_name,
                file_size=file_size,
                chunk_size=chunk_size,
            )

        chunk_path = upload_dir / f"{chunk_number}.chunk"
        bytes_written = await stream_upload_to_path(chunk, chunk_path)
        try:
            validate_chunk_size(
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                bytes_written=bytes_written,
                file_size=meta.get("file_size"),
                chunk_size=meta.get("chunk_size"),
            )
        except HTTPException:
            chunk_path.unlink(missing_ok=True)
            raise

    logger.info(
        f"Received chunk {chunk_number + 1}/{total_chunks} for {file_key} "
        f"(file_id={file_id}, size={bytes_written/1_000_000:.1f}MB)"
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
    return await finalize_chunk_upload(file_id)

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
    return await process_direct_uploads(incoming)

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
    if not container.loaded_files():
        return UploadStatus(
            loaded_files=[],
            missing_files=container.missing_files(),
            ready=False,
            rows_total=0,
            message="Waiting for uploads.",
        )
    return upload_status_for(container)


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


METER_PATTERN_RE = "^(shutdown|maintenance|gap|normal)$"


@app.get("/api/eda/meter-patterns")
def eda_meter_patterns(
    window: int = Query(3, ge=1, le=24, description="how many most-recent months to classify"),
    pattern: Optional[str] = Query(None, pattern=METER_PATTERN_RE,
                                   description="only rows of this pattern"),
    limit: Optional[int] = Query(None, ge=0, le=1000,
                                 description="page size; omit for all rows"),
    offset: int = Query(0, ge=0, description="rows to skip (paging)"),
):
    """Datasheet of every uploaded meter with its last-N-months bill amounts
    and a pattern label: shutdown (no bill at all), maintenance (only the
    meter charge), gap (billed intermittently, 'ฟันหลอ') or normal — plus
    unique meter counts."""
    return get_container().eda_meter_patterns(
        window=window, pattern=pattern, limit=limit, offset=offset)


@app.get("/api/eda/meter-patterns/export")
def eda_meter_patterns_export(
    window: int = Query(3, ge=1, le=24),
    pattern: Optional[str] = Query(None, pattern=METER_PATTERN_RE),
):
    """Stream the full meter-pattern datasheet as a CSV download."""
    container = get_container()
    filename = f"bill_patterns_{pattern or 'all'}.csv"

    def generate():
        yield "\ufeff"  # UTF-8 BOM so Excel renders Thai text correctly
        yield from container.meter_patterns_csv(window=window, pattern=pattern)

    return StreamingResponse(
        generate(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
