"""End-to-end smoke test: upload synthetic files, hit every endpoint.

Run with: pytest tests/
"""
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.data_container import DataBillContainer
from app.main import app, STATE


@pytest.fixture()
def client():
    # each test starts from a clean container
    STATE["container"] = DataBillContainer()
    return TestClient(app)


def pea_csv(site_prefix: str) -> bytes:
    # row 0 = throwaway header, row 1 = real column names, then data.
    # Layout the parser expects: id cols, 'avg', BE amount months, 'avg', BE unit months.
    lines = [
        "h1,h2,h3,h4,h5,h6,h7,h8,h9",
        "Site_ID,Meter_No.,Province,avg,256901,256902,avg,256901,256902",
        f"{site_prefix}4017,111,BKK,,500,600,,50,60",
        f"{site_prefix}4018,112,BKK,,100,150,,10,15",
    ]
    return "\n".join(lines).encode()


def mea_csv(site_prefix: str) -> bytes:
    # row 0 = junk banner (real header is row index 1); the duplicated month
    # names in the unit block get pandas' '.1' mangling, as in real CSV exports.
    lines = [
        "junk banner line,,,,,,",
        "Meter_No,Site_ID,MSC/RMSC/IBC/WIFI/Decom,201901,201902,201901,201902",
        f"201,{site_prefix}9001,0,300,400,30,40",
        f"202,{site_prefix}9002,WIFI,50,0,5,0",
        "total,,,,,,",  # trailing summary row that must be dropped
    ]
    return "\n".join(lines).encode()


def upload_all(client: TestClient):
    files = {
        "pea_bfkt": ("pea_bfkt.csv", io.BytesIO(pea_csv("CBR")), "text/csv"),
        "pea_tuc": ("pea_tuc.csv", io.BytesIO(pea_csv("TUC")), "text/csv"),
        "mea_bfkt": ("mea_bfkt.csv", io.BytesIO(mea_csv("MBF")), "text/csv"),
        "mea_tuc": ("mea_tuc.csv", io.BytesIO(mea_csv("MTU")), "text/csv"),
        "mea_tmv": ("mea_tmv.csv", io.BytesIO(mea_csv("MTM")), "text/csv"),
    }
    return client.post("/api/upload", files=files)


def upload_chunked_file(
    client: TestClient,
    *,
    file_key: str,
    file_id: str,
    data: bytes,
    chunk_size: int = 40,
):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    for i, chunk in enumerate(chunks):
        resp = client.post(
            "/api/upload/chunk",
            params={
                "file_id": file_id,
                "chunk_number": i,
                "total_chunks": len(chunks),
                "file_key": file_key,
                "file_name": f"{file_key}.csv",
                "file_size": len(data),
                "chunk_size": chunk_size,
            },
            files={"chunk": ("blob", io.BytesIO(chunk), "application/octet-stream")},
        )
        assert resp.status_code == 200, resp.text

    return chunks


def test_health_and_root(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_status_and_409_before_upload(client):
    status = client.get("/api/upload/status").json()
    assert status["ready"] is False
    assert client.get("/api/eda/summary").status_code == 409


def test_upload_and_all_eda_endpoints(client):
    up = upload_all(client)
    assert up.status_code == 200
    body = up.json()
    assert body["ready"] is True
    assert body["rows_total"] > 0
    assert body["missing_files"] == []

    for path in ("summary", "bill-range", "duplicates", "common-sites",
                 "site-types", "missing-consequence", "maintenance-sites",
                 "error-rates"):
        resp = client.get(f"/api/eda/{path}")
        assert resp.status_code == 200, f"{path}: {resp.text[:300]}"
        resp.json()  # must be valid JSON (no NaN)


def test_bad_windows_param_returns_400(client):
    upload_all(client)
    assert client.get("/api/eda/missing-consequence", params={"windows": "3,x"}).status_code == 400
    assert client.get("/api/eda/missing-consequence", params={"windows": "-1"}).status_code == 400


def test_sites_and_trends(client):
    upload_all(client)

    sites = client.get("/api/sites").json()["site_ids"]
    assert "CBR4017" in sites and "MBF9001" in sites

    trend = client.get("/api/site/CBR4017/trend").json()
    assert trend["found"] is True
    # PEA Buddhist-era month 256901 -> Gregorian 202601
    assert trend["series"][0] == {"month": 202601, "value": 50.0}

    trend = client.get("/api/site/MBF9001/trend", params={"metric": "bill_amount"}).json()
    assert trend["found"] is True
    assert trend["series"][0] == {"month": 201901, "value": 300.0}

    assert client.get("/api/site/NOPE123/trend").json()["found"] is False


def test_chunked_upload_and_finalize(client):
    """Split a synthetic PEA file into chunks, upload them, finalize, verify."""
    data = pea_csv("CBR")
    file_id = "pea_bfkt-1234567890-abc123"

    upload_chunked_file(client, file_key="pea_bfkt", file_id=file_id, data=data)

    resp = client.post("/api/upload/finalize", params={"file_id": file_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "pea_bfkt" in body["loaded_files"]
    assert body["rows_total"] > 0
    assert body["ready"] is False
    assert "pea_tuc" in body["missing_files"]

    # The assembled data parsed successfully, but dashboard endpoints stay
    # gated until all required billing files are loaded.
    assert client.get("/api/sites").status_code == 409


def test_chunk_upload_rejects_bad_sequence_and_metadata(client):
    data = pea_csv("CBR")
    chunk = data[:40]

    resp = client.post(
        "/api/upload/chunk",
        params={
            "file_id": "bad-sequence-1",
            "chunk_number": 1,
            "total_chunks": 1,
            "file_key": "pea_bfkt",
        },
        files={"chunk": ("blob", io.BytesIO(chunk), "application/octet-stream")},
    )
    assert resp.status_code == 422

    resp = client.post(
        "/api/upload/chunk",
        params={
            "file_id": "metadata-mismatch-1",
            "chunk_number": 0,
            "total_chunks": 3,
            "file_key": "pea_bfkt",
            "file_size": len(data),
            "chunk_size": 40,
        },
        files={"chunk": ("blob", io.BytesIO(chunk), "application/octet-stream")},
    )
    assert resp.status_code == 200

    resp = client.post(
        "/api/upload/chunk",
        params={
            "file_id": "metadata-mismatch-1",
            "chunk_number": 1,
            "total_chunks": 4,
            "file_key": "pea_bfkt",
            "file_size": len(data),
            "chunk_size": 40,
        },
        files={"chunk": ("blob", io.BytesIO(data[40:80]), "application/octet-stream")},
    )
    assert resp.status_code == 409


def test_concurrent_chunk_finalization_is_serialized(client):
    upload_chunked_file(
        client,
        file_key="pea_bfkt",
        file_id="pea-bfkt-concurrent",
        data=pea_csv("CBR"),
    )
    upload_chunked_file(
        client,
        file_key="pea_tuc",
        file_id="pea-tuc-concurrent",
        data=pea_csv("TUC"),
    )

    def finalize(file_id: str):
        return client.post("/api/upload/finalize", params={"file_id": file_id})

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(finalize, ["pea-bfkt-concurrent", "pea-tuc-concurrent"]))

    assert all(resp.status_code == 200 for resp in responses)
    status = client.get("/api/upload/status").json()
    assert {"pea_bfkt", "pea_tuc"}.issubset(set(status["loaded_files"]))
    assert status["ready"] is False


def test_finalize_rejects_missing_chunks_and_bad_ids(client):
    # finalize with no chunks at all
    assert client.post("/api/upload/finalize", params={"file_id": "nope-1-x"}).status_code == 400

    # upload chunk 0 of 3, then finalize -> must report missing chunks
    client.post(
        "/api/upload/chunk",
        params={"file_id": "partial-1-x", "chunk_number": 0,
                "total_chunks": 3, "file_key": "pea_bfkt"},
        files={"chunk": ("blob", io.BytesIO(b"abc"), "application/octet-stream")},
    )
    resp = client.post("/api/upload/finalize", params={"file_id": "partial-1-x"})
    assert resp.status_code == 400 and "Missing chunks" in resp.text

    # invalid file_key and path-traversal file_id are rejected
    resp = client.post(
        "/api/upload/chunk",
        params={"file_id": "x-1-y", "chunk_number": 0,
                "total_chunks": 1, "file_key": "not_a_key"},
        files={"chunk": ("blob", io.BytesIO(b"abc"), "application/octet-stream")},
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/upload/chunk",
        params={"file_id": "../escape", "chunk_number": 0,
                "total_chunks": 1, "file_key": "pea_bfkt"},
        files={"chunk": ("blob", io.BytesIO(b"abc"), "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_bad_uploads_return_422_with_clear_message(client):
    resp = client.post("/api/upload",
                       files={"pea_bfkt": ("e.csv", io.BytesIO(b"h1,h2\n"), "text/csv")})
    assert resp.status_code == 422 and "no data rows" in resp.text

    resp = client.post("/api/upload",
                       files={"mea_tuc": ("e.csv", io.BytesIO(b"x\na,b\n1,2\n"), "text/csv")})
    assert resp.status_code == 422 and "Meter_No" in resp.text
