"""End-to-end smoke test: upload synthetic files, hit every endpoint.

Run with: pytest tests/
"""
import io
import sys
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
    assert body["rows_total"] == 0
    assert body["missing_files"] == []

    for path in ("summary", "bill-range", "duplicates", "common-sites",
                 "site-types", "missing-consequence", "maintenance-sites",
                 "meter-patterns", "error-rates"):
        resp = client.get(f"/api/eda/{path}")
        assert resp.status_code == 200, f"{path}: {resp.text[:300]}"
        resp.json()  # must be valid JSON (no NaN)

    assert client.get("/api/upload/status").json()["rows_total"] > 0


def test_meter_patterns_classification(client):
    upload_all(client)
    resp = client.get("/api/eda/meter-patterns")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["window"] == 3
    assert body["unique_meters"] > 0
    assert set(body["unique_meters_per_provider"]) == {"PEA", "MEA"}
    # every uploaded meter is listed, normal ones included
    assert body["total_records"] == len(body["records"]) == body["unique_meters"]

    by_meter = {r["meter_no"]: r for r in body["records"]}
    # MEA meter 202 bills 50 THB then 0 -> intermittent "gap"
    assert by_meter["202"]["pattern"] == "gap"
    assert [m["bill_amount"] for m in by_meter["202"]["monthly"]] == [50.0, 0.0]
    # meters billed every month are "normal", small amounts included
    assert by_meter["111"]["pattern"] == "normal"
    assert by_meter["112"]["pattern"] == "normal"
    assert by_meter["201"]["pattern"] == "normal"
    assert body["counts"]["normal"] >= 3
    assert body["counts"]["gap"] >= 1
    assert "maintenance" not in body["counts"]

    # window param is validated
    assert client.get("/api/eda/meter-patterns", params={"window": 0}).status_code == 422


def test_meter_patterns_paging_filter_and_export(client):
    upload_all(client)
    full = client.get("/api/eda/meter-patterns").json()

    # limit/offset page through the same sorted rows
    page = client.get("/api/eda/meter-patterns", params={"limit": 1}).json()
    assert len(page["records"]) == 1
    assert page["total_records"] == full["total_records"]
    assert page["records"][0] == full["records"][0]
    # rows are sorted most severe first -> the gap meter leads here
    assert page["records"][0]["pattern"] == "gap"
    page2 = client.get("/api/eda/meter-patterns",
                       params={"limit": 1, "offset": 1}).json()
    assert page2["records"][0] == full["records"][1]

    # pattern filter narrows total_records but keeps global counts
    normal = client.get("/api/eda/meter-patterns", params={"pattern": "normal"}).json()
    assert normal["total_records"] == full["counts"]["normal"]
    assert all(r["pattern"] == "normal" for r in normal["records"])
    assert normal["counts"] == full["counts"]
    assert client.get("/api/eda/meter-patterns",
                      params={"pattern": "bogus"}).status_code == 422
    # the maintenance pattern was removed from this datasheet
    assert client.get("/api/eda/meter-patterns",
                      params={"pattern": "maintenance"}).status_code == 422

    # CSV export streams the datasheet with a BOM + header + every meter row
    resp = client.get("/api/eda/meter-patterns/export")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    assert "bill_patterns_all.csv" in resp.headers["content-disposition"]
    text = resp.text
    assert text.startswith("﻿")
    lines = [ln for ln in text.lstrip("﻿").split("\r\n") if ln]
    assert lines[0].startswith("Meter No,Site ID,Provider,Company,Type,Pattern")
    assert len(lines) == 1 + full["total_records"]
    assert any(ln.startswith("112,") for ln in lines)

    filtered = client.get("/api/eda/meter-patterns/export",
                          params={"pattern": "gap"}).text
    gap_lines = [ln for ln in filtered.lstrip("﻿").split("\r\n") if ln]
    assert len(gap_lines) == 1 + full["counts"]["gap"]


def test_maintenance_sites_include_meter_no(client):
    upload_all(client)
    sites = client.get("/api/eda/maintenance-sites").json()[
        "maintenance_sites_last_6_months"]
    assert sites, "synthetic data must produce at least one maintenance site"
    assert all("meter_no" in s for s in sites)


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
    assert body["rows_total"] == 0
    assert body["ready"] is False
    assert "pea_tuc" in body["missing_files"]

    # The assembled data parsed successfully, so partial dashboard endpoints
    # should be available even while the full five-file dataset is incomplete.
    sites = client.get("/api/sites")
    assert sites.status_code == 200, sites.text
    assert "CBR4017" in sites.json()["site_ids"]
    assert client.get("/api/upload/status").json()["rows_total"] > 0


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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hangs on Windows (TestClient + ThreadPoolExecutor deadlock, pre-existing); "
           "covered by the Linux CI run.",
)
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


def pea_csv_ml(site_prefix: str, n_sites: int = 8, spike_site_idx: int = 0) -> bytes:
    """12 BE months (256901..256912 -> 202601..202612) of non-zero usage so
    sites clear the >=7-consecutive-clean-months bar the ML pipeline needs.
    Site `spike_site_idx` gets a 10x kWh jump in 202610 (a test-window
    target month) so the quantile band reliably flags at least one anomaly.
    """
    months = [f"2569{m:02d}" for m in range(1, 13)]
    n_cols = 3 + 1 + len(months) + 1 + len(months)
    banner = ",".join(f"h{i}" for i in range(1, n_cols + 1))
    header = "Site_ID,Meter_No.,Province,avg," + ",".join(months) + ",avg," + ",".join(months)
    lines = [banner, header]
    for i in range(n_sites):
        base = 50 + 10 * i
        units = [base + (m % 3) for m in range(1, 13)]
        if i == spike_site_idx:
            units[9] = base * 10  # BE 256910 -> 202610
        amounts = [u * 4 for u in units]
        row = [f"{site_prefix}5{i:03d}", str(500 + i), "BKK", ""]
        row += [str(a) for a in amounts]
        row += [""]
        row += [str(u) for u in units]
        lines.append(",".join(row))
    return "\n".join(lines).encode()


def upload_all_ml(client: TestClient):
    files = {
        "pea_bfkt": ("pea_bfkt.csv", io.BytesIO(pea_csv_ml("CBR")), "text/csv"),
        "pea_tuc": ("pea_tuc.csv", io.BytesIO(pea_csv_ml("TUC")), "text/csv"),
        "mea_bfkt": ("mea_bfkt.csv", io.BytesIO(mea_csv("MBF")), "text/csv"),
        "mea_tuc": ("mea_tuc.csv", io.BytesIO(mea_csv("MTU")), "text/csv"),
        "mea_tmv": ("mea_tmv.csv", io.BytesIO(mea_csv("MTM")), "text/csv"),
    }
    return client.post("/api/upload", files=files)


def test_ml_pipeline_end_to_end(client):
    up = upload_all_ml(client)
    assert up.status_code == 200, up.text
    assert up.json()["ready"] is True

    opts = client.get("/api/ml/drop-options").json()["options"]
    assert {o["value"] for o in opts} == {
        "duplicate_site", "common_site", "shutdown_site", "maintenance_site"}

    # ML endpoints that need a model must 409 before any build
    assert client.get("/api/ml/abnormal").status_code == 409
    assert client.post("/api/ml/classify", json={}).status_code == 409
    assert client.get("/api/ml/examples", params={"anom_type": "spike_up"}).status_code == 409

    resp = client.post("/api/ml/preview", json={
        "drop_options": {}, "start_month": 202601, "end_month": 202611})
    assert resp.status_code == 200, resp.text
    prev = resp.json()
    assert prev["missing"]["n_months"] > 0
    assert prev["drop_report"]["sites_remaining"] > 0

    resp = client.post("/api/ml/build", json={
        "drop_options": {},
        "train_start": 202601, "train_end": 202608,
        "test_start": 202609, "test_end": 202610,
    })
    assert resp.status_code == 200, resp.text
    build = resp.json()
    assert build["n_train_rows"] > 0 and build["n_test_rows"] > 0
    assert 0.0 <= build["metrics"]["train"]["coverage"] <= 1.0
    assert build["metrics"]["n_flagged_test"] >= 1  # the 10x spike must escape the band

    resp = client.get("/api/ml/abnormal")
    assert resp.status_code == 200, resp.text
    ab = resp.json()
    assert ab["count"] == len(ab["rows"]) >= 1
    for row in ab["rows"]:
        assert set(row) >= {"site_id", "anom_month", "kwh",
                            "q05", "q50", "q95", "quantile_severity"}

    resp = client.post("/api/ml/classify", json={})
    assert resp.status_code == 200, resp.text
    cls = resp.json()
    assert set(cls["surfaced_types"]) == {"spike_up", "step_up"}
    for row in cls["rows"]:
        assert set(row) >= {"site_id", "anom_month", "anom_val",
                            "anom_type", "quantile_severity"}

    resp = client.get("/api/ml/examples", params={"anom_type": "spike_up", "limit": 3})
    assert resp.status_code == 200, resp.text
    ex = resp.json()
    assert ex["count"] == len(ex["images"])

    resp = client.get("/api/ml/plots/download")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"


def test_bad_uploads_return_422_with_clear_message(client):
    resp = client.post("/api/upload",
                       files={"pea_bfkt": ("e.csv", io.BytesIO(b"h1,h2\n"), "text/csv")})
    assert resp.status_code == 422 and "no data rows" in resp.text

    resp = client.post("/api/upload",
                       files={"mea_tuc": ("e.csv", io.BytesIO(b"x\na,b\n1,2\n"), "text/csv")})
    assert resp.status_code == 422 and "Meter_No" in resp.text
