"""End-to-end smoke test: upload synthetic files, hit every endpoint.

Run with: pytest tests/
"""
import io

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


def test_bad_uploads_return_422_with_clear_message(client):
    resp = client.post("/api/upload",
                       files={"pea_bfkt": ("e.csv", io.BytesIO(b"h1,h2\n"), "text/csv")})
    assert resp.status_code == 422 and "no data rows" in resp.text

    resp = client.post("/api/upload",
                       files={"mea_tuc": ("e.csv", io.BytesIO(b"x\na,b\n1,2\n"), "text/csv")})
    assert resp.status_code == 422 and "Meter_No" in resp.text
