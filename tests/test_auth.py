"""Password-gate tests: default-deny coverage, login flow, token lifecycle,
rate limiting, and open mode when APP_PASSWORD is unset.
"""
import time

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.auth import LoginRateLimiter, issue_token, verify_token
from app.data_container import DataBillContainer
from app.main import app, STATE

PASSWORD = "correct-horse-battery"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    # fresh limiter so one test's failures never bleed into another
    monkeypatch.setattr(auth, "LOGIN_RATE_LIMITER", LoginRateLimiter())
    STATE["container"] = DataBillContainer()
    return TestClient(app)


@pytest.fixture()
def open_client(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    STATE["container"] = DataBillContainer()
    return TestClient(app)


def login_token(client: TestClient) -> str:
    resp = client.post("/api/auth/login", json={"password": PASSWORD})
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def test_everything_denied_without_token(client):
    for path in ("/api/upload/status", "/api/eda/summary", "/api/sites",
                 "/api/ml/drop-options", "/docs", "/openapi.json"):
        resp = client.get(path)
        assert resp.status_code == 401, f"{path} leaked: {resp.status_code}"

    assert client.post("/api/ml/build", json={}).status_code == 401
    assert client.post("/api/upload/finalize", params={"file_id": "x-1-y"}).status_code == 401


def test_public_paths_stay_open(client):
    assert client.get("/api/health").status_code == 200  # keep-warm ping
    assert client.get("/").status_code == 200


def test_login_and_authorized_request(client):
    token = login_token(client)
    resp = client.get("/api/upload/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["ready"] is False


def test_wrong_password_and_garbage_tokens_rejected(client):
    assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401

    for bad in ("Bearer garbage", "Bearer v1.123.deadbeef", "Basic abc", "Bearer "):
        resp = client.get("/api/upload/status", headers={"Authorization": bad})
        assert resp.status_code == 401, f"accepted: {bad!r}"


def test_expired_and_tampered_tokens_rejected():
    token, _ = issue_token(PASSWORD, ttl_seconds=-1)  # already expired
    assert verify_token(token, PASSWORD) is False

    token, _ = issue_token(PASSWORD)
    assert verify_token(token, PASSWORD) is True
    version, expiry, sig = token.split(".")
    extended = f"{version}.{int(expiry) + 999999}.{sig}"  # forge a later expiry
    assert verify_token(extended, PASSWORD) is False
    assert verify_token(token, "other-password") is False


def test_rotating_password_invalidates_tokens(client, monkeypatch):
    token = login_token(client)
    monkeypatch.setenv("APP_PASSWORD", "brand-new-password")
    resp = client.get("/api/upload/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_login_rate_limited_after_repeated_failures(client):
    for _ in range(auth.MAX_LOGIN_ATTEMPTS):
        assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401
    # further attempts blocked, even with the right password
    assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 429
    assert client.post("/api/auth/login", json={"password": PASSWORD}).status_code == 429


def test_rate_limiter_window_slides():
    limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)
    now = time.time()
    limiter._attempts["1.2.3.4"] = __import__("collections").deque([now - 120, now - 90])
    assert limiter.allow("1.2.3.4") is True  # old failures aged out


def test_open_mode_without_app_password(open_client):
    assert open_client.get("/api/upload/status").status_code == 200
    resp = open_client.post("/api/auth/login", json={"password": "anything"})
    assert resp.status_code == 503
