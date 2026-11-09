"""Regression coverage for the service's mandatory external model backend."""

import http.client
import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from app import create_app
from engine import D3Engine
from ollama_backend import engine_from_env
from options import SETTINGS

BODY = {"state": "charged twice", "questions": {"team": {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Charges", "returns": "Refunds"}}}}
SYSTEMONE = {"model": "m", "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0}, "confidence": 1.0}},
             "usage": {"input_tokens": 3, "output_tokens": 0}}


def test_explicitly_empty_backend_url_is_an_error(monkeypatch):
    monkeypatch.setenv("SERVICE_OLLAMA_URL", "  ")
    with pytest.raises(ValueError, match="SERVICE_OLLAMA_URL must not be empty"):
        engine_from_env("SERVICE_", D3Engine, "m")


def test_local_backend_is_default_when_url_is_unset(monkeypatch):
    monkeypatch.delenv("SERVICE_OLLAMA_URL", raising=False)
    assert engine_from_env("SERVICE_", D3Engine, "m") is not None


def test_unreachable_backend_keeps_service_available_and_decide_is_503(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    monkeypatch.setenv("SERVICE_OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    with TestClient(create_app(warm=True)) as http:
        assert http.get("/health").json()["ready"] is False
        assert http.get("/admin/").status_code == 200
        reply = http.post("/v1/decide", json=BODY)
        assert reply.status_code == 503
        assert "press Download on the Updates card" in reply.json()["detail"]
        assert "127.0.0.1" not in reply.text


def test_settings_only_describe_supported_controls():
    assert {item.key for item in SETTINGS} == {"update_check_hours", "max_inflight", "cache_size", "ollama_model", "warm"}


def test_missing_model_returns_503_then_recovers_when_pulled(monkeypatch):
    monkeypatch.setenv("SERVICE_OLLAMA_URL", "http://ollama.invalid:11434")
    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    installed = False

    def request(req, **kwargs):
        if not installed:
            raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, io.BytesIO(b'{"error":"internal path"}'))
        return io.BytesIO(json.dumps({"done": True} if req.full_url.endswith("/api/show") else SYSTEMONE).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    with TestClient(create_app(warm=True)) as http:
        reply = http.post("/v1/decide", json=BODY)
        assert reply.status_code == 503 and "internal path" not in reply.text
        assert http.get("/health").json()["ready"] is False
        installed = True
        assert http.post("/v1/decide", json=BODY).json()["answers"]["team"]["choice"] == "billing"
        assert http.get("/health").json()["ready"] is True


def test_update_download_pulls_then_invalidates_replicas(monkeypatch, tmp_path):
    seen = []

    def request(req, **kwargs):
        seen.append(req.full_url.rsplit("/", 1)[-1])
        body = {"models": [{"name": "m", "digest": "same"}]} if req.full_url.endswith("/tags") else {"status": "success"}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    monkeypatch.setattr("urllib.request.urlopen", request)
    engine = engine_from_env("SERVICE_", D3Engine, "m", marker=tmp_path / "marker")
    with TestClient(create_app(engine, admin_password="admin")) as http:
        http.post("/admin/updates/check", headers={"X-Internal-Token": "admin"})
        reply = http.post("/admin/updates/weights", headers={"X-Internal-Token": "admin"})
        assert reply.status_code == 200
        assert "pull" in seen
        assert engine.generation == 1 and (tmp_path / "marker").exists()


@pytest.mark.parametrize("broken", ["timeout", "reset", "incomplete", "invalid-json", "invalid-body", "no-answers"])
def test_backend_read_failures_are_sanitized_503_and_clear_readiness(monkeypatch, broken):
    def request(req, **kwargs):
        if req.full_url.endswith("/show"):
            return io.BytesIO(b'{"done":true}')
        if broken == "invalid-json":
            return io.BytesIO(b"bad backend.internal/path")
        if broken == "invalid-body":
            return io.BytesIO(b"[]")
        if broken == "no-answers":
            return io.BytesIO(b'{"model":"m"}')

        class BrokenResponse(io.BytesIO):
            def read(self, *args):
                if broken == "timeout":
                    raise TimeoutError("backend.internal/path")
                if broken == "reset":
                    raise ConnectionResetError("backend.internal/path")
                raise http.client.IncompleteRead(b"backend.internal/path")

        return BrokenResponse()

    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    monkeypatch.setattr("urllib.request.urlopen", request)
    with TestClient(create_app()) as http_client:
        reply = http_client.post("/v1/decide", json=BODY)
        assert reply.status_code == 503
        assert "press Download on the Updates card" in reply.json()["detail"]
        assert "backend.internal/path" not in reply.text
        assert http_client.get("/health").json()["ready"] is False


def test_token_routes_refuse_restrictions_and_hide_legacy_scope_labels(tmp_path):
    from tokens import TokenStore

    store = TokenStore(tmp_path / "t.json")
    old = store.create("legacy", ["old-project"])
    with TestClient(create_app(admin_password="admin", tokens=store)) as http_client:
        headers = {"X-Internal-Token": "admin"}
        for projects in ([], ["app-a"]):
            assert http_client.post("/admin/tokens", headers=headers, json={"name": "new", "projects": projects}).status_code == 422
            assert http_client.put(f"/admin/tokens/{old['id']}", headers=headers, json={"projects": projects}).status_code == 422
        assert http_client.get("/admin/tokens", headers=headers).json()["tokens"][0]["projects"] is None
        assert http_client.post("/admin/tokens", headers=headers, json={"name": "new", "projects": None}).status_code == 200
