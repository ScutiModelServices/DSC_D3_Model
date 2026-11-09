"""Cached Ollama loading and metadata-only registry comparison."""
import asyncio
import hashlib
import io
import json
import urllib.error

from fastapi.testclient import TestClient
from app import create_app
from ollama_backend import engine_from_env
from engine import D3Engine

BODY = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it x?", "criteria": {"true": "yes", "false": "no"}}}}


def test_registry_manifest_matches_local_digest_without_pulling(monkeypatch):
    manifest = json.dumps({"schemaVersion": 2, "config": {}, "layers": []}).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    calls = []

    def request(req, **kwargs):
        calls.append(req.full_url)
        assert "/api/pull" not in req.full_url
        if "/manifests/" in req.full_url:
            assert req.full_url == "https://registry.ollama.ai/v2/library/d3/manifests/2b"
            return io.BytesIO(manifest)
        return io.BytesIO(json.dumps({"models": [{"name": "d3:2b", "digest": "sha256:" + digest}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "d3:2b")
    engine = engine_from_env("SERVICE_", D3Engine, "unused")
    assert asyncio.run(engine.cached_sha()) == digest
    assert asyncio.run(engine.remote_sha()) == digest
    with TestClient(create_app(engine, admin_password="admin")) as http:
        status = http.post("/admin/updates/check", headers={"X-Internal-Token": "admin"}).json()
        assert status["weights"][0]["update"] is False
    assert all("blobs" not in url for url in calls)


def test_missing_model_never_pulls_and_admin_download_recovers(monkeypatch):
    installed = False
    paths = []

    def request(req, **kwargs):
        nonlocal installed
        paths.append(req.full_url)
        if req.full_url.endswith("/api/pull"):
            installed = True
            return io.BytesIO(b'{"status":"success"}')
        if req.full_url.endswith("/api/show") and not installed:
            raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)
        return io.BytesIO(b'{"done":true,"answers":{"q":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1}}')

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    with TestClient(create_app(warm=True, admin_password="admin")) as http:
        headers = {"X-Internal-Token": "admin"}
        assert http.get("/health").json()["ready"] is False
        public = {"X-Internal-Token": http.post("/admin/tokens", headers=headers, json={"name": "test"}).json()["token"]}
        failed = http.post("/v1/decide", json=BODY, headers=public)
        assert failed.status_code == 503
        assert "Updates" in failed.json()["detail"] and "Download" in failed.json()["detail"]
        assert not any(path.endswith("/pull") for path in paths)
        assert http.post("/admin/updates/weights", headers=headers).status_code == 200
        assert http.post("/v1/decide", json=BODY, headers=public).status_code == 200
        assert http.get("/health").json()["ready"] is True
