import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import settings
from app import create_app
from conftest import make_engine
from engine import EngineError
from options import SETTINGS
from tokens import TokenStore

BODY = {"state": "charged twice", "questions": {"team": {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Charges", "returns": "Refunds"}}}}


def client(engine=None, admin_password=None, **options):
    return TestClient(create_app(engine or make_engine(), admin_password, **options))


def test_health_reports_readiness():
    c = client()
    assert c.get("/health").json() == {"status": "ok", "model": "m", "ready": False, "num_thread": None}
    c.post("/v1/decide", json=BODY)
    assert c.get("/health").json()["ready"] is True


def test_each_project_gets_its_own_revocable_token(tmp_path):
    c = client(admin_password="admin", tokens=TokenStore(tmp_path / "t.json"))
    admin = {"X-Internal-Token": "admin"}
    assert c.post("/v1/decide", json=BODY).status_code == 401
    made = c.post("/admin/tokens", json={"name": "Garrison"}, headers=admin).json()
    assert made["token"].startswith("d3_")
    use = {"X-Internal-Token": made["token"]}
    assert c.post("/v1/decide", json=BODY, headers=use).status_code == 200
    assert c.delete(f"/admin/tokens/{made['id']}", headers=admin).status_code == 200
    assert c.post("/v1/decide", json=BODY, headers=use).status_code == 401
    assert c.get("/health").status_code == 200  # /health needs no token


def test_token_scope_lists_are_refused_but_null_is_compatible(tmp_path):
    c = client(admin_password="admin", tokens=TokenStore(tmp_path / "t.json"))
    admin = {"X-Internal-Token": "admin"}
    assert c.post("/admin/tokens", json={"name": "p", "projects": ["app-a"]}, headers=admin).status_code == 422
    made = c.post("/admin/tokens", json={"name": "p", "projects": None}, headers=admin).json()
    use = {"X-Internal-Token": made["token"]}
    assert c.post("/v1/decide", json=BODY, headers=use).status_code == 200
    assert c.post("/v1/decide", json={**BODY, "project": "app-a"}, headers=use).status_code == 422


def test_rate_limit_edit_stats_and_metrics(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    c = client(admin_password="admin", tokens=store)
    admin = {"X-Internal-Token": "admin"}
    made = c.post("/admin/tokens", json={"name": "p", "rate": 1}, headers=admin).json()
    use = {"X-Internal-Token": made["token"]}
    assert c.post("/v1/decide", json=BODY, headers=use).status_code == 200
    limited = c.post("/v1/decide", json=BODY, headers=use)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
    assert c.put(f"/admin/tokens/{made['id']}", json={"projects": None, "rate": None}, headers=admin).status_code == 200
    assert c.post("/v1/decide", json=BODY, headers=use).status_code == 200  # the limit is gone
    assert c.put("/admin/tokens/nope", json={}, headers=admin).status_code == 404
    assert store.list()[0]["timed"] >= 1
    assert c.get("/metrics").status_code == 401
    assert 'd3_requests_total{token="p"}' in c.get("/metrics", headers=admin).text
    assert 'd3_request_seconds_bucket{le="+Inf"}' in c.get("/metrics", headers=admin).text


def test_requests_over_the_in_flight_limit_get_429():
    async def main():
        engine = make_engine()
        release = asyncio.Event()
        real = engine.decide

        async def slow(*args, **kwargs):
            await release.wait()
            return await real(*args, **kwargs)

        engine.decide = slow
        app = create_app(engine, max_inflight=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as http:
            first = asyncio.create_task(http.post("/v1/decide", json=BODY))
            await asyncio.sleep(0.2)
            second = await http.post("/v1/decide", json=BODY)
            release.set()
            return second, await first

    second, first = asyncio.run(main())
    assert second.status_code == 429 and second.headers["Retry-After"] == "5"
    assert first.status_code == 200


def test_workers_are_read_only_and_admin_only(monkeypatch):
    monkeypatch.setenv("SERVICE_REPLICAS", "3")
    c = client(admin_password="admin")
    assert c.get("/admin/workers").status_code == 401
    assert c.get("/admin/workers", headers={"X-Internal-Token": "admin"}).json() == {"replicas": 3, "ollama_parallel": 1}


def test_num_thread_is_reported_and_set_by_the_admin(tmp_path):
    engine = make_engine()
    c = client(engine, "admin", tokens=TokenStore(tmp_path / "t.json"))
    admin = {"X-Internal-Token": "admin"}
    assert c.put("/admin/threads", json={"num_thread": 0}, headers=admin).status_code == 422
    assert c.put("/admin/threads", json={"num_thread": 3}, headers=admin).json() == {"num_thread": 3}
    assert c.get("/health").json()["num_thread"] == 3
    assert settings.load(tmp_path / "settings.json")["num_thread"] == 3


def test_settings_are_admin_only_checked_and_apply_live(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    c = client(admin_password="admin", tokens=store, config=config)
    admin = {"X-Internal-Token": "admin"}
    assert c.get("/admin/settings").status_code == 401
    assert c.put("/admin/settings", json={"values": {"max_inflight": -1}}, headers=admin).status_code == 422
    assert c.put("/admin/settings", json={"values": {"max_inflight": 7}}, headers=admin).status_code == 200
    saved = {item["key"]: item for item in c.get("/admin/settings", headers=admin).json()["settings"]}
    assert saved["max_inflight"]["value"] == 7
    assert client(admin_password="admin").put("/admin/settings", json={"values": {}}, headers=admin).status_code == 409


def test_what_another_replica_saved_reaches_this_one(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    engine = make_engine()
    with TestClient(create_app(engine, "admin", tokens=store, config=config)):
        settings.save(store.settings_path, num_thread=5)
        for _ in range(300):
            if engine.num_thread == 5:
                break
            time.sleep(0.02)
    assert engine.num_thread == 5


def test_startup_warms_the_model():
    engine = make_engine()
    with TestClient(create_app(engine, warm=True)) as c:
        for _ in range(100):
            if c.get("/health").json()["ready"]:
                break
            time.sleep(0.02)
        assert engine.loads == ["m"] and c.get("/health").json()["ready"] is True


def test_health_answers_while_the_model_is_still_loading():
    import threading
    release = threading.Event()

    def slow(source):
        release.wait(5)
        return make_engine()._loader(source)

    with TestClient(create_app(make_engine(loader=slow), warm=True)) as c:
        assert c.get("/health").json()["ready"] is False
        release.set()


def test_warmup_failure_is_logged_without_stopping_service(caplog):
    def failing(source):
        raise RuntimeError("no weights")

    with TestClient(create_app(make_engine(loader=failing), warm=True)) as c:
        assert c.get("/health").status_code == 200
        for _ in range(100):
            if "warm-up failed" in caplog.text:
                break
            time.sleep(0.02)
    assert "warm-up failed" in caplog.text


def test_warmup_is_optional():
    engine = make_engine()
    with TestClient(create_app(engine)):
        assert engine.loads == []


def test_main_builds_the_engine_from_settings(monkeypatch, tmp_path):
    import app as module

    monkeypatch.setenv("SERVICE_TOKENS_FILE", str(tmp_path / "tokens.json"))
    monkeypatch.setenv("SERVICE_NUM_THREAD", "2")
    monkeypatch.setenv("SERVICE_PORT", "9999")
    monkeypatch.setattr(module.D3Engine, "num_thread", property(lambda self: None, lambda self, count: seen.append(count)))
    seen, started = [], {}
    monkeypatch.setattr("uvicorn.Server.run", lambda self: started.update(app=self.config.app, host=self.config.host, port=self.config.port))
    module.main()
    assert started["port"] == 9999 and started["host"] == "127.0.0.1" and seen == [2]


def events(reply):
    return [json.loads(text) for text in reply.text.splitlines()]


def test_cache_size_is_a_setting_that_applies_live(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    c = client(make_engine(), admin_password="admin", tokens=store, config=config)
    admin = {"X-Internal-Token": "admin"}
    use = {"X-Internal-Token": c.post("/admin/tokens", json={"name": "t"}, headers=admin).json()["token"]}
    assert [c.post("/v1/decide", json=BODY, headers=use).headers["X-Cache"] for _ in range(2)] == ["miss", "hit"]
    assert c.put("/admin/settings", json={"values": {"cache_size": 0}}, headers=admin).status_code == 200
    assert c.post("/v1/decide", json=BODY, headers=use).headers["X-Cache"] == "miss"


def test_decide_returns_answers_usage_and_model():
    engine = make_engine()
    reply = client(engine).post("/v1/decide", json=BODY)
    body = reply.json()
    assert reply.status_code == 200 and body["answers"]["team"]["choice"] == "billing"
    assert body["model"] == "m" and body["usage"] == {"input_tokens": 7} and "duration_ms" in body
    assert engine._base.calls[0]["state"] == "charged twice"
    assert engine._base.calls[0]["questions"]["team"]["criteria"] == BODY["questions"]["team"]["criteria"]


def test_request_is_validated():
    c = client()
    bad = lambda q: c.post("/v1/decide", json={"state": "x", "questions": q}).status_code
    assert bad({}) == 422
    assert bad({"a": {"type": "choice", "instructions": "q", "criteria": {}}}) == 422
    assert bad({"a": {"type": "noul", "instructions": "q", "criteria": {"true": "t"}}}) == 422
    assert bad({"a": {"type": "score", "instructions": "q", "criteria": ["only one"]}}) == 422
    assert bad({"a": {"type": "video", "instructions": "q", "criteria": {"x": "y"}}}) == 422
    assert c.post("/v1/decide", json={**BODY, "images": ["x"]}).status_code == 422
    assert c.post("/v1/decide", json={"state": "x"}).status_code == 422


def test_engine_errors_map_to_http_statuses():
    for fail, status in ((EngineError("too long", 400), 422), (EngineError("down", 503), 503), (RuntimeError("boom"), 500)):
        assert client(make_engine(fail=fail)).post("/v1/decide", json=BODY).status_code == status


def test_the_same_request_is_answered_from_the_cache_until_weights_change():
    engine = make_engine()
    c = client(engine, cache_size=8)
    assert [c.post("/v1/decide", json=BODY).headers["X-Cache"] for _ in range(2)] == ["miss", "hit"]
    assert len(engine._base.calls) == 1
    asyncio.run(engine.unload())
    assert c.post("/v1/decide", json=BODY).headers["X-Cache"] == "miss"
    off = client(make_engine(), cache_size=0)
    assert [off.post("/v1/decide", json=BODY).headers["X-Cache"] for _ in range(2)] == ["miss", "miss"]
