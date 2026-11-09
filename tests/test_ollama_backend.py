import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from engine import EngineError, D3Engine
from ollama_backend import OllamaLoaded, engine_from_env

seen = []
ANSWERS = {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "returns": 0.1}, "confidence": 0.8}}
QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "returns": "Refunds"}}}
reply_body = {"model": "m", "answers": ANSWERS, "usage": {"input_tokens": 308, "output_tokens": 0}}


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_json({"models": [{"name": "m", "digest": "abc"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        seen.append((self.path, body))
        self.send_json({"done": True} if self.path == "/api/show" else reply_body)

    def send_json(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def url(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen.clear()
    monkeypatch.setitem(reply_body, "answers", ANSWERS)
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_decide_posts_to_systemone_and_maps_the_reply(url):
    reply = OllamaLoaded(url, "m").decide("charged twice", QUESTIONS, 5)
    assert reply == {"answers": ANSWERS, "input_tokens": 308}
    path, body = seen[-1]
    assert path == "/v1/systemone" and body == {"model": "m", "state": "charged twice", "questions": QUESTIONS}


def test_a_reply_without_answers_is_a_503(url, monkeypatch):
    loaded = OllamaLoaded(url, "m")
    monkeypatch.setitem(reply_body, "answers", "not an object")
    with pytest.raises(EngineError) as error:
        loaded.decide("x", QUESTIONS, 5)
    assert error.value.status == 503


def test_an_unreachable_server_is_an_engine_error():
    with pytest.raises(EngineError):
        OllamaLoaded("http://127.0.0.1:1", "m", load_timeout=1)


def test_engine_from_env_uses_local_default_and_serves_through_explicit_url(url, monkeypatch):
    monkeypatch.delenv("SERVICE_OLLAMA_URL", raising=False)
    assert engine_from_env("SERVICE_", D3Engine, "x") is not None
    monkeypatch.setenv("SERVICE_OLLAMA_URL", url)
    monkeypatch.setenv("SERVICE_OLLAMA_MODEL", "m")
    engine = engine_from_env("SERVICE_", D3Engine, "x")
    reply = asyncio.run(engine.decide("charged twice", QUESTIONS))
    assert reply["answers"] == ANSWERS and reply["model"] == "m"
    assert asyncio.run(engine.cached_sha()) == "abc"


def test_parallel_comes_from_service_ollama_parallel(monkeypatch):
    monkeypatch.setenv("SERVICE_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("SERVICE_OLLAMA_PARALLEL", raising=False)
    assert engine_from_env("SERVICE_", D3Engine, "m").parallel == 2
    monkeypatch.setenv("SERVICE_OLLAMA_PARALLEL", "4")
    assert engine_from_env("SERVICE_", D3Engine, "m").parallel == 4
    monkeypatch.setenv("SERVICE_OLLAMA_PARALLEL", "0")
    with pytest.raises(ValueError):
        engine_from_env("SERVICE_", D3Engine, "m")
