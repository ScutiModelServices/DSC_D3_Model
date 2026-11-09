"""Ollama decision adapter; blocking network calls run in engine worker threads."""

from __future__ import annotations

import json
import hashlib
from urllib.parse import quote
from http.client import HTTPException
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator

from engine import EngineError
from updates import Updates

DEFAULT_URL = "http://127.0.0.1:11434"


def _call(url: str, path: str, body: dict[str, Any] | None, timeout: float):
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 400:
            raise EngineError("Ollama refused the request (400)", 400) from None
        model = (body or {}).get("model", "the configured model")
        raise EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503) from None
    except OSError:
        model = (body or {}).get("model", "the configured model")
        raise EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503) from None


_READ_ERRORS = (OSError, HTTPException, ValueError, TypeError, KeyError, AttributeError)


def _unavailable(model: str) -> EngineError:
    return EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503)


def _json_call(url: str, path: str, body: dict[str, Any] | None, timeout: float, model: str) -> dict[str, Any]:
    """Read and validate a backend object without exposing response bodies or transport details."""
    try:
        with _call(url, path, body, timeout) as response:
            reply = json.load(response)
        if not isinstance(reply, dict) or reply.get("error"):
            raise _unavailable(model)
        return reply
    except _READ_ERRORS:
        raise _unavailable(model) from None


class OllamaLoaded:
    """Answer typed questions through Ollama's /v1/systemone (d3 decision models)."""

    def __init__(self, url: str, model: str, load_timeout: float = 300.0) -> None:
        """Check the model is installed (d3 refuses /api/generate; Ollama loads it on the first decision). An unpulled model is a 503, never a download."""
        self.url, self.model = url.rstrip("/"), model
        _json_call(self.url, "/api/show", {"model": model}, load_timeout, model)

    def decide(self, state: str, questions: dict[str, Any], max_time: float) -> dict[str, Any]:
        reply = _json_call(self.url, "/v1/systemone", {"model": self.model, "state": state, "questions": questions},
                           max_time, self.model)
        answers, usage = reply.get("answers"), reply.get("usage") or {}
        if not isinstance(answers, dict) or not isinstance(usage, dict):
            raise _unavailable(self.model)
        return {"answers": answers, "input_tokens": usage.get("input_tokens", 0)}


class OllamaUpdates(Updates):
    """Explicit downloads pull; metadata checks compare registry manifests without fetching blobs."""

    async def update_weights(self) -> dict[str, Any]:
        async with self._lock:
            try:
                self._unload_pending = True
                await self._download()
                await self._unload()
                self._unload_pending = False
                return await self._check()
            except Exception as error:
                self._status = {**self._status, "error": f"{type(error).__name__}: {error}"}
                raise


def _parallel() -> int:
    """Generations one replica runs at once: SERVICE_OLLAMA_PARALLEL (default 2), the OLLAMA_NUM_PARALLEL of your Ollama."""
    raw = os.environ.get("SERVICE_OLLAMA_PARALLEL", "").strip() or "2"
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError("SERVICE_OLLAMA_PARALLEL must be a whole number of 1 or more")
    return int(raw)

def engine_from_env(prefix: str, engine_class: type, default_model: str, **options: Any) -> Any:
    """Always build an Ollama engine; an explicitly empty URL is a configuration error."""
    url = os.environ.get(f"{prefix}OLLAMA_URL", DEFAULT_URL).strip().rstrip("/")
    if not url:
        raise ValueError(f"{prefix}OLLAMA_URL must not be empty; set an Ollama server URL")
    model = os.environ.get(f"{prefix}OLLAMA_MODEL", "").strip() or default_model

    def local_digest(_: str) -> str | None:
        reply = _json_call(url, "/api/tags", None, 15, model)
        try:
            digest = next((m["digest"] for m in reply["models"] if m["name"] in (model, f"{model}:latest")), None)
            return digest.removeprefix("sha256:").lower() if digest else None
        except _READ_ERRORS:
            raise _unavailable(model) from None

    def remote_digest(_: str) -> str:
        """Hash the registry manifest bytes, matching Ollama's installed digest."""
        name = model
        host = "registry.ollama.ai"
        if name.startswith(("hf.co/", "huggingface.co/")):
            host, name = name.split("/", 1)
        name, separator, tag = name.partition(":")
        if not separator:
            tag = "latest"
        if host == "registry.ollama.ai" and "/" not in name:
            name = "library/" + name
        request = urllib.request.Request(
            f"https://{host}/v2/{quote(name, safe='/')}/manifests/{quote(tag, safe='')}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2
                or not isinstance(manifest.get("config"), dict) or not isinstance(manifest.get("layers"), list)):
            raise EngineError("Registry returned an invalid model manifest", 503)
        return hashlib.sha256(raw).hexdigest()

    def pull(_: str) -> None:
        from progress import ollama_pull_progress

        with _call(url, "/api/pull", {"model": model, "stream": True}, 3600) as response:
            status = ollama_pull_progress(response, model)
        if status != "success":
            raise _unavailable(model)

    return engine_class(model, parallel=_parallel(), loader=lambda source: OllamaLoaded(url, source),
                        cached_sha=local_digest, remote_sha=remote_digest, download=pull, **options)
