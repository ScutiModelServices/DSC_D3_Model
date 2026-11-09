"""HTTP API for the d3 decision model through Ollama, with shared admin, tokens and cache."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, model_validator

import strategies
from answers import check_questions
import hardware
import variants
from cache import AnswerCache
from engine import DEFAULT_MODEL, D3Engine, EngineError, default_marker
from ollama_backend import OllamaUpdates, engine_from_env
from options import SETTINGS
import settings
from tokens import LatencyHistogram, TokenStore
from updates import Updates

_log = logging.getLogger(__name__)

class Question(BaseModel):
    """One typed question. `choice`: criteria maps option -> meaning. `noul`: criteria has `true` and `false`.
    `score`: criteria lists 2-10 level descriptions, lowest first."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=10_000)
    criteria: dict[str, str] | list[str]

    @model_validator(mode="after")
    def _criteria_fit_type(self) -> "Question":
        if self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score criteria must be a list of 2 to 10 levels")
        elif not isinstance(self.criteria, dict) or not self.criteria:
            raise ValueError(f"{self.type} criteria must be a non-empty object")
        elif self.type == "noul" and set(self.criteria) != {"true", "false"}:
            raise ValueError("noul criteria needs exactly the keys true and false")
        return self


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # images, video and unknown fields are refused, not ignored

    state: str = Field(max_length=100_000)
    questions: dict[str, Question] = Field(min_length=1, max_length=16)


_Thresholds = dict[Literal["2", "3+"], dict[Literal["d3"], float]]


class AskRequest(BaseModel):
    """Laya's `/v1/ask` request without what d3 cannot do: `project`, `max_len` and `pair_ratio` are 422."""
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["d3-only", "d3-narrowing", "auto"] = strategies.DEFAULT_STRATEGY
    text: str = Field(min_length=1, max_length=100_000)
    questions: dict[str, Any]
    min_confidence: float | None = Field(None, gt=0, le=1)
    thresholds: _Thresholds = Field(default_factory=dict)
    prune_ratio: float = Field(strategies.PRUNE_RATIO, ge=0, lt=1)
    trace: bool = False
    stream: bool = False  # true: each question's answer arrives as a JSON line when it is settled


class ValidateRequest(BaseModel):
    questions: Any


class CombineRequest(BaseModel):
    voters: dict[str, dict[str, Any]] = Field(min_length=1, max_length=4)
    options: list[str] | None = Field(None, min_length=2, max_length=10)
    prune_ratio: float = Field(strategies.PRUNE_RATIO, ge=0, lt=1)
    pair_ratio: float = Field(strategies.PAIR_RATIO, gt=0, le=1)


class SettingsUpdate(BaseModel):
    values: dict[str, Any]  # None resets an option to its environment variable or default


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    projects: None = None  # null-only compatibility with the shared token editor
    rate: int | None = Field(None, ge=1, le=100_000)  # calls per minute
    expires_days: int | None = Field(None, ge=1, le=3650)


class TokenScope(BaseModel):
    projects: None = None  # null-only compatibility with the shared token editor
    rate: int | None = Field(None, ge=1, le=100_000)


class Threads(BaseModel):
    num_thread: int = Field(ge=1, le=256)


class VariantChoice(BaseModel):
    id: str = Field(min_length=1, max_length=64)


def create_app(client: D3Engine | None = None, admin_password: str | None = None,
               tokens: TokenStore | None = None, warm: bool = False,
               updates: Updates | None = None, check_hours: float = 0, max_inflight: int | None = None,
               cache_size: int | None = None, config: settings.Config | None = None) -> FastAPI:
    """`admin_password` (SERVICE_ADMIN_PASSWORD) unlocks /admin/, where each project gets a token of its own; when it
    is set, every call except /health must send one of those tokens in the X-Internal-Token header. Unset = no
    auth at all (local use only). `tokens` holds the project tokens (SERVICE_TOKENS_FILE). `cache_size` > 0 keeps that
    many answers per replica (SERVICE_CACHE_SIZE). `config` (the admin page's settings) replaces `warm`, `check_hours`,
    `max_inflight` and `cache_size` and keeps them live; without it they stay fixed."""
    tokens = tokens or TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    client = client or engine_from_env("SERVICE_", D3Engine, DEFAULT_MODEL, settings=config, marker=default_marker())
    inflight = 0
    cache = AnswerCache()

    def cache_limit() -> int:
        """Answers this replica keeps; 0 = off."""
        return config.get("cache_size") if config else (
            cache_size if cache_size is not None else int(os.environ.get("SERVICE_CACHE_SIZE") or 0))

    def inflight_limit() -> int:
        """Calls this replica lets wait for the model at once; more get 429. 0 = 4."""
        value = config.get("max_inflight") if config else (
            max_inflight if max_inflight is not None else int(os.environ.get("SERVICE_MAX_INFLIGHT") or 0))
        return value or 4 * client.parallel

    updates = updates or OllamaUpdates(client.model, cached_sha=lambda: client.cached_sha(), remote_sha=lambda: client.remote_sha(),
                                 download=lambda: client.pull(), unload=lambda: client.unload(),
                                 description="Checks the downloaded Ollama digest against the online registry manifest without downloading weights. Only Download pulls model files.",
                                 reload_note="Weights downloaded. Every replica drops the old copy within a few seconds and loads the new weights on its next request.")

    async def check_forever() -> None:
        """Checks every `update_check_hours` (0 = off). The value is read again every second, so the admin page can change it."""
        last = None
        while True:
            wait = (config.get("update_check_hours") if config else check_hours) * 3600
            now = time.monotonic()
            if wait > 0 and (last is None or now - last >= wait):
                last = now
                try:
                    status = await updates.check()
                    if status["error"]:
                        _log.warning("Update check failed: %s", status["error"])
                except Exception:
                    _log.exception("Update check failed")
            await asyncio.sleep(min(wait, 1.0) if wait > 0 else 1.0)

    async def sync_forever() -> None:
        """Applies saved admin values and drops models it replaced with new weights."""
        while True:
            saved = await asyncio.to_thread(settings.load, tokens.settings_path)
            if saved.get("num_thread") and saved["num_thread"] != client.num_thread:
                client.num_thread = int(saved["num_thread"])
            await asyncio.to_thread(client.sync)
            # A dropped model (download, threads change, sync) leaves /health "Starting" until a request loads it: reload once per drop.
            if config.get("warm") and not client.ready and client.generation != warmed:
                await warm_up()
            await asyncio.sleep(1)

    warmed = 0  # generation of the last warm-up attempt

    async def warm_up() -> None:
        nonlocal warmed
        warmed = client.generation
        try:
            await client.warm()
        except Exception:
            _log.exception("Model warm-up failed; the first request will try again")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        background: list[asyncio.Task] = []
        try:
            if config.get("warm") if config else warm:
                # In the background: only downloaded weights load; /health must answer meanwhile.
                background.append(asyncio.create_task(warm_up()))
            if config or check_hours > 0:
                background.append(asyncio.create_task(check_forever()))
            if config:
                background.append(asyncio.create_task(sync_forever()))
            yield
        finally:
            for task in background:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await client.close()

    def authorize(request: Request, x_internal_token: str = Header("")) -> None:
        """Authenticate and rate-limit public tokens. Legacy scope lists have no effect."""
        if not admin_password:
            return None
        row = tokens.lookup(x_internal_token)
        if row is None:
            raise HTTPException(401, "Invalid token")
        wait = tokens.throttle(row)
        if wait:
            raise HTTPException(429, "Too many requests for this token", headers={"Retry-After": str(wait)})
        request.state.token_id = row["id"]

    def admin(x_internal_token: str = Header("")) -> None:
        if not admin_password or not hmac.compare_digest(x_internal_token, admin_password):
            raise HTTPException(401, "Admin password required (set SERVICE_ADMIN_PASSWORD)")

    app = FastAPI(title="d3 decision service", lifespan=lifespan)

    async def slot():
        nonlocal inflight
        limit = inflight_limit()
        if limit and inflight >= limit:
            raise HTTPException(429, "d3 is busy: too many requests waiting", headers={"Retry-After": "5"})
        inflight += 1
        try:
            yield
        finally:
            inflight -= 1

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": client.model, "ready": client.ready,
                "num_thread": client.num_thread}

    def cache_key(body: DecideRequest) -> str:
        """Include the model generation so pulls invalidate old answers."""
        part = [client.generation, body.model_dump(mode="json")]
        return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()

    def line(event: dict[str, Any]) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    @app.post("/v1/decide")
    async def decide(body: DecideRequest, response: Response, _auth: None = Depends(authorize),
                     _slot: None = Depends(slot)) -> Any:
        """Scores are deterministic, so every request can be kept (`X-Cache: hit|miss`)."""
        started = time.perf_counter()
        size = cache_limit()
        key = cache_key(body) if size else None
        kept = cache.get(key, size) if key else None
        if kept is not None:
            response.headers["X-Cache"] = "hit"
            return {**kept, "duration_ms": round((time.perf_counter() - started) * 1000)}
        try:
            reply = await client.decide(body.state, {k: v.model_dump(mode="json") for k, v in body.questions.items()})
        except EngineError as error:
            raise HTTPException(422 if error.status == 400 else error.status or 500, str(error)) from None
        except Exception:
            _log.exception("d3 decide failed")
            raise HTTPException(500, "d3 could not answer") from None
        result = {"answers": reply["answers"], "model": reply["model"], "usage": {"input_tokens": reply["input_tokens"]},
                  "duration_ms": round((time.perf_counter() - started) * 1000)}
        if key:
            cache.put(key, result, size)
        response.headers["X-Cache"] = "miss"
        return result

    @app.get("/v1/strategies", dependencies=[Depends(authorize)])
    def list_strategies() -> dict[str, Any]:
        return {"default": strategies.DEFAULT_STRATEGY, "strategies": list(strategies.STRATEGIES)}

    def http_failure(error: Exception) -> HTTPException:
        calls = getattr(error, "calls", [])
        if isinstance(error, EngineError):
            return HTTPException(422 if error.status == 400 else error.status or 500, {"message": str(error), "calls": calls})
        if isinstance(error, ValueError):
            return HTTPException(502, {"message": str(error), "calls": calls})
        _log.error("d3 strategy failed", exc_info=error)
        return HTTPException(500, "d3 could not answer")

    def ask_key(body: AskRequest) -> str:
        part = [client.generation, body.model_dump(mode="json", exclude={"stream"})]
        return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()

    def ndjson_ask(lines: Any, cached: str) -> StreamingResponse:
        return StreamingResponse(lines, media_type="application/x-ndjson", headers={"X-Cache": cached})

    async def replay_ask(result: dict[str, Any]):
        """A kept answer in the shape of a stream: each answer, then `done`."""
        for qid, answer in result["answers"].items():
            yield line({"type": "answer", "id": qid, "answer": answer})
        yield line({"type": "done", **result})

    async def live_ask(events: Any, first: tuple[str, Any] | None, key: str | None, size: int):
        """One JSON line per question as it is settled, then `done` with the body a call without `stream` returns.
        A failure after the first answer can only be sent as an `error` line."""
        result: dict[str, Any] | None = None
        try:
            async def all_events():
                if first is not None:
                    yield first
                async for event in events:
                    yield event

            async for kind, value in all_events():
                if kind == "answer":
                    yield line({"type": "answer", "id": value[0], "answer": value[1]})
                else:
                    result = value
        except Exception as error:
            failure = http_failure(error)
            yield line({"type": "error", "status": failure.status_code, "detail": failure.detail})
            return
        finally:
            await events.aclose()
        if key and result is not None:
            cache.put(key, result, size)
        yield line({"type": "done", **(result or {})})

    @app.post("/v1/ask")
    async def ask(body: AskRequest, response: Response, _auth: None = Depends(authorize),
                  _slot: None = Depends(slot)) -> Any:
        """Laya-shaped: typed questions in, one settled answer per question out (see strategies.py)."""
        try:
            check_questions(body.questions)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        tuning = strategies.Tuning(body.min_confidence, body.thresholds, body.prune_ratio)
        size = cache_limit()
        key = ask_key(body) if size else None
        kept = cache.get(key, size) if key else None
        if kept is not None:
            if body.stream:
                return ndjson_ask(replay_ask(kept), "hit")
            response.headers["X-Cache"] = "hit"
            return kept
        try:
            if body.stream:
                events = strategies.run_iter(client, body.strategy, body.text, body.questions, tuning, body.trace)
                try:
                    first = await anext(events, None)  # the first question is settled here, so a refusal is still an HTTP error
                except BaseException:
                    await events.aclose()
                    raise
                return ndjson_ask(live_ask(events, first, key, size), "miss")
            result = await strategies.run(client, body.strategy, body.text, body.questions, tuning, body.trace)
        except Exception as error:
            raise http_failure(error) from None
        if key:
            cache.put(key, result, size)
        response.headers["X-Cache"] = "miss"
        return result

    @app.post("/v1/validate", dependencies=[Depends(authorize)])
    def validate(body: ValidateRequest) -> dict[str, bool]:
        """The one definition of a valid question set (same rules as DSC_Laya_Model's)."""
        try:
            check_questions(body.questions)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"ok": True}

    @app.post("/v1/combine", dependencies=[Depends(authorize)])
    def combine(body: CombineRequest) -> Any:
        try:
            return strategies.combine(body.voters, body.options, body.prune_ratio, body.pair_ratio)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise HTTPException(422, "voters must be answers to one choice question over the same options") from None

    @app.put("/admin/threads", dependencies=[Depends(admin)])
    def set_threads(body: Threads) -> dict[str, Any]:
        """Save and report the value for compatibility; Ollama chooses its own threads."""
        client.num_thread = body.num_thread
        settings.save(tokens.settings_path, num_thread=body.num_thread)
        return {"num_thread": client.num_thread}

    async def variant_rows() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """What this machine has (None when Ollama is on another machine) and every quantization with its estimate and
        whether it fits. Read when asked, never cached."""
        hw = await asyncio.to_thread(hardware.probe) if client.same_machine else None
        return hw, variants.listing(hw, client.model, await client.downloaded())

    @app.get("/admin/variants", dependencies=[Depends(admin)])
    async def list_variants() -> dict[str, Any]:
        hw, rows = await variant_rows()
        return {"hardware": hw, "summary": hardware.describe(hw), "variants": rows, "estimate": "Estimated from the file size; not measured."}

    @app.put("/admin/variant", dependencies=[Depends(admin)])
    async def choose_variant(body: VariantChoice) -> dict[str, Any]:
        """Saves the Ollama model name. It is read at start, so a restart applies it."""
        if config is None:
            raise HTTPException(409, "Settings are fixed in this setup")
        row = variants.find((await variant_rows())[1], body.id)
        if row is None:
            raise HTTPException(404, "No such variant.")
        if row["fits"] is False:
            raise HTTPException(409, row["reason"])
        try:
            config.set({"ollama_model": row["model"]})
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"saved": row["id"], "restart_required": not row["current"], "downloaded": row["downloaded"]}

    @app.get("/admin/workers", dependencies=[Depends(admin)])
    def workers() -> dict[str, Any]:
        """Read-only: the replica count is a Docker Compose setting the service cannot change."""
        def number(name: str) -> int | None:
            value = os.environ.get(name, "").strip()
            return int(value) if value.isdigit() else None

        return {"replicas": number("SERVICE_REPLICAS"), "ollama_parallel": client.parallel}

    @app.get("/admin/updates", dependencies=[Depends(admin)])
    async def update_status() -> dict[str, Any]:
        """Return the last check without I/O; zero check hours disables only the timer."""
        return {"enabled": True, **updates.status()}

    @app.post("/admin/updates/check", dependencies=[Depends(admin)])
    async def check_updates() -> dict[str, Any]:
        return {"enabled": True, **await updates.check()}

    @app.post("/admin/updates/weights", dependencies=[Depends(admin)])
    async def update_weights() -> dict[str, Any]:
        try:
            return {"enabled": True, **await updates.update_weights()}
        except Exception as error:
            _log.exception("Weight update failed")
            raise HTTPException(502, f"Weight update failed: {error}") from None

    @app.get("/admin/tokens", dependencies=[Depends(admin)])
    def list_tokens() -> dict[str, Any]:
        return {"tokens": [{**row, "projects": None} for row in tokens.list()]}

    @app.post("/admin/tokens", dependencies=[Depends(admin)])
    def create_token(body: NewToken) -> dict[str, Any]:
        return tokens.create(body.name.strip(), None, body.rate, body.expires_days)

    @app.get("/admin/settings", dependencies=[Depends(admin)])
    def get_settings() -> dict[str, Any]:
        return {"settings": config.describe() if config else []}

    @app.put("/admin/settings", dependencies=[Depends(admin)])
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        """Saves options; null resets one. Every replica reads the shared file, so a change applies everywhere within
        a second, except options marked `restart`."""
        if config is None:
            raise HTTPException(409, "Settings are fixed in this setup")
        try:
            config.set(body.values)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"settings": config.describe()}

    @app.put("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def update_token(token_id: str, body: TokenScope) -> dict[str, bool]:
        if not tokens.update(token_id, None, body.rate):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    latency = LatencyHistogram()

    @app.middleware("http")
    async def measure(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            latency.observe(time.perf_counter() - started)
        token_id = getattr(request.state, "token_id", None)
        if token_id:
            await run_in_threadpool(tokens.record, token_id, (time.perf_counter() - started) * 1000, response.status_code >= 500)
        return response

    @app.get("/metrics", dependencies=[Depends(admin)])
    def metrics() -> PlainTextResponse:
        """Prometheus text per token, from the shared token file (up to a minute behind). Send the admin password."""
        return PlainTextResponse(tokens.metrics("d3") + latency.render("d3"))

    @app.delete("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def revoke_token(token_id: str) -> dict[str, bool]:
        if not tokens.revoke(token_id):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    web = Path(os.environ.get("SERVICE_WEB_DIR", Path(__file__).resolve().parents[1] / "web"))
    if web.is_dir():
        app.mount("/admin", StaticFiles(directory=web, html=True))

    return app


def main() -> None:
    import sys

    import uvicorn

    if "--dotenv" in sys.argv[1:]:  # run.bat asks for it; a plain `python src/app.py` and Docker never read .env files
        loaded = settings.load_env_files(Path(__file__).resolve().parents[1])
        for name, names in loaded.items():
            print(f"{name}: loaded {', '.join(names) or 'no variables'} (names only; a real environment variable wins)", flush=True)
        if not loaded:
            print(".env.local / .env: no file in the service folder", flush=True)
    store = TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    config = settings.Config(store.settings_path, SETTINGS)
    client = engine_from_env("SERVICE_", D3Engine, DEFAULT_MODEL, settings=config, marker=default_marker())
    saved = settings.load(store.settings_path)
    threads = saved.get("num_thread") or os.environ.get("SERVICE_NUM_THREAD", "").strip()
    if threads:
        client.num_thread = int(threads)
    app = create_app(client=client, admin_password=os.environ.get("SERVICE_ADMIN_PASSWORD") or None,
                     tokens=store, config=config)
    host, port = os.environ.get("SERVICE_HOST", "127.0.0.1"), int(os.environ.get("SERVICE_PORT", "29221"))

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)  # after uvicorn's own "running on" line
            shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            print(f"INFO:     Admin page: http://{shown}:{port}/admin/", flush=True)

    Server(uvicorn.Config(app, host=host, port=port)).run()


if __name__ == "__main__":
    main()
