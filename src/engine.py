"""Queued Ollama decisions, cache invalidation, and replica update markers."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "d3:2b"


class EngineError(RuntimeError):
    """The model could not answer. `status` is the HTTP status the service maps it to."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def default_marker() -> Path:
    """File in the shared data volume whose content changes when one replica downloads new weights."""
    configured = os.environ.get("SERVICE_WEIGHTS_MARKER")
    return Path(configured) if configured else Path(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json")).parent / ".d3-weights-version"


class D3Engine:
    """One decision per replica; an injected loader supplies the Ollama adapter."""

    def __init__(self, model: str = DEFAULT_MODEL, *, loader: Callable[[str], Any],
                 timeout: float = 120.0, parallel: int = 1, marker: Path | None = None,
                 remote_sha: Callable[[str], str] = lambda _: "",
                 cached_sha: Callable[[str], str | None] = lambda _: None,
                 download: Callable[[str], None] = lambda _: None,
                 local_models: Callable[[], set[str]] = lambda: set(), same_machine: bool = True) -> None:
        self.model = model
        self._local_models = local_models
        self.same_machine = same_machine  # Ollama runs on the machine this service sees, so its free memory is known
        self._threads: int | None = None
        self._timeout = timeout
        self._marker = marker
        self._seen = self._read_marker()
        self._loader = loader
        self._remote_sha_of, self._cached_sha_of, self._download_of = remote_sha, cached_sha, download
        self._base: Any | None = None
        self._load_lock = threading.Lock()
        self.parallel = max(1, int(parallel))  # decisions one replica runs at once; match OLLAMA_NUM_PARALLEL
        self._gate = asyncio.Semaphore(self.parallel)
        self.generation = 0  # part of the answer cache key; changes whenever the loaded models are dropped

    @property
    def num_thread(self) -> int | None:
        return self._threads

    @num_thread.setter
    def num_thread(self, count: int) -> None:
        if count != self._threads:
            self._threads = count

    @property
    def ready(self) -> bool:
        return self._base is not None

    async def downloaded(self) -> set[str]:
        """Ollama model names on disk; empty when Ollama cannot be reached."""
        try:
            return await asyncio.to_thread(self._local_models)
        except EngineError:
            return set()

    async def close(self) -> None:
        self._base = None

    async def warm(self) -> None:
        """Load the base model now so the first request does not pay for it."""
        await asyncio.to_thread(self._base_model)

    def _base_model(self) -> Any:
        with self._load_lock:
            if self._base is None:
                self._base = self._loader(self.model)
            return self._base

    async def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Score the typed questions about `state` in one non-autoregressive pass."""
        async with self._gate:
            return await asyncio.to_thread(self._decide, state, questions)

    def _decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        try:
            return {**self._base_model().decide(state, questions, self._timeout), "model": self.model}
        except EngineError as error:
            if error.status == 503:
                self._base = None
            raise

    # Updates compare Ollama digests; a marker in the data volume invalidates replicas.

    async def cached_sha(self) -> str | None:
        return await asyncio.to_thread(self._cached_sha_of, self.model)

    async def remote_sha(self) -> str:
        return await asyncio.to_thread(self._remote_sha_of, self.model)

    async def pull(self) -> None:
        await asyncio.to_thread(self._download_of, self.model)

    async def unload(self) -> None:
        """Forget the loaded models here and tell the other replicas to do the same; the next call loads the new weights."""
        await asyncio.to_thread(self._invalidate)

    def _invalidate(self) -> None:
        if self._marker is not None:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._seen = uuid.uuid4().hex
            self._marker.write_text(self._seen, "utf-8")
        self._forget()

    def sync(self) -> None:
        """Called every few seconds: drop the loaded models when another replica downloaded new weights."""
        current = self._read_marker()
        if current != self._seen:
            self._seen = current
            self._forget()

    def _forget(self) -> None:
        with self._load_lock:
            self._base = None
            self.generation += 1  # kept answers are from the old weights

    def _read_marker(self) -> str | None:
        try:
            return self._marker.read_text("utf-8") if self._marker else None
        except OSError:
            return None
