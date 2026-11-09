import pytest

from engine import D3Engine

ANSWERS = {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "returns": 0.1}, "confidence": 0.8}}


@pytest.fixture(autouse=True)
def _tmp_tokens_file(tmp_path, monkeypatch):
    """Keeps tokens.json and settings.json out of the repo, and a saved setting from leaking into the next test."""
    monkeypatch.setenv("SERVICE_TOKENS_FILE", str(tmp_path / "tokens.json"))


class FakeLoaded:
    """Stands in for a loaded model: answers `answers` and records every call."""

    def __init__(self, source, answers=None, fail=None):
        self.source, self.answers, self.calls, self.fail = source, answers or ANSWERS, [], fail

    def decide(self, state, questions, max_time):
        self.calls.append({"state": state, "questions": questions})
        if self.fail:
            raise self.fail
        return {"answers": self.answers, "input_tokens": 7}


def make_engine(**options):
    """An engine whose loader builds FakeLoaded models; `engine.loads` lists the sources it was asked for."""
    loads = []
    answers, fail = options.pop("answers", None), options.pop("fail", None)

    def loader(source):
        loads.append(source)
        return FakeLoaded(source, answers, fail)

    options.setdefault("loader", loader)
    engine = D3Engine("m", **options)
    engine.loads = loads
    return engine
