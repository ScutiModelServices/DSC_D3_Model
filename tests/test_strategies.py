import json

import pytest
from fastapi.testclient import TestClient

import strategies
from app import create_app
from conftest import make_engine
from engine import EngineError

CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "returns": "Refunds", "tech": "Errors"}}
SCORE = {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "angry"]}
NOUL = {"type": "noul", "instructions": "Urgent?"}


def choice(pick, confidence, **p):
    return {"type": "choice", "choice": pick, "probabilities": p, "confidence": confidence}


class Scripted:
    """A model whose answers come from a list of functions of (questions), one per call."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, source):
        owner = self

        class Loaded:
            def decide(self, state, questions, max_time):
                owner.calls.append(questions)
                reply = owner.replies[min(len(owner.calls), len(owner.replies)) - 1]
                return {"answers": reply(questions) if callable(reply) else reply, "input_tokens": 5}

        return Loaded()


def app_with(*replies, **options):
    script = Scripted(*replies)
    return TestClient(create_app(make_engine(loader=script), **options)), script


def ask(c, questions, **extra):
    return c.post("/v1/ask", json={"text": "charged twice", "questions": questions, **extra})


def test_validate_uses_laya_rules_and_strategies_are_listed():
    c, _ = app_with({})
    assert c.post("/v1/validate", json={"questions": {"q": CHOICE}}).json() == {"ok": True}
    assert c.post("/v1/validate", json={"questions": {"q": NOUL}}).status_code == 200  # noul criteria may be left out
    assert c.post("/v1/validate", json={"questions": {"q": {**CHOICE, "criteria": {"a": "x"}}}}).status_code == 422
    assert c.post("/v1/validate", json={"questions": {}}).status_code == 422
    body = c.get("/v1/strategies").json()
    assert body["default"] == "auto" and body["strategies"] == ["d3-only", "d3-narrowing", "auto"]


def test_every_question_type_is_answered_in_laya_shape():
    c, script = app_with({"team": choice("billing", 0.96, billing=0.99, returns=0.005, tech=0.005),
                          "tone": {"type": "score", "score": 1.4012, "confidence": 0.25, "legend": {}, "probabilities": {}},
                          "urgent": {"type": "noul", "noul": 0.88}})
    body = ask(c, {"team": CHOICE, "tone": SCORE, "urgent": NOUL}, strategy="d3-only").json()
    answers = body["answers"]
    assert answers["team"] == {"type": "choice", "choice": "billing", "answer_confidence": 0.96, "uncertain": False,
                               "answered_by": "d3", "probabilities": {"billing": 0.99, "returns": 0.01, "tech": 0.01}}
    assert answers["tone"]["score"] == 1.4 and answers["tone"]["uncertain"] is True  # 0.25 is under the 3-level threshold
    assert answers["urgent"]["noul"] == 0.88 and answers["urgent"]["answer_confidence"] == 0.88
    assert body["uncertain"] is True and body["strategy"] == "d3-only" and body["model"] == "m" and body["fallback"] is False
    assert len(script.calls) == 1 and script.calls[0]["urgent"]["criteria"] == {"true": "Yes", "false": "No"}
    assert body["calls"][0]["ok"] is True and body["calls"][0]["input_tokens"] == 5


def test_an_unsure_choice_is_narrowed_and_asked_again():
    first = {"q": choice("billing", 0.5, billing=0.55, returns=0.43, tech=0.02)}
    second = {"q": choice("billing", 0.95, billing=0.9, returns=0.1)}
    c, script = app_with(first, second)
    answer = ask(c, {"q": CHOICE}, trace=True).json()["answers"]["q"]
    assert [list(call["q"]["criteria"]) for call in script.calls] == [["billing", "returns", "tech"], ["billing", "returns"]]
    assert answer["resolved_by"] == "narrowed" and answer["uncertain"] is False and answer["eliminated"] == ["tech"]
    assert answer["probabilities"]["tech"] == 0.0 and len(answer["rounds"]) == 2


def test_d3_only_never_narrows_and_a_never_sure_choice_stays_uncertain():
    def unsure(questions):  # like d3, answer only the options asked about
        options = list(questions["q"]["criteria"])
        shares = {"billing": 0.5, "returns": 0.45, "tech": 0.05}
        total = sum(shares[o] for o in options)
        return {"q": choice("billing", 0.5, **{o: shares[o] / total for o in options})}

    c, script = app_with(unsure)
    assert ask(c, {"q": CHOICE}, strategy="d3-only").json()["answers"]["q"]["uncertain"] is True
    assert len(script.calls) == 1
    c, script = app_with(unsure)
    body = ask(c, {"q": CHOICE}).json()
    assert body["answers"]["q"]["uncertain"] is True and len(script.calls) == 2  # narrowed once, then two options left


def test_min_confidence_and_thresholds_change_what_is_sure():
    reply = {"q": choice("billing", 0.7, billing=0.8, returns=0.1, tech=0.1)}
    c, _ = app_with(reply)
    assert ask(c, {"q": CHOICE}, strategy="d3-only").json()["answers"]["q"]["uncertain"] is True
    assert ask(c, {"q": CHOICE}, strategy="d3-only", min_confidence=0.6).json()["answers"]["q"]["uncertain"] is False
    assert ask(c, {"q": CHOICE}, strategy="d3-only", thresholds={"3+": {"d3": 0.65}}).json()["answers"]["q"]["uncertain"] is False


@pytest.mark.parametrize("extra", [{"project": "app-a"}, {"max_len": 512}, {"pair_ratio": 0.5}, {"strategy": "consensus"}])
def test_what_d3_cannot_do_is_refused(extra):
    c, _ = app_with({})
    assert ask(c, {"q": CHOICE}, **extra).status_code == 422


def test_bad_questions_are_422_and_a_bad_model_result_is_502_with_calls():
    c, _ = app_with({"q": {"type": "choice", "choice": "nope", "probabilities": {}, "confidence": 1}})
    assert ask(c, {"q": {**CHOICE, "criteria": {"a": "x"}}}).status_code == 422
    reply = ask(c, {"q": CHOICE})
    assert reply.status_code == 502 and reply.json()["detail"]["calls"][0]["ok"] is False


def test_a_backend_failure_keeps_its_status():
    c = TestClient(create_app(make_engine(fail=EngineError("down", 503))))
    assert ask(c, {"q": CHOICE}).status_code == 503


def test_stream_sends_one_line_per_question_then_done_and_fills_the_cache():
    reply = {"a": choice("billing", 0.95, billing=0.95, returns=0.03, tech=0.02), "b": {"type": "noul", "noul": 0.1}}
    c, script = app_with(reply, cache_size=8)
    q = {"a": CHOICE, "b": NOUL}
    live = c.post("/v1/ask", json={"text": "x", "questions": q, "stream": True})
    lines = [json.loads(x) for x in live.text.splitlines()]
    assert live.headers["X-Cache"] == "miss"
    assert [(e["type"], e.get("id")) for e in lines] == [("answer", "a"), ("answer", "b"), ("done", None)]
    assert lines[-1]["answers"]["a"]["choice"] == "billing"
    kept = c.post("/v1/ask", json={"text": "x", "questions": q})
    assert kept.headers["X-Cache"] == "hit" and kept.json()["answers"] == lines[-1]["answers"] and len(script.calls) == 1
    replay = c.post("/v1/ask", json={"text": "x", "questions": q, "stream": True})
    assert [json.loads(x)["type"] for x in replay.text.splitlines()] == ["answer", "answer", "done"]


def test_combine_mixes_a_callers_voter_with_d3():
    voters = {"d3": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}, "answer_confidence": 0.6},
              "cloud": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}, "answer_confidence": 0.7}}
    c, _ = app_with({})
    body = c.post("/v1/combine", json={"voters": voters, "options": ["a", "b"]}).json()
    assert body["kept"] == ["a", "b"] and "final" in body and set(body["voters"]) == {"d3", "cloud"}
    assert c.post("/v1/combine", json={"voters": {"x": {"nope": 1}}}).status_code == 422


def test_ask_needs_a_token_when_the_admin_password_is_set(tmp_path):
    from tokens import TokenStore

    c, _ = app_with({}, admin_password="admin", tokens=TokenStore(tmp_path / "t.json"))
    assert c.post("/v1/ask", json={"text": "x", "questions": {"q": CHOICE}}).status_code == 401
    assert c.get("/v1/strategies").status_code == 401
    assert c.post("/v1/validate", json={"questions": {"q": CHOICE}}).status_code == 401
