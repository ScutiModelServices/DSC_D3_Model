"""Validation of what callers send and of what d3 returns, per typed question.

`check_question(s)` is the same rule set as DSC_Laya_Model's, so a question set valid for one is valid for the other.
`normalize` turns d3's answers (Ollama /v1/systemone) into Laya's answer shape: every answer has `answer_confidence`."""

from __future__ import annotations

import math
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
MIN_OPTIONS, MAX_OPTIONS = 2, 10
MAX_QUESTIONS = 8
_NOUL_KEYS = {"false", "true"}
NOUL_DEFAULT = {"true": "Yes", "false": "No"}  # d3 needs criteria for noul; Laya lets callers leave them out


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def check_question(qid: str, spec: dict[str, Any]) -> None:
    """Raises ValueError naming the first thing wrong with one question."""
    if not _is_text(qid):
        raise ValueError("question ids must be non-empty strings")
    if not isinstance(spec, dict):
        raise ValueError(f"question {qid!r} must be an object")
    unknown = set(spec) - {"type", "instructions", "criteria"}
    if unknown:
        raise ValueError(f"question {qid!r} has unknown field(s): {', '.join(sorted(unknown))}")
    kind, criteria = spec.get("type"), spec.get("criteria")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid!r} type must be choice, score or noul")
    if not _is_text(spec.get("instructions")):
        raise ValueError(f"question {qid!r} needs non-empty instructions")
    if kind == "choice":
        ok = (isinstance(criteria, dict) and MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS
              and all(_is_text(k) and _is_text(v) for k, v in criteria.items()))
        rule = f"an object of {MIN_OPTIONS} to {MAX_OPTIONS} option: description strings"
    elif kind == "score":
        ok = (isinstance(criteria, list) and MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS
              and all(_is_text(v) for v in criteria))
        rule = f"a list of {MIN_OPTIONS} to {MAX_OPTIONS} level description strings, lowest first"
    else:
        ok = criteria is None or (isinstance(criteria, dict) and set(criteria) == _NOUL_KEYS
                                  and all(_is_text(v) for v in criteria.values()))
        rule = 'omitted, or an object with exactly the keys "false" and "true"'
    if not ok:
        raise ValueError(f"question {qid!r} ({kind}) criteria must be {rule}")


def check_questions(questions: Any) -> None:
    """Raises ValueError naming the first thing wrong with a request's questions."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"questions must be an object with 1 to {MAX_QUESTIONS} entries")
    for qid, spec in questions.items():
        check_question(qid, spec)


def for_model(spec: dict[str, Any]) -> dict[str, Any]:
    """The question as d3 wants it: a noul question gets default criteria when the caller left them out."""
    if spec["type"] == "noul" and spec.get("criteria") is None:
        return {**spec, "criteria": dict(NOUL_DEFAULT)}
    return spec


def _unit(value: Any) -> float:
    if not _is_number(value) or not 0 <= value <= 1:
        raise ValueError("invalid probability")
    return float(value)


def _answer(spec: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"type": spec["type"]}
    if spec["type"] == "choice":
        choice, probabilities = raw["choice"], raw["probabilities"]
        if choice not in spec["criteria"] or set(probabilities) != set(spec["criteria"]):
            raise ValueError("unknown choice")
        out["choice"] = choice
        out["probabilities"] = {key: _unit(value) for key, value in probabilities.items()}
        out["answer_confidence"] = _unit(raw["confidence"])
    elif spec["type"] == "score":
        score = raw["score"]
        if not _is_number(score) or not 0 <= score <= len(spec["criteria"]) - 1:
            raise ValueError("invalid score")
        out["score"] = round(float(score), 4)
        out["legend"] = {str(i): level for i, level in enumerate(spec["criteria"])}
        out["answer_confidence"] = _unit(raw["confidence"])
    else:
        noul = _unit(raw["noul"])
        out["noul"] = noul
        out["answer_confidence"] = max(noul, 1 - noul)  # d3 gives no confidence for yes/no: distance from a coin flip
    return out


def normalize(questions: dict[str, dict[str, Any]], prediction: Any) -> dict[str, dict[str, Any]]:
    """Validated answers by question id. Raises ValueError on anything unusable."""
    try:
        return {qid: _answer(spec, prediction["answers"][qid]) for qid, spec in questions.items()}
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("d3 returned an invalid result; no answers were returned") from error
