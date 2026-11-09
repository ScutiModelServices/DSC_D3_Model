"""Decision strategies for the one d3 model, answering in DSC_Laya_Model's shape.

Laya combines two local models; d3 is one, so only the single-model strategies exist here:

  d3-only       One call for every question; `uncertain` when the confidence is under the threshold.
  d3-narrowing  A choice question that is not settled drops clearly worse options and is asked again (up to 3 rounds).
  auto          d3-narrowing (the default).

Every answer has the pick (`choice`, `score` or `noul`), `answer_confidence`, `uncertain` and `answered_by`; `trace`
adds the `rounds`. Thresholds (confidence at or above): 2 options 0.90, 3+ options 0.75. These are Laya English's
numbers, used as a starting point; they have not been measured for d3. `min_confidence` replaces them,
`thresholds` ({"2": {"d3": 0.9}, "3+": {"d3": 0.7}}) replaces single ones."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from answers import for_model, normalize
from engine import D3Engine

VOTER = "d3"
_THRESHOLDS = {2: 0.90, 3: 0.75}
MAX_NARROWING = 3
PRUNE_RATIO = 0.25  # options under this share of the leader's averaged probability are dropped
PAIR_RATIO = 0.5    # `combine` only: a runner-up under this share of the leader's difference score leaves the leader standing


@dataclass(frozen=True)
class Tuning:
    """Per-request overrides. `min_confidence` replaces every threshold; `thresholds` replaces single ones."""

    min_confidence: float | None = None
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    prune_ratio: float = PRUNE_RATIO


class NoAnswer(ValueError):
    """d3 answered, but not usably; `calls` is the log of the calls made."""


class Session:
    """One request: asks d3 once for the full question set, and again only for narrowed questions."""

    def __init__(self, engine: D3Engine, text: str, questions: dict[str, dict], tuning: Tuning) -> None:
        self._engine, self._text = engine, text
        self.questions, self.tuning = questions, tuning
        self._full: dict[str, dict] | None = None
        self.failures: list[str] = []
        self.calls: list[dict[str, Any]] = []

    async def _call(self, questions: dict[str, dict]) -> dict[str, dict] | None:
        started, tokens = time.monotonic(), None
        try:
            prediction = await self._engine.decide(self._text, {q: for_model(s) for q, s in questions.items()})
            answers = normalize(questions, prediction)
            tokens = prediction.get("input_tokens", 0)
            return answers
        except ValueError as error:  # an unusable result; an EngineError (backend down) is not caught and ends the request
            self.failures.append(str(error))
            return None
        finally:
            self.calls.append({"model": self._engine.model, "fallback": False,
                               "input_tokens": tokens if isinstance(tokens, int) else None,
                               "duration_ms": round((time.monotonic() - started) * 1000), "ok": tokens is not None})

    async def ask(self, qid: str, spec: dict, narrowed: bool = False) -> dict | None:
        if narrowed:
            answers = await self._call({qid: spec})
        else:
            if self._full is None:
                self._full = await self._call(self.questions)
            answers = self._full
        return None if answers is None else answers[qid]

    def need(self, count: int) -> float:
        if self.tuning.min_confidence is not None:
            return self.tuning.min_confidence
        row = 2 if count == 2 else 3
        return self.tuning.thresholds.get("2" if row == 2 else "3+", {}).get(VOTER, _THRESHOLDS[row])


def _count(spec: dict) -> int:
    return len(spec["criteria"]) if spec["type"] == "choice" else (2 if spec["type"] == "noul" else 3)


def _done(answer: dict, uncertain: bool, **extra: Any) -> dict:
    return {**answer, "answered_by": VOTER, "uncertain": uncertain, **extra}


def _no_answer(s: Session, qid: str) -> NoAnswer:
    return NoAnswer(f"d3 could not answer {qid!r}" + (f" ({'; '.join(s.failures)})" if s.failures else ""))


# -- pure helpers, shared in shape with DSC_Laya_Model (also behind /v1/combine) ------------------------------
def _mean(answers: dict[str, dict]) -> dict[str, float]:
    options = list(next(iter(answers.values()))["probabilities"])
    return {o: sum(a["probabilities"][o] for a in answers.values()) / len(answers) for o in options}


def _restrict(answer: dict, options: list[str]) -> dict:
    kept = {o: answer["probabilities"].get(o, 0.0) for o in options}
    total = sum(kept.values())
    choice = answer["choice"] if answer["choice"] in kept else max(kept, key=kept.get)
    return {**answer, "choice": choice, "probabilities": {o: (v / total if total else 1 / len(options)) for o, v in kept.items()}}


def kept_options(average: dict[str, float], ratio: float) -> list[str]:
    """The options worth asking about again, in the question's order: those at least `ratio` of the leader's
    probability, never fewer than two. A ratio of 0 keeps them all."""
    if ratio <= 0:
        return list(average)
    top = max(average.values())
    kept = [o for o in average if average[o] >= ratio * top]
    if len(kept) < 2:
        kept = sorted(average, key=average.__getitem__, reverse=True)[:2]
    return [o for o in average if o in kept]


def combine(voters: dict[str, dict], options: list[str] | None, prune_ratio: float, pair_ratio: float) -> dict:
    """For a caller that adds its own voter (a cloud fallback) to d3: the voters over only `options` (renormalised),
    their averaged probabilities, the options still worth asking about and, with two or more voters, the pairwise
    comparison."""
    options = options or list(next(iter(voters.values()))["probabilities"])
    restricted = {n: _restrict(a, options) for n, a in voters.items()}
    average = _mean(restricted)
    out: dict[str, Any] = {"voters": restricted, "average": average, "kept": kept_options(average, prune_ratio)}
    if len(restricted) > 1:
        out["final"] = _final(restricted, pair_ratio)
    return out


def _final(voters: dict[str, dict], pair_ratio: float = PAIR_RATIO) -> dict:
    """Compare the voters' confidence-weighted values by pairwise difference per option."""
    options = list(next(iter(voters.values()))["probabilities"])
    scaled = {n: {o: a["answer_confidence"] * a["probabilities"][o] for o in options} for n, a in voters.items()}
    names = list(scaled)
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
    score = {o: sum(abs(scaled[x][o] - scaled[y][o]) for x, y in pairs) / len(pairs) for o in options}
    leader = max(voters.values(), key=lambda a: a["answer_confidence"])["choice"]
    ranked = sorted(options, key=lambda o: (round(score[o], 6), o == leader), reverse=True)
    top, runner_up = ranked[0], ranked[1]
    decided = score[top] > 0 and score[runner_up] < pair_ratio * score[top]
    total = sum(score.values())
    share = {o: (score[o] / total if total else 1 / len(options)) for o in options}
    out: dict[str, Any] = {
        "type": "choice", "choice": top, "probabilities": {o: round(v, 4) for o, v in share.items()},
        "answer_confidence": round(share[top], 4), "uncertain": True, "answered_by": "+".join(names),
        "resolved_by": "final", "disagreement": {n: a["choice"] for n, a in voters.items()}, "also_check": ranked[1:3],
    }
    if not decided:
        out.update(unknown=True, candidates=ranked[:2])
    return out


# -- strategies --------------------------------------------------------------------------------------------
async def _single(s: Session, qid: str, spec: dict) -> dict:
    answer = await s.ask(qid, spec)
    if answer is None:
        raise _no_answer(s, qid)
    return _done(answer, answer["answer_confidence"] < s.need(_count(spec)) - 1e-9)


async def _narrowing(s: Session, qid: str, spec: dict) -> dict:
    if spec["type"] != "choice":
        return await _single(s, qid, spec)
    options, eliminated, rounds = list(spec["criteria"]), [], []
    current, answer = spec, None
    for attempt in range(MAX_NARROWING + 1):
        answer = await s.ask(qid, current, narrowed=bool(eliminated))
        rounds.append({"step": VOTER, "options": list(options), **(
            {"choice": answer["choice"], "confidence": answer["answer_confidence"],
             "probabilities": {o: round(p, 4) for o, p in answer["probabilities"].items()},
             "threshold": s.need(len(options))} if answer else {"failed": "could not answer"})})
        if answer is None:
            break
        if answer["answer_confidence"] >= s.need(len(options)) - 1e-9:
            return _choice_result(spec, answer, False, eliminated, rounds, "narrowed" if eliminated else "local")
        if attempt == MAX_NARROWING or len(options) <= 2:
            break
        kept = kept_options(answer["probabilities"], s.tuning.prune_ratio)
        if len(kept) == len(options):
            break
        eliminated += [o for o in options if o not in kept]
        options = kept
        current = {**spec, "criteria": {o: spec["criteria"][o] for o in kept}}
    if answer is None:
        raise _no_answer(s, qid)
    return _choice_result(spec, _restrict(answer, options), True, eliminated, rounds, None)


def _choice_result(spec: dict, answer: dict, uncertain: bool, eliminated: list[str], rounds: list, resolved: str | None) -> dict:
    out = _done(answer, uncertain, rounds=rounds, **({"resolved_by": resolved} if resolved else {}))
    if eliminated:
        out["probabilities"] = {o: answer["probabilities"].get(o, 0.0) for o in spec["criteria"]}
        out["eliminated"] = eliminated
    return out


STRATEGIES = {"d3-only": _single, "d3-narrowing": _narrowing, "auto": _narrowing}
DEFAULT_STRATEGY = "auto"

_COMPACT = ("type", "choice", "score", "noul", "answer_confidence", "uncertain", "answered_by", "probabilities",
            "also_check", "unknown", "candidates", "disagreement")


def _compact(answer: dict) -> dict:
    out = {k: answer[k] for k in _COMPACT if answer.get(k) is not None}
    for key in ("answer_confidence", "score", "noul"):
        if key in out:
            out[key] = round(out[key], 2)
    if "probabilities" in out:
        out["probabilities"] = {o: round(p, 2) for o, p in out["probabilities"].items()}
    return out


async def run_iter(engine: D3Engine, strategy: str, text: str, questions: dict[str, dict],
                   tuning: Tuning = Tuning(), trace: bool = False) -> AsyncIterator[tuple[str, Any]]:
    """("answer", (qid, answer)) as each question is settled, then ("done", result). The first answer arrives after
    the one call for all questions; a narrowed choice question makes later answers wait for its extra calls.
    Raises NoAnswer (with `calls`) when d3 gave no usable answer; an EngineError (backend down) also carries `calls`."""
    settle = STRATEGIES[strategy]
    session = Session(engine, text, questions, tuning)
    answers: dict[str, dict] = {}
    try:
        for qid, spec in questions.items():
            answers[qid] = await settle(session, qid, spec)
            yield "answer", (qid, answers[qid] if trace else _compact(answers[qid]))
    except Exception as error:
        error.calls = session.calls  # type: ignore[attr-defined]
        raise
    yield "done", {
        "strategy": strategy,
        "model": engine.model,
        "fallback": False,
        "project_used": False,  # no project models here; kept so a caller reads the same fields as Laya's
        "answers": answers if trace else {qid: _compact(a) for qid, a in answers.items()},
        "uncertain": any(a["uncertain"] for a in answers.values()),
        "calls": session.calls,
    }


async def run(engine: D3Engine, strategy: str, text: str, questions: dict[str, dict],
              tuning: Tuning = Tuning(), trace: bool = False) -> dict[str, Any]:
    async for kind, value in run_iter(engine, strategy, text, questions, tuning, trace):
        if kind == "done":
            return value
    raise AssertionError("run_iter always ends with done")
