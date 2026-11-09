import asyncio

from conftest import ANSWERS, make_engine

QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "returns": "Refunds"}}}


def ask(engine):
    return asyncio.run(engine.decide("charged twice", QUESTIONS))


def test_the_model_loads_once_and_answers():
    engine = make_engine()
    reply = ask(engine)
    assert reply["answers"] == ANSWERS and reply["model"] == "m" and reply["input_tokens"] == 7
    ask(engine)
    assert engine.loads == ["m"] and engine.ready


def test_another_replicas_download_makes_this_one_load_again(tmp_path):
    marker = tmp_path / "marker"
    mine, other = make_engine(marker=marker), make_engine(marker=marker)
    ask(mine)
    mine.sync()
    assert mine.ready
    asyncio.run(other.unload())
    mine.sync()
    assert not mine.ready
    ask(mine)
    assert mine.loads == ["m", "m"]


def test_unload_drops_the_model_and_changes_the_cache_generation():
    engine = make_engine()
    ask(engine)
    before = engine.generation
    asyncio.run(engine.unload())
    assert not engine.ready and engine.generation == before + 1


def test_thread_count_is_reported_without_configuring_ollama():
    engine = make_engine()
    engine.num_thread = 4
    assert engine.num_thread == 4
