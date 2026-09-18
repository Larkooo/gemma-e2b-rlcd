import json

import pytest

from gemma_rlcd.comparison import discrete_answers, generation_task, parse_generated
from gemma_rlcd.core import Choice, Independent, Noul, Score, State


def questions():
    return {
        "brand": Choice("Which brand is this?", {"Porsche": "Porsche", "Mercedes": "Mercedes"}),
        "quality": Score("Grade its condition", ["Poor", "Good", "Excellent"]),
        "moving": Noul("Is it moving?"),
        "presence": Independent("Which are present?", {"car": "A car", "person": "A person"}),
    }


def test_generation_requests_all_fields_in_one_json_answer():
    task = generation_task(questions())
    assert "Porsche" in task and "Mercedes" in task
    assert "Grade its condition" in task and "Excellent" in task
    assert "a boolean" in task and "proposition name to a boolean" in task
    expected = {
        "brand": "Porsche",
        "quality": 2,
        "moving": False,
        "presence": {"car": True, "person": False},
    }
    assert parse_generated(json.dumps(expected), questions()) == expected
    assert parse_generated("```json\n" + json.dumps(expected) + "\n```", questions()) == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"brand":"BMW","quality":2,"moving":false,"presence":{"car":true,"person":false}}',
        '{"brand":"Porsche","quality":true,"moving":false,"presence":{"car":true,"person":false}}',
        '{"brand":"Porsche","quality":3,"moving":false,"presence":{"car":true,"person":false}}',
        '{"brand":"Porsche","quality":2,"moving":false,"presence":{"car":true}}',
        '{"brand":"Porsche","brand":"Mercedes"}',
        'The answer is {"brand":"Porsche"}',
    ],
)
def test_generated_output_must_match_the_contract(text):
    with pytest.raises(ValueError):
        parse_generated(text, questions())


def test_agreement_uses_grade_mode_and_boolean_thresholds():
    assert discrete_answers(
        {
            "grade": {
                "type": "score",
                "score": 1.4,
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            },
            "moving": {"type": "noul", "noul": 0.7},
            "presence": {"type": "independent", "probabilities": {"car": 0.9, "person": 0.1}},
        }
    ) == {"grade": 2, "moving": True, "presence": {"car": True, "person": False}}


def test_streaming_generation_uses_real_chunks_and_validates_complete_answer(monkeypatch):
    import sys
    from types import SimpleNamespace

    from gemma_rlcd import comparison

    seen = []
    parts = ['{"visible":', "true", "}"]

    def stream(*args, **kwargs):
        for index, text in enumerate(parts):
            assert len(seen) == index
            yield SimpleNamespace(
                text=text,
                generation_tokens=index + 1,
                prompt_tokens=50,
                finish_reason="stop" if index == len(parts) - 1 else None,
            )

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(generate=None, stream_generate=stream)
    )
    monkeypatch.setattr(comparison, "prepare_generation", lambda *args: ("prompt", {}))
    backend = SimpleNamespace(
        model=None,
        processor=None,
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1]),
        mx=SimpleNamespace(synchronize=lambda: None),
    )
    result = comparison.generate_answers(
        backend,
        State(text="test"),
        {"visible": Noul("Visible?")},
        on_token=lambda text, count: seen.append((text, count)),
    )
    assert seen == list(zip(parts, [1, 2, 3], strict=True))
    assert result["valid"] and result["answers"] == {"visible": True}
    assert result["raw_text"] == "".join(parts)


def test_concurrent_comparison_streams_both_paths_before_either_finishes(monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    from gemma_rlcd import comparison
    from gemma_rlcd.core import TokenScores

    scored, generated = Event(), Event()
    events = []

    class Backend:
        last_stats = {}
        processor = SimpleNamespace(tokenizer=SimpleNamespace(mutable=[]))

        def symbols(self, count):
            return ("A", "B")

        def score_questions(self, state, questions, on_scores, on_progress=None):
            assert generated.wait(2), "Generation never started alongside scoring"
            result = TokenScores((5, 0), 1, 20)
            on_scores([(0, result)])
            scored.set()
            return [result]

    backend = Backend()

    def generate(worker, state, questions, on_token, on_progress=None):
        assert worker.processor is not backend.processor
        assert worker.processor.tokenizer is not backend.processor.tokenizer
        on_token('{"visible":', 1)
        generated.set()
        assert scored.wait(2), "Scoring never completed while generation was active"
        on_token("true}", 2)
        return {"answers": {"visible": True}, "valid": True, "inference_seconds": 0}

    monkeypatch.setattr(comparison, "generate_answers", generate)
    _, result = comparison.compare(
        backend,
        State(text="scene"),
        {"visible": Noul("Visible?")},
        0,
        emit=events.append,
        concurrent=True,
    )
    assert events[0]["type"] == "race_start"
    first_token = next(i for i, event in enumerate(events) if event["type"] == "token")
    first_answer = next(i for i, event in enumerate(events) if event["type"] == "answer")
    first_finish = next(i for i, event in enumerate(events) if event["type"] == "phase_complete")
    assert first_token < first_answer < first_finish
    assert result["agreement"] == {"visible": True}
    assert result["methodology"]["execution"] == "concurrent_shared_gpu"


@pytest.mark.parametrize("cancel_at", ["race_start", "phase_start"])
def test_concurrent_comparison_can_stop_before_or_after_worker_launch(cancel_at):
    from types import SimpleNamespace

    from gemma_rlcd.comparison import compare

    def emit(event):
        if event["type"] == cancel_at:
            raise RuntimeError("Client stopped")

    with pytest.raises(RuntimeError, match="Client stopped"):
        compare(
            SimpleNamespace(),
            State(text="scene"),
            {"visible": Noul("Visible?")},
            0,
            emit=emit,
            concurrent=True,
        )
