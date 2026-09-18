import math
from string import ascii_uppercase

import pytest

from gemma_rlcd import Choice, DecisionEngine, Independent, Noul, Score, State
from gemma_rlcd.core import TokenScores, parse_question, softmax


class StubBackend:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def symbols(self, count):
        return ascii_uppercase[:count]

    def score(self, state, prompt, symbols):
        return TokenScores(tuple(math.log(p) for p in next(self.probabilities)), 0.8, 100)


def test_choice_returns_all_labels_and_correct_winner():
    answer = DecisionEngine(StubBackend([[0.2, 0.8]])).decide(
        State(text="test"), Choice("Which?", {"cat": "Cat", "dog": "Dog"})
    )
    assert answer["choice"] == "dog"
    assert answer["probabilities"] == pytest.approx({"cat": 0.2, "dog": 0.8})
    assert answer["selected_probability"] == pytest.approx(0.8)
    assert answer["calibration_status"] == "unvalidated"


def test_score_uses_weighted_level_positions():
    answer = DecisionEngine(StubBackend([[0.1, 0.6, 0.3]])).decide(
        State(text="test"), Score("How severe?", ["Cosmetic", "Workaround", "Blocking"])
    )
    assert answer["score"] == pytest.approx(1.2)
    assert answer["legend"]["2"] == "Blocking"
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)


def test_noul_handles_reversed_criteria_order():
    answer = DecisionEngine(StubBackend([[0.8, 0.2]])).decide(
        State(text="test"), Noul("Is there a cat?", {"false": "No cat", "true": "Cat"})
    )
    assert answer["noul"] == pytest.approx(0.2)
    assert "confidence" not in answer


def test_independent_probabilities_do_not_compete():
    answer = DecisionEngine(StubBackend([[0.9, 0.1], [0.8, 0.2]])).decide(
        State(text="test"), Independent("What is present?", {"cat": "Cat", "dog": "Dog"})
    )
    assert answer["probabilities"] == pytest.approx({"cat": 0.9, "dog": 0.8})


def test_all_primitives_share_state_in_one_request():
    engine = DecisionEngine(StubBackend([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]))
    answers = engine.system_one(
        State(text="test"),
        {
            "category": Choice("Which?", {"a": "A", "b": "B"}),
            "grade": Score("How much?", ["Low", "High"]),
            "present": Noul("Present?"),
        },
    )["answers"]
    assert set(answers) == {"category", "grade", "present"}
    assert answers["category"]["confidence"] == pytest.approx(0.0)
    assert answers["grade"]["score"] == pytest.approx(0.7)


def test_native_backend_receives_complete_schema_and_preserves_answer_contracts():
    state = State(text="Two cats. No dogs.")
    questions = {
        "animal": Choice("Which animal?", {"cat": "Cat", "dog": "Dog"}),
        "count": Score("How many cats?", ["Zero", "One", "Two"]),
        "dog": Noul("Any dogs?", {"false": "No dogs", "true": "Dogs present"}),
        "presence": Independent("Which animals?", {"cat": "A cat", "dog": "A dog"}),
    }

    class Native(StubBackend):
        def score_batch(self, *args):
            raise AssertionError("Native backend must receive the original questions")

        def score_questions(self, actual_state, actual_questions):
            assert actual_state is state
            assert actual_questions is questions
            probabilities = [[0.9, 0.1], [0.1, 0.1, 0.8], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]
            return [TokenScores(tuple(map(math.log, values)), 1, 100) for values in probabilities]

    answers = DecisionEngine(Native([])).system_one(state, questions)["answers"]
    assert answers["animal"]["choice"] == "cat"
    assert answers["count"]["score"] == pytest.approx(1.7)
    assert answers["dog"]["noul"] == pytest.approx(0.1)
    assert answers["presence"]["probabilities"] == pytest.approx({"cat": 0.9, "dog": 0.1})


@pytest.mark.parametrize(
    "data",
    [
        {"type": "score", "instructions": "x", "criteria": ["only"]},
        {"type": "noul", "instructions": "x", "criteria": {"yes": "Yes", "no": "No"}},
        {"type": "choice", "instructions": "x", "criteria": {"one": "one"}},
        {"type": "typo", "instructions": "x", "criteria": {}},
        {"type": "noul", "instructions": "x", "extra": 1},
    ],
)
def test_invalid_questions_rejected(data):
    with pytest.raises(ValueError):
        parse_question(data)


def test_choice_supports_255_criteria_but_rejects_256():
    Choice("Which?", {str(i): str(i) for i in range(255)})
    with pytest.raises(ValueError):
        Choice("Which?", {str(i): str(i) for i in range(256)})


@pytest.mark.parametrize("logits, temperature", [([math.nan, 0], 1), ([0, 1], 0), ([0], math.inf)])
def test_invalid_numerics_rejected(logits, temperature):
    with pytest.raises(ValueError):
        softmax(logits, temperature)


def test_large_logits_are_stable():
    assert softmax([10000, 10000]) == [0.5, 0.5]


def test_media_cannot_be_silently_dropped(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    assert State(videos=(str(video),)).videos == (str(video),)
    with pytest.raises(ValueError):
        State(images=(str(tmp_path / "missing.png"),))
    with pytest.raises(ValueError):
        State(audio=(str(video), str(video)))


def test_backend_invalid_output_is_rejected():
    class Invalid(StubBackend):
        def score(self, *args):
            return TokenScores((0.0, math.nan), 1, 100)

    with pytest.raises(ValueError):
        DecisionEngine(Invalid([])).decide(State(text="test"), Noul("True?"))


def test_all_fields_and_independent_labels_are_sent_in_one_batch():
    class Batched(StubBackend):
        def __init__(self):
            self.calls = []

        def score(self, *args):
            raise AssertionError("Batched backend must not use sequential score calls")

        def score_batch(self, state, requests):
            self.calls.append(requests)
            return [TokenScores((math.log(0.8), math.log(0.2)), 0.9, 120) for _ in requests]

    backend = Batched()
    result = DecisionEngine(backend).system_one(
        State(text="A cat and a dog."),
        {
            "choice": Choice("Which?", {"cat": "Cat", "dog": "Dog"}),
            "grade": Score("How clear?", ["Clear", "Unclear"]),
            "labels": Independent("What is present?", {"cat": "A cat", "dog": "A dog"}),
            "noul": Noul("Any animal?"),
        },
    )
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 5
    assert list(result["answers"]) == ["choice", "grade", "labels", "noul"]
    assert result["answers"]["labels"]["probabilities"] == pytest.approx({"cat": 0.8, "dog": 0.8})


def test_batch_result_count_is_checked():
    class Broken(StubBackend):
        def score_batch(self, state, requests):
            return []

    with pytest.raises(ValueError, match="wrong number of question results"):
        DecisionEngine(Broken([])).system_one(State(text="test"), {"test": Noul("True?")})


def test_streamed_decisions_arrive_before_later_batches_and_match_final_answers():
    seen = []

    class Streaming(StubBackend):
        def score_questions(self, state, questions, on_scores=None):
            scores = [TokenScores((2.0, 0.0), 0.8, 50), TokenScores((0.0, 2.0), 0.8, 50)]
            on_scores([(0, scores[0])])
            assert len(seen) == 1
            on_scores([(1, scores[1])])
            return scores

    result = DecisionEngine(Streaming([])).system_one(
        State(text="evidence"),
        {"visible": Independent("Which?", {"cat": "Cat", "dog": "Dog"})},
        on_answer=lambda path, answer: seen.append((path, answer)),
    )
    assert [path for path, _ in seen] == [("visible", "cat"), ("visible", "dog")]
    assert result["answers"]["visible"]["probabilities"] == {
        path[1]: answer["probabilities"]["yes"] for path, answer in seen
    }
