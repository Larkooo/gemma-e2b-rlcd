import json

import pytest

from gemma_decisions.comparison import discrete_answers, generation_task, parse_generated
from gemma_decisions.core import Choice, Independent, Noul, Score


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
