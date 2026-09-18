import json

import pytest

from gemma_rlcd.core import Choice, Independent, Noul, Score
from gemma_rlcd.json_scoring import JSONField, candidate_fields, compile_field


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))


def test_all_contracts_preserve_value_order_and_nested_fields():
    questions = {
        "animal": Choice("Pick one", {"cat": "cat", "dog": "dog"}),
        "count": Score("Count", ["Zero", "One", "Two"]),
        "present": Noul("Present?", {"false": "No", "true": "Yes"}),
        "labels": Independent("Present?", {"dog": "A dog", "cat": "A cat"}),
    }
    fields = candidate_fields(questions)
    assert [(field.path, field.values) for field in fields] == [
        (("animal",), ("cat", "dog")),
        (("count",), (0, 1, 2)),
        (("present",), (False, True)),
        (("labels", "dog"), (True, False)),
        (("labels", "cat"), (True, False)),
    ]


@pytest.mark.parametrize("values", [(0, 1, 2), (True, False), ("cat", "catfish", 'cat "A"\n雪')])
def test_json_paths_and_complete_values_survive_token_compilation(values):
    field = JSONField(('animal "kind"', "猫"), values)
    compiled = compile_field(CharacterTokenizer(), field, "Assistant:\n")
    for value, tail in zip(values, compiled.candidates, strict=True):
        result = "".join(map(chr, compiled.prefix + tail)) + "}}"
        assert json.loads(result) == {'animal "kind"': {"猫": value}}


def test_native_number_position_includes_expected_whitespace():
    field = compile_field(CharacterTokenizer(), JSONField(("count",), tuple(range(10))), "")
    assert "".join(map(chr, field.prefix)).endswith(": ")
    assert ["".join(map(chr, candidate)) for candidate in field.candidates] == list("0123456789")


def test_shared_first_token_is_not_treated_as_a_complete_choice():
    field = compile_field(
        CharacterTokenizer(), JSONField(("label",), ("cat", "catfish", "dog")), ""
    )
    assert field.candidates[0][0] == field.candidates[1][0]
    assert field.candidates[0] != field.candidates[1]
    assert all(candidate[-1] == ord('"') for candidate in field.candidates)


def test_tokenizer_boundary_merges_fail_instead_of_scoring_wrong_position():
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            return super().encode(text.replace("x{", "X"), **kwargs)

    with pytest.raises(ValueError, match="merged tokens"):
        compile_field(MergingTokenizer(), JSONField(("label",), ("a", "b")), "x")
