"""Compile complete native JSON values without answer-code remapping."""

import json
from dataclasses import dataclass

from .core import Choice, DecisionEngine, Independent, Noul, Score


@dataclass(frozen=True)
class JSONField:
    path: tuple[str, ...]
    values: tuple[str | int | bool, ...]


@dataclass(frozen=True)
class FieldTokens:
    prefix: tuple[int, ...]
    candidates: tuple[tuple[int, ...], ...]


def candidate_fields(questions: dict) -> list[JSONField]:
    fields = []
    for name, question in questions.items():
        if isinstance(question, Independent):
            fields.extend(JSONField((name, child), (True, False)) for child in question.criteria)
            continue
        if isinstance(question, Choice):
            values = tuple(question.criteria)
        elif isinstance(question, Score):
            values = tuple(range(len(question.criteria)))
        elif isinstance(question, Noul):
            values = tuple(key == "true" for key in DecisionEngine._criteria(question))
        else:
            raise TypeError(f"Unsupported question: {type(question).__name__}")
        fields.append(JSONField((name,), values))
    return fields


def compile_field(tokenizer, field: JSONField, prompt: str) -> FieldTokens:
    # Whitespace is part of the model's answer context. In particular, scoring
    # digits immediately after ':' instead of ': ' can score a whitespace slot.
    opening = "".join("{" + json.dumps(key, ensure_ascii=False) + ": " for key in field.path)
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sequences = []
    for value in field.values:
        suffix = opening + json.dumps(value, ensure_ascii=False)
        ids = tokenizer.encode(suffix, add_special_tokens=False)
        if tokenizer.encode(prompt + suffix, add_special_tokens=False) != base_ids + ids:
            raise ValueError("Tokenizer merged tokens across the JSON prefix boundary")
        sequences.append(tuple(ids))
    if len(set(sequences)) != len(sequences):
        raise ValueError("Distinct JSON candidates must have distinct token sequences")
    shared = 0
    for position in zip(*sequences):
        if len(set(position)) != 1:
            break
        shared += 1
    if shared == 0 or any(len(sequence) == shared for sequence in sequences):
        raise ValueError("JSON candidates must have a nonempty prefix and distinct complete values")
    return FieldTokens(sequences[0][:shared], tuple(sequence[shared:] for sequence in sequences))
