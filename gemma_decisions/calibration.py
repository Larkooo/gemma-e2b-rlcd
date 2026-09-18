"""Proper scoring rules and a calibration-set-only temperature baseline.

These are supervised objectives, not a reproduction of TypeSafe's RLCD algorithm.
"""

import math
from collections.abc import Sequence

from .core import softmax


def validate_distribution(values: Sequence[float]) -> None:
    if len(values) < 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("A distribution needs at least two finite probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Choice probabilities must sum to one")


def validate_target(target: int, size: int) -> None:
    if type(target) is not int or not 0 <= target < size:
        raise ValueError("Target must be an in-range integer class index")


def brier_loss(probabilities: Sequence[float], target: int) -> float:
    validate_distribution(probabilities)
    validate_target(target, len(probabilities))
    return sum((p - int(i == target)) ** 2 for i, p in enumerate(probabilities))


def log_loss(probabilities: Sequence[float], target: int) -> float:
    validate_distribution(probabilities)
    validate_target(target, len(probabilities))
    return -math.log(probabilities[target]) if probabilities[target] > 0 else math.inf


def evaluate(rows: Sequence[Sequence[float]], targets: Sequence[int], bins: int = 10) -> dict:
    if not rows or len(rows) != len(targets) or type(bins) is not int or bins < 1:
        raise ValueError("Need matched nonempty predictions/targets and positive integer bins")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    correct = 0
    nll = 0.0
    brier = 0.0
    for probabilities, target in zip(rows, targets, strict=True):
        nll += log_loss(probabilities, target)
        brier += brier_loss(probabilities, target)
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = probabilities[winner]
        hit = winner == target
        correct += hit
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, hit))
    count = len(rows)
    reliability = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(hit for _, hit in bucket) / len(bucket)
        ece += len(bucket) / count * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "bin": index,
                "count": len(bucket),
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "count": count,
        "accuracy": correct / count,
        "nll": nll / count,
        "brier": brier / count,
        "top_label_ece": ece,
        "reliability": reliability,
    }


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Fit on a dedicated calibration split, never on the final evaluation split."""
    if not logits or len(logits) != len(targets):
        raise ValueError("Need matched nonempty logits and targets")
    for row, target in zip(logits, targets, strict=True):
        softmax(row)
        if len(row) < 2:
            raise ValueError("Need at least two class logits")
        validate_target(target, len(row))

    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        total = 0.0
        for row, target in zip(logits, targets, strict=True):
            shifted = [(value - max(row)) / temperature for value in row]
            total += math.log(sum(math.exp(value) for value in shifted)) - shifted[target]
        return total / len(logits)

    lo, hi = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(80):
        if loss(left) <= loss(right):
            hi, right = right, left
            left = hi - ratio * (hi - lo)
        else:
            lo, left = left, right
            right = lo + ratio * (hi - lo)
    candidates = [0.0, math.log(0.05), math.log(20.0), (lo + hi) / 2]
    return math.exp(min(candidates, key=loss))
