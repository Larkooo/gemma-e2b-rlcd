import math

import pytest

from gemma_decisions.calibration import brier_loss, evaluate, fit_temperature, log_loss
from gemma_decisions.core import softmax


def test_proper_scores_reward_truthful_probabilities_in_expectation():
    def expected_brier(p):
        return 0.7 * brier_loss([p, 1 - p], 0) + 0.3 * brier_loss([p, 1 - p], 1)

    assert expected_brier(0.7) < expected_brier(0.5)
    assert expected_brier(0.7) < expected_brier(0.99)
    assert log_loss([1.0, 0.0], 1) == math.inf


def test_metrics_and_confidence_bins():
    result = evaluate([[1, 0], [0.25, 0.75]], [0, 0])
    assert result["accuracy"] == 0.5
    assert result["brier"] == pytest.approx(0.5625)
    assert result["top_label_ece"] == pytest.approx(0.375)


def test_temperature_recovers_known_calibration_on_separate_outcomes():
    # Predictions assign .99 to class 0, while its known frequency is .75.
    logits = [[math.log(99), 0]] * 100
    calibration_targets = [0] * 75 + [1] * 25
    temperature = fit_temperature(logits, calibration_targets)
    assert softmax(logits[0], temperature)[0] == pytest.approx(0.75, abs=1e-5)
    held_out_targets = [1] * 10 + [0] * 30
    before = evaluate([softmax(row) for row in logits[:40]], held_out_targets)
    after = evaluate([softmax(row, temperature) for row in logits[:40]], held_out_targets)
    assert after["nll"] < before["nll"]
    assert after["brier"] < before["brier"]


@pytest.mark.parametrize(
    "probabilities,target", [([0.8, 0.8], 0), ([0.5, 0.5], -1), ([math.nan, 0], 0)]
)
def test_bad_predictions_rejected(probabilities, target):
    with pytest.raises(ValueError):
        log_loss(probabilities, target)
