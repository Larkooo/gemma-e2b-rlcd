import math
from types import SimpleNamespace

import pytest

from gemma_rlcd.json_backend import JSONMLXBackend
from gemma_rlcd.json_scoring import FieldTokens

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_complete_candidate_likelihood_uses_suffix_and_ignores_padding(batch_size):
    backend = JSONMLXBackend.__new__(JSONMLXBackend)
    backend.mx = mx
    backend.branch_batch_size = batch_size
    backend.tokenizer = SimpleNamespace(pad_token_id=0)
    backend.fork_cache = lambda cache, size: []
    table = [[-30.0] * 16 for _ in range(16)]
    table[9][1], table[9][4] = math.log(0.9), math.log(0.1)
    table[1][2], table[1][3] = math.log(0.1), math.log(0.9)
    transition = mx.array(table)
    backend.model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=lambda *, inputs, cache, logits_to_keep: inputs[:, -logits_to_keep:, None],
            logits_from_hidden=lambda hidden: transition[hidden[..., 0]],
        )
    )
    fields = [(0, FieldTokens((9,), ((1, 2), (1, 3), (4,))))]
    streamed = []
    results, batches = backend._sequence_scores([], 100, fields, on_scores=streamed.append)
    likelihoods = [math.exp(value) for value in results[0].logits]
    assert likelihoods == pytest.approx([0.09, 0.81, 0.1], abs=1e-6)
    assert results[0].allowed_token_mass == pytest.approx(1, abs=1e-6)
    assert sum(batches) == 3
    assert max(batches) <= batch_size
    assert streamed == [[(0, results[0])]]
