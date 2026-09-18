from types import SimpleNamespace

import pytest

from gemma_rlcd.cached_backend import CachedMLXBackend, PreparedState
from gemma_rlcd.core import ScoringRequest

mx = pytest.importorskip("mlx.core")
cache_module = pytest.importorskip("mlx_vlm.models.cache")
KVCache = cache_module.KVCache
RotatingKVCache = cache_module.RotatingKVCache


@pytest.mark.parametrize("kind", ["full", "rotating"])
def test_cache_forks_preserve_prefix_and_isolate_rows(kind):
    backend = CachedMLXBackend.__new__(CachedMLXBackend)
    backend.mx = mx
    original = KVCache() if kind == "full" else RotatingKVCache(max_size=8)
    keys = mx.arange(48).reshape(1, 1, 12, 4).astype(mx.float32)
    original.update_and_fetch(keys, keys + 100)
    before_keys, before_values = mx.array(original.keys), mx.array(original.values)
    before_offset = original.offset
    before_index = getattr(original, "_idx", None)
    branch = backend.fork_cache([original], 3)[0]
    updates = mx.array([1000, 2000, 3000], dtype=mx.float32)[:, None, None, None]
    updates = mx.broadcast_to(updates, (3, 1, 2, 4))
    output_keys, _ = branch.update_and_fetch(updates, updates + 50)
    mx.eval(output_keys, original.keys, before_keys)
    assert original.offset == before_offset
    assert getattr(original, "_idx", None) == before_index
    assert bool(mx.array_equal(original.keys, before_keys).item())
    assert bool(mx.array_equal(original.values, before_values).item())
    assert output_keys[:, 0, -1, 0].tolist() == [1000, 2000, 3000]
    assert branch.offset == before_offset + 2


def test_unrecognized_cache_type_fails_explicitly():
    backend = CachedMLXBackend.__new__(CachedMLXBackend)
    backend.mx = mx
    with pytest.raises(TypeError, match="Unsupported cache type"):
        backend.fork_cache([object()], 2)


def test_batched_trim_selects_each_real_answer_before_padding():
    backend = CachedMLXBackend.__new__(CachedMLXBackend)
    backend.mx = mx
    backend.branch_batch_size = 4
    backend.last_stats = {}
    backend.tokenizer = SimpleNamespace(pad_token_id=0)
    backend._ids = {"A": 0, "B": 1}
    backend.fork_cache = lambda prefix, size: []
    calls = []

    def decoder(*, inputs, cache, logits_to_keep):
        calls.append((inputs.shape, logits_to_keep))
        return inputs[:, -logits_to_keep:, None].astype(mx.float32)

    backend.model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=decoder,
            logits_from_hidden=lambda hidden: mx.concatenate([hidden, -hidden], axis=-1),
        )
    )
    prepared = PreparedState({}, [[2, 3, 7, 9], [1, 11], [5, 13, 17]], 100)
    requests = [ScoringRequest("test", ("A", "B"))] * 3
    scores = backend.branches(prepared, [], requests)
    assert calls == [((3, 4), 3)]
    assert [score.logits for score in scores] == [(9, -9), (11, -11), (17, -17)]
    assert [score.input_tokens for score in scores] == [104, 102, 103]
    assert backend.last_stats["branch_batch_sizes"] == [3]


def test_gathered_causal_mask_preserves_each_answer_position():
    from gemma_rlcd.answer_positions import select_answer_mask

    mask = select_answer_mask("causal", mx.array([3, 1]), 4, 10)
    assert mask.shape == (2, 1, 1, 10)
    assert mask.sum(axis=-1).tolist() == [[[10]], [[8]]]


@pytest.mark.parametrize("batched", [False, True])
def test_gathered_window_mask_preserves_existing_mask(batched):
    from gemma_rlcd.answer_positions import select_answer_mask

    mask = (mx.arange(10)[None, :] <= mx.arange(6, 10)[:, None]) & (
        mx.arange(10)[None, :] >= mx.arange(4, 8)[:, None]
    )
    if batched:
        mask = mask[None, None, :, :]
    selected = select_answer_mask(mask, mx.array([3, 1]), 4, 10)
    assert selected.shape == (2, 1, 1, 10)
    assert selected[0, 0, 0].tolist() == [False] * 7 + [True] * 3
    assert selected[1, 0, 0].tolist() == [False] * 5 + [True] * 3 + [False] * 2
