import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

head_module = pytest.importorskip("gemma_rlcd.decision_head")
backend_module = pytest.importorskip("gemma_rlcd.head_backend")
HeadConfig = head_module.HeadConfig
ParallelDecisionHead = head_module.ParallelDecisionHead
grouped_cross_entropy = head_module.grouped_cross_entropy
DecisionHeadBackend = backend_module.DecisionHeadBackend


def fixture():
    mx.random.seed(7)
    model = ParallelDecisionHead(HeadConfig(input_dims=16, dims=8, heads=2))
    model.eval()
    state = mx.random.normal((1, 7, 16))
    queries = mx.random.normal((5, 4, 16))
    mask = mx.array([[True, True, False, False]] * 5)
    fields = mx.array([0, 0, 1, 1, 1])
    return model, state, queries, mask, fields


def test_all_fields_equal_independent_head_calls():
    model, state, queries, mask, fields = fixture()
    together = model(state, queries, mask, fields, 2)
    separate = mx.concatenate(
        [
            model(state, queries[:2], mask[:2], mx.zeros(2, dtype=mx.int32), 1),
            model(state, queries[2:], mask[2:], mx.zeros(3, dtype=mx.int32), 1),
        ]
    )
    assert bool(mx.allclose(together, separate, atol=1e-5).item())


def test_candidate_and_field_permutations_preserve_scores():
    model, state, queries, mask, fields = fixture()
    original = model(state, queries, mask, fields, 2)
    permutation = mx.array([4, 2, 3, 1, 0])
    reordered = model(state, queries[permutation], mask[permutation], mx.array([0, 0, 0, 1, 1]), 2)
    assert bool(mx.allclose(reordered, original[permutation], atol=1e-5).item())


def test_padding_values_and_other_fields_cannot_change_a_field():
    model, state, queries, mask, fields = fixture()
    original = model(state, queries, mask, fields, 2)
    altered = mx.where(mask[:, :, None], queries, queries + 1000)
    padded = model(state, altered, mask, fields, 2)
    assert bool(mx.allclose(original, padded, atol=1e-5).item())
    altered = queries.at[2:].add(100)
    changed = model(state, altered, mask, fields, 2)
    assert bool(mx.allclose(original[:2], changed[:2], atol=1e-5).item())


def test_head_has_finite_nonzero_training_gradients():
    model, state, queries, mask, fields = fixture()

    def loss(head):
        return grouped_cross_entropy(head(state, queries, mask, fields, 2), (0, 2, 5), (1, 2))

    from mlx.utils import tree_flatten

    value, gradients = nn.value_and_grad(model, loss)(model)
    arrays = [array for _, array in tree_flatten(gradients)]
    mx.eval(value, arrays)
    assert bool(mx.isfinite(value).item())
    assert all(bool(mx.all(mx.isfinite(array)).item()) for array in arrays)
    assert sum(float(mx.sum(mx.abs(array)).item()) for array in arrays) > 0


def test_untrained_backend_cannot_return_decisions():
    backend = DecisionHeadBackend.__new__(DecisionHeadBackend)
    backend.training_metadata = None
    with pytest.raises(RuntimeError, match="Untrained decision head"):
        backend.score_batch(None, [])


def test_grouped_loss_rejects_invalid_target():
    with pytest.raises(ValueError, match="Target index"):
        grouped_cross_entropy(mx.zeros(5), (0, 2, 5), (2, 0))


def test_head_checkpoint_round_trip_preserves_candidate_scores(tmp_path):
    model, state, queries, mask, fields = fixture()
    original = model(state, queries, mask, fields, 2)
    model.save_weights(str(tmp_path / "head.safetensors"))
    restored = ParallelDecisionHead(model.config)
    restored.load_weights(str(tmp_path / "head.safetensors"), strict=True)
    restored.eval()
    actual = restored(state, queries, mask, fields, 2)
    assert bool(mx.array_equal(original, actual).item())


@pytest.mark.parametrize("updates", [None, 0, -1, True, 1.5, "600"])
def test_checkpoint_rejects_invalid_training_update_count(tmp_path, updates):
    backend = DecisionHeadBackend.__new__(DecisionHeadBackend)
    with pytest.raises(ValueError, match="positive integer training updates"):
        backend.save(tmp_path / "invalid-head", {"updates": updates})
    assert not (tmp_path / "invalid-head").exists()
