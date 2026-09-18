import pytest

mx = pytest.importorskip("mlx.core")
vision = pytest.importorskip("gemma_rlcd.vision")
config_module = pytest.importorskip("mlx_vlm.models.gemma4.config")


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.bfloat16])
def test_indexed_positions_match_dense_lookup_including_padding(dtype):
    config = config_module.VisionConfig(hidden_size=8, position_embedding_size=16)
    original = vision.VisionPatchEmbedder(config)
    original.position_embedding_table = mx.random.normal((2, 16, 8)).astype(dtype)
    indexed = vision.IndexedVisionPatchEmbedder(original)
    positions = mx.array([[[2, 3], [0, 0], [-1, -1], [16, 2], [4, 5]]])
    padding = mx.array([[False, False, True, False, True]])
    expected = original._position_embeddings(positions, padding)
    actual = indexed._position_embeddings(positions, padding)
    # The M5 dense float32 matmul can round table entries internally; gather
    # returns their stored values directly. Half/bfloat paths match exactly.
    if dtype == mx.float32:
        assert bool(mx.allclose(actual, expected, atol=2e-3, rtol=2e-3).item())
    else:
        assert bool(mx.array_equal(actual, expected).item())
