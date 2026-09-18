"""Position-table lookup without a dense one-hot matrix multiplication."""

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma4.vision import VisionPatchEmbedder


class IndexedVisionPatchEmbedder(VisionPatchEmbedder):
    def __init__(self, original: VisionPatchEmbedder):
        nn.Module.__init__(self)
        self.hidden_size = original.hidden_size
        self.patch_size = original.patch_size
        self.position_embedding_size = original.position_embedding_size
        self.input_proj = original.input_proj
        self.position_embedding_table = original.position_embedding_table

    def _position_embeddings(self, patch_positions, padding_positions):
        size = self.position_embedding_size
        valid = (patch_positions >= 0) & (patch_positions < size)
        indices = mx.clip(patch_positions, 0, size - 1)
        horizontal = self.position_embedding_table[0, indices[..., 0]]
        vertical = self.position_embedding_table[1, indices[..., 1]]
        horizontal = mx.where(valid[..., 0, None], horizontal, 0.0)
        vertical = mx.where(valid[..., 1, None], vertical, 0.0)
        positions = horizontal + vertical
        return mx.where(padding_positions[..., None], 0.0, positions)


def install_indexed_positions(model):
    original = model.vision_tower.patch_embedder
    if not isinstance(original, IndexedVisionPatchEmbedder):
        model.vision_tower.patch_embedder = IndexedVisionPatchEmbedder(original)
