"""Trainable candidate-conditioned head with one shared state attention memory."""

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class HeadConfig:
    input_dims: int = 1536
    dims: int = 256
    heads: int = 4
    query_layers: int = 1
    decision_layers: int = 2

    def __post_init__(self):
        if min(self.input_dims, self.dims, self.heads, self.query_layers, self.decision_layers) < 1:
            raise ValueError("Head dimensions and layer counts must be positive")
        if self.dims % self.heads or self.dims % 2:
            raise ValueError("Head dimensions must be even and divisible by attention heads")

    def to_dict(self):
        return asdict(self)


class DecisionBlock(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.query_norm = nn.LayerNorm(config.dims)
        self.state_norm = nn.LayerNorm(config.dims)
        self.attention = nn.MultiHeadAttention(config.dims, config.heads)
        self.output_norm = nn.LayerNorm(config.dims)
        self.up = nn.Linear(config.dims, config.dims * 2)
        self.down = nn.Linear(config.dims * 2, config.dims)

    def __call__(self, queries, state):
        memory = self.state_norm(state)
        # Shape [1, all_candidates, D] attends to [1, state_tokens, D]. State
        # projections have batch size one; no per-field state/KV copies exist.
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        return queries + self.down(nn.gelu(self.up(self.output_norm(queries))))


class ParallelDecisionHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        self.query_norm = nn.LayerNorm(config.input_dims)
        self.query_projection = nn.Linear(config.input_dims, config.dims)
        self.query_layers = [
            nn.TransformerEncoderLayer(config.dims, config.heads, config.dims * 2)
            for _ in range(config.query_layers)
        ]
        self.state_norm = nn.LayerNorm(config.input_dims)
        self.state_projection = nn.Linear(config.input_dims, config.dims)
        self.field_context = nn.Linear(config.dims, config.dims, bias=False)
        self.blocks = [DecisionBlock(config) for _ in range(config.decision_layers)]
        self.score_norm = nn.LayerNorm(config.dims)
        self.score = nn.Linear(config.dims, 1, bias=False)

    def encode_queries(self, token_embeddings, token_mask, field_ids, field_count: int):
        query = self.query_projection(self.query_norm(token_embeddings))
        length = query.shape[1]
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        frequencies = mx.exp(-mx.arange(0, self.config.dims, 2) * (9.210340372 / self.config.dims))
        phases = positions * frequencies[None, :]
        positional = mx.stack([mx.sin(phases), mx.cos(phases)], axis=-1).reshape(length, -1)
        query = query + positional.astype(query.dtype)[None, :, :]
        attention_mask = mx.where(token_mask[:, None, None, :], 0.0, -1e9)
        for layer in self.query_layers:
            query = layer(query, attention_mask)
        weights = token_mask[:, :, None].astype(query.dtype)
        query = (query * weights).sum(axis=1) / weights.sum(axis=1)
        totals = (
            mx.zeros((field_count, self.config.dims), dtype=query.dtype).at[field_ids].add(query)
        )
        counts = (
            mx.zeros((field_count, 1), dtype=query.dtype)
            .at[field_ids]
            .add(mx.ones((query.shape[0], 1), dtype=query.dtype))
        )
        # Each candidate sees its own field's complete candidate set. No field
        # attends to another field, and permuting candidate order permutes scores.
        return query + self.field_context((totals / counts)[field_ids])

    def __call__(self, state_features, query_embeddings, token_mask, field_ids, field_count: int):
        query = self.encode_queries(query_embeddings, token_mask, field_ids, field_count)
        state = self.state_projection(self.state_norm(state_features))
        query = query[None, :, :]
        for block in self.blocks:
            query = block(query, state)
        return self.score(self.score_norm(query))[0, :, 0].astype(mx.float32)


def grouped_cross_entropy(logits, offsets: tuple[int, ...], targets: tuple[int, ...]):
    """One proper categorical loss per field, including two-choice propositions."""
    if len(offsets) != len(targets) + 1 or offsets[0] != 0 or offsets[-1] != logits.shape[0]:
        raise ValueError("Targets and candidate offsets do not match logits")
    losses = []
    for start, end, target in zip(offsets[:-1], offsets[1:], targets, strict=True):
        if end <= start or not 0 <= target < end - start:
            raise ValueError("Target index is outside its field's candidate set")
        scores = logits[start:end]
        losses.append(mx.logsumexp(scores) - scores[target])
    return mx.stack(losses).mean()
