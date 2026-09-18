"""Compute only each row's requested position in Gemma's KV-sharing layers."""

import mlx.core as mx


def select_answer_mask(mask, positions, width: int, key_length: int):
    """Retain the original causal/window mask at each real answer position."""
    batch = positions.shape[0]
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"Unsupported attention mask: {mask}")
        return (mx.arange(key_length)[None, :] <= (key_length - width + positions)[:, None])[
            :, None, None, :
        ]
    if mask.ndim == 2:
        return mask[positions, :][:, None, None, :]
    if mask.ndim == 4:
        expanded = mx.broadcast_to(mask, (batch, *mask.shape[1:]))
        return expanded[mx.arange(batch), :, positions, :][:, :, None, :]
    raise ValueError(f"Unsupported attention mask shape: {mask.shape}")


def gather_answer_hidden(language, tokens, cache, lengths):
    """Keep the token sequence and all layers, with one late-layer query per field.

    Earlier layers process every suffix token and populate complete KV. Later
    layers use those KVs, so other output positions cannot affect the selected
    answer. Per-row RoPE offsets preserve each answer's original position.
    This adapter targets the pinned mlx-vlm Gemma4 implementation.
    """
    first_shared = language.first_kv_shared_layer_idx
    if not 0 < first_shared < len(language.layers):
        raise ValueError("Answer gathering requires Gemma's KV-sharing layers")
    batch, width = tokens.shape
    rows = mx.arange(batch)
    positions = mx.array(lengths) - 1
    hidden = language.embed_tokens(tokens) * language.embed_scale
    per_layer = language.project_per_layer_inputs(hidden, language.get_per_layer_inputs(tokens))
    cache = cache + [None] * (len(language.layers) - len(cache))
    masks = language._make_masks(hidden, cache)
    intermediates = [(None, None)] * len(language.layers)
    for index, layer in enumerate(language.layers):
        shared, offset = intermediates[language.previous_kvs[index]]
        mask = masks[index]
        pli = per_layer[:, :, index, :]
        if index == first_shared:
            hidden = hidden[rows, positions, :][:, None, :]
        if index >= first_shared:
            if shared is None:
                raise ValueError("A late Gemma layer did not receive shared KV")
            pli = pli[rows, positions, :][:, None, :]
            mask = select_answer_mask(mask, positions, width, shared[0].shape[-2])
            offset = offset + positions
        hidden, shared, offset = layer(
            hidden, mask, cache[index], per_layer_input=pli, shared_kv=shared, offset=offset
        )
        intermediates[index] = (shared, offset)
    return language.norm(hidden)[:, 0, :]
