"""Experimental frozen Gemma state encoder and small parallel decision head.

Untrained heads expose timing probes only. Scoring requires a trained checkpoint.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .cached_backend import CachedMLXBackend
from .core import ScoringRequest, State, TokenScores
from .decision_head import HeadConfig, ParallelDecisionHead
from .mlx_backend import MLXBackend


def validate_training_metadata(training: dict):
    updates = training.get("updates")
    if type(updates) is not int or updates < 1:
        raise ValueError("A trained checkpoint must record positive integer training updates")


@dataclass
class HeadInputs:
    state_features: object
    query_embeddings: object
    token_mask: object
    field_ids: object
    offsets: tuple[int, ...]
    input_tokens: int

    def arrays(self):
        return self.state_features, self.query_embeddings, self.token_mask, self.field_ids


class DecisionHeadBackend(CachedMLXBackend):
    probability_source = "candidate_conditioned_decision_head"

    def __init__(
        self,
        model: str,
        *,
        state_layers: int = 35,
        compute_dtype: str = "float32",
        head_config: HeadConfig | None = None,
        checkpoint: str | None = None,
        max_input_tokens: int = 8192,
        max_query_tokens: int = 512,
        max_padded_query_tokens: int = 16384,
        image_soft_tokens: int = 280,
        video_max_frames: int = 32,
    ):
        MLXBackend.__init__(self, model, max_input_tokens=max_input_tokens)
        mx = self.mx
        total_layers = len(self.model.language_model.model.layers)
        if type(state_layers) is not int or not 1 <= state_layers <= total_layers:
            raise ValueError(f"state_layers must be an integer from 1 to {total_layers}")
        if compute_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("Unknown compute dtype")
        self.model.set_dtype(getattr(mx, compute_dtype))
        # The optimizer only owns self.head; extracted backbone features are
        # stop_gradient values. Avoid recursive freeze on native audio helpers.
        self.state_layers = state_layers
        self.compute_dtype = compute_dtype
        self.max_query_tokens = max_query_tokens
        self.max_padded_query_tokens = max_padded_query_tokens
        if image_soft_tokens not in {70, 140, 280}:
            raise ValueError("image_soft_tokens must be 70, 140, or 280")
        if type(video_max_frames) is not int or not 2 <= video_max_frames <= 32:
            raise ValueError("video_max_frames must be an integer from 2 to 32")
        self.image_soft_tokens = image_soft_tokens
        self.video_max_frames = video_max_frames
        self.processor.image_processor.max_soft_tokens = image_soft_tokens
        self.processor.video_processor.num_frames = video_max_frames
        self.last_stats = {}
        self.training_metadata = None
        self.head = ParallelDecisionHead(head_config or HeadConfig())
        if self.head.config.input_dims != self.model.config.text_config.hidden_size:
            raise ValueError("Decision head dimensions must match the Gemma state encoder")
        if checkpoint is not None:
            directory = Path(checkpoint)
            manifest = json.loads((directory / "head.json").read_text())
            expected = {
                "model": self.model_id,
                "state_layers": state_layers,
                "compute_dtype": compute_dtype,
                "head_config": self.head.config.to_dict(),
                "image_soft_tokens": image_soft_tokens,
                "video_max_frames": video_max_frames,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "Checkpoint does not match backbone, depth, precision, or head config"
                )
            validate_training_metadata(manifest.get("training", {}))
            self.head.load_weights(str(directory / "head.safetensors"), strict=True)
            self.training_metadata = manifest["training"]
        self.head.eval()
        mx.eval(self.model.parameters(), self.head.parameters())

    def _state_features(self, inputs):
        """Exact prefix of the installed native Gemma decoder, without output logits.

        Layer count is an explicit architectural experiment, not a claim that
        truncating the pretrained decoder preserves decision quality.
        """
        features = self.model.get_input_embeddings(**inputs)
        language = self.model.language_model.model
        hidden = features.inputs_embeds
        pli = features.per_layer_inputs
        if pli is None:
            pli = language.get_per_layer_inputs(inputs["input_ids"])
        pli = language.project_per_layer_inputs(hidden, pli)
        masks = language._make_masks(
            hidden,
            [None] * len(language.layers),
            inputs.get("mm_token_type_ids", inputs.get("token_type_ids")),
        )
        intermediates = [(None, None)] * len(language.layers)
        for index, layer in enumerate(language.layers[: self.state_layers]):
            shared, offset = intermediates[language.previous_kvs[index]]
            hidden, shared, offset = layer(
                hidden,
                masks[index],
                per_layer_input=pli[:, :, index, :],
                shared_kv=shared,
                offset=offset,
            )
            intermediates[index] = (shared, offset)
        return self.mx.stop_gradient(hidden.astype(self.mx.float32))

    def _queries(self, requests: Sequence[ScoringRequest]):
        if not requests or len(requests) > 64:
            raise ValueError("The decision head accepts 1 to 64 fields in one batch")
        rows, field_ids, offsets = [], [], [0]
        for field, request in enumerate(requests):
            if request.instructions is None or request.criteria is None:
                raise ValueError("The decision head requires structured instructions and criteria")
            if len(request.criteria) != len(request.symbols) or len(request.criteria) < 2:
                raise ValueError("Candidate descriptions must match all answer codes")
            for name, description in request.criteria:
                text = json.dumps(
                    {"question": request.instructions, "name": name, "meaning": description},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if not tokens or len(tokens) > self.max_query_tokens:
                    raise ValueError("Candidate query exceeds token limit; no truncation applied")
                rows.append(tokens)
                field_ids.append(field)
            offsets.append(len(rows))
        width = max(map(len, rows))
        if width * len(rows) > self.max_padded_query_tokens:
            raise ValueError("Padded query batch exceeds the memory budget; no fields dropped")
        mx = self.mx
        ids = mx.array([row + [self.tokenizer.pad_token_id] * (width - len(row)) for row in rows])
        mask = mx.arange(width)[None, :] < mx.array([len(row) for row in rows])[:, None]
        embeddings = self.model.language_model.model.embed_tokens(ids).astype(mx.float32)
        return (
            mx.stop_gradient(embeddings),
            mask,
            mx.array(field_ids),
            tuple(offsets),
            sum(map(len, rows)),
        )

    def encode(self, state: State, requests: Sequence[ScoringRequest]) -> HeadInputs:
        started = time.perf_counter()
        # A constant suffix requests only state preprocessing from the reference
        # renderer. Field text never enters the large Gemma decoder on this path.
        prepared = super().prepare(state, [ScoringRequest("Read the evidence.", ("A", "B"))])
        processed = time.perf_counter()
        state_features = self._state_features(prepared.inputs)
        self.mx.eval(state_features)
        encoded = time.perf_counter()
        query, mask, fields, offsets, query_tokens = self._queries(requests)
        self.mx.eval(query, mask, fields)
        queried = time.perf_counter()
        self.last_stats = {
            "execution": "shared_state_parallel_decision_head",
            "state_encodes": 1,
            "state_tokens": prepared.prefix_tokens,
            "state_layers": self.state_layers,
            "compute_dtype": self.compute_dtype,
            "field_count": len(requests),
            "candidate_count": offsets[-1],
            "query_tokens": query_tokens,
            "padded_query_tokens": int(mask.size),
            "state_memory_batch_size": 1,
            "image_soft_tokens": self.image_soft_tokens,
            "video_max_frames": self.video_max_frames,
            "video_frames": prepared.inputs.get("num_frames_per_video", []),
            "preprocess_seconds": processed - started,
            "state_encode_seconds": encoded - processed,
            "query_embed_seconds": queried - encoded,
            "trained": self.training_metadata is not None,
        }
        return HeadInputs(
            state_features, query, mask, fields, offsets, prepared.prefix_tokens + query_tokens
        )

    def forward(self, inputs: HeadInputs):
        return self.head(*inputs.arrays(), len(inputs.offsets) - 1)

    def probe(self, state: State, requests: Sequence[ScoringRequest]) -> dict:
        """Execute every stage; never present random weights as usable predictions."""
        started = time.perf_counter()
        inputs = self.encode(state, requests)
        head_started = time.perf_counter()
        logits = self.forward(inputs)
        self.mx.eval(logits)
        if not bool(self.mx.all(self.mx.isfinite(logits)).item()):
            raise ValueError("Decision head produced non-finite logits")
        self.last_stats["head_seconds"] = time.perf_counter() - head_started
        self.last_stats["total_seconds"] = time.perf_counter() - started
        self.last_stats["status"] = (
            "runtime_probe_only" if self.training_metadata is None else "trained_runtime_probe"
        )
        return dict(self.last_stats)

    def score_batch(self, state: State, requests: Sequence[ScoringRequest]) -> list[TokenScores]:
        if self.training_metadata is None:
            raise RuntimeError(
                "Untrained decision head: use probe() for timing or load a trained checkpoint"
            )
        started = time.perf_counter()
        inputs = self.encode(state, requests)
        head_started = time.perf_counter()
        logits = self.forward(inputs)
        self.mx.eval(logits)
        values = logits.tolist()
        self.last_stats["head_seconds"] = time.perf_counter() - head_started
        self.last_stats["total_seconds"] = time.perf_counter() - started
        return [
            TokenScores(tuple(values[start:end]), None, inputs.input_tokens)
            for start, end in zip(inputs.offsets[:-1], inputs.offsets[1:], strict=True)
        ]

    def save(self, directory: Path, training: dict):
        validate_training_metadata(training)
        directory.mkdir(parents=True, exist_ok=True)
        self.head.save_weights(str(directory / "head.safetensors"))
        (directory / "head.json").write_text(
            json.dumps(
                {
                    "model": self.model_id,
                    "state_layers": self.state_layers,
                    "compute_dtype": self.compute_dtype,
                    "head_config": self.head.config.to_dict(),
                    "image_soft_tokens": self.image_soft_tokens,
                    "video_max_frames": self.video_max_frames,
                    "training": training,
                    "calibration_status": "unvalidated",
                },
                indent=2,
            )
            + "\n"
        )
