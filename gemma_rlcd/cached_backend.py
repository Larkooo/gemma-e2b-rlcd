"""One multimodal state prefill, then GPU-batched independent question branches."""

import json
import time
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass

from .core import ScoringRequest, State, TokenScores
from .mlx_backend import MLXBackend, audio_paths


@dataclass
class PreparedState:
    inputs: dict
    suffixes: list[list[int]]
    prefix_tokens: int


class CachedMLXBackend(MLXBackend):
    def __init__(
        self,
        model: str,
        *,
        branch_batch_size: int = 8,
        max_input_tokens: int = 8192,
        answer_mode: str = "gather",
    ):
        if type(branch_batch_size) is not int or not 1 <= branch_batch_size <= 64:
            raise ValueError("branch_batch_size must be an integer between 1 and 64")
        super().__init__(model, max_input_tokens=max_input_tokens)
        # Keep packed 4-bit integer weights; promote floating scales/norms and
        # activations to reduce shape-dependent probability drift across branches.
        self.model.set_dtype(self.mx.float32)
        self.mx.eval(self.model.parameters())
        self.branch_batch_size = branch_batch_size
        if answer_mode not in {"gather", "tail"}:
            raise ValueError("answer_mode must be gather or tail")
        self.answer_mode = answer_mode
        self.last_stats: dict = {}

    def prefix_task(self) -> str:
        return "\n\nDecision task:\n"

    def prepare(self, state: State, requests: Sequence[ScoringRequest]) -> PreparedState:
        from mlx_vlm.utils import prepare_inputs

        if not requests:
            raise ValueError("At least one scoring request is required")
        marker = "__GEMMA_DECISION_BRANCH_BOUNDARY__"
        with audio_paths(state) as audios:
            content = [{"type": "image"} for _ in state.images]
            content += [{"type": "video"} for _ in state.videos]
            content.append({"type": "text", "text": json.dumps({"state": state.text})})
            content += [{"type": "audio"} for _ in audios]
            content.append({"type": "text", "text": self.prefix_task() + marker})
            messages = [
                {
                    "role": "system",
                    "content": "Evaluate the supplied evidence. Follow the decision task after the state. Return only its allowed answer code.",
                },
                {"role": "user", "content": content},
            ]
            # Render explicit content order directly. Some convenience wrappers
            # move audio after all text, which would put it outside the shared prefix.
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix, boundary, trailer = formatted.rpartition(marker)
            if not boundary:
                raise ValueError("Chat template lost the branch boundary")
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            suffixes = []
            for request in requests:
                suffix_text = request.prompt + trailer
                suffix = self.tokenizer.encode(suffix_text, add_special_tokens=False)
                combined = self.tokenizer.encode(prefix + suffix_text, add_special_tokens=False)
                if combined != prefix_ids + suffix:
                    raise ValueError("Tokenizer merged tokens across the shared-prefix boundary")
                suffixes.append(suffix)
            inputs = dict(
                prepare_inputs(
                    self.processor,
                    prompts=prefix,
                    images=list(state.images) or None,
                    audio=audios or None,
                    videos=list(state.videos) or None,
                    fps=1.0,
                    max_frames=60,
                )
            )
        inputs.pop("attention_mask", None)
        length = int(inputs["input_ids"].shape[-1])
        if length + max(map(len, suffixes)) > self.max_input_tokens:
            raise ValueError(
                "State plus question exceeds the input token limit; no truncation applied"
            )
        return PreparedState(inputs, suffixes, length)

    def prefill(self, prepared: PreparedState):
        mx = self.mx
        inputs = prepared.inputs
        features = self.model.get_input_embeddings(**inputs)
        cache = self.model.make_cache()
        hidden = self.model.language_model.model(
            inputs=inputs["input_ids"],
            inputs_embeds=features.inputs_embeds,
            per_layer_inputs=features.per_layer_inputs,
            mm_token_type_ids=inputs.get("mm_token_type_ids", inputs.get("token_type_ids")),
            cache=cache,
            logits_to_keep=1,
        )
        mx.eval(hidden, *[array for entry in cache for array in entry.state])
        return cache

    def fork_cache(self, prefix_cache, batch_size: int):
        """Independent mutable cache objects; prefix computations are reused.

        This backend replicates KV storage across batch rows. It does not yet
        provide a paged-attention kernel with zero-copy shared prefix storage.
        """
        from mlx_vlm.models.cache import KVCache, RotatingKVCache

        result = []
        for original in prefix_cache:
            if type(original) not in (KVCache, RotatingKVCache):
                raise TypeError(f"Unsupported cache type: {type(original).__name__}")
            branch = copy(original)
            branch.keys = self.mx.repeat(self.mx.array(original.keys), batch_size, axis=0)
            branch.values = self.mx.repeat(self.mx.array(original.values), batch_size, axis=0)
            result.append(branch)
        return result

    def _extract(
        self, logits, requests: Sequence[ScoringRequest], lengths: Sequence[int]
    ) -> list[TokenScores]:
        mx = self.mx
        scores = []
        for row, request in enumerate(requests):
            selected = logits[row, mx.array([self._ids[symbol] for symbol in request.symbols])]
            mass = mx.exp(mx.logsumexp(selected) - mx.logsumexp(logits[row]))
            mx.eval(selected, mass)
            scores.append(
                TokenScores(tuple(selected.tolist()), min(1.0, float(mass.item())), lengths[row])
            )
        return scores

    def branches(
        self,
        prepared: PreparedState,
        prefix_cache,
        requests: Sequence[ScoringRequest],
        on_batch=None,
    ) -> list[TokenScores]:
        mx = self.mx
        scores = []
        batch_sizes = []
        tail_lengths = []
        for start in range(0, len(requests), self.branch_batch_size):
            suffixes = prepared.suffixes[start : start + self.branch_batch_size]
            batch = requests[start : start + self.branch_batch_size]
            lengths = [len(suffix) for suffix in suffixes]
            width = max(lengths)
            tokens = mx.array(
                [
                    suffix + [self.tokenizer.pad_token_id] * (width - len(suffix))
                    for suffix in suffixes
                ]
            )
            cache = self.fork_cache(prefix_cache, len(batch))
            # Retain each row's real answer before padding. The tail mode keeps
            # the previous implementation available for paired comparisons.
            tail_length = width - min(lengths) + 1
            if getattr(self, "answer_mode", "tail") == "gather":
                from .answer_positions import gather_answer_hidden

                final_hidden = gather_answer_hidden(
                    self.model.language_model.model, tokens, cache, lengths
                )
                tail_length = 1
            else:
                hidden = self.model.language_model.model(
                    inputs=tokens, cache=cache, logits_to_keep=tail_length
                )
                # Right padding cannot affect an earlier answer under causal attention.
                removed = width - hidden.shape[1]
                final_hidden = hidden[mx.arange(len(batch)), mx.array(lengths) - 1 - removed, :]
            logits = self.model.language_model.logits_from_hidden(final_hidden[:, None, :])[
                :, 0, :
            ].astype(mx.float32)
            mx.eval(logits)
            completed = self._extract(logits, batch, [prepared.prefix_tokens + n for n in lengths])
            scores.extend(completed)
            if on_batch is not None:
                on_batch(start, completed)
            batch_sizes.append(len(batch))
            tail_lengths.append(tail_length)
        self.last_stats["branch_batch_sizes"] = batch_sizes
        self.last_stats["answer_tail_tokens"] = tail_lengths
        return scores

    def score_batch(self, state: State, requests: Sequence[ScoringRequest]) -> list[TokenScores]:
        started = time.perf_counter()
        prepared = self.prepare(state, requests)
        processed = time.perf_counter()
        cache = self.prefill(prepared)
        prefilled = time.perf_counter()
        self.last_stats = {
            "execution": "shared_prefix_gpu_batched_branches",
            "prefix_prefills": 1,
            "prefix_tokens": prepared.prefix_tokens,
            "question_suffix_tokens": [len(suffix) for suffix in prepared.suffixes],
            "preprocess_seconds": processed - started,
            "prefill_seconds": prefilled - processed,
            "kv_storage": "replicated_per_batch_row",
            "compute_dtype": "float32",
            "answer_mode": self.answer_mode,
        }
        results = self.branches(prepared, cache, requests)
        self.last_stats["branch_seconds"] = time.perf_counter() - prefilled
        return results

    def score(self, state: State, prompt: str, symbols: Sequence[str]) -> TokenScores:
        return self.score_batch(state, [ScoringRequest(prompt, tuple(symbols))])[0]

    def uncached(
        self, prepared: PreparedState, requests: Sequence[ScoringRequest]
    ) -> list[TokenScores]:
        """Reference: identical token sequence, full independent model forward per field."""
        mx = self.mx
        results = []
        for request, suffix in zip(requests, prepared.suffixes, strict=True):
            inputs = dict(prepared.inputs)
            inputs["input_ids"] = mx.concatenate([inputs["input_ids"], mx.array([suffix])], axis=1)
            for key in ("mm_token_type_ids", "token_type_ids"):
                if key in inputs:
                    inputs[key] = mx.concatenate(
                        [inputs[key], mx.zeros((1, len(suffix)), dtype=inputs[key].dtype)], axis=1
                    )
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :].astype(mx.float32)
            results.extend(self._extract(logits, [request], [prepared.prefix_tokens + len(suffix)]))
        return results
