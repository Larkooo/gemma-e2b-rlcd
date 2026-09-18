"""Score native JSON values against the same multimodal prompt as normal Gemma."""

import math
import time

from .cached_backend import CachedMLXBackend, PreparedState
from .comparison import prepare_generation
from .core import ScoringRequest, TokenScores
from .json_scoring import candidate_fields, compile_field


class JSONMLXBackend(CachedMLXBackend):
    probability_source = "restricted_json_value_likelihoods"

    def _sequence_scores(self, prefix_cache, prefix_tokens, fields):
        """Teacher-force every complete candidate in bounded GPU batches.

        Score all tokens, including string terminators. Shared first tokens and
        prefix labels never fall back to an arbitrary choice or synthetic confidence.
        """
        mx = self.mx
        jobs = [
            (field_index, choice, field.prefix, candidate)
            for field_index, field in fields
            for choice, candidate in enumerate(field.candidates)
        ]
        values = {index: [None] * len(field.candidates) for index, field in fields}
        batches = []
        for start in range(0, len(jobs), self.branch_batch_size):
            batch = jobs[start : start + self.branch_batch_size]
            suffixes = [prefix + candidate[:-1] for _, _, prefix, candidate in batch]
            width = max(map(len, suffixes))
            tokens = mx.array(
                [
                    list(suffix) + [self.tokenizer.pad_token_id] * (width - len(suffix))
                    for suffix in suffixes
                ]
            )
            tail = width - min(len(prefix) for _, _, prefix, _ in batch) + 1
            hidden = self.model.language_model.model(
                inputs=tokens, cache=self.fork_cache(prefix_cache, len(batch)), logits_to_keep=tail
            )
            totals = []
            for row, (_, _, prefix, candidate) in enumerate(batch):
                position = len(prefix) - 1 - (width - hidden.shape[1])
                states = hidden[row : row + 1, position : position + len(candidate), :]
                logits = self.model.language_model.logits_from_hidden(states)[0].astype(mx.float32)
                log_probs = logits[mx.arange(len(candidate)), mx.array(candidate)] - mx.logsumexp(
                    logits, axis=-1
                )
                totals.append(mx.sum(log_probs))
            mx.eval(*totals)
            for (field_index, choice, _, _), total in zip(batch, totals, strict=True):
                values[field_index][choice] = float(total.item())
            batches.append(len(batch))
        output = {}
        for index, field in fields:
            scores = tuple(values[index])
            mass = min(1.0, sum(math.exp(value) for value in scores))
            length = prefix_tokens + len(field.prefix) + max(map(len, field.candidates)) - 1
            output[index] = TokenScores(scores, mass, length)
        return output, batches

    def score_questions(self, state, questions):
        started = time.perf_counter()
        prompt, inputs = prepare_generation(self, state, questions)
        fields = [
            compile_field(self.tokenizer, field, prompt) for field in candidate_fields(questions)
        ]
        prefix_tokens = int(inputs["input_ids"].shape[1])
        if any(
            prefix_tokens + len(field.prefix) + max(map(len, field.candidates))
            > self.max_input_tokens
            for field in fields
        ):
            raise ValueError(
                "State, questions, and JSON candidate exceed the token limit; no truncation applied"
            )
        prepared = PreparedState(inputs, [], prefix_tokens)
        processed = time.perf_counter()
        prefix_cache = self.prefill(prepared)
        prefilled = time.perf_counter()
        self.last_stats = {}
        scores = [None] * len(fields)
        single = [
            (index, field)
            for index, field in enumerate(fields)
            if all(len(candidate) == 1 for candidate in field.candidates)
        ]
        batches = []
        if single:
            requests = []
            for _, field in single:
                symbols = tuple(f"json-token:{candidate[0]}" for candidate in field.candidates)
                self._ids.update(
                    {
                        symbol: candidate[0]
                        for symbol, candidate in zip(symbols, field.candidates, strict=True)
                    }
                )
                requests.append(ScoringRequest("", symbols))
            simple = PreparedState(
                inputs, [list(field.prefix) for _, field in single], prefix_tokens
            )
            results = self.branches(simple, prefix_cache, requests)
            batches.extend(self.last_stats["branch_batch_sizes"])
            for (index, _), result in zip(single, results, strict=True):
                scores[index] = result
        multiple = [
            (index, field)
            for index, field in enumerate(fields)
            if any(len(candidate) != 1 for candidate in field.candidates)
        ]
        candidate_batches = []
        if multiple:
            results, candidate_batches = self._sequence_scores(
                prefix_cache, prefix_tokens, multiple
            )
            for index, result in results.items():
                scores[index] = result
        self.last_stats = {
            "execution": "shared_json_prefix_gpu_batched_fields",
            "prefix_prefills": 1,
            "prefix_tokens": prefix_tokens,
            "primitive_fields": len(fields),
            "question_suffix_tokens": [len(field.prefix) for field in fields],
            "candidate_token_lengths": [
                [len(candidate) for candidate in field.candidates] for field in fields
            ],
            "preprocess_seconds": processed - started,
            "prefill_seconds": prefilled - processed,
            "branch_seconds": time.perf_counter() - prefilled,
            "branch_batch_sizes": batches,
            "candidate_batch_sizes": candidate_batches,
            "compute_dtype": "float32",
            "kv_storage": "replicated_per_batch_row",
            "conditioning": "shared complete question schema; no previous field answers",
        }
        return scores
