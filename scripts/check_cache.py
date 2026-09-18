"""Compare cached GPU batches to identical uncached multimodal token sequences."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from gemma_rlcd import Choice, DecisionEngine, Noul, Score, State
from gemma_rlcd.cached_backend import CachedMLXBackend
from gemma_rlcd.core import ScoringRequest, decision_prompt, softmax


def compare(left, right):
    logit_delta = max(
        abs(a - b)
        for x, y in zip(left, right, strict=True)
        for a, b in zip(x.logits, y.logits, strict=True)
    )
    probability_delta = max(
        abs(a - b)
        for x, y in zip(left, right, strict=True)
        for a, b in zip(softmax(x.logits), softmax(y.logits), strict=True)
    )
    winners_match = all(
        max(range(len(x.logits)), key=x.logits.__getitem__)
        == max(range(len(y.logits)), key=y.logits.__getitem__)
        for x, y in zip(left, right, strict=True)
    )
    return {
        "max_logit_delta": logit_delta,
        "max_probability_delta": probability_delta,
        "all_winners_match": winners_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    backend = CachedMLXBackend(args.model, branch_batch_size=4)
    mx = backend.mx
    media = args.media.resolve()
    questions = [
        Choice(
            "Which animal is mentioned or visible?",
            {"cat": "A cat", "dog": "A dog", "other": "Neither"},
        ),
        Noul("Is a dog mentioned?"),
        Score("How much red is visible?", ["No red", "Some red", "Mostly red"]),
        Noul(
            "Does the supplied evidence contain any mention of a sofa? Evaluate the complete supplied state."
        ),
    ]
    requests = []
    for question in questions:
        criteria = DecisionEngine._criteria(question)
        symbols = tuple(backend.symbols(len(criteria)))
        requests.append(
            ScoringRequest(decision_prompt(question.instructions, criteria, symbols), symbols)
        )
    states = {
        "text": State(text="A cat sleeps on a sofa. No dogs are present."),
        "image": State(images=(str(media / "red.png"),)),
        "speech": State(audio=(str(media / "dog.wav"),)),
        "video": State(videos=(str(media / "red-blue.mp4"),)),
        "video_speech": State(videos=(str(media / "red-blue-speech.mp4"),)),
        "long_text": State(
            text=("Background: the room has a table, a window, and a lamp. " * 80)
            + "A cat sleeps on a sofa. No dogs are present."
        ),
    }
    report = {
        "model": args.model,
        "compute_dtype": "float32",
        "status": "cache_numerics_and_timing_probe_not_quality_benchmark",
        "results": [],
    }
    for name, state in states.items():
        try:
            prepared = backend.prepare(state, requests)
            prefix = backend.prefill(prepared)
            before = [(c.offset, mx.array(c.keys), mx.array(c.values)) for c in prefix]
            mx.eval(*[a for _, k, v in before for a in (k, v)])
            reference = backend.uncached(prepared, requests)
            backend.branch_batch_size = 1
            serial = backend.branches(prepared, prefix, requests)
            backend.branch_batch_size = 4
            batched = backend.branches(prepared, prefix, requests)
            reverse_prepared = replace(prepared, suffixes=list(reversed(prepared.suffixes)))
            reversed_scores = list(
                reversed(backend.branches(reverse_prepared, prefix, list(reversed(requests))))
            )
            unchanged = all(
                c.offset == offset
                and bool(mx.array_equal(c.keys, k).item())
                and bool(mx.array_equal(c.values, v).item())
                for c, (offset, k, v) in zip(prefix, before, strict=True)
            )
            timing = {"uncached": [], "cached_batched": []}
            # All shapes have been exercised. Alternate order to reduce order bias.
            for repeat in range(3):
                for method in (
                    ("uncached", "cached_batched")
                    if repeat % 2 == 0
                    else ("cached_batched", "uncached")
                ):
                    start = time.perf_counter()
                    if method == "uncached":
                        backend.uncached(prepared, requests)
                    else:
                        current_prefix = backend.prefill(prepared)
                        backend.branches(prepared, current_prefix, requests)
                    timing[method].append(time.perf_counter() - start)
            medians = {key: statistics.median(values) for key, values in timing.items()}
            record = {
                "state": name,
                "prefix_tokens": prepared.prefix_tokens,
                "suffix_lengths": [len(x) for x in prepared.suffixes],
                "cached_serial_vs_uncached": compare(reference, serial),
                "cached_batch_vs_uncached": compare(reference, batched),
                "reversed_batch_vs_original": compare(batched, reversed_scores),
                "prefix_unchanged": unchanged,
                "branch_batch_sizes": backend.last_stats["branch_batch_sizes"],
                "warm_seconds": timing,
                "warm_median_seconds": medians,
                "speedup": medians["uncached"] / medians["cached_batched"],
            }
        except Exception as exc:
            record = {"state": name, "error": f"{type(exc).__name__}: {exc}"}
        report["results"].append(record)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
