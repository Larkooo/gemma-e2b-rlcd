"""Separate shared-video reuse from GPU batching for two independent fields."""

import argparse
import json
import statistics
import time
from pathlib import Path

from check_cache import compare

from gemma_decisions import State
from gemma_decisions.cached_backend import CachedMLXBackend
from gemma_decisions.core import ScoringRequest, decision_prompt, softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    backend = CachedMLXBackend(args.model)
    requests = []
    for animal in ("dogs", "cats"):
        instruction = f"What proportion of sampled video frames contain visible {animal}?"
        criteria = {
            "0": f"No sampled frames contain visible {animal}",
            "1": f"More than zero and at most 25 percent contain visible {animal}",
            "2": f"More than 25 and at most 50 percent contain visible {animal}",
            "3": f"More than 50 and at most 75 percent contain visible {animal}",
            "4": f"More than 75 percent contain visible {animal}",
        }
        symbols = tuple(backend.symbols(len(criteria)))
        requests.append(
            ScoringRequest(
                decision_prompt(instruction, criteria, symbols),
                symbols,
                instruction,
                tuple(criteria.items()),
            )
        )
    modes = ["separate_requests", "shared_video_serial_fields", "shared_video_batched_fields"]
    report = {
        "status": "two_field_runtime_probe_not_animal_recognition_validation",
        "model": args.model,
        "compute_dtype": "float32",
        "state_layers": 35,
        "timing_scope": "resident_model_fresh_inputs_including_media_processing_and_probabilities",
        "input_note": "Four-second 224x224 synthetic color clips, with or without speech; no animals are visually present.",
        "question_note": "Two independent grades of the proportion of sampled frames containing dogs or cats.",
        "results": [],
    }
    for name, filename in [
        ("silent_video", "red-blue.mp4"),
        ("video_with_speech", "red-blue-speech.mp4"),
    ]:
        state = State(videos=(str((args.media / filename).resolve()),))

        def run(mode):
            backend.branch_batch_size = 2 if mode == "shared_video_batched_fields" else 1
            started = time.perf_counter()
            if mode == "separate_requests":
                scores, executions = [], []
                for request in requests:
                    scores.extend(backend.score_batch(state, [request]))
                    executions.append(dict(backend.last_stats))
            else:
                scores = backend.score_batch(state, requests)
                executions = [dict(backend.last_stats)]
            probabilities = [softmax(score.logits) for score in scores]
            return scores, {
                "seconds": time.perf_counter() - started,
                "execution": executions,
                "probabilities": probabilities,
            }

        for mode in modes:
            run(mode)
        samples = {mode: [] for mode in modes}
        comparisons = []
        for repeat in range(args.repeats):
            order = modes[repeat % 3 :] + modes[: repeat % 3]
            if repeat % 2:
                order = list(reversed(order))
            scores = {}
            for mode in order:
                scores[mode], sample = run(mode)
                samples[mode].append(sample)
            comparisons.append(
                {mode: compare(scores["separate_requests"], scores[mode]) for mode in modes[1:]}
            )
        medians = {
            mode: statistics.median(sample["seconds"] for sample in records)
            for mode, records in samples.items()
        }
        row = {
            "input": name,
            "fields": 2,
            "median_seconds": medians,
            "total_speedup": medians[modes[0]] / medians[modes[2]],
            "batching_only_speedup": medians[modes[1]] / medians[modes[2]],
            "samples": samples,
            "comparisons": comparisons,
        }
        report["results"].append(row)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in row.items() if key not in {"samples", "comparisons"}}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
