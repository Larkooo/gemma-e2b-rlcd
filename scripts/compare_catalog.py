"""Paired full-request timing of independent questions and shared schema selectors."""

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_head import requests_for

from gemma_decisions import State
from gemma_decisions.cached_backend import CachedMLXBackend
from gemma_decisions.catalog_backend import CatalogMLXBackend
from gemma_decisions.core import softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    base = CachedMLXBackend(args.model, branch_batch_size=32)
    catalog = CatalogMLXBackend.__new__(CatalogMLXBackend)
    catalog.__dict__.update(base.__dict__)
    catalog.last_stats = {}
    media = args.media.resolve()
    jobs = [
        ("text", State(text="A cat sleeps on a sofa. No dogs are present."), count)
        for count in (1, 4, 16, 28)
    ] + [
        ("image", State(images=(str(media / "red.png"),)), 4),
        ("speech", State(audio=(str(media / "dog.wav"),)), 4),
        ("video", State(videos=(str(media / "red-blue.mp4"),)), 4),
        ("video_speech", State(videos=(str(media / "red-blue-speech.mp4"),)), 4),
    ]
    report = {
        "status": "prompt_ablation_not_equivalence_or_quality_benchmark",
        "compute_dtype": "float32",
        "results": [],
    }
    for name, state, count in jobs:
        requests = requests_for(base, count)
        paths = {"independent_questions": base, "shared_catalog": catalog}
        for backend in paths.values():
            backend.score_batch(state, requests)
        samples = {key: [] for key in paths}
        outputs = {}
        for repeat in range(3):
            for key in paths if repeat % 2 == 0 else reversed(paths):
                start = time.perf_counter()
                outputs[key] = paths[key].score_batch(state, requests)
                for score in outputs[key]:
                    softmax(score.logits)
                samples[key].append(time.perf_counter() - start)
        left, right = outputs.values()
        agree = sum(
            max(range(len(a.logits)), key=a.logits.__getitem__)
            == max(range(len(b.logits)), key=b.logits.__getitem__)
            for a, b in zip(left, right, strict=True)
        )
        record = {
            "state": name,
            "fields": count,
            "winner_agreement": agree,
            "raw_seconds": samples,
            "median_seconds": {key: statistics.median(value) for key, value in samples.items()},
            "independent_execution": dict(base.last_stats),
            "catalog_execution": dict(catalog.last_stats),
        }
        report["results"].append(record)
        print(
            json.dumps(
                {key: value for key, value in record.items() if not key.endswith("execution")}
            ),
            flush=True,
        )
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
