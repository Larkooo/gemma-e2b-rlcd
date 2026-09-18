"""Paired full-detail execution comparison against version 0.3 computations."""

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_head import requests_for
from check_cache import compare
from mlx_vlm.models.gemma4.vision import VisionPatchEmbedder

from gemma_rlcd import State
from gemma_rlcd.cached_backend import CachedMLXBackend
from gemma_rlcd.core import softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    backend = CachedMLXBackend(args.model, branch_batch_size=32)
    indexed = backend.model.vision_tower.patch_embedder
    dense = VisionPatchEmbedder(backend.model.config.vision_config)
    dense.input_proj = indexed.input_proj
    dense.position_embedding_table = indexed.position_embedding_table
    media = args.media.resolve()
    jobs = [
        ("text", State(text="A cat sleeps on a sofa. No dogs are present."), 4),
        ("text_many_fields", State(text="A cat sleeps on a sofa. No dogs are present."), 28),
        ("image", State(images=(str(media / "red.png"),)), 4),
        ("speech", State(audio=(str(media / "dog.wav"),)), 4),
        ("video", State(videos=(str(media / "red-blue.mp4"),)), 4),
        ("video_speech", State(videos=(str(media / "red-blue-speech.mp4"),)), 4),
    ]
    report = {
        "status": "paired_full_detail_execution_probe_not_quality_benchmark",
        "compute_dtype": "float32",
        "state_layers": 35,
        "timing_scope": "warm_model_fresh_state_including_preprocessing_and_all_fields",
        "reference": "v0.3_native_answer_tail_and_dense_vision_positions",
        "optimized": "per_row_answer_gather_and_indexed_vision_positions",
        "results": [],
    }
    for name, state, count in jobs:
        requests = requests_for(backend, count)

        def run(mode):
            backend.answer_mode = "tail" if mode == "reference" else "gather"
            backend.model.vision_tower.patch_embedder = dense if mode == "reference" else indexed
            start = time.perf_counter()
            scores = backend.score_batch(state, requests)
            for score in scores:
                softmax(score.logits)
            elapsed = time.perf_counter() - start
            return scores, elapsed, dict(backend.last_stats)

        modes = ["reference", "optimized"]
        for mode in modes:
            run(mode)
        records = {mode: [] for mode in modes}
        scores = {}
        stats = {}
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else reversed(modes):
                scores[mode], elapsed, stats[mode] = run(mode)
                records[mode].append(elapsed)
        medians = {mode: statistics.median(values) for mode, values in records.items()}
        result = {
            "state": name,
            "fields": count,
            "comparison": compare(scores["reference"], scores["optimized"]),
            "raw_seconds": records,
            "median_seconds": medians,
            "speed_ratio": medians["reference"] / medians["optimized"],
            "execution": stats,
        }
        report["results"].append(result)
        print(
            json.dumps({key: value for key, value in result.items() if key != "execution"}),
            flush=True,
        )
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
