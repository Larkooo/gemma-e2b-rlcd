"""Full warm-request runtime probes. Untrained head outputs are not quality evidence."""

import argparse
import json
import statistics
from pathlib import Path

from gemma_decisions import State
from gemma_decisions.core import ScoringRequest, decision_prompt
from gemma_decisions.head_backend import DecisionHeadBackend


def requests_for(backend, count):
    tasks = [
        (
            "Which animal is mentioned or visible?",
            {"cat": "A cat", "dog": "A dog", "other": "Neither"},
        ),
        ("Is a dog mentioned?", {"true": "Yes", "false": "No"}),
        ("How much red is visible?", {"0": "No red", "1": "Some red", "2": "Mostly red"}),
        ("Is a sofa mentioned?", {"true": "Yes", "false": "No"}),
    ]
    requests = []
    for i in range(count):
        instruction, criteria = tasks[i % len(tasks)]
        symbols = tuple(backend.symbols(len(criteria)))
        requests.append(
            ScoringRequest(
                decision_prompt(instruction, criteria, symbols),
                symbols,
                instruction,
                tuple(criteria.items()),
            )
        )
    return requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--state-layers", type=int, default=35)
    parser.add_argument("--image-soft-tokens", type=int, default=280)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--fields", type=int, nargs="+", default=[1, 4, 16, 28])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--states", nargs="+", default=["text", "image", "speech", "video", "video_speech"]
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    media = args.media.resolve()
    states = {
        "text": State(text="A cat sleeps on a sofa. No dogs are present."),
        "image": State(images=(str(media / "red.png"),)),
        "speech": State(audio=(str(media / "dog.wav"),)),
        "video": State(videos=(str(media / "red-blue.mp4"),)),
        "video_speech": State(videos=(str(media / "red-blue-speech.mp4"),)),
    }
    backend = DecisionHeadBackend(
        args.model,
        state_layers=args.state_layers,
        compute_dtype=args.dtype,
        image_soft_tokens=args.image_soft_tokens,
        video_max_frames=args.video_max_frames,
    )
    report = {
        "status": "untrained_runtime_probe_not_quality_or_calibration_evidence",
        "model": args.model,
        "state_layers": args.state_layers,
        "compute_dtype": args.dtype,
        "head_config": backend.head.config.to_dict(),
        "image_soft_tokens": args.image_soft_tokens,
        "video_max_frames": args.video_max_frames,
        "timing_scope": "warm_model_fresh_state_including_media_decode_preprocess_encoding_and_all_fields",
        "field_scaling": "four representative question templates repeated to the requested count",
        "results": [],
    }
    for name in args.states:
        state = states[name]
        requests = {count: requests_for(backend, count) for count in args.fields}
        for count in args.fields:
            backend.probe(state, requests[count])
        samples = {count: [] for count in args.fields}
        for repeat in range(args.repeats):
            for count in args.fields if repeat % 2 == 0 else reversed(args.fields):
                try:
                    samples[count].append(backend.probe(state, requests[count]))
                except Exception as exc:
                    samples[count].append({"error": f"{type(exc).__name__}: {exc}"})
        for count, records in samples.items():
            successful = [record for record in records if "error" not in record]
            record = {
                "state": name,
                "fields": count,
                "attempted": len(records),
                "completed": len(successful),
                "samples": records,
            }
            if successful:
                timing_keys = [key for key in successful[0] if key.endswith("_seconds")]
                record["median_seconds"] = {
                    key: statistics.median(row[key] for row in successful) for key in timing_keys
                }
                record["max_seconds"] = max(row["total_seconds"] for row in successful)
            report["results"].append(record)
            print(
                json.dumps({key: value for key, value in record.items() if key != "samples"}),
                flush=True,
            )
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
