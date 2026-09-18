"""Exercise real inference. Tiny synthetic fixtures are not a benchmark."""

import argparse
import json
import time
from pathlib import Path

from gemma_decisions import Choice, DecisionEngine, Noul, Score, State
from gemma_decisions.cached_backend import CachedMLXBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--backend", choices=["cached", "catalog"], default="cached")
    args = parser.parse_args()
    media = args.media.resolve()
    animals = {"cat": "A cat", "dog": "A dog", "other": "Neither cat nor dog"}
    counts = [
        "No animals are present",
        "One kind of animal is present",
        "Two or more kinds of animals are present",
    ]
    jobs = [
        (
            "text",
            State(text="A cat sleeps on a sofa. No dogs are present."),
            {
                "category": Choice("Which animal is present?", animals),
                "dog": Noul("Is a dog present?"),
                "grade": Score("How many kinds of animals are present?", counts),
            },
            {"category": "cat", "dog": False, "grade": 1},
        ),
        (
            "image",
            State(images=(str(media / "red.png"),)),
            {
                "category": Choice(
                    "What is the dominant color?", {"red": "Red", "green": "Green", "blue": "Blue"}
                ),
                "red": Noul("Is the image predominantly red?"),
                "grade": Score(
                    "How much red is in the image?",
                    [
                        "No red",
                        "Red covers some but not most of the image",
                        "Red covers most or all of the image",
                    ],
                ),
            },
            {"category": "red", "red": True, "grade": 2},
        ),
        (
            "speech",
            State(audio=(str(media / "dog.wav"),)),
            {
                "category": Choice("Which animal does the speaker say is present?", animals),
                "dog": Noul("Does the speaker say a dog is present?"),
                "grade": Score(
                    "How many kinds of animals does the speaker say are present?", counts
                ),
            },
            {"category": "dog", "dog": True, "grade": 1},
        ),
        (
            "video",
            State(videos=(str(media / "red-blue.mp4"),)),
            {
                "category": Choice(
                    "In what order do the colors appear in the video?",
                    {
                        "red_blue": "Red first then blue",
                        "blue_red": "Blue first then red",
                        "unchanging": "One unchanging color",
                    },
                ),
                "both": Noul("Does the video show both red and blue?"),
                "grade": Score(
                    "How many of red and blue appear in the video?",
                    ["Neither red nor blue", "Only one of red or blue", "Both red and blue"],
                ),
            },
            {"category": "red_blue", "both": True, "grade": 2},
        ),
        (
            "video_with_speech",
            State(videos=(str(media / "red-blue-speech.mp4"),)),
            {
                "animal": Choice(
                    "Which animal does the speaker mention in the soundtrack?", animals
                ),
                "dog": Noul("Does the soundtrack mention a dog?"),
                "grade": Score(
                    "How many of red and blue appear in the video?",
                    ["Neither red nor blue", "Only one of red or blue", "Both red and blue"],
                ),
            },
            {"animal": "dog", "dog": True, "grade": 2},
        ),
    ]
    started = time.perf_counter()
    if args.backend == "catalog":
        from gemma_decisions.catalog_backend import CatalogMLXBackend

        backend = CatalogMLXBackend(args.model)
    else:
        backend = CachedMLXBackend(args.model)
    loaded = time.perf_counter()
    engine = DecisionEngine(backend)
    report = {
        "status": "integration_smoke_only",
        "trained": False,
        "calibration_validated": False,
        "model": args.model,
        "backend": args.backend,
        "load_seconds": loaded - started,
        "results": [],
        "batches": [],
    }
    for name, state, questions, expected in jobs:
        start = time.perf_counter()
        batch_error = None
        try:
            answers = engine.system_one(state, questions)["answers"]
        except Exception as exc:
            batch_error = f"{type(exc).__name__}: {exc}"
            answers = {}
        report["batches"].append(
            {
                "modality": name,
                "seconds": time.perf_counter() - start,
                "execution": dict(backend.last_stats),
                "error": batch_error,
            }
        )
        for question_id, question in questions.items():
            record = {"modality": name, "question": question_id, "expected": expected[question_id]}
            try:
                if batch_error:
                    raise RuntimeError(batch_error)
                answer = answers[question_id]
                if isinstance(question, Choice):
                    actual = answer["choice"]
                elif isinstance(question, Noul):
                    actual = answer["noul"] >= 0.5
                else:
                    actual = int(max(answer["probabilities"], key=answer["probabilities"].get))
                record.update(answer=answer, actual=actual, correct=actual == expected[question_id])
            except Exception as exc:
                record.update(error=f"{type(exc).__name__}: {exc}", correct=False)
            report["results"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "answer"}), flush=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["attempted"] = len(report["results"])
    report["completed"] = sum("answer" in result for result in report["results"])
    report["correct"] = sum(result["correct"] for result in report["results"])
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
