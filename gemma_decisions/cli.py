import argparse
import json
import sys
import time
from pathlib import Path

from .core import DecisionEngine, State, parse_question


def read_request(path: Path) -> tuple[State, dict]:
    data = json.loads(path.read_text())
    if set(data) != {"state", "questions"}:
        raise ValueError("Request requires exactly state and questions")
    raw_state = data["state"]
    if isinstance(raw_state, str):
        raw_state = {"text": raw_state}
    if not isinstance(raw_state, dict) or set(raw_state) - {"text", "images", "audio", "videos"}:
        raise ValueError("State must be a string or an object with text/images/audio/videos")
    paths = {}
    for kind in ("images", "audio", "videos"):
        values = raw_state.get(kind, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{kind} must be a list of local paths")
        paths[kind] = tuple(str((path.parent / value).resolve()) for value in values)
    state = State(text=raw_state.get("text", ""), **paths)
    if not isinstance(data["questions"], dict) or not data["questions"]:
        raise ValueError("questions must be a nonempty object")
    questions = {key: parse_question(value) for key, value in data["questions"].items()}
    return state, questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma multimodal typed decisions (uncalibrated baseline)"
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("--model", required=True, help="Full multimodal Gemma 4 E2B MLX checkpoint")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--branch-batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=["cached", "catalog", "head"], default="cached")
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state, questions = read_request(args.request)
    from .cached_backend import CachedMLXBackend

    started = time.perf_counter()
    if args.backend == "head":
        if args.head_checkpoint is None:
            parser.error("--backend head requires --head-checkpoint")
        from .decision_head import HeadConfig
        from .head_backend import DecisionHeadBackend

        manifest = json.loads((args.head_checkpoint / "head.json").read_text())
        backend = DecisionHeadBackend(
            args.model,
            checkpoint=str(args.head_checkpoint),
            state_layers=manifest["state_layers"],
            compute_dtype=manifest["compute_dtype"],
            head_config=HeadConfig(**manifest["head_config"]),
            image_soft_tokens=manifest["image_soft_tokens"],
            video_max_frames=manifest["video_max_frames"],
        )
    elif args.backend == "catalog":
        from .catalog_backend import CatalogMLXBackend

        backend = CatalogMLXBackend(args.model, branch_batch_size=args.branch_batch_size)
    else:
        backend = CachedMLXBackend(args.model, branch_batch_size=args.branch_batch_size)
    loaded = time.perf_counter()
    result = DecisionEngine(backend, args.temperature).system_one(state, questions)
    result.update(
        {
            "model": args.model,
            "load_seconds": loaded - started,
            "decision_seconds": time.perf_counter() - loaded,
            "execution": backend.last_stats,
            "video_sampling_fps": 1.0 if state.videos else None,
            "video_soundtrack": "included_if_present" if state.videos else None,
        }
    )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
