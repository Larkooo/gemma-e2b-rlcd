"""Supervised head training on frozen features; no RL or calibration claims."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from gemma_decisions import DecisionEngine, Independent, Noul, State
from gemma_decisions.calibration import evaluate
from gemma_decisions.core import TokenScores, parse_question, softmax
from gemma_decisions.decision_head import grouped_cross_entropy
from gemma_decisions.head_backend import DecisionHeadBackend


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    for row in rows:
        if set(row) != {"id", "state", "questions", "targets"}:
            raise ValueError("Each JSONL record requires id, state, questions, and targets")
        if set(row["questions"]) != set(row["targets"]):
            raise ValueError("Every question requires exactly one target")
        for kind in ("images", "audio", "videos"):
            if kind in row["state"]:
                row["state"][kind] = [
                    str((path.parent / value).resolve()) for value in row["state"][kind]
                ]
    return rows


def check_splits(splits):
    seen_ids, seen_states = set(), set()
    for name, rows in splits.items():
        for row in rows:
            state_hash = hashlib.sha256(
                json.dumps(row["state"], sort_keys=True).encode()
            ).hexdigest()
            if row["id"] in seen_ids or state_hash in seen_states:
                raise ValueError(f"Duplicate source id or exact state in split {name}")
            seen_ids.add(row["id"])
            seen_states.add(state_hash)


def compile_row(backend, row):
    state = State(**row["state"])
    questions = {key: parse_question(value) for key, value in row["questions"].items()}
    requests = []

    class Capture:
        symbols = backend.symbols

        def score_batch(self, state, batch):
            requests.extend(batch)
            return [TokenScores((0.0,) * len(request.symbols), None, 0) for request in batch]

    DecisionEngine(Capture()).system_one(state, questions)
    target_names = []
    for key, question in questions.items():
        target = row["targets"][key]
        if isinstance(question, Independent):
            if not isinstance(target, dict) or set(target) != set(question.criteria):
                raise ValueError("Independent labels require one boolean target per label")
            for label in question.criteria:
                if type(target[label]) is not bool:
                    raise ValueError("Independent targets must be booleans")
                target_names.append("yes" if target[label] else "no")
        elif isinstance(question, Noul) and type(target) is bool:
            target_names.append("true" if target else "false")
        else:
            target_names.append(str(target))
    targets = tuple(
        [name for name, _ in request.criteria].index(target)
        for request, target in zip(requests, target_names, strict=True)
    )
    return state, requests, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--state-layers", type=int, default=35)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error("steps and learning rate must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty to preserve prior experiments")
    splits = {name: read_rows(getattr(args, name)) for name in ("train", "validation", "test")}
    check_splits(splits)
    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    backend = DecisionHeadBackend(
        args.model, state_layers=args.state_layers, compute_dtype=args.dtype
    )
    encoded = {}
    compiled = {}
    for name, rows in splits.items():
        compiled[name] = [compile_row(backend, row) for row in rows]
        encoded[name] = [
            (backend.encode(state, requests), targets)
            for state, requests, targets in compiled[name]
        ]
        print(json.dumps({"encoded_split": name, "source_items": len(rows)}), flush=True)

    def measure(split):
        rows, targets = [], []
        backend.head.eval()
        for inputs, expected in encoded[split]:
            logits = backend.forward(inputs).tolist()
            for start, end in zip(inputs.offsets[:-1], inputs.offsets[1:], strict=True):
                rows.append(softmax(logits[start:end]))
            targets.extend(expected)
        return evaluate(rows, targets)

    initial_validation = measure("validation")
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    def loss(head, inputs, targets):
        logits = head(*inputs.arrays(), len(inputs.offsets) - 1)
        return grouped_cross_entropy(logits, inputs.offsets, targets)

    value_and_grad = nn.value_and_grad(backend.head, loss)
    trace = []
    order = list(range(len(encoded["train"])))
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        inputs, targets = encoded["train"][order[step % len(order)]]
        backend.head.train()
        value, gradients = value_and_grad(backend.head, inputs, targets)
        optimizer.update(backend.head, gradients)
        mx.eval(backend.head.parameters(), optimizer.state, value)
        if not bool(mx.isfinite(value).item()):
            raise ValueError(f"Non-finite training loss at update {step + 1}")
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            record = {"update": step + 1, "loss": float(value.item())}
            trace.append(record)
            print(json.dumps(record), flush=True)
    training = {
        "method": "supervised_categorical_nll_frozen_backbone",
        "updates": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "source_counts": {name: len(rows) for name, rows in splits.items()},
        "dataset_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in splits
        },
    }
    report = {
        "status": "pilot_evidence_only_not_generalization_or_calibration_validation",
        "training": training,
        "initial_validation": initial_validation,
        "train": measure("train"),
        "validation": measure("validation"),
        "test": measure("test"),
        "trace": trace,
    }
    if args.reference:
        from gemma_decisions.cached_backend import CachedMLXBackend

        reference = CachedMLXBackend(args.model)
        rows, targets = [], []
        for state, requests, expected in compiled["test"]:
            rows.extend(softmax(score.logits) for score in reference.score_batch(state, requests))
            targets.extend(expected)
        report["frozen_decoder_test"] = evaluate(rows, targets)
    backend.head.eval()
    backend.save(args.output, training)
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trace"}), flush=True)


if __name__ == "__main__":
    main()
