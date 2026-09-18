"""Matched, repeated workload comparisons; preserve errors and raw predictions."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from gemma_rlcd.comparison import discrete_answers, generate_answers
from gemma_rlcd.core import DecisionEngine, Independent, State, parse_question
from gemma_rlcd.json_backend import JSONMLXBackend


def flatten(values, prefix=""):
    result = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            result.update(flatten(value, key))
        else:
            result[key] = value
    return result


def quality(actual, expected):
    actual = flatten(actual or {})
    expected = flatten(expected)
    checks = {
        key: key in actual
        and (actual[key] in value if isinstance(value, list) else actual[key] == value)
        for key, value in expected.items()
    }
    return {"correct": sum(checks.values()), "scored": len(checks), "checks": checks}


def agreement(left, right):
    if left is None or right is None:
        return None
    left, right = flatten(left), flatten(right)
    keys = left.keys() | right.keys()
    return {key: key in left and key in right and left[key] == right[key] for key in sorted(keys)}


def summarize(samples, methods):
    result = {}
    for method in methods:
        runs = [sample for sample in samples if sample["method"] == method]
        times = [sample["seconds"] for sample in runs]
        result[method] = {
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "valid_runs": sum(sample["valid"] for sample in runs),
            "attempted_runs": len(runs),
            "correct": sum(sample["quality"]["correct"] for sample in runs),
            "scored": sum(sample["quality"]["scored"] for sample in runs),
            "stable_answers": all(sample["values"] == runs[0]["values"] for sample in runs),
        }
    valid = all(
        result[method]["valid_runs"] == result[method]["attempted_runs"] for method in methods
    )
    result["normal_over_batched"] = (
        result["normal"]["median_seconds"] / result["batched"]["median_seconds"] if valid else None
    )
    if "serial" in result:
        result["serial_over_batched"] = (
            result["serial"]["median_seconds"] / result["batched"]["median_seconds"]
            if valid
            else None
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=Path("examples/demo-workloads.json"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repetitions to vary execution order")
    cases = json.loads(args.cases.read_text())["cases"]
    if args.only:
        unknown = set(args.only) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.only]
    backend = JSONMLXBackend(
        args.model, branch_batch_size=args.batch_size, max_input_tokens=args.max_input_tokens
    )
    report = {
        "status": "development_workloads_not_a_held_out_benchmark",
        "platform": platform.platform(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "host_swap_at_start": subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True
        ).strip(),
        "model_source": json.loads(Path("model-source.json").read_text()),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "methodology": {
            "compute": "Same resident frozen 4-bit Gemma weights, float32 compute, all 35 layers.",
            "timing": "One excluded warmup per method per case, then repeated runs in rotating/reversed order. GPU synchronized at boundaries. Medians include prompt preparation and inference; exclude model loading, upload, and allocator reset.",
            "cache": "Fresh input/media KV per run. Every scorer run prefills all state and question definitions once. Serial control changes only branch batch size to one, retaining the same shared prefix.",
            "normal": "Greedy compact JSON with decisions only, no reasoning or requested probabilities; same complete input and schema as the scorer. No padded output requirement.",
            "quality": "Predeclared development expectations are scored where available. Invalid structured responses count as failed decisions. Unannotated policy questions are compared only for agreement. Agreement does not establish accuracy.",
            "input_limit": args.max_input_tokens,
            "input_limit_note": "The 255-choice example exceeds the default 8192-token UI limit. This standalone benchmark raises the limit without truncating the schema for either method.",
            "repeats": args.repeats,
            "batch_size": args.batch_size,
        },
        "cases": [],
    }

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for case in cases:
        state = State(text=case["text"])
        questions = {name: parse_question(value) for name, value in case["questions"].items()}
        methods = ["batched", "normal"] + (["serial"] if case.get("serial_control") else [])
        row = {
            "name": case["name"],
            "title": case["title"],
            "primitive_fields": sum(
                len(question.criteria) if isinstance(question, Independent) else 1
                for question in questions.values()
            ),
            "max_choices": max(
                2 if value["type"] in {"independent", "noul"} else len(value["criteria"])
                for value in case["questions"].values()
            ),
            "fixture": case,
            "warmups": [],
            "samples": [],
        }
        report["cases"].append(row)
        print(json.dumps({"start": case["name"], "methods": methods}), flush=True)

        def run(method):
            backend.branch_batch_size = 1 if method == "serial" else args.batch_size
            backend.mx.synchronize()
            backend.mx.clear_cache()
            backend.mx.reset_peak_memory()
            started = time.perf_counter()
            record = {"method": method, "valid": False, "values": None}
            try:
                if method == "normal":
                    output = generate_answers(backend, state, questions)
                    record.update(
                        valid=output["valid"], values=output["answers"] if output["valid"] else None
                    )
                else:
                    output = DecisionEngine(backend).system_one(state, questions)
                    output["execution"] = dict(backend.last_stats)
                    record.update(valid=True, values=discrete_answers(output["answers"]))
                record["output"] = output
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            backend.mx.synchronize()
            record["seconds"] = time.perf_counter() - started
            record["peak_mlx_bytes"] = backend.mx.get_peak_memory()
            record["quality"] = quality(record["values"], case["expected"])
            return record

        for method in methods:
            row["warmups"].append(run(method))
            save()
        for repeat in range(args.repeats):
            order = methods[repeat % len(methods) :] + methods[: repeat % len(methods)]
            if len(methods) > 2 and repeat % 2:
                order.reverse()
            current = {}
            for method in order:
                sample = run(method)
                sample["repeat"] = repeat
                current[method] = sample["values"]
                row["samples"].append(sample)
                save()
            row.setdefault("agreements", []).append(
                {
                    "repeat": repeat,
                    "normal": agreement(current["batched"], current["normal"]),
                    "serial": agreement(current["batched"], current["serial"])
                    if "serial" in current
                    else None,
                }
            )
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "repeat": repeat,
                        "seconds": {
                            sample["method"]: round(sample["seconds"], 3)
                            for sample in row["samples"]
                            if sample["repeat"] == repeat
                        },
                    }
                ),
                flush=True,
            )
        row["summary"] = summarize(row["samples"], methods)
        save()
        print(json.dumps({"complete": case["name"], "summary": row["summary"]}), flush=True)


if __name__ == "__main__":
    main()
