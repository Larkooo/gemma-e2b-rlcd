import argparse
import json
import statistics
import time
from pathlib import Path

from gemma_decisions import Choice, DecisionEngine, Noul, Score, State
from gemma_decisions.cached_backend import CachedMLXBackend
from gemma_decisions.core import ScoringRequest, decision_prompt, softmax


def main():
    parser = argparse.ArgumentParser(description="Profile cache, decoder, and projection stages")
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    b = CachedMLXBackend(args.model, branch_batch_size=4)
    mx = b.mx
    qs = [
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
    rs = []
    for q in qs:
        c = DecisionEngine._criteria(q)
        sy = tuple(b.symbols(len(c)))
        rs.append(ScoringRequest(decision_prompt(q.instructions, c, sy), sy))
    report = []
    for name, state in [
        ("text", State(text="A cat sleeps on a sofa. No dogs are present.")),
        ("video_speech", State(videos=(str((args.media / "red-blue-speech.mp4").resolve()),))),
    ]:
        p = b.prepare(state, rs)
        start = time.perf_counter()
        cache = b.prefill(p)
        prefill = time.perf_counter() - start
        lengths = [len(s) for s in p.suffixes]
        width = max(lengths)
        keep = width - min(lengths) + 1
        tokens = mx.array([s + [b.tokenizer.pad_token_id] * (width - len(s)) for s in p.suffixes])
        mx.eval(tokens)

        def run(optimized):
            timings = {}
            start = time.perf_counter()
            fork = b.fork_cache(cache, len(rs))
            mx.eval(*[a for c in fork for a in (c.keys, c.values)])
            timings["fork"] = time.perf_counter() - start
            start = time.perf_counter()
            h = b.model.language_model.model(
                inputs=tokens, cache=fork, logits_to_keep=keep if optimized else None
            )
            mx.eval(h)
            timings["decoder"] = time.perf_counter() - start
            indices = mx.array(lengths) - 1 - (width - h.shape[1])
            last = h[mx.arange(len(rs)), indices, :]
            start = time.perf_counter()
            logits = b.model.language_model.logits_from_hidden(last[:, None, :])[:, 0, :].astype(
                mx.float32
            )
            mx.eval(logits)
            timings["vocab_projection"] = time.perf_counter() - start
            start = time.perf_counter()
            scores = b._extract(logits, rs, [p.prefix_tokens + n for n in lengths])
            timings["extract"] = time.perf_counter() - start
            return timings, scores

        _, ref = run(False)
        _, opt = run(True)
        data = {"full": [], "trimmed": []}
        for iteration in range(5):
            for mode in ["full", "trimmed"] if iteration % 2 == 0 else ["trimmed", "full"]:
                t, s = run(mode == "trimmed")
                data[mode].append(t)
        med = {
            mode: {stage: statistics.median(t[stage] for t in samples) for stage in samples[0]}
            for mode, samples in data.items()
        }
        delta = max(
            abs(a - z)
            for x, y in zip(ref, opt)
            for a, z in zip(softmax(x.logits), softmax(y.logits))
        )
        result = {
            "state": name,
            "prefix_tokens": p.prefix_tokens,
            "suffix_lengths": lengths,
            "tail_tokens": keep,
            "prefill_first_seconds": prefill,
            "prefix_kv_logical_bytes": sum(c.keys.nbytes + c.values.nbytes for c in cache),
            "median_seconds": med,
            "raw_seconds": data,
            "max_probability_delta": delta,
            "winners_match": all(
                max(range(len(x.logits)), key=x.logits.__getitem__)
                == max(range(len(y.logits)), key=y.logits.__getitem__)
                for x, y in zip(ref, opt)
            ),
        }
        report.append(result)
        print(json.dumps(result), flush=True)
        args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
