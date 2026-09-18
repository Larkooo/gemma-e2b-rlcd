# Shared-state demo workload comparison

These local development workloads test whether many structured decisions benefit from parallel scoring. They compare the fixed native-JSON Gemma scorer with ordinary compact JSON generation from the same resident model. The public Qwen project supplies some workload examples; both inference paths use the same Gemma weights.

The 28-field support workload is 3.55× faster with identical discrete outputs; the 32-label workload is 6.88× faster but misses one expected fact that normal generation gets right. The inbox has a similar one-decision quality loss. The 64-choice catalog is 1.33× slower. These results support a benefit for many small decisions, not a universal speedup or accuracy-equivalence claim.

## Method

- Apple M5 Mac, full 35-layer Gemma 4 E2B, pinned 4-bit checkpoint, float32 compute. One excluded warmup per method and case, then four measured repeats in changing order. Raw order, predictions, errors, timings, and peak MLX memory are retained.
- Every run starts with fresh input KV. Both paths see the complete state, instructions, and candidate definitions. Normal Gemma is asked for one compact object of decisions without explanations or probability prose; its actual formatting is retained. The parallel scorer also returns probability distributions.
- The serial scorer control retains the same shared prefix and complete schema; only the branch batch size changes from eight to one. This measures the incremental benefit of batching separately from replacing autoregressive output.
- Timings include prompt preparation and inference, exclude loading and upload. The idle playground process was stopped before this run. The Mac had substantial pre-existing swap use and timing variability; these are observed local measurements, not clean-hardware throughput claims.
- The first attempt was interrupted after latency variability and host memory pressure were observed. It remains in [the separate partial report](../../reports/demo-workload-host-pressure.json), and is not pooled with this run.
- Expected answers were written before inference. Some external-demo policy questions have no reliable expected answer and are left unscored. Repeats measure timing; they do not create new independent accuracy examples.

## Results

Accuracy below counts only the predeclared, annotated primitive decisions. Ranges show the minimum and maximum of four measured runs. Every failure remains in the denominator.

An invalid overall response receives zero usable decisions under the strict output contract. This does not mean every value in its raw text is semantically wrong: see the security and 255-choice notes below.

| Workload | Parallel seconds, median [range] | Normal seconds, median [range] | Normal / parallel | Parallel expected answers | Normal expected answers | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_1 | 0.512 [0.470–0.579] | 0.813 [0.694–0.843] | 1.59× | 1/1 | 1/1 | 1/1 |
| support_4 | 1.501 [1.455–1.584] | 2.801 [2.756–2.850] | 1.87× | 3/3 | 3/3 | 4/4 |
| support_8 | 1.890 [1.778–1.991] | 4.971 [4.892–5.134] | 2.63× | 5/5 | 5/5 | 8/8 |
| support_16 | 2.555 [2.238–3.519] | 9.971 [8.504–10.712] | 3.90× | 10/12 | 11/12 | 15/16 |
| support_28 | 4.415 [4.343–4.757] | 15.660 [15.375–16.082] | 3.55× | 15/17 | 15/17 | 28/28 |
| code_security | 4.521 [4.409–4.730] | 16.992 [16.548–17.275] | Not reported: invalid run | 9/10 | 0/10 | Invalid normal output |
| inbox_32 | 5.237 [5.076–5.463] | 13.966 [13.864–14.196] | 2.67× | 31/32 | 32/32 | 31/32 |
| ticket_flags_32 | 0.864 [0.577–1.350] | 5.945 [4.001–9.611] | 6.88× | 31/32 | 32/32 | 31/32 |
| catalog_16 | 0.981 [0.948–1.030] | 2.021 [1.774–2.234] | 2.06× | 1/1 | 1/1 | 3/4 |
| catalog_64 | 3.274 [3.229–3.376] | 2.457 [2.390–2.501] | 0.75× | 1/1 | 1/1 | 3/4 |
| catalog_255 | 15.417 [15.081–18.155] | 6.511 [6.366–6.690] | Not reported: invalid run | 1/1 | 0/1 | Invalid normal output |

A ratio above 1 favors the parallel scorer; below 1 favors normal generation. Agreement is exact output equality and is separate from expected-answer accuracy. Equivalent duplicate category IDs can differ while both match the annotated category.

## Incremental batching benefit

| Workload | Shared-prefix serial scorer | Batched scorer | Serial / batched | Same discrete answers? |
| --- | ---: | ---: | ---: | --- |
| support_28 | 6.386 s | 4.415 s | 1.45× | Yes |
| inbox_32 | 8.327 s | 5.237 s | 1.59× | Yes |
| ticket_flags_32 | 1.902 s | 0.864 s | 2.20× | Yes |

## Answer differences

- `support_16`: batched missed security_incident, compliance_breach; normal missed compliance_breach.
- `support_28`: batched missed compliance_breach, suggested_action; normal missed compliance_breach, suggested_action.
- `code_security`: the scorer missed `public_cve_match`. Normal returned an unlisted `secondary_cwe` value, invalidating all four measured responses. Its raw text still matches 9/10 annotated answers; the table's 0/10 measures valid usable output, not zero understanding.
- `inbox_32`: the scorer called the polite cancellation request negative despite the instruction to treat emotionally neutral requests as neutral. Normal got all 32 annotated answers right.
- `ticket_flags_32`: the scorer incorrectly rejected the explicitly stated 99.99% SLA. Normal got all 32 facts right.
- `catalog_255`: the scorer selected the expected category. Normal selected that same category but emitted strings `"true"` instead of booleans for two companion fields, invalidating all four measured responses. Its shorter completion time is reported, but no speedup for valid complete responses is claimed.

The same inbox and incident-label misses occur in the serial control. Batching did not introduce these observed decision differences. Neither path was retrained or retuned on these fixtures.

For the first measured 255-choice run, native scoring spent 0.80 s preparing input, 6.34 s prefilling 9,118 tokens, and 11.00 s scoring branches. It required 33 candidate batches plus one binary-field batch. This is direct evidence that large choice sets still add substantial work in this implementation.

## Scope and reproduction

The support, security, and catalog inputs come from [the public Qwen parallel-decoding demo](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9/presets), translated to our answer contracts. The original missing currency numerals and ambiguous policy questions remain intact. The 255-choice catalog repeats category meanings under different numeric IDs; it is not 255 distinct semantic classes. Its synthetic categories are not an authoritative tariff schedule. Source URLs, hashes, and modifications are recorded in [the cases](../../examples/demo-workloads.json).

The 255-choice complete schema exceeds the playground’s 8,192-token limit. This standalone test explicitly uses 16,384 for both methods, with no input truncation. The browser includes the smaller 64-choice version.

The current implementation scores every token of every complete candidate label. Increasing the number of multi-token choices can therefore increase work substantially even when the number of output fields is small. It is not constant-time in the number or length of choices.

```bash
python scripts/benchmark_workloads.py \
  --model models/gemma-4-e2b-it-4bit \
  --report work/demo-workloads.json --repeats 4
```

Use `--only support_28 inbox_32 ticket_flags_32 catalog_64` for a shorter run. [Raw complete results](../../reports/demo-workload-benchmark.json) include all warmups and measured attempts.

The web example selector includes Support triage (28 fields), Customer inbox (32 decisions), Incident facts (32 labels), and Catalog routing (64 choices). Select an example and click Compare with Gemma. The UI shows a single fresh observation rather than these benchmark medians.

All four presets were checked in the browser. An initial check found the editor substituting different default yes/no descriptions, changing two predictions. Presets now include the core defaults explicitly, and a regression test checks prompt equality with the benchmark. The corrected live support comparison returned matching answers in all 28 rows (3.407 s scorer, 8.573 s normal). This individual UI run is separate from the warmed benchmark; see [browser validation](../../reports/demo-workload-web-validation.json).
