# Shared-prefix, parallel decision inference

This report preserves the **version 0.2** measurements before answer-tail trimming. Current version 0.3 profiling and numerical checks are in [PERFORMANCE.md](performance.md). The older smoke results below remain integration evidence for that earlier version.

The default backend now encodes the text/media state once and prefills one KV cache. It evaluates independent question suffixes in a GPU tensor batch, with all options for each question scored together. Choice, Score, and Noul use the same execution path. The CLI uses this backend by default.

The prompt order is **generic instructions → complete multimodal state → field question and criteria**. Question-specific instructions cannot precede the shared state if their prefix is to be reused. Each branch has its own mutable cache object and can attend only to the common prefix and its own suffix.

KV computation is reused; the present backend replicates prefix KV storage across batch rows. Zero-copy prefix storage would require a different attention/cache implementation. The default batch holds up to eight fields, configurable from 1 to 64; larger request maps use multiple batches. All options within a Choice are read from one output vector, so 255 options do not mean 255 separate forwards.

## Numerical and isolation checks

Six inputs were exercised with four fields each: short text, an image, speech, video, video with speech, and a 1,250-token text prefix that crosses the sliding-attention window. Field suffixes had unequal lengths. The reference and cache paths used identical prepared tokens, media tensors, model parameters, and output positions.

- **24/24 winning answers agreed** between the cached four-row batch and independent full forwards. This measures agreement, not correctness.
- Reversing branch order preserved logits exactly on every probe.
- Prefix cache contents and offsets remained unchanged after serial, batched, and reordered branches.
- Right-padded rows gather their own final real-token position. Later padding tokens cannot affect that position through causal attention; those branch caches are discarded after scoring.
- With float32 floating parameters/activations and packed 4-bit weights, the maximum observed absolute probability difference was **0.00773** for the four-row batch versus full forwards (0.773 percentage points). Cached serial versus full forwards differed by up to **0.01047**. These results are not bit-exact equivalence or a universal tolerance guarantee.

The initial bfloat16 compute path differed by as much as **0.20776** in probability under the same cache comparison. That path was rejected as the default. Its results are preserved in [CACHE-BFLOAT16.json](../../reports/cache-bfloat16.json), and the float32 results are in [CACHE-FLOAT32.json](../../reports/cache-float32.json). The dtype experiment reduces observed drift; it does not identify every source of the remaining numerical difference.

Calibrate and evaluate the actual serving precision and batch configuration. The same argmax is insufficient evidence that a probability distribution is unchanged.

## Small local timing probe

All relevant shapes were warmed first. Each method ran three times, with alternating order, on the same Apple Silicon machine. Numbers below are medians for four fields. Both paths reused the same preprocessed media/input tensors; timings exclude model loading, file decoding, and tokenization. They include model encoding, decoder work, and, on the cached path, a fresh shared prefill plus branch evaluation. They are not end-to-end production latency measurements.

| Shared input | Independent full forwards | Shared prefix + four-row batch | Speed ratio |
| --- | ---: | ---: | ---: |
| Short text, 50 prefix tokens | 0.267 s | 0.403 s | 0.66× |
| Image, 295 prefix tokens | 2.465 s | 0.944 s | 2.61× |
| Short speech, 86 prefix tokens | 0.424 s | 0.453 s | 0.94× |
| Video, 328 prefix tokens | 2.851 s | 1.134 s | 2.51× |
| Video + speech, 431 prefix tokens | 2.963 s | 1.127 s | 2.63× |
| Longer text, 1,250 prefix tokens | 2.120 s | 1.083 s | 1.96× |

The cache path helped when repeated state/media work dominated. It was slower for these short text/speech inputs. Batch capacity and latency do not remain constant as field count, criteria length, or media size increases. Peak memory and scaling to large workloads have not been benchmarked.

## Current multimodal smoke checks

The cached backend completed all 15 fixture questions, in five batches of three fields with exactly one prefix prefill per batch. **12/15 expected answers matched.** The incorrect answers were the text animal-count grade, the silent video's color-count grade, and the dog-mentioned Noul on the video with speech. All failures are retained in [CACHED-SMOKE.json](../../reports/cached-smoke.json).

The initial sequential baseline achieved 13/15 on these fixtures. Prompt placement and compute precision changed to enable validated cache reuse, so this is not a controlled estimate of a batching-induced quality difference. The matched cache comparison above uses identical prompts. These synthetic examples are integration checks, not a representative quality or calibration test.

**No decision-head training, LoRA training, or RL has run.** Numerical cache validation and syntactically valid typed results do not establish calibrated decisions.

## Reproduce

From the project directory:

```bash
python scripts/check_cache.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media \
  --report work/cache-check.json
python scripts/smoke.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media \
  --report work/cached-smoke.json
```

Create the media first with `scripts/create_fixtures.py --output-dir work/media` if needed. The current script uses float32; the retained bfloat16 report documents the earlier experiment rather than a current default setting.

28 automated tests pass, including one-batch dispatch for mixed primitives, independent-label batching, backend result-count validation, full and rotating KV cache isolation, probability/grade semantics, and calibration math. Lint and formatting checks pass.
