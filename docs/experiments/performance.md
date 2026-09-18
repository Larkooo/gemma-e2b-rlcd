# Parallel fields and KV reuse: version 0.3

This is the historical version 0.3 report. See [FULL-DETAIL-RESULTS.md](full-detail.md) for version 0.4, paired full-request timings, and current training results. Running the scripts with current defaults exercises version 0.4; `compare_execution.py` explicitly reconstructs both execution paths.

The state is encoded and prefilled once per request. All fields within a branch batch then run together through one GPU model call. The default batch holds eight fields; larger requests use sequential batches. Each field can attend to the common prefix and its own question, with no attention between field suffixes. All categories within one Choice are scored from one output vector, not separate model forwards.

The current implementation repeats the prefix KV tensors across batch rows. It reuses their computation but does not maintain one read-only prefix storage buffer throughout attention. Each branch has separate mutable cache state. A broadcast view alone would not provide end-to-end shared storage when the cache appends each question's KV. That would require compatible attention and cache machinery.

## Measured bottleneck and change

A paired local profile separates cache expansion, question decoding, vocabulary projection, and probability extraction. The same prefilled state was reused, both paths were warmed, and five repetitions alternated execution order. Explicit MLX evaluation synchronized each stage. These are stage medians for four fields on this Apple Silicon machine, not end-to-end latency guarantees.

| Shared state | KV expansion, previous path | Decoder, previous path | KV expansion, trimmed path | Decoder, trimmed path |
| --- | ---: | ---: | ---: | ---: |
| Short text | 0.38 ms | 383 ms | 0.35 ms | 178 ms |
| Video with speech | 0.80 ms | 357 ms | 0.66 ms | 188 ms |

Vocabulary projection took approximately 3–5 ms and probability extraction approximately 1–1.5 ms in these medians. Cache expansion was under 1 ms; eliminating that measured stage alone would not remove the dominant decoder cost. Shared storage could still matter for memory and scaling, which this four-field probe does not establish. Raw repetitions and logical prefix sizes are retained in [COMPONENT-PROFILE.json](../../reports/component-profile.json).

The previous batch path processed every question position through all decoder layers. Gemma E2B's final 20 of 35 layers reuse earlier layers' KV, and the installed model implementation supports trimming their queries to the needed output positions. Version 0.3 enables that existing optimization. With suffix lengths `[111, 99, 111, 111]`, the final layers retain the last 13 positions so every row's real answer position survives right padding. Earlier layers still process the full questions. Equal-length rows need only one final position. This roughly halved the measured decoder stage without changing the requested answer positions.

The paired full-versus-trimmed profile retained all eight winning answers. Its maximum probability difference was 0.001413 (0.1413 percentage points). Changing execution shapes can change floating-point results; this is not bit-exact equivalence.

## Updated comparison against independent forwards

The six-state cache check uses identical prepared tokens, media tensors, weights, and answer positions for both paths. Each method was warmed and timed three times with alternating order. These medians include a new shared state encoding/prefill on the cached path, plus question work. They exclude model loading, media file decoding, and tokenization. They must not be interpreted as the latency of branching from an already cached state.

| Shared input | Independent full forwards | One shared prefill + four-field batch | Speed ratio |
| --- | ---: | ---: | ---: |
| Short text | 0.271 s | 0.224 s | 1.21× |
| Image | 2.679 s | 0.775 s | 3.45× |
| Speech | 0.305 s | 0.205 s | 1.48× |
| Video | 3.724 s | 1.142 s | 3.26× |
| Video with speech | 3.858 s | 1.375 s | 2.81× |
| Longer text | 2.594 s | 0.855 s | 3.03× |

The current report is [CACHE-TRIMMED.json](../../reports/cache-trimmed.json). Prior reports are preserved in [CACHE-RESULTS.md](cache.md). Host conditions varied between runs, so comparing the old and new whole-call medians does not isolate the trimming effect. Use the paired component profile above for that attribution. Larger field counts and contexts have not been benchmarked.

- All 24 cached-batch winners matched independent full forwards; this measures agreement, not answer correctness.
- Reversing field order preserved logits exactly on all six probes.
- Shared prefix contents and offsets remained unchanged.
- Maximum cached-batch probability drift versus independent forwards was 0.003811 (0.3811 percentage points). Cached serial drift reached 0.013604 (1.3604 percentage points). These observations are not universal error bounds.
- The longer text prefix crossed the sliding-attention window, and all probes used unequal suffix lengths.
- All 29 automated tests pass, including an unequal-length trimming test that checks the selected answer before each row's padding. Lint and formatting checks pass.

Packed 4-bit weights remain in use with float32 floating parameters/activations. No training, calibration validation, or new representative quality evaluation has run. The earlier synthetic smoke results remain available in the version 0.2 report.

## What larger speed gains would require

GPU batching shares scheduling and model-weight use, but each field still performs question-token computation and attention over the state. A new multimodal state also requires its encoders and prefill. The public request path does not retain state caches between calls.

A compact trained decision-query or scoring head could reduce per-field token work substantially; its speed and quality would need a separate matched experiment. Zero-copy prefix storage is another possible memory/scaling improvement, but the current profile does not support treating KV copying as the main four-field latency bottleneck.

## Reproduce

From the project directory after installing the current package:

```bash
python scripts/profile_components.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media \
  --report work/component-profile.json
python scripts/check_cache.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media \
  --report work/cache-trimmed.json
```

Use `scripts/create_fixtures.py --output-dir work/media` if fixtures are absent. Tested dependency versions are recorded in [requirements-tested.txt](../../requirements-tested.txt).
