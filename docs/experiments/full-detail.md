# Full-detail execution and decision-head experiments: version 0.4

The default scorer is faster on the local paired probe while retaining the same winning answers. It preserves all input tokens, complete question/option descriptions, all 35 decoder layers, float32 floating parameters/activations, and the existing image/audio/video settings. Accuracy has not improved on the current smoke set. The new trained heads fail the quality comparison and remain experimental.

The objective is speed and accuracy without discarding detail. Reduced-depth and reduced-media probes are not adopted.

The follow-up [two-field video comparison](two-fields.md) separates shared video processing from GPU field batching. Parallel execution is supported, but two fields do not guarantee a 2x speedup.

## Default execution change

Gemma E2B's first 15 layers process each complete question and produce its KV. The last 20 layers reuse those KVs. Previously a batch with unequal question lengths retained the entire tail spanning all answer positions in those last layers. The new path gathers each row's real answer position, preserves its mask, rotary position, and per-layer inputs, and processes just that position through all remaining layers. Other late-layer output positions cannot contribute to the selected answer because these layers read the earlier shared KV.

Vision position embeddings use table indexing in place of dense one-hot multiplication. This preserves the mathematical operation but, like changing attention shapes, can change floating-point rounding. The native and extracted full-depth state features matched exactly on the four live encoder probes. This is measured agreement, not a universal numerical guarantee.

Fields still execute in GPU batches, with one shared state prefill per request. Prefix KV computation is reused; the default backend still replicates its storage per branch. The separate experimental head uses one state memory for all candidate queries. No token-by-token answer generation or generated JSON is involved.

## Paired complete-request timing

Apple M5, 32 GB unified memory; resident four-bit Gemma 4 E2B checkpoint, float32 computation; MLX 0.32.2 / MLX-VLM 0.7.1. Each shape was warmed before five repetitions with alternating reference/optimized order. Every timed call reprocessed the input, encoded/prefilled the state, scored all fields, and normalized the logits. Model loading, JSON file reading, and final serialization are excluded. Both paths use identical inputs and questions. The test used batch size 32; the public default remains 8. The 28-field case repeats four question templates and measures scaling, not 28 distinct tasks.

| Input | Fields | Previous median | New median | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Text | 4 | 300 ms | 213 ms | 29.0% |
| Text | 28 | 1,393 ms | 870 ms | 37.6% |
| Image | 4 | 708 ms | 617 ms | 12.8% |
| Speech | 4 | 323 ms | 259 ms | 19.8% |
| Silent video | 4 | 789 ms | 684 ms | 13.2% |
| Video with speech | 4 | 876 ms | 791 ms | 9.7% |

All 48 winning answers matched. Maximum probability difference was 0.003059, or 0.306 percentage points. Host variation was material: one optimized video-with-speech call took 1.354 seconds. Five repetitions do not establish a reliable tail-latency estimate. Keep the [raw paired report](../../reports/full-detail-comparison.json) when interpreting these medians; no latency guarantee follows from this probe.

## Correctness and quality

- The separate six-state cache comparison retained all 24 winners versus independent full forwards, including a context crossing the sliding window and unequal question lengths. Reversing field order changed no logits and the shared prefix stayed unchanged. Maximum batch probability drift versus independent forwards was 0.008152 (0.815 percentage points); serial cached drift reached 0.013604. [Cache report](../../reports/cache-gathered.json).
- The default path completed all 15 synthetic multimodal smoke questions and answered 12 correctly, unchanged from the prior baseline. Failures remain in text count grading, silent-video color-count grading, and a dog proposition over the soundtrack. [Smoke report](../../reports/smoke-gathered.json).
- The experimental shared catalog scored 14/15 on the same smoke set, correcting those cases but introducing a speech proposition error. Field sets share context on this path, so independence and quality cannot be inferred from the higher aggregate score. It stays opt-in. [Catalog smoke](../../reports/smoke-catalog.json), [separate prompt/timing comparison](../../reports/catalog-comparison.json).
- Two supervised text head pilots scored 21/64 and 23/64 against the frozen scorer's 64/64. They do not meet the quality requirement. No learned multimodal quality or calibration result is established. [Training details](../training.md).

The smoke fixtures are already inspected and are integration checks, not a representative held-out benchmark. Agreement with an earlier model can preserve its errors. The current change demonstrates reduced execution work and no observed winner regression in the matched probes; it does not prove general accuracy preservation or improvement.

All existing video sampling limits still apply: target 1 fps, processor cap of 32 frames, native image reduction, and audio duration limits. Preserving those settings means preserving the existing input representation, not lossless processing of every source-video frame. There is no new compression or reduced-media setting in the default path.

## Reproduce

From the project directory after reinstalling the package:

```bash
python scripts/compare_execution.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/full-detail-comparison.json
python scripts/check_cache.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/cache-gathered.json
python scripts/smoke.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/smoke-gathered.json
python scripts/check_encoder.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/encoder-check-full-depth.json
```

Use `scripts/create_fixtures.py --output-dir work/media` if fixtures are absent. The code pins MLX-VLM because the answer-position adapter depends on its native Gemma implementation. [Tested dependencies](../../requirements-tested.txt) and [earlier performance evidence](performance.md) are retained. Final automated validation is recorded in [validation.json](../../reports/validation.json).

Archived four-layer timing probes are retained in `reports/head-runtime-fp16.json`, `reports/head-runtime-indexed.json`, and `reports/head-runtime-compact.json`. They used untrained heads and do not establish usable decision quality. The compact probe additionally reduced media detail and is superseded by the full-detail requirement. Current full-depth runtime and native encoder checks are recorded separately in `reports/head-runtime-full-depth.json` and `reports/encoder-check-full-depth.json`.
