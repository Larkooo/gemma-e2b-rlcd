# Two video fields: shared input and GPU batching

Both fields run together when submitted in one request with a branch batch size of at least two (the default is eight). They share the video prefill and execute as two rows of one GPU batch. This does not give each field a separate GPU or guarantee twice the throughput.

The local comparison uses separate dog and cat grades describing the proportion of sampled frames containing that animal. Both questions include all five grade descriptions. Each suffix contains 201 tokens. The inputs are four-second, 224x224 synthetic color clips, one silent and one with speech. This is a timing probe, not animal-recognition validation.

All three execution modes use the same resident Gemma checkpoint, full model depth, float32 computation, and unchanged media settings. Each timed call includes fresh video/audio processing, prefill, both field computations, and probability normalization. Model loading and final JSON serialization are excluded. Six repetitions follow warmup, using all six mode orderings.

| Input | Two separate requests | Shared input, serial fields | Shared input, batched fields |
| --- | ---: | ---: | ---: |
| Silent video | 1.475 s | 1.217 s | 1.198 s |
| Video with speech | 1.481 s | 0.866 s | 0.803 s |

These medians correspond to 1.23x and 1.84x total speedups versus two separate requests. Comparing serial and batched fields after sharing the input gives only 1.02x and 1.08x in the complete-request medians. Most of the observed overall benefit comes from sharing video/audio processing; batching alone is not a demonstrated 2x end-to-end improvement.

Run variation was substantial. For silent video, separate requests ranged from 1.211 to 2.748 s, shared serial from 0.550 to 1.515 s, and shared batched from 0.661 to 1.479 s. These results do not isolate a stable small batching gain or establish universal speed ratios. The original source video, sampling workload, field lengths, and GPU contention all matter.

All compared winners matched in every repetition. Maximum probability drift was 0.003211 for silent video and 0.000002191 for video with speech. Agreement is not a correctness or calibration claim. See the [raw repetitions, stage timings, and distributions](../../reports/two-field-video.json).

Reproduce from the project directory:

```bash
python scripts/compare_two_fields.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/two-field-video.json
```
