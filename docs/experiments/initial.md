# Local integration results — 2026-09-18

This report preserves the initial sequential bfloat16 baseline. The current default uses cached, batched float32 computation; its validation is in [CACHE-RESULTS.md](cache.md). The smoke script now exercises the current backend, so rerunning it need not reproduce this historical report's predictions.

The frozen Gemma 4 E2B model successfully processed text, image, speech, video, and video-with-speech inputs through native multimodal inference. All three primitive contracts were exercised on every input category.

**15/15 inference calls completed; 13/15 simple expected answers matched. No model training or calibration validation has occurred.** These deliberately tiny synthetic fixtures are integration checks, not a quality benchmark or evidence of general multimodal reliability.

| Input | Choice | Noul | Score's highest-probability level |
| --- | --- | --- | --- |
| Text describing a cat and no dogs | Correct | Correct | Incorrect |
| Uniform red image | Correct | Correct | Correct |
| Synthesized speech mentioning a dog | Correct | Correct | Correct |
| Red-then-blue video | Correct temporal order | Correct | Correct |
| Red-then-blue video with dog speech | Correct speech classification | Incorrect | Correct visual grade |

The text grading failure assigned roughly 99.98% probability to the wrong level. The combined video/speech input classified the spoken animal correctly, then incorrectly answered the separate dog-presence proposition. These errors directly show why a typed output and highly concentrated probabilities are not enough. Both failures remain in the saved report.

The inference run used one loaded model and sequential questions. First-use kernel compilation, preprocessing, media encoders, and decoder execution all contribute to the recorded per-call times. These are single observations, not controlled warm/cold latency measurements. The processor also emitted an empty-mel-filter warning at load time; broader speech-quality validation remains necessary.

## Reproduce

From the project directory, using the existing environment:

```bash
python scripts/create_fixtures.py --output-dir work/media
python scripts/smoke.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media \
  --report work/smoke-reproduced.json
pytest -q -o cache_dir=work/pytest-cache
```

Fixture creation requires macOS's `say` with the Samantha voice, Pillow, and ffmpeg. It creates a plain color image, synthetic speech, a four-second color sequence, and the same sequence with speech. No user audio or video is read.

The full inference outputs, expected answers, failures, timings, and local model path are in [RESULTS.json](../../reports/initial-smoke.json). The tested dependency versions are in [requirements-tested.txt](../../requirements-tested.txt).

Model provenance read from the reused checkpoint's `source.json`:

- Repository: `mlx-community/gemma-4-e2b-it-4bit`
- Revision: `238767527555cb75a05732a84dff5d6ba0dd6809`
- Backend: MLX-VLM 0.7.1; MLX 0.32.2; Transformers 5.17.0.
- Host: Apple Silicon, 32 GiB physical memory.

## Contract and math checks

23 automated tests passed. They cover exclusive and independent probability semantics, weighted grading, Noul criteria ordering, validation failures, finite/stable scores, input-path validation, proper scoring rules, and temperature recovery on a synthetic known-frequency example with separate evaluation outcomes. Lint and formatting checks also passed.

The temperature recovery test verifies the calibration implementation's math. It does not establish calibration of Gemma's predictions. Frozen-model smoke fixtures were not used to fit a temperature or train a head.
