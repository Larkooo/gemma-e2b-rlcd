# Experiment reports

These reports preserve local prototype experiments from September 18, 2026. Each result applies to the execution settings and fixtures recorded in that report. They are not a representative model evaluation, a latency guarantee, or evidence of calibrated probabilities.

- `initial-smoke.json`: original uncached bfloat16 integration smoke test.
- `cache-bfloat16.json`, `cache-float32.json`, `cached-smoke.json`: historical precision and cache checks.
- `component-profile.json`, `cache-trimmed.json`: earlier answer-tail optimization.
- `full-detail-comparison.json`, `cache-gathered.json`, `smoke-gathered.json`: full-depth answer-position gathering.
- `two-field-video.json`: separate requests, shared serial fields, and shared batched fields.
- `catalog-comparison.json`, `smoke-catalog.json`: experimental shared-question catalog.
- `head-pilots/`: supervised text pilot configuration, dataset hashes, and failed quality results. No weights are included; reproduce using `scripts/train_head.py` and `examples/head-pilot/`.
- `head-runtime-*.json`, `encoder-check*.json`: experimental head runtime and feature extraction checks. Runtime probes with untrained heads do not establish useful predictions.
- `validation.json`, `web-validation.json`: historical local checks, not hosted CI results.

Absolute workstation model paths were replaced with `models/gemma-4-e2b-it-4bit`. Local command paths were normalized for publication; measurements and predictions were preserved. The exact model repository and revision are recorded in [model-source.json](../model-source.json).

All reproduction commands in [the experiment notes](../docs/experiments/full-detail.md) run from the repository root. Write fresh outputs to the ignored `work/` directory so archived measurements remain intact.
