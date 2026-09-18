# Experiment reports

Local measurements from September 18, 2026, with inputs, outputs, timings, and execution settings retained for each run. Start with the [shared-state workload comparison](../docs/experiments/demo-workloads.md) for the current native JSON scorer.

- `initial-smoke.json`: original uncached bfloat16 integration smoke test.
- `cache-bfloat16.json`, `cache-float32.json`, `cached-smoke.json`: historical precision and cache checks.
- `component-profile.json`, `cache-trimmed.json`: earlier answer-tail optimization.
- `full-detail-comparison.json`, `cache-gathered.json`, `smoke-gathered.json`: full-depth answer-position gathering.
- `two-field-video.json`: separate requests, shared serial fields, and shared batched fields.
- `catalog-comparison.json`, `smoke-catalog.json`: experimental shared-question catalog.
- `head-pilots/`: supervised text pilot configuration, dataset hashes, and failed quality results. No weights are included; reproduce using `scripts/train_head.py` and `examples/head-pilot/`.
- `head-runtime-*.json`, `encoder-check*.json`: experimental head runtime and feature extraction checks. Runtime probes with untrained heads do not establish useful predictions.
- `validation.json`, `web-validation.json`: historical local checks, not hosted CI results.
- `native-json-validation.json`: native JSON scoring versus compact normal generation on text, images, speech, video, soundtracks, and the reported counting regression. Contains disagreements with expected answers and invalid normal outputs; see [interpretation](../docs/experiments/native-json.md).
- `native-json-web-validation.json`: checks through the restarted local web server for the full skunk video with its soundtrack and speech counts/negation. Both paths match the expected answers; the video expectation is baseline agreement, not independent annotation.
- `demo-workload-benchmark.json`: warmed repeated comparisons on shared-state support, security, inbox, independent-label, and large-choice workloads. Includes serial scoring controls, all raw responses, and predeclared development expectations; see [interpretation](../docs/experiments/demo-workloads.md).
- `demo-workload-host-pressure.json`: interrupted earlier attempt while the idle web model was also resident and host memory pressure/latency variation were observed. Retained separately and excluded from the subsequent run's medians.
- `demo-workload-web-validation.json`: browser checks of all four workload presets and the live 28-field comparison, including the corrected default yes/no wording mismatch.

Absolute workstation model paths were replaced with `models/gemma-4-e2b-it-4bit`. Local command paths were normalized for publication; measurements and predictions were preserved. The exact model repository and revision are recorded in [model-source.json](../model-source.json).

All reproduction commands in [the experiment notes](../docs/experiments/full-detail.md) run from the repository root. Write fresh outputs to the ignored `work/` directory so archived measurements remain intact.
