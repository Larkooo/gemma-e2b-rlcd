# Contributing

Use Python 3.12+ and an Apple Silicon Mac for MLX tests and inference. Follow the README setup, then run:

```bash
python -m pytest -q
ruff check gemma_decisions tests scripts
ruff format --check gemma_decisions tests scripts
node --check gemma_decisions/static/app.js
```

Keep changes focused. Add tests for changed behavior, preserve the probability semantics of each output type, and keep the normal-generation comparison honest about failures and timing boundaries.

The GitHub workflow checks the portable contracts and web backend without downloading model weights. MLX-only tests skip on Linux. Passing that workflow does not establish model quality, calibration, GPU performance, or multimodal accuracy.

For model changes, retain the frozen scorer as a reference. Report held-out quality, failures, latency, and preprocessing settings separately. Supervised negative log likelihood is not an RL implementation; do not label a model calibrated without independent evidence. See [the training plan](docs/training.md).

Do not commit uploaded media, credentials, model weights, local machine paths, or runtime logs. Generated fixtures and new experimental outputs belong in the ignored `work/` directory. Published experiment reports should identify their data, model revision, settings, and limitations.
