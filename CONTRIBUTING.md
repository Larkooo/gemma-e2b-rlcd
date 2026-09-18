# Contributing

Follow the README setup, then run:

```bash
python -m pytest -q
ruff check gemma_rlcd tests scripts
ruff format --check gemma_rlcd tests scripts
node --check gemma_rlcd/static/app.js
```

CI runs portable contract and web tests on Linux without model weights. MLX tests and inference require Apple Silicon and run locally.

Keep changes focused. Add tests for changed behavior, preserve each output type's probability semantics, and include answer quality alongside performance measurements. Use the same input, model, and media settings for comparison paths; retain invalid responses in reports.

Training changes should report held-out accuracy, log loss, Brier score, and reliability alongside latency. Fit temperatures on a separate calibration split. See [training and calibration](docs/training.md) for the data format and objectives.

Keep uploads, credentials, model weights, machine paths, and runtime logs out of commits. Use the ignored `work/` directory for generated fixtures and new runs. Published reports should identify their data, model revision, settings, and measurement boundaries.
