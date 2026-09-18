# Rejected four-layer text pilot

This experimental head scored 21/64 test decisions versus the full frozen scorer's 64/64. It is retained to reproduce the failed experiment, not recommended for inference. Its four-layer float16 state encoder is superseded by the full-depth default and does not satisfy the current detail-preservation goal.

See [training-report.json](training-report.json) and [head.json](head.json) for results, data hashes, and configuration. Reproduce using the command in [TRAINING.md](../../../docs/training.md), adding `--state-layers 4 --dtype float16` and using a new output directory. There were 600 supervised updates; no RL or calibration validation ran. Only text was used for training and quality evaluation.

The public repository includes metadata and results, not the rejected weight file. Train a new checkpoint with the documented script.
