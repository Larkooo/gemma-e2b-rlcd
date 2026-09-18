# Rejected full-depth text pilot

This experimental head scored 23/64 test decisions versus the full frozen scorer's 64/64. It preserves all 35 state-encoder layers and float32 computation, but does not preserve the pretrained decoder's decision quality. It is retained to reproduce the failed experiment, not recommended for inference.

See [training-report.json](training-report.json) and [head.json](head.json) for results, data hashes, and configuration. Reproduce using the default command in [TRAINING.md](../../../docs/training.md) and a new output directory. There were 600 supervised updates; no RL or calibration validation ran. Only text was used for training and quality evaluation.

The public repository includes metadata and results, not the rejected weight file. Train a new checkpoint with the documented script.
