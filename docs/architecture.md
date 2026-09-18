# Architecture and output contracts


The common prefix contains generic instructions and the complete multimodal state. Each question and its criteria follow that prefix. This order is essential: a question-dependent prefix cannot be reused for different questions.

`CachedMLXBackend` processes media and computes the shared state KV cache once per `system_one` request. It then forks independent cache objects and evaluates question suffixes in a tensor batch on the GPU. This is model batching, not a Python loop presented as concurrency. Unequal suffix lengths use right padding and each row gathers its own final real-token position. Each branch can attend to the shared state and its own suffix, never another field's suffix. The shared cache is not mutated.

Version 0.4 gathers each row's actual answer position before the final 20 KV-sharing layers. They process one position per field, preserving its original attention mask, position, and per-layer inputs. The first 15 layers still process every question token and supply complete KV to the later layers. All 35 layers run. Version 0.3 retained the entire tail spanning unequal answer positions; it remains available as `answer_mode="tail"` for comparisons. Vision position embeddings now use indexed table reads instead of a dense one-hot matrix multiplication. Neither change reduces model depth, media settings, or instruction content.

Each question's options receive distinct, tokenizer-verified single-token codes. All option logits are read from the same output vector, so a 255-option Choice does not require 255 model passes. Multiple Noul fields become separate batched rows. Score computes the expectation over level indices. The three primitives share the same backbone and execution path.

The default branch batch size is 8 (configurable from 1 to 64). More fields are processed in bounded batches against the same prefix. Within a batch they are evaluated together; different batches run sequentially. Memory and compute still grow with branch count, suffix length, and media context. State encoding and prefill reuse can provide substantial savings for larger shared inputs; tiny inputs can be slower because of batching overhead.

KV **computation** is reused. The current MLX implementation replicates prefix KV **storage** across batch rows. It is not zero-copy paged attention, and it is not a reconstruction of Jev's architecture. Caches are scoped to one request; no stale prefix is reused across changed states or model versions.

The cached backend preserves packed 4-bit weights and promotes floating parameters/activations to float32. The original bfloat16 compute path showed unacceptable probability drift when execution shapes changed. Float32 reduced this substantially, but cached and uncached scores are still not bit-identical. See [CACHE-RESULTS.md](experiments/cache.md) for measured differences and timings. Any training/calibration must target the actual serving precision and batch path.

`allowed_token_mass` reports how much probability the original full vocabulary assigned to allowed codes before normalization. A very small value is evidence of a mismatch with the requested answer format. Even a high value does not establish correctness. Code order and wording can change the answer; permutation robustness must be evaluated before deployment.

The default temperature is 1. `calibration.py` provides proper log/Brier scores, reliability bins, and temperature fitting on a dedicated calibration split. Passing a temperature never changes `calibration_status` to validated. Calibration needs independent evaluation, including per-modality and shifted-data checks.

## Current input limits

- Text and one or more local images.
- One speech/audio clip up to 30 seconds.
- One video up to 60 seconds, sampled at a target 1 frame per second. The verified checkpoint's processor caps the resulting clip at 32 frames and may adjust sampling for short clips. This is frame-based video understanding, not continuous high-frame-rate perception.
- A video's soundtrack is extracted and included automatically. Video with a soundtrack is limited to 30 seconds because of the audio limit. A second separate audio stream is rejected, not silently discarded.
- An 8,192-token prototype limit is enforced after preprocessing; excessive inputs fail instead of truncating silently. This is a local operational limit, not the model's maximum context length.
- Descriptions and instructions currently use strings. Jev's additional object/array descriptions, SDK compatibility, and structured-output combinators are not implemented.

These limits are explicit so a successful request actually processes the supplied media. Sampling can miss short visual events; native audio input does not by itself prove dependable general sound-event recognition.
