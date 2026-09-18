# Training a multimodal calibrated decision model

## Objective and current status

The target is one Gemma 4 E2B backbone supporting arbitrary described Choice, Score, and Noul questions over text, images, speech, and video. Version 0.4 includes a trainable parallel decision head, frozen native multimodal feature extraction, a supervised training runner, and two completed text pilots. Both pilots failed to match the frozen scorer, which remains the default. No LoRA or reinforcement-learning update has run. No checkpoint is validated for calibration or general multimodal accuracy.

The current priority is preserving input detail and improving both speed and quality. Training and runtime probes therefore default to all 35 backbone layers, float32 compute, 280 image soft tokens, and the original video frame cap. The earlier four-layer and reduced-media probes are archived experiments, not the recommended configuration.

TypeSafe calls its method Reinforcement Learning for Calibrated Decisions. The public announcement and documentation describe its intended behavior, but the sources reviewed do not disclose enough detail to reproduce its complete architecture, training data, reward, and optimization procedure. The proposal below is an independent experiment toward similar behavior, not an implementation of TypeSafe's undisclosed recipe. This RLCD is also distinct from the older paper named Reinforcement Learning from Contrastive Distillation.

## Shared model and output semantics

Retain the native image/audio encoders, video frame handling, and multimodal projections. Start with those components frozen. Train a shared candidate-conditioned scoring head and, if needed, small LoRA adapters in the final decoder layers. Candidate descriptions are input, so the classifier can receive new labels without creating a new output layer for every taxonomy.

The implemented head reads the complete sequence of frozen state features and separate question/candidate descriptions. A small query transformer encodes every description using frozen Gemma token embeddings; candidates also receive a pooled representation of their own field's candidate set. Two cross-attention blocks score all candidates against a single state memory, without replicating it for each field. The head has no fixed cat/dog output vocabulary. Its query encoder starts randomly initialized, so pretrained Gemma's question understanding is not automatically retained.

Independent propositions use two-choice softmax probabilities, equivalent to a Bernoulli model. Exclusive options produce scores normalized within their field. Ordinal grading uses a distribution over described levels; code calculates the expected level. The same head and categorical NLL service all three contracts. Candidate and field permutation tests check that the scores follow the candidates and independent fields cannot affect one another.

A fixed classifier with one permanent cat neuron and one permanent dog neuron would not meet the arbitrary-criteria requirement. A randomly initialized head would also discard useful zero-shot behavior; compare initialization/distillation from the frozen baseline, preserving label uncertainty and auditing errors.

Shared multimodal prefill and GPU-batched independent question branches are now implemented in `CachedMLXBackend`. See `docs/experiments/cache.md` for measured latency, numerical drift, and cache-isolation checks. This implementation replicates KV storage per batch row; shared prefix storage through paged attention remains future work. Use the actual serving precision and batching policy during training and calibration. A stock causal decoder does not make arbitrary query tokens in one concatenated sequence independent automatically.

## Completed supervised pilots

Both runs use 64 training scenes, 16 validation scenes, and 16 test scenes, each with four questions: animal Choice, color Choice, count Score, and cat-presence Noul. Scenes are synthetic combinations; candidate order is shuffled. Each run uses 600 AdamW updates, learning rate 0.0003, and seed 11. All candidate scores receive a proper categorical likelihood loss. Backbone weights stay unchanged.

| Scorer | Test correct | Test NLL | Test Brier |
| --- | ---: | ---: | ---: |
| Four-layer, float16 state + trained head | 21/64 | 1.0769 | 0.6510 |
| Full 35-layer, float32 state + trained head | 23/64 | 0.9659 | 0.5818 |
| Full frozen Gemma scorer, current path | 64/64 | 0.0063 | 0.0010 |

Neither learned head passes the quality gate. The full-depth model's training accuracy is also only 120/256, so the failure is not just a small test-set fluctuation. More backbone depth alone did not recover the pretrained question-answering behavior. These runs do not establish that this architecture cannot work, but they do reject deploying these checkpoints as an improvement. Do not trade that quality loss for their runtime.

The archived [four-layer pilot](../reports/head-pilots/text-pilot/README.md) and [full-depth pilot](../reports/head-pilots/text-pilot-full-depth/README.md) retain exact training metadata, dataset hashes, and full reports. Weight files are not distributed; the scripts and synthetic data reproduce the training procedure. The same small test set was used to compare both configurations, making this exploratory ablation evidence, not an untouched final benchmark. No trained multimodal evaluation has run.

The runner rejects overlapping IDs and exact state objects across splits. This does not detect semantic duplicates, different paths to the same media, or shared underlying scenes/speakers. Real datasets must be partitioned by source before creating question variants. The synthetic templates exercise implementation and a narrow learning task; they do not evaluate unseen rubrics or arbitrary descriptions.

From the project directory, train a new full-depth pilot without overwriting existing checkpoints:

```bash
python scripts/train_head.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --train examples/head-pilot/train.jsonl \
  --validation examples/head-pilot/validation.jsonl \
  --test examples/head-pilot/test.jsonl \
  --output work/new-head --reference
```

Each JSONL record contains `id`, `state`, `questions`, and `targets`. Targets are Choice names, Score level indices, Noul booleans, or maps of Independent label booleans. Media paths resolve relative to that JSONL file. Descriptions are retained in full: the head rejects queries over 512 tokens, batches over 16,384 padded query tokens, and requests over 64 primitive fields rather than truncating them. Training materializes frozen state features locally; dataset size and media length therefore affect memory requirements.

Runtime probes are separate from quality measurements:

```bash
python scripts/benchmark_head.py \
  --model /absolute/path/to/full-gemma-4-e2b-mlx-checkpoint \
  --media work/media --report work/head-runtime.json
```

An untrained head exposes timings only and refuses to return predictions. A trained checkpoint requires matching backbone path, depth, precision, media settings, and head configuration. This checks configuration consistency, not immutable backbone content: a production registry must additionally pin and verify weight hashes. Cached schema representations and pretrained/distilled query initialization remain future work.

## Learn accurate probabilities before claiming RL benefits

When outcomes are labeled, start with proper scoring objectives:

- Choice: categorical negative log likelihood, with Brier score as an evaluation measure.
- Noul: binary negative log likelihood and binary Brier score.
- Score: categorical likelihood over rubric levels; also evaluate the ranked probability score and expected-grade error. The full distribution matters, since different distributions can have the same mean.

These are ordinary supervised objectives. Calling their negative loss a reward does not make the procedure a new RL method. Proper scoring rules encourage truthful probabilities in expectation when their assumptions hold; finite data, model limitations, and distribution shift still require empirical calibration checks.

For a later feedback experiment, a bounded reward for a probability forecast can be the negative Brier score:

`r(p, y) = -sum_k (p_k - 1[y=k])**2`

If the outcome is fully observed and this loss is differentiable, direct optimization is the baseline to beat. An RL policy that only earns a point for sampling the correct class generally learns to concentrate on the most likely class; that objective alone does not recover the true probability distribution.

For actions that change the world, define precisely what is being predicted: for example, success conditional on taking each action. Feedback for a chosen action does not label all unchosen actions. Any bandit/RL extension needs exploration, logged selection probabilities, and appropriate evaluation. Keep calibrated outcome probabilities separate from expected utility and action-selection probabilities.

## Bounded experiment

1. Build a labeled pilot containing all three question types across all four modalities. Include negative examples, multiple simultaneous labels, irrelevant media, conflicting text/media, ambiguous evidence, and video questions requiring temporal order or soundtrack information.
2. Split by underlying source item, speaker, video, and scene before creating question variants. Keep train, development, calibration, and final test partitions disjoint. Keep renamed labels and some new rubrics/tasks for a separate generalization test. Do not use the current synthetic smoke fixtures as test evidence after inspecting or tuning on them.
3. Compare the unchanged frozen baseline, frozen + temperature scaling, supervised decision-head/LoRA training, and only then an outcome-feedback variant. Match data, starting checkpoint, inference settings, and evaluation budget. Repeat training with multiple seeds.
4. Fit temperatures/thresholds on the calibration split only. Keep class priors representative of the target workload or report the consequences of any balancing/reweighting.
5. Report NLL, Brier, accuracy, per-class/binary reliability, ordinal ranked probability score, grade error, abstention coverage/error, failures, option-order sensitivity, and calibration by modality. ECE alone is insufficient and depends on binning and sample size.
6. Report cold load, first inference, warm latency, preprocessing/encoder/decoder time, peak memory, question count, media duration, and frame count separately. Keep failures in denominators.

Advance only if the trained model improves held-out proper scores over both frozen baselines without an unacceptable accuracy or modality regression. Report uncertainty intervals; a few successful demos do not establish generalization, calibration, or an RL advantage.

## References

- [TypeSafe announcement and RLCD framing](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe's calibration explanation](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [Gneiting and Raftery: Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
- [Guo et al.: On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Gemma 4 E2B model card](https://huggingface.co/google/gemma-4-E2B-it)
