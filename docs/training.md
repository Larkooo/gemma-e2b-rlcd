# RLCD, training, and calibration

RLCD—Reinforcement Learning for Calibrated Decisions—focuses on learning decision probabilities from outcome feedback. The target is a model that selects useful answers and expresses uncertainty consistently: among comparable predictions assigned 80% probability, the predicted outcome should occur about 80% of the time.

The repository separates inference, supervised head training, and probability evaluation:

| Component | Implementation |
| --- | --- |
| Default inference | Pretrained Gemma candidate likelihoods through `JSONMLXBackend` |
| Trainable decision head | Candidate queries cross-attend to shared Gemma state features |
| Training runner | Supervised categorical likelihood with AdamW |
| Probability evaluation | Log loss, Brier score, reliability bins, and temperature fitting |
| Outcome-feedback extension | Reward formulation described below |

## Candidate-conditioned head

`DecisionHeadBackend` extracts the complete sequence of state features from Gemma. A small query transformer reads question and candidate descriptions using Gemma token embeddings. Each candidate also receives a pooled representation of its own field's candidate set. Two cross-attention blocks score these queries against one shared state memory.

Labels are inputs, so the head can accept a new taxonomy without creating a new fixed output layer. Exclusive choices normalize together, independent propositions use separate two-choice distributions, and grades use a distribution over ordered levels. The head trains with grouped categorical negative log likelihood while the backbone remains fixed.

The query encoder begins with randomly initialized parameters. Training and runtime probes default to all 35 backbone layers, float32 compute, 280 image soft tokens, and the original video frame cap. Candidate/field permutation tests check that scores follow the candidates and that independent fields remain isolated.

## Train a head

```bash
python scripts/train_head.py \
  --model models/gemma-4-e2b-it-4bit \
  --train examples/head-pilot/train.jsonl \
  --validation examples/head-pilot/validation.jsonl \
  --test examples/head-pilot/test.jsonl \
  --output work/new-head --reference
```

Each JSONL record contains `id`, `state`, `questions`, and `targets`. Targets are Choice names, Score level indices, Noul booleans, or maps of Independent label booleans. Media paths resolve relative to the JSONL file.

The runner rejects overlapping IDs and exact state objects across splits. Split real datasets by source scene, document, speaker, or video before creating question variants. Queries above 512 tokens, batches above 16,384 padded query tokens, and requests above 64 primitive fields return errors rather than being truncated.

Frozen state features are materialized locally. Dataset size and media length determine memory requirements. A checkpoint records backbone path, depth, precision, media settings, and head configuration; these must match at inference time.

Runtime measurements can be collected separately:

```bash
python scripts/benchmark_head.py \
  --model models/gemma-4-e2b-it-4bit \
  --media work/media --report work/head-runtime.json
```

## Probability objectives

For labeled outcomes, the training runner minimizes categorical negative log likelihood:

```text
L_NLL(p, y) = -log p(y)
```

`calibration.py` also evaluates Brier score and reliability bins:

```text
L_Brier(p, y) = Σₖ (pₖ - 1[y = k])²
```

Both objectives evaluate the full probability forecast. Choice questions use a categorical distribution; yes/no propositions use a binary distribution. Grades use categorical likelihood over rubric levels, with expected-grade error and ranked probability score as useful additional evaluation measures.

Temperature fitting rescales candidate scores before softmax. Fit the temperature on a dedicated calibration split and evaluate its effect on a separate test set. The API records `calibration_status` independently of the temperature value. Accuracy, log loss, Brier score, and reliability describe different aspects of prediction quality.

## Outcome-feedback extension

A feedback-based training extension can use negative Brier score as a forecast reward:

```text
r(p, y) = -Σₖ (pₖ - 1[y = k])²
```

When complete labels are available, direct supervised optimization of this loss provides the reference method. If an action changes what is observed, define the predicted event explicitly—for example, success conditional on taking that action. Feedback on the selected action does not provide labels for every unchosen action. Such an extension needs exploration, logged selection probabilities, and evaluation that accounts for that selection.

Keep outcome probabilities distinct from expected utility and the policy's action-selection probabilities. A reward for choosing the correct class alone encourages selecting the most likely class; it does not by itself train the whole probability distribution.

## Recorded head experiments

Two text pilots used 64 training scenes, 16 validation scenes, and 16 test scenes, each with four questions: animal Choice, color Choice, count Score, and cat-presence Noul. Both used 600 AdamW updates, learning rate 0.0003, and seed 11, with shuffled candidate order.

| Scorer | Test correct | Test NLL | Test Brier |
| --- | ---: | ---: | ---: |
| Four-layer state + trained head | 21/64 | 1.0769 | 0.6510 |
| Full 35-layer state + trained head | 23/64 | 0.9659 | 0.5818 |
| Pretrained Gemma answer-code reference | 64/64 | 0.0063 | 0.0010 |

The head configurations scored below the pretrained reference, so the default inference path uses Gemma's own candidate likelihoods. These synthetic pilots share one small test set and measure head development, separately from the native JSON workload benchmarks.

The [four-layer report](../reports/head-pilots/text-pilot/README.md) and [full-depth report](../reports/head-pilots/text-pilot-full-depth/README.md) retain training metadata, dataset hashes, and complete results. The included scripts and synthetic data reproduce the training procedure.

For broader evaluation, hold out sources and some label/rubric descriptions, include all four modalities, and report per-modality quality, option-order sensitivity, failures, latency, and peak memory. Use the serving precision and batching settings during evaluation.
