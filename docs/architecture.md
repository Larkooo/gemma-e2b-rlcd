# Parallel decision inference

The default `JSONMLXBackend` scores defined answers using Gemma's native JSON format. It shares prompt construction and media preprocessing with the normal-generation path used by the playground comparison.

## Shared state

One prefix contains the complete text/media input, all questions and criteria, and the assistant generation header. Gemma computes this prefix once per request. Each field then branches at its own JSON key, such as `{"priority": `, and attends to the shared prefix and its own suffix.

The prefix cache remains unchanged while branches run. KV computation is reused; the MLX implementation replicates KV storage across batch rows. Caches are scoped to a request. The branch batch size defaults to eight and is configurable from 1 to 64; additional batches run sequentially.

Every field sees the full schema. Fields do not see one another's answers, so tasks with answer dependencies need an explicit sequence of requests.

## Candidate likelihoods

For a field with allowed values `c₁ … cₖ`, the scorer computes the conditional log likelihood of each complete JSON value:

```text
s(c) = Σₜ log P(cₜ | shared input, field prefix, c₍<ₜ₎)
p(c) = exp(s(c) / T) / Σⱼ exp(s(cⱼ) / T)
```

A common candidate-token prefix can be factored into the field prefix without changing the normalized distribution. The default temperature `T` is 1.

For single-token candidates, all candidate scores come from one vocabulary output vector per field. For multi-token candidates, known token sequences are teacher-forced in bounded batches. The causal model scores every remaining token, including string terminators; it does not iteratively sample the next answer token. This preserves distinctions between labels sharing initial tokens or entire word prefixes. Whitespace at the value boundary is retained during tokenization.

Work grows with the number and length of candidates. A field with 64 long labels can cost more than several binary fields.

## Output contracts

| Contract | Scoring | Returned value |
| --- | --- | --- |
| `Choice` | Normalize allowed category scores together | Winning category and full distribution |
| `Independent` | Expand each label into its own boolean field | One yes probability per label |
| `Noul` | Score JSON `true` and `false` | Probability of the true proposition |
| `Score` | Normalize scores over ordered level indices | Distribution and probability-weighted level |

Nested independent fields retain their parent and child JSON keys. JSON is assembled in code after scoring, so returned values follow the requested schema.

`confidence` is one minus normalized entropy: a measure of how concentrated the distribution is. `allowed_token_mass` measures probability assigned to allowed tokens, or summed complete candidate-sequence probabilities for multi-token fields. Neither is an empirical accuracy estimate. The response records the probability source and calibration status; temperature fitting is available in `calibration.py`.

## Backbone and precision

Inference uses all 35 Gemma layers with packed 4-bit weights and float32 floating parameters/activations. The native vision and audio towers, video timestamps, soundtrack extraction, and processor sampling settings are retained.

Single-token branches gather each row's actual answer position before the final 20 KV-sharing layers. Those layers process the answer position using its original mask, position, and per-layer inputs. The first 15 layers still process every suffix token and provide the complete KV state. Unequal suffix lengths use right padding and row-specific answer positions. Vision position embeddings use indexed table reads.

Float32 reduced the probability drift observed when changing execution shapes under bfloat16. Numerical measurements are preserved in the [cache comparison](experiments/cache.md) and [full-depth execution report](experiments/full-detail.md). [Native JSON regression checks](experiments/native-json.md) cover text, images, speech, video, and soundtracks.

## Media and limits

- Text and local images; the web interface accepts up to eight images.
- One audio clip up to 30 seconds.
- One silent video up to 60 seconds, sampled at a target of 1 fps with a 32-frame processor cap.
- A video's soundtrack is included automatically. Videos with audio have a 30-second limit; a second audio source returns an error.
- Up to 8,192 processed input tokens by default. Inputs exceeding the limit are rejected without truncation.
- Up to 32 named fields and 128 primitive decisions in the playground.
- Questions and descriptions are strings; the Python contracts use name-to-description mappings.

Video sampling can miss brief events. The limits describe the current serving configuration rather than the checkpoint's maximum context capacity.

## Additional backends

| CLI option | Implementation | Purpose |
| --- | --- | --- |
| `--backend json` | `JSONMLXBackend` | Default native JSON candidate scoring |
| `--backend cached` | `CachedMLXBackend` | Earlier single-token answer-code scoring |
| `--backend catalog` | `CatalogMLXBackend` | Shared-question catalog experiment |
| `--backend head` | `DecisionHeadBackend` | Candidate-conditioned head; requires a checkpoint |

The cached path assigns tokenizer-verified codes to options and scores per-question suffixes against shared state. Historical answer-code and catalog measurements are retained for reproducing those experiments. The [training guide](training.md) describes the head separately.

## Streaming startup

The visual demo starts scoring and normal generation together on separate worker streams with separate processors and KV caches. Both process the complete media and question schema before producing answers; the UI reports input preparation, prefill, first token, and first completed decision separately. End-to-end concurrent timings include GPU contention.

The scorer encodes the common prompt once for field compilation and keeps up to eight compiled schemas in an LRU cache. Cache keys include the complete prompt and candidate definitions. This cache contains only tokenized field definitions; every request recomputes media features, input KV state, and answer probabilities. Boundary-merge validation and complete-candidate scoring remain unchanged.
