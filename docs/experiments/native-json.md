# Native JSON scoring regression fix

The default CLI and web runtime now use `JSONMLXBackend`. The model weights and native media processing are unchanged. This is an inference-interface fix; no RL training or calibration is claimed.

## Diagnosis

On the four-second baby-skunks/dog excerpt, normal Gemma generated 5 baby skunks and 1 dog. The previous answer-code scorer selected 1 and 1, assigning 99.33% of its restricted probability to the wrong skunk count.

Disabling cache reuse reproduced the same 1/1 result. Replacing letters with number codes or simply shortening the question did not recover the normal answer. This isolated the observed discrepancy from cache batching: it was already present in the decision prompt and answer representation.

Using the complete normal-generation prompt, then branching at each native JSON value position, recovered 5/1 in both cached and uncached evaluation. JSON whitespace matters: digits must be scored after the expected colon-space context, rather than at a position where the model predicts whitespace. The old skunk and dog suffixes were 287 and 265 tokens; native JSON prefixes are 6 and 4 tokens. The full question descriptions now belong to the common prefix, so this does not discard instructions.

## Implementation

- Normal generation and native scoring share `prepare_generation`, including image, audio, video, and soundtrack handling.
- All questions and candidate meanings stay in the shared prompt. Each field branches at its actual JSON key, including nested independent labels.
- Single-token values are scored together at their actual answer positions.
- Multi-token values are scored over their complete token sequences, including closing quotes. Tests cover prefix labels, shared first tokens, quoted/Unicode names, different candidate lengths, padding, and bounded batches.
- JSON values are assembled programmatically. Native scoring does not generate an output string one token at a time.
- Probability calibration remains unvalidated. Fields share schema context but do not condition on other fields' generated answers, so universal equality with joint generation is not promised.

## Local model checks

The saved [results](../../reports/native-json-validation.json) contain ten cases on Apple M5 / 32 GB, with the pinned full Gemma 4 E2B 4-bit checkpoint and float32 compute. They are development regression checks, not a held-out model benchmark.

| Case | Native scorer outcome |
| --- | --- |
| Four-second skunk excerpt | 5 baby skunks, 1 dog; matches normal |
| Text: animals, count, negative label, independent labels | All expected answers; matches normal |
| Invoice image | Invoice, USD, unpaid; matches normal |
| Subscription speech | Cancel, end of month, no refund; matches normal |
| Motion video | Correct objects; both paths incorrectly say no movement |
| Video with synthesized soundtrack | Correct spoken animal and both visible colors; normal JSON invalid |
| Speech: five skunks, one dog, do not call vet | All expected answers; matches normal |
| Speech: no skunks, two dogs, call vet | All expected answers; normal JSON invalid |
| Video plus contradictory separate speech | Speech answers correct; both paths incorrectly let speech override visible skunk count |
| Full 21-second user video, original soundtrack retained | 5 baby skunks, 1 dog; matches normal |

All eight schema-valid normal responses matched the native scorer's discrete answers. Two cases had errors shared by both paths. The full-video case has no independently annotated ground truth; its result establishes agreement, not counting accuracy. Test inputs do not establish general audio understanding, temporal reasoning, calibration, or robustness to conflicting media.

One-run processing times, excluding model loading and upload:

- Four-second excerpt: native 1.446 s, normal 1.353 s.
- Full video with original soundtrack: native 3.171 s, normal 3.328 s.

These observations do not establish a 3× gain. Input processing dominates these video requests; multi-token candidates also cost more than single-token codes. Run order and first-use compilation can affect timing.

The restarted local web server was also checked through `/api/compare`: the full video retained its soundtrack and returned 5/1 in both paths (native 3.129 s, normal 3.179 s). A speech clip returned five skunks, one dog, and “Do not call the vet” in both paths (native 0.387 s, normal 0.792 s). Raw responses and media hashes are in [web validation](../../reports/native-json-web-validation.json). These are individual integration runs with the same timing limitations.

The user's clip is not redistributed in the repository. Unit tests use synthetic token models and do not need weights. Actual-model validation requires local media and the full multimodal checkpoint.

The existing fixture smoke runner now defaults to native JSON scoring. After creating the fixtures with `scripts/create_fixtures.py`, run `python scripts/smoke.py --model MODEL --media FIXTURE_DIRECTORY --report REPORT.json --compare-normal` to check text, image, speech, video, and video soundtracks against both expected answers and ordinary generation. Failures and invalid normal JSON remain in the report.
