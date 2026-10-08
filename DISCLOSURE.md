# metask-jev-rain-4B disclosure

## Identity

`metask-jev-rain-4B` is a separate inference system using the same pinned
policy-mix weights as `metask-jev-4b`. The code identity is the commit given in
the submission email. The configuration's
`model_key="metask-jev-4b"` identifies the weights, routing threshold and calibration
blocks; it is not the public system name.

Weights: [wayfind/metask-jev-4b-policy-mix](https://huggingface.co/wayfind/metask-jev-4b-policy-mix/tree/ea20fe85b28733b1522721dec119bec50947a869).
`model.safetensors` is 9,078,620,536 bytes, with file SHA256
`f4b40475d18e0a38638b985ed51816c012619e0d4138166704e2ea0aac7a12b3`.
This hash identifies the weight file, not a manifest of filenames.

Four auxiliary files are included byte-for-byte from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a):

| File | SHA256 |
|---|---|
| preprocessor_config.json | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| video_preprocessor_config.json | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| merges.txt | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| vocab.json | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

Their Apache-2.0 attribution and license are in `model_aux/NOTICE` and
`model_aux/LICENSE`. Service code is licensed under MIT in `LICENSE`. The model
folder is mounted read-only. Instead of creating a new model revision, this
package carries the four upstream files and verifies them by SHA256 at startup.
The loaded input is therefore fixed by two immutable identifiers: the model
revision above and this code commit. The separate `merges.txt` and `vocab.json`
provide the BPE merges and vocabulary as sidecar files, using entries already in
the checkpoint's tokenizer. `merges.txt` contains the same 247,587 BPE merges
and `vocab.json` the same 248,044-token vocabulary as the checkpoint's
`tokenizer.json`. The checkpoint's tokenizer file, including its additional
tokens and tokenizer settings, is unchanged and checked by SHA256 at startup.
Qwen3.5's vLLM architecture uses the Qwen3VL processor, which loads image/video
processor configuration as architecture metadata; this service accepts text only.

Startup constructs a symlink view containing only `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`,
`model.safetensors` and the four auxiliary files. All five identity-file hashes
and auxiliary hashes are checked; the weight file is checked by byte size at
startup, not by rereading its full SHA256. If all four auxiliary files are absent
from the model folder, the view uses the package copies; a complete model-folder
set must have the same hashes. A partial set or unexpected weight/tokenizer/
processor files are rejected.

The weight repository's `README.md`, `eval/`, `install.sh`, `selftest.sh`,
`serve.py`, `serve.sh`, `temperature.json` and `.gitattributes` are not linked into
the view and are not used by this service. The old model card and evaluation
material describe `metask-jev-4b`; this system does not adopt their ranking or
validation claims. Same-named scripts in this code package are separate files.

The model-file verification recorded on 2026-10-08 found that the clean model
directory's 21 revision files matched the pinned repository by size and SHA256.
That directory was assembled with hard links from the measured host's existing
files; the four auxiliary files were supplied from this package. Its five
identity-file hashes are:

| File | SHA256 |
|---|---|
| config.json | `4d4ea499a469baaeb518473ff0e2b1e823697024639a93319219e27c9759336e` |
| tokenizer.json | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| tokenizer_config.json | `66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b` |
| generation_config.json | `afa48c3c3f7a3e873e82b5e77af288143c5b400d5df39ae4405e40e8e92d9fcb` |
| chat_template.jinja | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

The historical freeze record of 2026-10-06 identifies the same weight SHA256 and
byte size stated above. The auxiliary files on the measured host were checked
against the package on 2026-10-08 and matched the four hashes above. These later
checks do not by themselves establish the complete file contents of the model
directory at the time of each 2026-10-06 run.

## Changes from the reviewed code

Relative to reviewed commit `7feb587df7fddcaa7b27b042661eb1715b99e4aa`, this package
adds explicit 422/503 failure handling, fatal-engine detection and a three-item
failure limit; isolated model views and writable state; and three-attempt
attention-backend startup with per-attempt logs and isolated cache, home and
temporary paths. The existing offline supervisor is retained. Explicit
`attention_backend` is passed only for the `FLASH_ATTN` and `TRITON_ATTN` fallback
attempts; auto remains the first selection policy.

The environment gate no longer requires Python 3.10 and all 201 versions in the
historical lock. It requires Python >=3.10 and the five base package versions
listed in Environment. The four auxiliary model files are carried in
`model_aux/` and verified by SHA256; the evaluator no longer runs
`prepare_model.sh`.

In `config.json`, the edits add `pricing.submission_models.metask-jev-rain-4B`
and remove the unused metadata keys `model.vllm_expected_version` and
`generation_reference`. In `calibration.json`, only three `status` strings change.
All remaining JSON values and types, including the numerical routing,
calibration, generation and accounting parameters, are unchanged; the
`json_equivalence.py` check verifies JSON equivalence against that reviewed
commit and separately checks the selected inference AST nodes. That selected-node
check is not a proof of every request path or GPU numerical equivalence.

| File | Reviewed commit SHA256 | Current SHA256 |
|---|---|---|
| config.json | `693881a2f0de01b31770d6f19db1bc1cef764b81b6a7eb81651f51d83403ee19` | `348df67c0f625afca1a720db2e4ed3d25b2a477948816c502445bf391943c864` |
| calibration.json | `3c595f948a7ec361179ba0482cee11d2366d4ec68cf63d7b2c2dd71bb2524614` | `892d83759d09f3dfac64f31c5c3b107517ba0ce031e65f6485f300893ab20d3e` |

Parameter identity does not establish bit-for-bit inference identity across
Python, GPU or kernel versions. Python 3.12 changed floating-point `sum()` from
the Python 3.10 implementation; probability calculations can differ at rounding
precision even with the same logits. GPU inference can add numerical variation.
The comparisons in Results distinguish parameter checks from measured outcomes.

## Training data

This package loads the policy-mix checkpoint identified above; it performs no
training at startup or during evaluation.

This system uses the same weights and Hugging Face revision as the listed
`metask-jev-4b` checkpoint, with `adapter=null`. The
[model card at that revision](https://huggingface.co/wayfind/metask-jev-4b-policy-mix/tree/ea20fe85b28733b1522721dec119bec50947a869)
describes the training as follows:

- Qwen/Qwen3.5-4B as the backbone, with merged LoRA rank 16 and alpha 32 on the
  language model's linear layers;
- 44.8k view-augmented decisions from 11 public datasets, with three criteria
  orderings per item;
- 16.1k MASSIVE utterance-to-domain decisions across 14 locales, upsampled four times;
- 390 synthetic policy-family decisions with teacher soft labels, upsampled twice.
  Their families mirror JevBench's hard tier: long_policy, multi_hop,
  temporal_numeric, judge_hard, trap, probability, ambiguous, adversarial and
  tradeoff; the card describes states of at most 2,048 tokens.

These are the model card's reported counts and descriptions. The operator also
records this benchmark-directed development in its
[v1.4.x result notes for metask-jev-4b](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/results/v1.4.2/jevbench-v1.4.2-results.json).

Separate in-house fine-tuning experiments used the public items for monitoring;
none of their weights is used by this system.

The model card's "Score evolution" figure shows repeated evaluation on all
231 public JevBench items across six development variants of the 0.8B and 4B
models, with scores from 55.8% to 80.1%. Its training section explicitly relates
the synthetic policy families to JevBench's hard tier. This system's calibration
is fitted separately as described below; the checkpoint repository's
`temperature.json` is excluded from the runtime model view.

## Use of public JevBench items

The package's threshold and calibration metadata refer to 231 public items from
[JevBench at bb05a335bc809e61b20c0f745d25499a82b326fc](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
The original bytes under `datasets/public/`, as recorded in that revision's
`datasets/manifest.json`, are:

| File | SHA256 |
|---|---|
| easy.jsonl | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` |
| original.jsonl | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` |
| hard.jsonl | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` |

Historical venv runs used CRLF copies of the same files. All 231 items are
identical after line-by-line JSON parsing. The hashes above identify the
committed LF bytes used for the image-run task files.

The threshold metadata records `n=231` and a fitted slow eligibility share of
`58/231`. Calibration records raw fitted values, sample counts, a manual fast
probability floor and slow-block shrinkage. Probability fitting uses target labels;
it is not an unlabelled procedure. `held_out_validated` is false. Results on these
231 items are in-sample observations, not held-out performance estimates.

The public items informed the following development work. "Labels" below means
the public gold answers; these uses make the reported public results in-sample.

1. **Prompt comparisons (labels).** On 2026-10-01 the checkpoint was evaluated
   with its original prompt and a shorter wrapper; the shorter wrapper scored
   lower and was not adopted. The shared prompt used by this package was written
   on 2026-10-02 before results with it existed. Its single-pass evaluation on
   this checkpoint on 2026-10-05 answered 181/231 correctly. Six alternative
   presentations evaluated on the 12B model were not adopted.
2. **Router and unparsed-draft policy (labels, 2026-10-05 to 2026-10-06).** For
   predicting "single pass wrong and draft right", structural features gave
   AUROC 0.508, the fast top-two log-ratio margin 0.807 and probability difference
   0.804. The log-ratio margin was selected. Keeping the fast answer for an
   unparsed draft scored 2.1 percentage points above treating it as wrong.
3. **Draft length (labels, 2026-10-05 to 2026-10-06).** With all items drafted and
   unparsed drafts retaining the fast answer, word/token limits of 40/64, 80/128,
   150/256 and 250/512 gave accuracies 0.818, 0.831, 0.848 and 0.874. The selected
   limit is 250 words and 512 tokens.
4. **Slow share (labels, 2026-10-05 to 2026-10-06).** Offline simulations tested
   the lowest-margin 5% through 60% in 5% steps for 40-, 80- and 150-word drafts,
   and 20% through 40% for 250-word drafts. With 250-word drafts, 25% and 35%
   corrected 10 and 15 items net. The selected 25% kept the 75th latency
   percentile on the fast path in the local latency projection.
5. **Threshold (without labels, 2026-10-06).** The configured
   `2.3108509669020307` is the midpoint of the 58th and 59th smallest fast margins
   in a fast-only service run; 58/231 items fall below it.
6. **Calibration (labels, 2026-10-06).** The fast block uses the same fast-only
   run: 58/74 yes/no answers correct, with choice and score temperatures
   1.681792830507429 and 1.189207115002721. Slow and fallback fits use a routed run
   at quota 0.25 on the same items; the table below gives each block's counts.
   The fast yes/no probability was raised by hand to 0.81 to stay above the
   METHOD-v1.5 decisiveness boundary of 0.80 with numerical margin. Values strictly
   between 0.20 and 0.80 count as abstentions under that method. Slow-block
   shrinkage was selected from six treatments using in-sample and leave-one-out
   ECE on the 111 public hard items: leave-one-out ECE 0.0873 versus 0.1019 for the
   unadjusted fit. The comparison evaluated a single prior sample size, `n0=20`;
   it was not a search over prior sizes.
7. **Long inputs (budget choice without labels; robustness scoring with labels).**
   The 16,384-token budget and fast-only rule use the public aggregate description
   of long inputs and local cost and memory limits. On 2026-10-08, adding about
   880,000 unrelated characters to each of 20 public items produced 20/20 answers;
   correct answers changed from 15 to 13. That test changed no parameter.
8. **Cost and engine settings.** The cost-fuse warmup and hysteresis settings were
   fixed before the final harness measurements. Memory fraction 0.85 and
   `max_num_seqs=1` were selected as conservative serving settings for the
   anticipated evaluation GPU; their measured effects are separated in Results.
9. **Additional measurement (labels, 2026-10-06).** Besides the cold runs below,
   an earlier 150-word service answered 194/231 correctly, and a source-tree
   service run after the same items had warmed its cache answered 196/231.
   Service-side benchmark runs also supplied the threshold and calibration data.
10. **Weights.** The submitted system points to the fixed checkpoint identified
    above. Separate fine-tuning experiments that monitored public-item scores are
    disclosed in Training data; their weights are not loaded by this package.

## Sealed items and ImageJevBench

The threshold and calibration inputs documented for this serving package are
the 231 public items identified above. Its long-tail settings use the public
aggregate description below. The checkpoint training account is limited to the
pinned model card and operator result notes cited above; it is not a declaration
about unreleased training material, historical sealed-item access or
ImageJevBench training use.

Configuration contains long-tail statistics of 23 items, approximately 79,000
tokens each, and 1,624 decisions (the v1.5 total stated in
[METHOD-v1.5](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5.md)).
The configuration uses aggregate counts, not item contents. They feed projection and
qualification calculations. The USD 0.0646 per 1,000 decisions cap also controls
the cost fuse; the 1.23-second median limit is used in local qualification and
projection calculations. `limits.raw_median_latency_seconds=0.54` is the local
raw-latency threshold used by `bench.py` for its fast-quantile check. It equals
`(1.23-0.15)/2` under the legacy latency projection; it is not a measured latency.

The [JevBench v1.6.0 leaderboard page](https://benchmarkheaven.com/jev-models),
captured on 2026-10-05, states "caps = 2x (1.23 s, USD 0.0646)" and describes
"23 items of this draw" with "about 77,000 to 81,000 input tokens". It applies
the latency cap to adjusted p50 and describes the self-hosted adjustment as
raw latency multiplied by 2 plus 0.15 seconds. The configured 79,000 tokens is
the midpoint of that range, not a separately published measurement.

The [leaderboard API](https://benchmarkheaven.com/api/jevbench/latest), captured
on the same date, gives unrounded limits of USD 0.06459465517241379 per 1,000
decisions and 1.2329566404223442 seconds. The configured cost limit uses the page's
rounded 0.0646 value, approximately USD 0.0000053 above the API value. The local
0.54-second raw threshold is `(1.23-0.15)/2`. The configuration's 1,624 decision
count comes from METHOD-v1.5; the v1.6 page describes 1,500 self-hosted items.
These are dated rule snapshots used for local projections.

## Serving algorithm

The fast path reads candidate-token logits and aggregates accepted candidate
forms. The slow path extends the fast prompt with a draft that the model is
instructed to keep within 250 words, with a hard maximum of 512 generated tokens,
then reads the answer distribution
at the `ANSWER:` marker. Failure to parse a slow answer falls back to the fast
answer after accounting for the draft's work.

Routing uses a logratio margin threshold of `2.3108509669020307`. The slow quota
is 0.25 and the share fuse is 0.35. Slow admissions use an accumulated budget
based on decisions already served: at quota 0.25 the first three decisions cannot enter the slow
path. Moving an item to another position can therefore change its route and
answer. Each independent evaluation starts a new process in a new `WORK` directory
and container; README keeps the run ID `run1` because its state directory is new.

The cost fuse takes no action during the first 100 decisions; it records their
costs and can limit subsequent admissions. It engages when mean cost reaches 0.85 times the USD 0.0646 per 1,000 decisions limit and releases at
0.80 times that limit. While engaged it reduces slow admissions using observed
fast/slow cost. Costs use the fixed package tariff and engine usage below. These
controls are stateful and depend on item order and cache history.

The cost-fuse field records 0 transitions throughout the frozen-package cold
run and the source-tree cold run of 2026-10-06. Their last recorded running costs
were respectively USD 0.04099 and 0.04065 per 1,000 decisions. The additional
source-tree warm-cache run and the retained fast-only and routed fitting runs
also record 0 transitions. The earlier 150-word run did not record this field.
Final image run: [[FACTS-PENDING: S6-FINAL cost-fuse transition count and final running_cost_per_1000 from the last response's usage.cost_fuse.]]

Question-aware extractive compression limits long prompts to 16,384 tokens and
forces the fast path. The physical prompt guard is 32,768 tokens and the configured
model context is 34,816 tokens. Compression can discard relevant evidence. The
physical guard is not a lossless-context guarantee.

A recoverable slow-path exception or timeout retains the computed fast answer
and returns HTTP 200 with the answer's `warning` and top-level `partial_errors`.
Known work is included in usage; where accounting is estimated, the counters are
conservatively increased. Slow-to-fast parse recovery also remains part of the
algorithm. Fatal engine errors bypass these recoveries.

An unrecovered item-processing failure returns HTTP 422 with
`{"error":"item could not be processed"}`. On the third consecutive item failure,
the service becomes dead and that request returns HTTP 503 with
`{"error":"service failure"}`. A successfully processed item resets the failure
count. Unrecoverable engine errors immediately mark the service dead, return
503, make `/health` report `status="error"` and make all later POST requests
return 503. Other service faults also return 503. The service does not invent a
uniform answer distribution. Unsupported request shapes return 422
`{"error":"Unsupported systemone request"}`; malformed requests return 400
`{"error":"Invalid systemone request"}`.

A 422 response contains neither an answer nor usage. The official TypeSafe
harness counts it as an incorrect answer. At the pinned harness revision,
`jevbench summarize` sets the run's `price_per_1000_decisions_usd` to null whenever
any attempted item's cost is missing; the ledger retains the failed item's
reserved amount, which is not a measured price. This describes the harness
summary and ledger, not a determination of the operator's leaderboard score.
Failures before quota admission do not advance quota or cost-fuse counters.
In a multi-question request, one unrecovered failure aborts the entire response,
while completed earlier questions retain their state updates; the official
TypeSafe adapter sends one question per request.

## Usage and price

`usage.accounting` is `engine`. `usage.input_tokens` counts actual engine prefill
work, subtracting prefix-cache hits including reuse across requests.
`usage.output_tokens` counts generated engine tokens. The response also includes:

| Field | Definition |
|---|---|
| `usage.alternatives.engine` | Engine prefill and engine-generated tokens, matching reported usage. |
| `usage.alternatives.submitted` | All submitted engine prompt tokens before cache deductions, plus the recorded completion count. Repeated scoring calls can submit a prefix more than once. |
| `usage.alternatives.each_once` | Each decision's fast prompt and, when routed slow, its slow prompt, each counted once, plus the recorded completion count. |

Detailed cache and call counters are under `usage.input_tokens_details`; per-item
records are under `usage.questions`. If token accounting is estimated,
the response marks `usage_estimated` and conservatively increases the counters.

The fixed local tariff is USD 0.03 per million input tokens and USD 0.15 per million
output tokens, keyed as `Qwen3.5-4B`. This is a budgeting and declared-price
assumption, not a measured compute/electricity cost. For N decisions, the cost per
1,000 decisions is `(input_tokens*0.03 + output_tokens*0.15)/1e6 * 1000/N`.

The [v1.5 pricing disclosure correction](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5-ADDENDUM-PRICING-DEEPINFRA-DISCLOSURE-CORRECTION-1.md)
identifies USD 0.03/M input and USD 0.15/M output as the frozen 25 September 2026
DeepInfra Qwen3.5-4B M2 base-model reference. That snapshot records the listing's
11 June 2026 deprecation and replacement by Qwen3.5-9B; the correction explicitly
retains the frozen 4B rates. These rates are not a claim of a currently bookable
Qwen3.5-4B service.

Prefix caching is enabled and GPU memory utilization is 0.85, with one sequence
at a time. The memory fraction allocates a different cache capacity on different
GPU sizes; cache reuse and engine-basis price can consequently differ by hardware.

## Calibration

`noul` is the binary yes/no task. Counts and raw fits below are copied from
`calibration.json`; final values are the values loaded by the service.
`source_placeholder=true` records the fitting tool's flag that a block contains
at least one question type with fewer than 20 fitting samples: fast score has 18,
slow score has 5, and all fallback types are below 20. `finalize.py` copies this
flag; the runtime does not read it. It indicates the sample-count limitation,
not a substituted calibration value.

| Block | Type | n | Raw fit | Final value | Processing |
|---|---|---:|---:|---:|---|
| fast | noul | 74 | p=0.7837837837837838 | p=0.81 | Manual lower bound 0.81. |
| fast | choice | 139 | T=1.681792830507429 | T=1.681792830507429 | Raw fit retained. |
| fast | score | 18 | T=1.189207115002721 | T=1.189207115002721 | Raw fit retained; sparse fit. |
| slow | noul | 20 | p=0.9 | p=0.842 | Shrink toward the raw fast p with 20 prior samples, rounded to 3 decimals. |
| slow | choice | 21 | T=5.656854249492381 | T=6.156 | Log-temperature shrinkage toward the pooled prior, rounded to 3 decimals. |
| slow | score | 5 | T=32.0 | T=9.190 | Same shrinkage; sparse raw fit lies on grid boundary. |
| fallback | noul | 0 | No fit | p=0.81 | Inherit the final fast value. |
| fallback | choice | 9 | T=1.0905077326652577 | T=1.681792830507429 | Inherit the final fast value. |
| fallback | score | 1 | T=1.681792830507429 | T=1.189207115002721 | Inherit the final fast value. |

The temperature prior is the pooled slow choice/score grid fit. To reproduce the
stored values, retain the full grid value `T0=2**(22/8)=6.727171322029716` and
the raw fast probability `p0=0.7837837837837838`. With prior sample size `n0=20`,
`T_slow=exp((n*ln(T_raw)+20*ln(T0))/(n+20))` and
`p_slow=(n*p_raw+20*p0)/(n+20)`. The unrounded values are
`6.155820736159339`, `9.189586839976283` and `0.841891891891892`; rounding to
3 decimals gives `6.156`, `9.190` and `0.842`.
No block has held-out validation recorded. Fast score, slow score and all fallback
types are marked insufficient in their source metadata; fallback always inherits
the final fast block. Fitting ECE is not evidence of held-out calibration quality.

## Results

The historical runs below evaluate the same 231 public items on one RTX 4090
24 GB with the official TypeSafe harness, one request at a time. The two
2026-10-06 cold runs started fresh service processes; their timestamps below are
the first and last item records, not run-manifest boundaries. Prices apply the
package tariff to the token totals from each same run. Historical venv runs used
CRLF task copies; image runs used the committed LF files with identical parsed
items. Model identity evidence and its historical limits are given above.

The target-image row is reserved for the final package's measurement. Earlier
image measurements used the same pinned image contents through a registry-mirror
index reference, rather than the exact repository reference in README. The
2026-10-08 run record also reports a later four-step README test of the preceding
package revision; neither historical run substitutes for the final-package row.

| Run | Environment and configuration | Evidence and measured values |
|---|---|---|
| Target-image cold run | Pinned image below; network disabled; root filesystem, model and code read-only; memory fraction 0.85; max_num_seqs=1; fresh prefix cache and state. | [[FACTS-PENDING: S6-FINAL start and end UTC from manifest.json; actual image reference and digest; attempted/failed and schema validity; accuracy n/231 and macro; ECE/Brier; raw p50/p95; peak GPU memory MiB; empty-cache and restart startup seconds with actual cache state; observed attention backend from health and startup log; cost-fuse transitions; engine/each_once/submitted token totals and prices; slow-path count with parsed/unparsed split; per-item answer and route differences against the historical frozen run.]] |
| First image test of an earlier revision | 2026-10-08; pinned image contents through a registry-mirror reference; memory fraction 0.85; max_num_seqs=1. | 231/231 answered, 0 failed, schema validity 1.000; 198/231 (0.8571); ECE 0.042; USD 0.04005 per 1,000 decisions (engine); raw p50 0.054 s, p95 5.15 s; peak GPU memory 19,702 MiB; 56 slow decisions; empty-cache startup 207 s. |
| Later image test of the preceding revision | 2026-10-08; four-step README workflow recorded for the preceding package revision. | 231/231 answered, 0 failed; 196/231; USD 0.04073 per 1,000 decisions (engine); raw p50 0.053 s, p95 4.86 s; peak GPU memory 19,702 MiB; startup 201 s; health ok with no warnings; no residual process after stop and removal. |
| Earlier revision in historical venv | 2026-10-08; Python 3.10 venv; an earlier package revision with the verified clean model directory; memory fraction 0.85; max_num_seqs=1. | Recorded on 2026-10-08: health ok without warnings; startup about 180 s; 196/231 (0.8485); USD 0.04073 per 1,000 decisions (engine); raw p50 0.057 s, p95 4.91 s. The run record reports 8 answer differences from the historical frozen run (net +1 correct), and 6 from the first image run. |
| Historical frozen package | RTX 4090 24 GB; Python 3.10 venv; memory fraction 0.85; max_num_seqs=1. | 2026-10-06T10:46:35Z to 2026-10-06T10:49:47Z (first/last item). Config and calibration hashes are the reviewed-commit hashes above. 231/231 answered, 0 failed, schema validity 1.000; 195/231 (0.8442), macro 0.8606; ECE 0.0354, Brier 0.2509; raw p50 0.055 s, p95 5.15 s; recorded peak GPU memory 19,702 MiB. 56 slow decisions (46 parsed, 10 unparsed retaining fast answers), 175 fast. Tokens in/out: engine 232,308/16,670 (USD 0.04099 per 1,000), each_once 228,167/16,777 (0.04053), submitted 494,196/16,777 (0.07508). Cost fuse 0 transitions; no estimated usage. The historical server check records 231 raw responses and zero files containing fast_failed or service_failed. |
| Earlier source-tree service | RTX 4090 24 GB; Python 3.10 venv; default vLLM memory/sequence settings. | 2026-10-06T07:14:39Z to 2026-10-06T07:17:46Z (first/last item). Same recorded prompt identity, threshold 2.3108509669020307, quota 0.25 and 250-word/512-token drafts; unadjusted calibration (fast yes/no 0.7838; slow yes/no 0.90 and choice/score temperatures 5.657/32.0). The retained evidence lacks this run's original parameter-file hashes and peak GPU memory. 231/231 answered, 0 failed, schema validity 1.000; 196/231 (0.8485), macro 0.8567; ECE 0.0214, Brier 0.2434; raw p50 0.047 s, p95 4.83 s. 56 slow decisions (46 parsed, 10 unparsed). Tokens in/out: engine 230,588/16,490 (USD 0.04065 per 1,000), each_once 232,109/16,607 (0.04093), submitted 530,492/16,607 (0.07968). Cost fuse 0 transitions. |

The source-tree and frozen-package cold runs differ in engine settings,
calibration and launcher. They answered 196 and 195 items correctly. Eight
answers differ: 2 correct only in the frozen run, 3 only in the source-tree run,
and 3 wrong in both. Four routes differ. Fast margins differ by more than 0.1 on
110 items, with a maximum absolute difference of 0.4996. Their engine outputs
therefore were not numerically identical. Temperature scaling and the stated
yes/no calibration preserve the winner, but these runs changed multiple serving
conditions together; they do not isolate a cause for the answer differences.
Their ECE/Brier difference also includes the change in calibration.

The first image run answered 198/231 versus the historical frozen run's 195/231,
with identical routes and 3 different answers, all hard items routed slow and
correct only in the image run. This exceeded the predeclared plus-or-minus-2-item
accuracy tolerance by one item; the tolerance was not changed. Its engine price
was 2.3% lower. The contemporary venv run differed from the frozen run on 8
answers (net +1 correct) and from the image run on 6. These comparisons do not
separate code, Python, image and run-to-run numerical effects.

[[FACTS-PENDING: S6-FINAL per-item answer and route comparison with the historical frozen run, listing routes and margins for each difference; accuracy, price and latency differences and the predeclared acceptance checks.]]

Public-set results are in-sample. Legacy
`latency_scale=2.0` and `latency_offset_seconds=0.15` are projections; they do not
change the raw harness latency or constitute a measurement. Long-tail projected
cost likewise is not a measured long-input result.

## Environment

The runtime uses the official vLLM release image `v0.31.0` (Linux amd64):
`vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
The corresponding index digest is
`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`,
and its amd64 image configuration ID is
`sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`.
The image contents provide Python 3.12.3 and installed packages vLLM 0.31.0,
torch 2.13.0+cu130, transformers 5.17.0, tokenizers 0.23.2 and triton 3.7.1;
startup verifies the package versions and health `info.runtime` reports them.
Startup requires Python >=3.10 and these exact base package versions, ignoring
local build suffixes such as `+cu130`. README starts the image by its linux/amd64
manifest digest, waits for the ready log line, then runs and summarizes the harness
with `docker exec`; startup checks package versions. The final step stops and
removes the container, retaining Docker logs after an exit until that removal.
Runtime health reports loaded versions, CUDA and GPU. The README workflow
assumes the exact image repository digest is already available on an offline
host and Docker client configuration does not inject HTTP proxy settings.

Attention selection tries auto, then `FLASH_ATTN`, then `TRITON_ATTN`. Explicit
choices use the vLLM constructor argument. The observed backend is read from
startup logs; selecting a backend and observing the backend are separate facts.
Each startup attempt has a 1800-second deadline; a timeout aborts startup instead
of advancing to another backend. The runtime is offline and directs service
state, home, temporary files and caches into the explicit state directory.
Auto is also the effective selection policy of the historical service; its old
environment setting did not force an implementation in this vLLM version.
Auto selection is hardware-dependent. vLLM's CUDA selection prioritizes
FlashAttention on SM89/SM90 and FlashInfer for causal attention on SM10x;
FlashAttention prefers FA3 on supported SM90 systems and otherwise can use FA2.
These are selection rules, not proof of the backend used by a particular run.
`info.attention_backend` reports the backend observed in startup logs. Measurements
on one GPU architecture do not establish numerical parity on another.
If startup reaches a fallback backend, `info.attention_backend` reports it.
The measurements reported here use auto selection; they do not establish
numerical parity for fallback backends.

The model view is isolated under each run's startup directory. Its path enters
vLLM's compilation cache key, so reusing the state root does not guarantee reuse
of that cache or a faster next startup. Other compiler/cache layers can reuse
artifacts independently. Startup timings describe their actual cache state.

The historical environment record is Ubuntu 22.04.5, Python 3.10.12,
NVIDIA driver 580.105.08 and RTX 4090 24 GB. `environment.resolved.json` and
`requirements.lock` describe that old venv; they do not require the current image
to match every historical dependency. The lock contains versions, not wheel
hashes. Hardware requirements are one exclusive GPU with at least 24 GB VRAM,
compute capability >=8.0 and driver >=580.

## Limits

Public-set selection and fitting can overfit the public items. Sparse calibration
requires held-out evaluation. Stateful routing and cache accounting depend on
order, hardware and restart discipline. Extractive compression can lose evidence,
and an unparsed slow draft still consumes tokens.

The historical image test on 2026-10-08 used an earlier package revision on the
RTX 4090, with network disabled, read-only root filesystem and read-only package
and model mounts. Empty-cache startup took 207 s; health was ok with no warnings.
The retained test record reports zero occurrences of "Unknown vLLM environment
variable", "Failed to get the IP address", "Traceback", "Network is unreachable"
and "huggingface.co" in its startup log. User 1000:1000 also started successfully
from empty caches in 207 s. Stopping the container took 1 s and left no GPU
process. The image reference differed from README as stated in Results.

The frozen package's venv long-input test on 2026-10-08 inserted about 880,000
unrelated characters into each of 20 public items: all 20 were answered; correct
answers changed from 15 to 13; raw median latency was 1.76 s and maximum 1.81 s;
median input usage was 15,788 tokens, using the fast path only.

[[FACTS-PENDING: S6-FINAL real-GPU offline empty-cache and restart startup evidence with cache state; root and user 1000:1000 read-only startup; stop and process cleanup; malformed requests returning HTTP 400/422; raw-response failure/recovery and warning counts; long-input smoke in the image with HTTP status, latency and routes; exact README workflow and any image-reference substitution.]]
Quota persistence must work at startup. The existing counter behavior after a
later persistence failure is retained: it reports a health warning and restricts
admissions to the fast path using memory counts. Persistence errors that escape
that recovery are service faults (HTTP 503). A degraded run is not clean evidence.

The service reads the package and model folder, writes only below `--state-dir`
(plus shared memory in `/dev/shm`), and never opens the harness checkout, task
files or harness output. It performs no recursive deletion; the only deletion
is `quota.py` removing its own temporary file after a failed write. Quota
persistence also atomically replaces its own state file.

The static package audit flags the common deleting and replacing calls below.
This is a bounded source check, not a proof covering arbitrary dynamic code or
every form of file overwrite.

| Surface | Calls or commands flagged |
|---|---|
| Python attributes | `unlink`, `rmtree`, `TemporaryDirectory`, `rmdir`, `removedirs`; single-positional-argument `replace` and `rename`. |
| Python named calls | `os.remove`, `os.replace`, `os.rename`, `shutil.move`, including resolved import aliases. |
| Command literals in Python | Arguments to `subprocess` calls and `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, for `rm`, `rmdir`, `unlink`, `truncate`, `shred` and `find -delete`. |
| Shell files | The same command tokens: `rm`, `rmdir`, `unlink`, `truncate`, `shred` and `find -delete`. |

The audit permits only the exact quota-state replacement and failed-write
temporary-file cleanup expressions in `quota.py`.
