# metask-jev-rain-4B disclosure

## Identity and attribution

`metask-jev-rain-4B` is a separate inference system using the same pinned
policy-mix weights as `metask-jev-4b`. The code identity is the commit given in
the submission email. The configuration's `model_key="metask-jev-4b"`
identifies the weights, routing threshold and calibration blocks; it is not
the public system name.

Weights: [wayfind/metask-jev-4b-policy-mix](https://huggingface.co/wayfind/metask-jev-4b-policy-mix/tree/ea20fe85b28733b1522721dec119bec50947a869).
`model.safetensors` is 9,078,620,536 bytes, with file SHA256
`f4b40475d18e0a38638b985ed51816c012619e0d4138166704e2ea0aac7a12b3`.

Four auxiliary files are included byte-for-byte from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
Their Apache-2.0 attribution and license are in `model_aux/NOTICE` and
`model_aux/LICENSE`. Service code is licensed under MIT in `LICENSE`.
The checkpoint's tokenizer file, including its additional tokens and tokenizer
settings, is unchanged and checked by SHA256 at startup.

## Training data and public-item use

This package loads the policy-mix checkpoint identified above; it performs no
training at startup or during evaluation. It uses the checkpoint with `adapter=null`.
The pinned model card describes Qwen/Qwen3.5-4B as the backbone with merged LoRA,
and training on public datasets, MASSIVE utterance-to-domain decisions, and
synthetic policy-family decisions with teacher soft labels. The model card also reports repeated evaluation on public JevBench items.

Public JevBench items informed prompt comparisons, routing, draft policy,
threshold and calibration fitting, and evaluation. Public gold answers informed
prompt, routing and calibration choices. Separate in-house fine-tuning experiments
used the public items for monitoring; none of their weights is used by this system. Public-item results are in-sample, not
held-out performance estimates.

The public sources are `datasets/public/easy.jsonl`, `original.jsonl` and
`hard.jsonl` from
[JevBench at bb05a335bc809e61b20c0f745d25499a82b326fc](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
Their original bytes are identified by the pinned harness manifest:

| File | SHA256 |
|---|---|
| easy.jsonl | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` |
| original.jsonl | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` |
| hard.jsonl | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` |

The checkpoint training account is limited to the pinned model card; it is not
a declaration about unreleased training material, historical sealed-item access
or ImageJevBench training use.

## Failure handling

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

## Usage and price

`usage.accounting` is `engine`. `usage.input_tokens` counts actual engine prefill
work, subtracting prefix-cache hits including reuse across requests.
`usage.output_tokens` counts generated engine tokens. Detailed cache and call
counters are under `usage.input_tokens_details`; per-item records are under
`usage.questions`. Estimated accounting is marked `usage_estimated`.

The fixed local tariff is USD 0.03 per million input tokens and USD 0.15 per
million output tokens, keyed as `Qwen3.5-4B`. This is a budgeting and
declared-price assumption, not a measured compute/electricity cost or a claim
of a currently bookable Qwen3.5-4B service. The source is the
[v1.5 pricing disclosure correction](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5-ADDENDUM-PRICING-DEEPINFRA-DISCLOSURE-CORRECTION-1.md).

See [README.md](README.md) for requirements, pinned downloads, startup,
evaluation and shutdown commands.
