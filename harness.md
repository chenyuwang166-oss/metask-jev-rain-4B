# Evaluator: official TypeSafe adapter

Install the official JevBench harness in its own environment. Its source is not
vendored here. Use the supplied official clone revision bb05a33 (2026-09-29).
The harness and service must run on the same machine (loopback binding).
Start `serve.sh` in the model environment, wait for the printed `/health` result,
then run the official harness from its environment:

```bash
python -m jevbench.cli run --adapter typesafe \
  --endpoint http://127.0.0.1:8000 --model metask-jev-rain-4B --key-env '' \
  --tasks "$TASKS_JSONL" --results "$RUN_DIR/results.jsonl" \
  --ledger "$RUN_DIR/ledger.jsonl" --raw-dir "$RUN_DIR/raw" \
  --manifest "$RUN_DIR/manifest.json" --cap-usd 15 \
  --price-in-per-m 0.03 --price-out-per-m 0.15
python -m jevbench.cli summarize --tasks "$TASKS_JSONL" \
  --results "$RUN_DIR/results.jsonl" --ledger "$RUN_DIR/ledger.jsonl"
```

Set TASKS_JSONL and a new RUN_DIR explicitly. Override 8000 consistently if using
another port. These options match the supplied official CLI: `typesafe` appends
`/v1/systemone`, sends `state`, `model`, `questions.decision`, and reads typed
probabilities and `usage`. Empty `--key-env ''` disables token authentication.
The quoted empty argument is intentional (Bash). Never add `/v1/systemone` to
the base endpoint. All output locations are explicit rather than harness defaults.

The shown prices are the existing submission's declared rates (USD per million
input/output tokens), not measured cloud or electricity costs. Confirm them before
submission. Do not run harness requests concurrently with selftest: reconciliation
requires exclusive engine counters. Restart service for each independent evaluation
to obtain a fresh quota/cost-fuse state.
