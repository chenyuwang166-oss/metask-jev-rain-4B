# metask-jev-rain-4B

Standalone inference-service package for the official JevBench TypeSafe adapter.
Fitted parameters and the measured version lock are included. Identity placeholders,
clean Linux installation and evaluation-machine cold-run evidence remain pending;
an offline release check is not a declaration that these steps have passed.

## Evaluator workflow

Use Linux, Bash and Python 3.10. Set VENV, MODEL_DIR, STATE_DIR and RUN_ID explicitly;
all directories must be outside this package, and each run needs a fresh RUN_ID.
Replace both revision placeholders with immutable 40-character commits first.
Run from this package directory:

```bash
sha256sum -c MANIFEST.sha256
bash install.sh "$VENV"
PATH="$VENV/bin:$PATH" bash prepare_model.sh "$MODEL_DIR" 'wayfind/metask-jev-4b-policy-mix' '<fill:revision>' '<fill:base_revision>'
PATH="$VENV/bin:$PATH" bash serve.sh --model "$MODEL_DIR" --port 8000 --state-dir "$STATE_DIR" --run-id "$RUN_ID"
```

Then use the official harness as described in harness.md, on the same machine.
Preparation needs network access; serving is offline and disables usage telemetry.
serve.sh forwards --state-dir and --run-id to launch.py and then server.py.
If omitted, launch creates a fresh external temporary state directory and run ID.
The foreground supervisor binds to loopback, waits up to 900 seconds for health,
requires prefix caching and slow sessions, and aborts on persistence errors.
Stop with Ctrl-C. Start a fresh service for every independent evaluation.

## Optional evaluator selftest

Stop any existing model service first. Set PUBLIC_DIR to `datasets/public` in the
official harness checkout at bb05a33 (2026-09-29). Set OUTPUT_DIR to a new, absent
directory outside both the package and public sources. No maintainer index is needed.

```bash
PATH="$VENV/bin:$PATH" bash selftest.sh --model "$MODEL_DIR" --public "$PUBLIC_DIR" --output-dir "$OUTPUT_DIR" --port 8001
```

Reads all 231 tasks from easy.jsonl, original.jsonl and hard.jsonl in that order,
preserving source row order. Derives an id/type/tier index in OUTPUT_DIR; this order
is explicit and is not claimed to reconstruct the historic maintainer-index order.
Runs fresh fast_only and routed cache-on/off services sequentially, with run-specific
state directories, reconciliation and an exploratory calibration candidate. This
candidate is produced by the legacy fitter and is NOT the finalized n0=20/clamped
calibration; it never replaces packaged parameters. Reconciliation is strict about
paired decode drift, counters and preemptions. Failure on different hardware is a
failure to validate that run, not automatic proof of an accounting bug.

## Maintainer verification

`requirements.lock` is derived from the supplied measured freeze, with torch's
measured CUDA suffix made explicit. All entries are retained. It contains exact
versions but no downloaded wheel hashes; see environment.md for that limitation.
Do not substitute guessed versions. `resolve_env.py` deterministically rebuilds it.

config.json and calibration.json are the formal parameter files. The retained
fitted.example.json is a SAMPLE ONLY template. finalize.py accepts its schema plus
full block audit metadata (by_type, status, held-out markers), updating both routing
levels. Preserve that metadata when preparing any future fitted input. Keep the
input outside the package. After changing parameters, rerun finalize with that input;
after documentation edits, fill FREEZE.md's time when actually frozen and run:

```bash
python -B finalize.py --manifest-only
python -B pkg_check.py --release
sha256sum -c MANIFEST.sha256
```

All package text uses LF. MANIFEST covers every package file except itself.
The checker permits only explicitly named outstanding disclosure fields and prints
them; it rejects generic/unknown placeholders, runtime output, CRLF and bad hashes.
A passing check does not replace immutable model identity, clean installation,
held-out validation, target-GPU cold runs or peak-memory measurement.

## Operational notes

Measured hardware was RTX 4090 24 GB. The target is RTX 5090 32 GB, not measured here.
The package's 0.85 memory utilization and one sequence differ from measured defaults;
confirm cold-run behavior on the evaluation machine before submission. Do not alter
context budgets or thresholds to work around OOM without a new validated revision.

prepare_model.sh streams hashes for all safetensors and the four auxiliary files;
PREPARATION.json and WEIGHTS.sha256 are written to the external model directory.
Keep environments, weights, data, logs and all selftest output outside the package.
Copied algorithm modules retain historical internal module names. quota.py changes
startup failure handling only; routing, decoding, costs and fitting algorithms are
otherwise preserved. No real model was run during this packaging task.
