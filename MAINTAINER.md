# Maintenance and validation

The evaluator workflow is in [README.md](README.md): set its five variables, start
the pinned-image container, follow its logs until ready, run and summarize the
harness, then stop and remove the container. Each evaluation uses a new `WORK`
and container; `run1` can remain the run ID because the state directory is new.
The container is retained after an exit so its logs remain available until removal.
Keep models, environments,
data, logs and test outputs outside the source package. Use absolute paths and a
new output directory for each operation. Output creation is exclusive; preserve
prior evidence under its existing name.

Commands here run on a Linux maintenance host. Export `PKG`, `WORK` and
`JEVBENCH` as its absolute package, writable maintenance-output and official
harness directories. README uses the same host-path variables for mounts;
its `TASKS` and paths passed to `docker exec` are container paths.
`WORK` must be outside the package, model and public data directories. Each named
output below must be new. Unit tests target Linux; portable fixture checks on
other systems do not establish that the Linux GPU service runs there.

```bash
: "${PKG:?Export an absolute host package path}"
: "${WORK:?Export an absolute host maintenance-output path}"
: "${JEVBENCH:?Export an absolute host harness path}"
```

## Historical venv reproduction

`requirements.measured.lock` is the recorded 201-entry environment freeze.
`requirements.lock` records the measured torch build `2.13.0+cu130`; the source
freeze lists its base version. `environment.resolved.json` records this provenance
and the historical Ubuntu/Python/GPU environment. The lock is an exact version
lock, not a wheel-hash lock. A clean installation and wheel availability are
separate checks from reproducing those version strings.

Rebuild the version-lock record into a new external directory without installing
packages:

```bash
python3 -B "$PKG/resolve_env.py" --output-dir "$WORK/resolved-lock"
```

On a connected Linux maintenance host, reconstruct the historical environment
with its Python 3.10 interpreter in a fresh external venv:

```bash
PYTHON=python3.10 bash "$PKG/install.sh" "$WORK/repro-venv"
```

`install.sh` installs `requirements.lock` and runs `pip check`. This is a
maintenance operation; the evaluator runs the prebuilt image without installing
packages. The historical venv used Python 3.10.12; using another interpreter does
not establish that historical results have been reproduced. The runtime and
installer minimum is Python >=3.10; that minimum alone is not a claim that every
package in this historical lock installs on every newer Python version.

The download helper takes one absolute external model directory and pins the
checkpoint revision internally:

```bash
PYTHON="$WORK/repro-venv/bin/python" bash "$PKG/prepare_model.sh" "$WORK/model-snapshot"
```

It records safetensors weight-file hashes and the hash of the weights list. It does not fetch
auxiliary files from another model. The evaluator's pinned-revision directory of
regular files can be used directly without this helper; an isolated HF cache
snapshot with relative links to its blob store cannot. The runtime verifies the
package's auxiliary files.
The auxiliary files and license notices must be preserved byte-for-byte.

## Selftest

Run with exclusive GPU access and no concurrent harness requests. The output
directory must be new, external to the package and separate from public sources.

```bash
PYTHON="$WORK/repro-venv/bin/python" bash "$PKG/selftest.sh" \
  --model "$WORK/model-snapshot" --public "$JEVBENCH/datasets/public" \
  --output-dir "$WORK/selftest-run1"
```

The selftest owns and stops its services. It runs fast-only and routed passes,
reconciles prefix-cache-on/off usage and writes `calibration.candidate.json` into
the output directory. Its target slow share remains 0.25. Candidate files are
evidence for review; selftest does not replace package parameters.

## Parameter identity and packaging

`config.json` and `calibration.json` are the formal parameter files.
`fitted.example.json` is a sample template, not fitted evidence. Preserve the
render function, choice codes and threshold identity when making packaging-only
changes. The fixed prompt SHA is
`e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938`.

Create a separate package snapshot with a new freeze record and manifest:

```bash
python3 -B "$PKG/finalize.py" --freeze-record --output-dir "$WORK/frozen-package"
```

The output directory must not already exist and may neither contain nor be inside
the source package. The record includes the current UTC time, actual config and
calibration hashes, prompt identity and runtime-image digest. Manifest generation
excludes repository metadata, caches and runtime state. It does not change
parameter values. The source package remains intact.

Validate that output package using distinct external evidence directories:

```bash
python3 -B "$WORK/frozen-package/regression_checks.py" --output-dir "$WORK/regression-run1"
python3 -B "$WORK/frozen-package/test_attention.py" --output-dir "$WORK/attention-run1"
python3 -B "$WORK/frozen-package/test_service_failures.py" --output-dir "$WORK/service-run1"
python3 -B "$WORK/frozen-package/test_pkg_check.py" --output-dir "$WORK/checker-run1"
python3 -B "$WORK/frozen-package/pkg_check.py" --output-dir "$WORK/pkg-check-run1"
python3 -B "$WORK/frozen-package/pkg_check.py" --release --output-dir "$WORK/pkg-check-release1"
cd "$WORK/frozen-package"
sha256sum -c MANIFEST.sha256
```

Run without Python optimization. The publication check rejects unresolved factual
markers in the disclosure. Resolve them from primary evidence before creating
the publication snapshot.

The source audit flags the common deleting and replacing calls: Python
`unlink`, `rmtree`, `TemporaryDirectory`, `rmdir`, `removedirs`, `os.remove`,
`os.replace`, `os.rename`, `shutil.move`, and single-argument `.replace` or
`.rename`; literal shell invocations through `subprocess`, `os.system`,
`os.popen`, `os.exec*` and `os.spawn*`; and shell-script uses of `rm`, `rmdir`,
`unlink`, `truncate`, `shred` or `find -delete`. The two exact exceptions in
`quota.py` atomically replace its own state file and remove its own temporary
file after a failed write. This audit is not a general proof against file
overwrites, dynamic shell commands or every possible filesystem mutation.

After validation, verify that all package members other than `FREEZE.md` and
`MANIFEST.sha256` still match the source tree byte-for-byte, using the member list
from `package_files.py`. Copy those two generated files from the validated
snapshot back to the source repository:

```bash
cp -- "$WORK/frozen-package/FREEZE.md" "$PKG/FREEZE.md"
cp -- "$WORK/frozen-package/MANIFEST.sha256" "$PKG/MANIFEST.sha256"
cd "$PKG"
sha256sum -c MANIFEST.sha256
python3 -B "$PKG/pkg_check.py" --release --output-dir "$WORK/source-release-check1"
```

Do not make further package edits after this check. When the publication commit
is made, include the copied freeze and manifest, record the full commit
SHA, and run manifest verification and all package checks again in a fresh clone
of that exact commit and in its separate `git archive` export. This verifies the
committed bytes, not only the uncommitted snapshot. Any later document change
requires a new snapshot and the same sequence.

Unit tests with fake backends do not establish GPU inference
compatibility; repeat the cold-run, restart, shutdown, accounting, long-input and
README workflow checks in the fixed image with networking disabled and read-only
code/model mounts.

## GPU comparison evidence

Record the actual files created under the writable mount. Supervised quota state
is under `state/run1-startup/state-N/run1/`, not `state/run1/`; each startup
attempt has its own state. The startup directory also contains `attempt-N.log`,
`attempts.jsonl`, `home/`, `tmp/`, `model/` and `write-probe`, with shared cache
paths under `state/cache/`. The README harness writes under `run1/`. These are
relative to its host `WORK`. Compare this tree with the read-only code and model
mounts to establish the write boundary.

For historical-venv and pinned-image comparisons, require exact per-item equality
of `usage.questions` decision fields `fast_prompt_tokens`, `kept_prompt_tokens`
and `original_prompt_tokens`; where both runs take the slow path, compare slow
prompt token counts exactly as well. Report missing fields explicitly. Record
the maximum and median absolute fast-margin difference, route/argmax changes,
probability differences, all three usage totals and prices, failures, cost-fuse
transitions, latency and peak VRAM. Python 3.10/3.12 floating-point aggregation is
not a bitwise-equivalence test: use a stated rounding tolerance such as `1e-12`
for the CPU probability arithmetic at identical logits and assess GPU variation
separately. The static `json_equivalence.py` guard also checks 69 selected
unchanged inference functions and constants against the reviewed commit, while
rejecting direct module/class-level rebinding or mutation of those names.
The canonical AST representation retains empty fields across Python versions.
The guard does not establish complete request-path equivalence, detect arbitrary
dynamic rebinding, or replace the GPU comparison. A deterministic fake-backend
CPU comparison of responses and engine calls is separate evidence; it cannot
establish real GPU logits, timing or memory behavior.

Every startup has an isolated model-view path, which enters vLLM's compile-cache
identity. A reused state root does not promise a reused vLLM compilation or a
faster restart. Record cache state and actual startup duration without treating
"warm must be faster" as an acceptance criterion.

Use an explicitly configured GitHub noreply identity for any future authorized
commit; inspect that commit's author and committer identities before publishing.
Do not rewrite existing history. The immutable code commit is
communicated in the submission email; a tag does not replace that commit identity.
