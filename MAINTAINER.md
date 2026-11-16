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
