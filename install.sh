#!/usr/bin/env bash
set -euo pipefail
# Maintainer-only online reconstruction of the recorded environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${PYTHON:-python3}"
VENV=${1:?Usage: bash install.sh /path/to/new/venv}
[[ $# -eq 1 ]] || { echo 'Pass exactly one new external environment path' >&2; exit 2; }
[[ -f "$SCRIPT_DIR/requirements.lock" ]] || { echo 'Missing measured requirements.lock' >&2; exit 2; }
[[ ! -e "$VENV" ]] || { echo 'Use a new environment outside the package' >&2; exit 2; }
"$PY" -B - "$SCRIPT_DIR" "$VENV" <<'PY'
import pathlib, sys
if sys.version_info < (3, 10):
    raise SystemExit('Python >= 3.10 required')
package=pathlib.Path(sys.argv[1]).resolve()
target=pathlib.Path(sys.argv[2])
if not target.is_absolute():
    raise SystemExit('Environment path must be absolute')
target=target.resolve()
if target.is_relative_to(package) or package.is_relative_to(target):
    raise SystemExit('Environment must be separate from the package')
if target.exists():
    raise SystemExit('Use a new environment directory')
target.mkdir(parents=True,exist_ok=False)
PY
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VENV/tmp" XDG_CACHE_HOME="$VENV/cache"
export PIP_CACHE_DIR="$VENV/cache/pip"
mkdir -- "$TMPDIR" "$XDG_CACHE_HOME"
"$PY" -B -m venv "$VENV"
"$VENV/bin/python" -B -m pip install -r "$SCRIPT_DIR/requirements.lock"
"$VENV/bin/python" -B -m pip check
