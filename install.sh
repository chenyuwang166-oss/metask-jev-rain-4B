#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
VENV=${1:?Usage: bash install.sh /path/to/new/venv}
[[ -f requirements.lock ]] || { echo 'Missing measured requirements.lock' >&2; exit 2; }
python3.10 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
[[ ! -e "$VENV" ]] || { echo 'Use a new environment outside the package' >&2; exit 2; }
python3.10 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); assert not p.is_relative_to(pathlib.Path.cwd()), "Environment must be outside package"' "$VENV"
python3.10 -m venv "$VENV"
"$VENV/bin/python" -m pip install -r requirements.lock
"$VENV/bin/python" -m pip check
