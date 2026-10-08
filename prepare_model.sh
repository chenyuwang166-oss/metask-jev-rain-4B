#!/usr/bin/env bash
set -euo pipefail
# Maintainer-only online download helper; runtime never invokes this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${PYTHON:-python3}"
"$PY" -B - "$SCRIPT_DIR" "$@" <<'PY'
import sys, hashlib, json, os, tempfile
from pathlib import Path
if len(sys.argv)!=3:raise SystemExit('Usage: bash prepare_model.sh /absolute/path/to/new/MODEL_DIR')
if sys.version_info < (3,10):raise SystemExit('Python >= 3.10 required')
package=Path(sys.argv[1]).resolve()
p=Path(sys.argv[2])
if not p.is_absolute():raise SystemExit('Model directory must be absolute')
p=p.resolve()
if p.is_relative_to(package) or package.is_relative_to(p):raise SystemExit('Model directory must be separate from the package')
if p.exists():raise SystemExit('Use a new model directory')
p.mkdir(parents=True,exist_ok=False)
cache=p/'.download-cache'
temporary=cache/'tmp'
temporary.mkdir(parents=True,exist_ok=False)
os.environ.update(HF_HOME=str(cache/'hf'),HF_HUB_DISABLE_TELEMETRY='1',
                  DO_NOT_TRACK='1',TMPDIR=str(temporary),PYTHONDONTWRITEBYTECODE='1')
tempfile.tempdir=str(temporary)
from huggingface_hub import snapshot_download
repo='wayfind/metask-jev-4b-policy-mix'
rev='ea20fe85b28733b1522721dec119bec50947a869'
snapshot_download(repo_id=repo,revision=rev,local_dir=p,cache_dir=cache/'hub')
if 'Qwen3_5ForConditionalGeneration' not in json.loads((p/'config.json').read_text(encoding='utf-8')).get('architectures',[]):
    raise SystemExit('Unexpected model architecture')
if not any(p.glob('*.safetensors')):raise SystemExit('Missing weights')
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1<<24),b''):h.update(chunk)
    return h.hexdigest()
weights={q.name:sha(q) for q in sorted(p.glob('*.safetensors'))}
weight_manifest=''.join(f'{h}  {n}\n' for n,h in weights.items())
with (p/'WEIGHTS.sha256').open('x',encoding='utf-8',newline='\n') as out:out.write(weight_manifest)
meta={'model_repo':repo,'model_revision':rev,'weights_sha256':weights,
      'weights_list_sha256':hashlib.sha256(weight_manifest.encode()).hexdigest()}
with (p/'PREPARATION.json').open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(meta,indent=2)+'\n')
print('Pinned model downloaded; weight hashes recorded. Auxiliary files are supplied by the runtime package.')
PY
