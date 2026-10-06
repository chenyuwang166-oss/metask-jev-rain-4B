#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - "$@" <<'PY'
import sys, shutil, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
if len(sys.argv)!=5:raise SystemExit('Usage: bash prepare_model.sh MODEL_DIR HF_REPO MODEL_COMMIT BASE_COMMIT')
dest,repo,rev,base=sys.argv[1:]
if repo!='wayfind/metask-jev-4b-policy-mix' or any(len(x)!=40 or any(c not in '0123456789abcdef' for c in x) for x in (rev,base)):
    raise SystemExit('Use wayfind/metask-jev-4b-policy-mix and replace <fill:revision> / <fill:base_revision> with immutable 40-character commits')
p=Path(dest).resolve()
package=Path.cwd().resolve()
if p.is_relative_to(package):raise SystemExit('Model directory must be outside package')
if p.exists():raise SystemExit('Use a new model directory')
snapshot_download(repo_id=repo,revision=rev,local_dir=p)
required=('preprocessor_config.json','video_preprocessor_config.json','merges.txt','vocab.json')
for name in required:
    if not (p/name).is_file():
        shutil.copyfile(hf_hub_download('Qwen/Qwen3.5-4B',name,revision=base),p/name)
assert 'Qwen3_5ForConditionalGeneration' in json.loads((p/'config.json').read_text())['architectures']
assert any(p.glob('*.safetensors')), 'Missing weights'
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1<<24),b''):h.update(chunk)
    return h.hexdigest()
weights={q.name:sha(q) for q in sorted(p.glob('*.safetensors'))}
weight_manifest=''.join(f'{h}  {n}\n' for n,h in weights.items())
(p/'WEIGHTS.sha256').write_text(weight_manifest,encoding='utf-8',newline='\n')
meta={'model_repo':repo,'model_revision':rev,'base_repo':'Qwen/Qwen3.5-4B','base_revision':base,
      'processor_sha256':{n:sha(p/n) for n in required},'weights_sha256':weights,
      'weight_manifest_sha256':hashlib.sha256(weight_manifest.encode()).hexdigest()}
(p/'PREPARATION.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Model files prepared; weight and processor hashes recorded.')
PY
