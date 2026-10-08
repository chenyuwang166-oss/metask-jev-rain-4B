"""Rebuild the version lock from the supplied measured freeze; no new resolution."""
import hashlib
import json
import argparse
from pathlib import Path
from core import external_output

ROOT = Path(__file__).resolve().parent

def build(output_dir):
    output=external_output(output_dir,create_parent=False)
    if output.exists():
        raise FileExistsError('Use a new external output directory')
    source = (ROOT/'requirements.measured.lock').read_text(encoding='utf-8')
    versions = {}
    lines = []
    for line in source.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, version = line.split('==')
        if name == 'torch':
            version = '2.13.0+cu130'  # Explicit measured build supplied with the freeze.
        if name in versions:
            raise ValueError('Duplicate measured package')
        versions[name] = version
        lines.append(name+'=='+version)
    lock = ('# Measured version lock; artifact hashes were not supplied.\n'
            '# All measured entries retained; torch CUDA build made explicit.\n'
            '--extra-index-url https://download.pytorch.org/whl/cu130\n'
            + '\n'.join(lines)+'\n')
    evidence = {
        'source': 'requirements.measured.lock; historical measured environment',
        'os': 'Ubuntu 22.04.5', 'python': '3.10.12', 'python_required': '>=3.10',
        'nvidia_driver': '580.105.08', 'gpu': 'RTX 4090', 'gpu_memory_gb': 24,
        'torch_cuda_build': 'cu130', 'cuda_toolkit_system': 'not separately supplied',
        'measured_freeze': source, 'versions': versions, 'removed_entries': [],
        'overrides': {'torch': {'freeze': '2.13.0', 'measured_build': '2.13.0+cu130'}},
        'lock_kind': 'exact versions, not wheel hashes',
        'installation_verified_here': False,
        'lock_sha256': hashlib.sha256(lock.encode()).hexdigest(),
        'measured_freeze_sha256': hashlib.sha256(source.encode()).hexdigest(),
    }
    output.mkdir(parents=True,exist_ok=False)
    with external_output(output/'requirements.lock').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(lock)
    with external_output(output/'environment.resolved.json').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(evidence,indent=2)+'\n')
    print('Measured version lock created in '+str(output)+'; installation and artifact hashes not verified here.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    build(parser.parse_args().output_dir)
