"""Offline package checks. No network, GPU imports or inference."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
REQUIRED='server.py core.py backends.py quota.py bench.py reconcile_usage.py config.json calibration.json requirements.txt environment.md resolve_env.py install.sh prepare_model.sh serve.sh launch.py selftest.sh selftest.py finalize.py fitted.example.json pkg_check.py harness.md DISCLOSURE.md README.md FREEZE.md MANIFEST.sha256'.split()
GENERATED={'requirements.lock','environment.resolved.json'}
PENDING_FIELDS={
    'DISCLOSURE.md': {'revision','weights_sha256','base_revision','dataset_sha256','reconciliation_coverage','freeze_time','peak_vram','final_calibration_metrics'},
    'FREEZE.md': {'freeze_time'},
}

def check_manifest(root):
    assert b'\r' not in (root/'MANIFEST.sha256').read_bytes(),'MANIFEST must use LF for sha256sum -c'
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        sha,name=line.split('  ',1)
        p=root/name
        assert p.resolve().is_relative_to(root.resolve()) and p.is_file(), 'Unsafe/missing manifest member'
        assert name not in expected,'Duplicate manifest member'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,'Hash mismatch: '+name
        expected[name]=sha
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    assert actual==set(expected),'Manifest membership mismatch'

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release',action='store_true');a=ap.parse_args()
    for name in REQUIRED:assert (ROOT/name).is_file(),'Missing package file: '+name
    check_manifest(ROOT)
    modules={p.stem for p in ROOT.glob('*.py')}
    external={'torch','transformers','vllm','peft'}
    for path in ROOT.glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                top=name.split('.')[0]
                assert top in modules or top in sys.stdlib_module_names or top in external, 'Unresolved Python import: '+name
        # Check literal references to packaged scripts/configs; dataset and runtime
        # output names are explicitly external/generated and are not vendored.
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'[\w.-]+\.py',node.value):
                assert (ROOT/node.value).is_file(),'Missing script reference: '+node.value
    for script in ('serve.sh','selftest.sh'):
        t=(ROOT/script).read_text()
        assert 'VLLM_USE_FLASHINFER_SAMPLER=0' in t and 'VLLM_ATTENTION_BACKEND=FLASH_ATTN' in t
    from core import load_config,load_json,threshold_identity,validate_calibration
    c=load_config();assert c['model']['path']==''
    assert set(c['profiles'])=={'metask-jev-4b'}
    assert (ROOT/c['calibration_file']).is_file()
    for block in ('fast','slow','fallback'):validate_calibration(load_json(ROOT/c['calibration_file']),'metask-jev-4b',block)
    # Check new authored/copied text for accidental deployment identifiers.
    banned=[r'[A-Za-z]:[\\/](?:Users|home)[\\/]',r'/(?:home|Users|root)/[\w.-]+',r'(?<!\w)~[/\\]',r'(?i)\bssh\s+(?:-[^\s]+\s+)*[\w.-]+',
            r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',r'\b(?:\d{1,3}\.){3}\d{1,3}\b']
    for f in ROOT.rglob('*'):
        if f.is_file():
            assert b'\r' not in f.read_bytes(),'Package text must use LF: '+f.name
            text=f.read_text(encoding='utf-8')
            # Numeric package versions are not deployment addresses. Only exempt
            # exact version tokens from the measured lock, not arbitrary addresses.
            measured=(ROOT/'requirements.measured.lock').read_text(encoding='utf-8')
            version_tokens={line.split('==',1)[1] for line in measured.splitlines() if '==' in line}
            for version in sorted(version_tokens,key=len,reverse=True):
                text=re.sub(r'(?<![\w.])'+re.escape(version)+r'(?![\w.])','VERSION',text)
            for pattern in banned:
                hits=re.findall(pattern,text)
                assert all(h=='127.0.0.1' for h in hits),'Unexpected deployment identifier in '+f.name
    with tempfile.TemporaryDirectory(prefix='pkg-check-') as temp:
        clone=Path(temp)/'pkg';shutil.copytree(ROOT,clone)
        sample=json.loads((clone/'fitted.example.json').read_text())
        assert sample['prompt_sha']==threshold_identity(c)['prompt_sha']
        sample['calibration']['slow'].update(by_type={'score':{'count':5,'status':'insufficient'}},held_out_validated=False,validation_status='candidate; held-out validation required')
        (clone/'fitted.example.json').write_text(json.dumps(sample)+'\n',encoding='utf-8',newline='\n')
        cmd=[sys.executable,'-B',str(clone/'finalize.py'),str(clone/'fitted.example.json'),'--sample']
        subprocess.run(cmd,check=True,capture_output=True)
        check_manifest(clone)
        new=json.loads((clone/'config.json').read_text());cal=json.loads((clone/'calibration.json').read_text())
        assert not new['package']['finalized'] and new['package']['sample_only']
        for r in (new['routing'],new['profiles']['metask-jev-4b']['routing']):
            assert r['margin_threshold']==sample['threshold'] and r['quota']==.25 and r['fuse']==.35
        for name,b in sample['calibration'].items():
            v=cal['models']['metask-jev-4b'][name]
            assert v['noul_calibration']['p_cal']==b['p_cal']
            assert v['temperature_by_kind']['choice']==b['T_choice']
            assert v['temperature_by_kind']['score']==b['T_score']
        preserved=cal['models']['metask-jev-4b']['slow']
        for field in ('by_type','held_out_validated','validation_status'):
            assert preserved[field]==sample['calibration']['slow'][field],'Lost calibration audit metadata'
        before=(clone/'config.json').read_bytes()
        sample['prompt_sha']='0'*64
        (clone/'bad.json').write_text(json.dumps(sample))
        bad=subprocess.run([sys.executable,'-B',str(clone/'finalize.py'),str(clone/'bad.json'),'--sample'],capture_output=True)
        assert bad.returncode!=0 and (clone/'config.json').read_bytes()==before
        # Every Python file is byte-compiled into a disposable external cache.
        subprocess.run([sys.executable,'-B','-X','pycache_prefix='+str(Path(temp)/'cache'),'-m','py_compile',
                        *map(str,ROOT.glob('*.py'))],check=True,capture_output=True)
    if a.release:
        assert c['package']['finalized'] and not c['package'].get('sample_only'),'Real fitted parameters missing'
        from core import validate_threshold
        validate_threshold(c)
        cal=load_json(ROOT/c['calibration_file'])['models']['metask-jev-4b']
        assert not cal.get('placeholder') and all(not cal[b].get('placeholder') for b in ('fast','slow','fallback'))
        for name in GENERATED:assert (ROOT/name).is_file(),'Missing release environment artifact: '+name
        for f in ROOT.rglob('*'):
            rel=f.relative_to(ROOT).parts
            assert not f.is_symlink(),'Symlink in release'
            assert '__pycache__' not in rel and not rel[0].startswith('selftest-') and f.suffix not in ('.pyc','.tmp'),'Runtime output in release'
        for name,allowed in PENDING_FIELDS.items():
            t=(ROOT/name).read_text(encoding='utf-8')
            assert not re.search(r'<(?:fill|HF_REPO)>',t),'Unnamed release placeholder: '+name
            pending=set(re.findall(r'<fill:([^>]+)>',t))
            assert pending<=allowed,'Unknown release placeholder: '+name
            if pending:print('Outstanding disclosure fields ('+name+'): '+', '.join(sorted(pending)))
        env=load_json(ROOT/'environment.resolved.json')
        assert env['python']=='3.10.12' and env['python_required']=='3.10'
        assert env['lock_sha256']==hashlib.sha256((ROOT/'requirements.lock').read_bytes()).hexdigest()
        assert env['measured_freeze_sha256']==hashlib.sha256((ROOT/'requirements.measured.lock').read_bytes()).hexdigest()
        pins=dict(line.split('==',1) for line in (ROOT/'requirements.lock').read_text().splitlines() if line and not line.startswith(('#','--')))
        assert pins==env['versions'],'Lock versions differ from measured evidence'
        measured=dict(line.split('==',1) for line in (ROOT/'requirements.measured.lock').read_text().splitlines() if line)
        measured['torch']='2.13.0+cu130'
        assert pins==measured,'Unexplained change to measured freeze'
        assert cal['fast']['by_type']['score']['status']=='insufficient'
        assert cal['slow']['by_type']['score']['count']==5
        for b in ('fast','slow','fallback'):
            assert cal[b]['held_out_validated'] is False
            assert cal[b]['validation_status']=='candidate; held-out validation required'
        assert 'PENDING' not in (ROOT/'FREEZE.md').read_text(),'Environment pending'
    subprocess.run([sys.executable,'-B',str(ROOT/'regression_checks.py')],check=True)
    print('PASS: references, config, sample finalize, rejection before mutation, hashes, compilation, identifier scan')
    print('Offline packaging check only; no dependency installation or GPU validation performed.')

if __name__=='__main__':main()
