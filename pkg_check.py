"""Offline package validation; keeps test evidence in an external directory."""
import argparse
import ast
import hashlib
import json
import py_compile
from pathlib import Path
import re
import subprocess
import sys
import tempfile

if not __debug__:
    raise SystemExit('run without -O')

from package_files import package_files
from finalize import files

ROOT=Path(__file__).resolve().parent
REQUIRED='server.py core.py backends.py quota.py bench.py reconcile_usage.py config.json calibration.json requirements.txt resolve_env.py install.sh prepare_model.sh serve.sh launch.py selftest.sh selftest.py finalize.py fitted.example.json pkg_check.py DISCLOSURE.md README.md MAINTAINER.md FREEZE.md MANIFEST.sha256 LICENSE attention.py package_files.py test_attention.py regression_checks.py json_equivalence.py model_aux/NOTICE model_aux/LICENSE test_pkg_check.py test_service_failures.py inference_identity.json'.split()

def check_manifest(root):
    root=Path(root).resolve()
    assert b'\r' not in (root/'MANIFEST.sha256').read_bytes(),'MANIFEST must use LF'
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        sha,name=line.split('  ',1)
        path=root/name
        assert re.fullmatch('[0-9a-f]{64}',sha),'Invalid digest'
        assert not Path(name).is_absolute() and path.resolve().is_relative_to(root) and path.is_file(),'Unsafe/missing manifest member'
        assert name not in expected,'Duplicate manifest member'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,'Hash mismatch: '+name
        expected[name]=sha
    actual={p.relative_to(root).as_posix() for p in files(root)}
    assert actual==set(expected),'Manifest membership mismatch: '+str(actual.symmetric_difference(expected))

def check_freeze():
    from core import load_config,threshold_identity
    record=(ROOT/'FREEZE.md').read_text(encoding='utf-8')
    for name in ('config.json','calibration.json'):
        match=re.search(re.escape(name)+r' SHA256: ([0-9a-f]{64})',record)
        assert match and match[1]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),'FREEZE digest mismatch: '+name
    assert 'prompt_sha: '+threshold_identity(load_config())['prompt_sha'] in record,'FREEZE prompt identity mismatch'
    from launch import IMAGE
    assert 'Runtime image digest: '+IMAGE in record,'FREEZE image mismatch'

# Scanner policy is explicit. Only these definitions and the scanner test fixtures
# are excluded from text scans; executable code is still checked through the AST.
SCAN_MARKER = '[[FACTS-PENDING'
SCAN_LEGACY = 'VLLM_ATTENTION_BACKEND'
SCAN_PROCESS = r'\bonce\b[^.!?\]]*\bavailable\b|\bbefore\s+publication\b|\bmust\s+supply\b|\bnot\s+yet\b|\bTB[DA]\b|\bto\s+be\s+(?:added|filled|determined|confirmed)\b|\bwill\s+be\s+(?:added|recorded|filled)\b'
SCAN_DOCUMENT = r'<fill:|\bpending\b|\bplaceholder\b|\bconfirm\b|\boptional\b|\bTODO\b|Maintainers must|待定|待核'
SCAN_SOURCE = r'<fill:|\bTODO\b'
SCAN_SHELL_DELETE = r'(^|[\s;/|&()])(?:rm|rmdir|unlink|truncate|shred)(?=\s|$|[;|&()])|-delete\b'
PUBLIC_DATA_SHA256 = {
    'easy.jsonl': '231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b',
    'original.jsonl': '5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180',
    'hard.jsonl': '89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb',
}


def scan_text(path, text):
    """Preserve line numbers while removing documented policy/renderer exclusions."""
    if path.suffix != '.py':
        return text
    lines=text.splitlines()
    for node in ast.parse(text).body:
        protected = path.name=='core.py' and isinstance(node,ast.FunctionDef) and node.name in ('render','choice_codes')
        policy = (isinstance(node,ast.Assign) and
                  ((path.name=='pkg_check.py' and all(isinstance(t,ast.Name) and t.id.startswith('SCAN_') for t in node.targets)) or
                   (path.name=='test_pkg_check.py' and all(isinstance(t,ast.Name) and t.id=='SCAN_TEST_FIXTURES' for t in node.targets))))
        if protected or policy:
            lines[node.lineno-1:node.end_lineno]=['']*(node.end_lineno-node.lineno+1)
    return '\n'.join(lines)


def check_private_text(text, versions=()):
    for version in sorted(set(versions),key=len,reverse=True):
        text=re.sub(r'(?<![\w.])'+re.escape(version)+r'(?![\w.])','VERSION',text)
    addresses=re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',text)
    assert all(value in ('127.0.0.1','0.0.0.0') for value in addresses),'Unexpected address'
    assert not re.search(r'[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:home|Users|root)/[\w.-]+',text),'Private deployment path'
    assert not re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',text),'Email address'
    assert not re.search(r'(?<!\w)~[/\\]|\bssh\s+(?:-[^\s]+\s+)*[\w.-]+',text,re.I),'Private home path or remote-shell command'


def dotted_name(node, aliases):
    if isinstance(node,ast.Name):return aliases.get(node.id,node.id)
    if isinstance(node,ast.Attribute):return dotted_name(node.value,aliases)+'.'+node.attr
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)>=2:
        def literal(value):
            if isinstance(value,ast.Constant) and isinstance(value.value,str):return value.value
            if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add):return literal(value.left)+literal(value.right)
            return ''
        return dotted_name(node.args[0],aliases)+'.'+literal(node.args[1])
    return ''


def check_deletions(path, tree):
    aliases={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):aliases.update({e.asname or e.name:e.name for e in node.names})
        elif isinstance(node,ast.ImportFrom):aliases.update({e.asname or e.name:(node.module or '')+'.'+e.name for e in node.names})
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        name=dotted_name(node.func,aliases);method=name.rsplit('.',1)[-1]
        temporary_cleanup=(path.name=='quota.py' and ast.dump(node,include_attributes=False)==ast.dump(ast.parse('Path(temporary).unlink(missing_ok=True)').body[0].value,include_attributes=False))
        atomic_replace=(path.name=='quota.py' and ast.dump(node,include_attributes=False)==ast.dump(ast.parse('os.replace(temporary, self.state_path)').body[0].value,include_attributes=False))
        if method in {'unlink','rmtree','TemporaryDirectory','rmdir','removedirs'}:
            assert temporary_cleanup,'Destructive call: '+path.name+':'+str(node.lineno)
        if name in {'os.remove','os.replace','os.rename','shutil.move'}:
            assert atomic_replace,'Destructive file call: '+path.name+':'+str(node.lineno)
        if method in {'replace','rename'} and len(node.args)==1:
            assert atomic_replace,'Destructive path call: '+path.name+':'+str(node.lineno)
        if name.startswith(('subprocess.','os.system','os.popen','os.exec','os.spawn')):
            literals=[n.value for arg in [*node.args,*(kw.value for kw in node.keywords)] for n in ast.walk(arg) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            for value in literals:check_shell_deletions(path,value)


def check_shell_deletions(path, text):
    assert not re.search(SCAN_SHELL_DELETE,text),'Destructive shell command: '+path.name


def check_legacy_environment(path, text, tree):
    if SCAN_LEGACY not in text:return
    if path.name in ('regression_checks.py','test_attention.py'):
        return  # Regression fixtures exercise removal of this obsolete variable.
    assert path.name=='launch.py','Deprecated environment variable: '+path.name
    calls={id(arg) for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='pop'
           and isinstance(node.func.value,ast.Name) and node.func.value.id=='env'
           for arg in node.args[:1]}
    literals=[node for node in ast.walk(tree) if isinstance(node,ast.Constant) and node.value==SCAN_LEGACY]
    assert literals and all(id(node) in calls for node in literals),'Deprecated variable must only be removed'
    assert text.count(SCAN_LEGACY)==len(literals),'Deprecated variable outside removal call'


def check_sources():
    modules={p.stem for p in ROOT.glob('*.py')}
    external={'torch','transformers','vllm','peft'}
    measured=(ROOT/'requirements.measured.lock').read_text(encoding='utf-8')
    versions={s.split('==',1)[1] for s in measured.splitlines() if '==' in s}
    for path in package_files(ROOT):
        if path.is_symlink():raise AssertionError('Symlink in release: '+str(path))
        if path.relative_to(ROOT).parts[0]=='model_aux':continue
        data=path.read_bytes()
        assert b'\r' not in data,'Package text must use LF: '+path.name
        original=data.decode('utf-8')
        text=scan_text(path,original)
        if path.suffix=='.py':
            tree=ast.parse(original)
            for node in ast.walk(tree):
                imports=[x.name for x in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
                for name in imports:assert name.split('.')[0] in modules|sys.stdlib_module_names|external,'Unresolved import: '+name
            check_deletions(path,tree)
            check_legacy_environment(path,text,tree)
        else:
            assert SCAN_LEGACY not in text,'Deprecated environment variable: '+path.name
        if path.suffix=='.sh':
            check_shell_deletions(path,text)
            assert not re.search(r'(?m)(?:^|[;\s])(?:exec\s+)?python(?:\s|$)',text),'Unversioned Python: '+path.name
        try:check_private_text(text,versions)
        except AssertionError as exc:raise AssertionError(str(exc)+': '+path.name) from exc
    print('PASS: source imports, LF, deletion audit, private identifiers and shell interpreters')


def release_wording_errors(path, original):
    text=scan_text(path,original)
    errors=[]
    marker_pattern=re.escape(SCAN_MARKER)+r':.*?\]\]'
    # Blank marker contents, including multiline markers, without losing line numbers.
    cleaned=re.sub(marker_pattern,lambda m:'\n'*m.group(0).count('\n'),text,flags=re.S)
    for number,line in enumerate(text.splitlines(),1):
        if SCAN_MARKER in line:errors.append(f'{path.name}:{number}: '+SCAN_MARKER)
    patterns=SCAN_PROCESS+'|'+(SCAN_DOCUMENT if path.suffix=='.md' else SCAN_SOURCE)
    for match in re.finditer(patterns,cleaned,re.I):
        number=cleaned.count('\n',0,match.start())+1
        errors.append(f'{path.name}:{number}: prohibited release wording')
    if path.suffix=='.md':
        for match in re.finditer(r'\]\]',cleaned):
            number=cleaned.count('\n',0,match.start())+1
            errors.append(f'{path.name}:{number}: unmatched release marker terminator')
    return errors


def release_reference_errors(path, text):
    errors=[]
    if re.search(r'\b(?=[0-9a-f]{7,39}\b)(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]+\b',text):errors.append(path.name+': abbreviated revision')
    if re.search(r'(?i)(?:package|code)\s+commit\s*[:|=]\s*v[0-9]+\.',text):errors.append(path.name+': tag used as commit')
    for url in re.findall(r'https://huggingface\.co/[^\s<>\)\]]+',text):
        if not re.fullmatch(r'https://huggingface\.co/[^/]+/[^/]+/tree/[0-9a-f]{40}',url.rstrip('.,;')):
            errors.append(path.name+': unpinned model URL')
    for url in re.findall(r'https://github\.com/[^\s<>\)\]]+',text):
        if re.match(r'https://github\.com/[^/]+/[^/]+/(?:tree|blob)/',url) and not re.fullmatch(r'https://github\.com/[^/]+/[^/]+/(?:tree|blob)/[0-9a-f]{40}(?:/[^\s]+)?',url.rstrip('.,;')):
            errors.append(path.name+': unpinned source URL')
    return errors


def check_release_documents():
    errors=[]
    for path in package_files(ROOT):
        if path.relative_to(ROOT).parts[0]=='model_aux':continue
        text=path.read_text(encoding='utf-8')
        errors.extend(release_wording_errors(path,text))
        if path.name not in ('README.md','DISCLOSURE.md','MAINTAINER.md','FREEZE.md'):continue
        errors.extend(release_reference_errors(path,text))
    return errors


def check_public_hashes():
    text=(ROOT/'DISCLOSURE.md').read_text(encoding='utf-8')
    for name,digest in PUBLIC_DATA_SHA256.items():
        row=re.search(r'(?m)^\|[^\n|]*'+re.escape(name)+r'[^\n]*$',text)
        assert row and digest in row[0],'Public dataset hash mismatch: '+name
    print('PASS: public dataset hashes match pinned harness manifest')


def check_environment_evidence(evidence):
    env=json.loads((ROOT/'environment.resolved.json').read_text(encoding='utf-8'))
    for name,field in (('requirements.lock','lock_sha256'),('requirements.measured.lock','measured_freeze_sha256')):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==env[field],'Historical environment hash: '+name
    pins=dict(line.split('==',1) for line in (ROOT/'requirements.lock').read_text().splitlines() if line and not line.startswith(('#','--')))
    measured=dict(line.split('==',1) for line in (ROOT/'requirements.measured.lock').read_text().splitlines() if line)
    measured['torch']='2.13.0+cu130'
    assert pins==measured==env['versions'],'Historical version evidence mismatch'
    assert env['python_required']=='>=3.10','Historical runtime requirement mismatch'
    target=evidence/'resolved-environment'
    run([sys.executable,'-B',str(ROOT/'resolve_env.py'),'--output-dir',str(target)])
    for name in ('requirements.lock','environment.resolved.json'):
        assert (target/name).read_bytes()==(ROOT/name).read_bytes(),'Environment generator mismatch: '+name
    print('PASS: historical environment evidence and byte-identical regeneration (not the runtime version gate)')

def run(command):
    result=subprocess.run(command,check=False,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.stdout:print(result.stdout.rstrip())
    if result.stderr:print(result.stderr.rstrip())
    if result.returncode:raise RuntimeError('Check failed: '+' '.join(map(str,command)))

def sample_roundtrip(evidence):
    from core import load_config,threshold_identity
    sample=json.loads((ROOT/'fitted.example.json').read_text(encoding='utf-8'))
    assert sample['prompt_sha']==threshold_identity(load_config())['prompt_sha']
    sample['calibration']['slow'].update(by_type={'score':{'count':5,'status':'insufficient'}},held_out_validated=False,validation_status='candidate; held-out validation required')
    sample_path=evidence/'sample.json'
    with sample_path.open('x',encoding='utf-8') as stream:json.dump(sample,stream)
    target=evidence/'sample-package'
    run([sys.executable,'-B',str(ROOT/'finalize.py'),str(sample_path),'--sample','--output-dir',str(target)])
    check_manifest(target)
    new=json.loads((target/'config.json').read_text());cal=json.loads((target/'calibration.json').read_text())
    assert not new['package']['finalized'] and new['package']['sample_only']
    for routing in (new['routing'],new['profiles']['metask-jev-4b']['routing']):
        assert routing['margin_threshold']==sample['threshold'] and routing['quota']==.25 and routing['fuse']==.35
    for name,block in sample['calibration'].items():
        value=cal['models']['metask-jev-4b'][name]
        assert value['noul_calibration']['p_cal']==block['p_cal']
        assert value['temperature_by_kind']['choice']==block['T_choice']
        assert value['temperature_by_kind']['score']==block['T_score']
    preserved=cal['models']['metask-jev-4b']['slow']
    for field in ('by_type','held_out_validated','validation_status'):assert preserved[field]==sample['calibration']['slow'][field]
    sample['prompt_sha']='0'*64
    bad=evidence/'bad.json'
    with bad.open('x',encoding='utf-8') as stream:json.dump(sample,stream)
    rejected=evidence/'rejected-package'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'finalize.py'),str(bad),'--sample','--output-dir',str(rejected)],capture_output=True)
    assert result.returncode!=0 and not rejected.exists(),'Invalid fitting mutated output'
    print('PASS: sample finalize roundtrip and rejection before mutation')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    from core import external_output,load_config,load_json,validate_calibration,validate_threshold
    if args.output_dir:
        evidence=external_output(args.output_dir,create_parent=False);evidence.mkdir(parents=True,exist_ok=False)
    else:
        evidence=Path(tempfile.mkdtemp(prefix='jev4b-check-')).resolve()
        external_output(evidence,create_parent=False)
    print('Evidence retained: '+str(evidence))
    for name in REQUIRED:assert (ROOT/name).is_file(),'Missing file: '+name
    assert not any((ROOT/name).exists() for name in ('harness.md','environment.md')),'Obsolete document'
    assert {p.name for p in ROOT.glob('*.md')}=={'README.md','DISCLOSURE.md','MAINTAINER.md','FREEZE.md'},'Unexpected document'
    check_manifest(ROOT);check_freeze();check_sources();check_environment_evidence(evidence);check_public_hashes()
    from launch import AUX_SHA256
    for name,expected in AUX_SHA256.items():
        assert hashlib.sha256((ROOT/'model_aux'/name).read_bytes()).hexdigest()==expected,'Auxiliary hash: '+name
    config=load_config();assert config['model']['path']==''
    assert set(config['profiles'])=={'metask-jev-4b'}
    assert config['package']['finalized'] and not config['package'].get('sample_only')
    assert validate_threshold(config)==[]
    calibration=load_json(ROOT/config['calibration_file'])
    for block in ('fast','slow','fallback'):validate_calibration(calibration,'metask-jev-4b',block)
    from json_equivalence import verify
    verify()
    sample_roundtrip(evidence)
    for test in ('regression_checks.py','test_attention.py','test_service_failures.py','test_pkg_check.py'):
        run([sys.executable,'-B',str(ROOT/test),'--output-dir',str(evidence/test.removesuffix('.py'))])
    cache=evidence/'cache';cache.mkdir()
    for source in ROOT.glob('*.py'):
        py_compile.compile(str(source),cfile=str(cache/(source.stem+'.pyc')),doraise=True)
    print('PASS: manifest, freeze hashes, auxiliary hashes, config, calibration and compilation')
    errors=check_release_documents()
    if args.release and errors:
        for error in errors:print('RELEASE BLOCKER: '+error)
        print('Release blocked; all preceding offline checks passed.')
        return 1
    print('PASS: offline package checks'+(' (release)' if args.release else ' (non-release)'))
    print('No GPU inference or container validation performed.')
    return 0

if __name__=='__main__':raise SystemExit(main())
