"""Inject fitted parameters without loading a model. See fitted.example.json."""
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from core import load_config, threshold_identity, validate_config, validate_threshold, validate_calibration
from package_files import package_files

ROOT = Path(__file__).resolve().parent
KEY = 'metask-jev-4b'
IMAGE = 'vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files(root=ROOT):
    for p in sorted(package_files(root)):
        if p.is_symlink():
            raise ValueError('Symlinks are not allowed in a freeze')
        rel=p.relative_to(root).parts
        if rel[0].startswith('selftest-') or p.suffix in ('.pyc','.tmp'):
            raise ValueError('Runtime output in package: '+rel[0])
        if p.is_file() and p.name != 'MANIFEST.sha256':
            yield p

def freeze(root):
    # A manifest cannot contain its own hash. FREEZE is hashed by MANIFEST.
    from core import external_output
    root=external_output(root,create_parent=False)
    members=list(files(root))
    with (root/'MANIFEST.sha256').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in members))

def output_copy(output_dir, omit=()):
    from core import external_output
    out=external_output(output_dir, create_parent=False)
    members=list(files())
    out.mkdir(parents=True,exist_ok=False)
    for source in members:
        relative=source.relative_to(ROOT)
        if relative.as_posix() in omit:
            continue
        target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    return out

def freeze_record(root, sample=False):
    config=load_config(root/'config.json')
    text=('# metask-jev-rain-4B\n\nStatus: '+('SAMPLE ONLY' if sample else 'PARAMETERS FITTED')+
          '\n\nFreeze time (UTC): '+datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')+
          '\n\nRuntime image digest: '+IMAGE+
          '\n\nThis record records parameter hashes; MANIFEST.sha256 identifies package bytes. Neither certifies a model run. Measurements are described in DISCLOSURE.md.'+
          '\n\nconfig.json SHA256: '+digest(root/'config.json')+
          '\n\ncalibration.json SHA256: '+digest(root/'calibration.json')+
          '\n\nprompt_sha: '+threshold_identity(config)['prompt_sha']+
          '\n\nMANIFEST.sha256 covers package files except itself and prunes clone metadata, bytecode caches and local state.\n')
    with (root/'FREEZE.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(text)
    freeze(root)

def number(v, name, lo=0, hi=None):
    if type(v) not in (int,float) or not math.isfinite(v) or v < lo or (hi is not None and v > hi):
        raise ValueError('Invalid fitted field: '+name)
    return v

def finalize(fitted, sample=False, output_dir=None):
    list(files())  # Reject runtime artifacts before any mutation.
    raw=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    c=load_config()
    ident=threshold_identity(c)
    if fitted.get('sample_only',False) != sample:
        raise ValueError('Sample parameters require --sample; never submit sample output')
    if fitted.get('prompt_sha') != ident['prompt_sha']:
        raise ValueError('prompt_sha does not match packaged renderer')
    for key in ('backend','quant','model_key'):
        if fitted.get(key)!=ident[key]: raise ValueError('Fitted identity mismatch: '+key)
    threshold=number(fitted['threshold'],'threshold')
    achieved=number(fitted['achieved'],'achieved',0,1)
    if fitted['quota']!=.25 or fitted['fuse']!=.35:
        raise ValueError('This submission requires quota=0.25 and fuse=0.35')
    if fitted.get('n')!=231: raise ValueError('Expected 231 public calibration decisions')
    ident.update(placeholder=False, achieved=achieved, n=231)
    for r in (raw['routing'],raw['profiles'][KEY]['routing']):
        r.update(margin_threshold=threshold,margin_threshold_fitted_for=ident,
                 quota=fitted['quota'],fuse=fitted['fuse'],target_slow_share=.25)
    blocks={}
    for name in ('fast','slow','fallback'):
        b=fitted['calibration'][name]
        pc=number(b['p_cal'],name+'.p_cal',.5,1)
        tc=number(b['T_choice'],name+'.T_choice',1e-12)
        ts=number(b['T_score'],name+'.T_score',1e-12)
        block=copy.deepcopy(b)
        for field in ('p_cal','T_choice','T_score'):block.pop(field,None)
        block['source_placeholder']=b.get('placeholder',sample)
        block.update(placeholder=sample,
                     status='SAMPLE ONLY' if sample else b.get('status','candidate; held-out validation required'),
                     validation_status=b.get('validation_status','candidate; held-out validation required'),
                     noul_calibration=dict(b.get('noul_calibration',{}),p_cal=pc),
                     temperature_by_kind=dict(b.get('temperature_by_kind',{}),noul='one_bin',choice=tc,score=ts))
        blocks[name]=block
    cal={'status':'SAMPLE ONLY' if sample else fitted.get('calibration_status','candidate; held-out validation required'),
         'held_out_validated':False,
         'models':{KEY:dict(blocks,placeholder=sample,fitted_for=ident)}}
    from core import apply_profile
    checked=copy.deepcopy(raw);apply_profile(checked)
    validate_config(checked);validate_threshold(checked)
    for name in blocks: validate_calibration(cal,KEY,name)
    raw['package']['finalized']=not sample
    raw['package']['sample_only']=sample
    out=output_copy(output_dir,omit=('config.json','calibration.json','FREEZE.md'))
    for name,value in [('config.json',raw),('calibration.json',cal)]:
        with (out/name).open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    freeze_record(out,sample=sample)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fitted',nargs='?',type=Path);p.add_argument('--sample',action='store_true')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--manifest-only',action='store_true')
    modes.add_argument('--freeze-record',action='store_true')
    p.add_argument('--output-dir',required=True,type=Path,help='New absolute directory outside the source package')
    a=p.parse_args()
    if a.manifest_only or a.freeze_record:
        if a.fitted or a.sample:p.error('Record-only operations do not accept fitted parameters')
        out=output_copy(a.output_dir,omit=('FREEZE.md',) if a.freeze_record else ())
        if a.freeze_record:freeze_record(out)
        else:freeze(out)
        print('Package written: '+str(out));return
    if not a.fitted:p.error('fitted JSON required')
    out=finalize(json.loads(a.fitted.read_text(encoding='utf-8')),a.sample,a.output_dir)
    print('Finalization complete: '+str(out))

if __name__=='__main__':main()
