"""Inject fitted parameters without loading a model. See fitted.example.json."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from core import load_config, threshold_identity, validate_config, validate_threshold, validate_calibration

ROOT = Path(__file__).resolve().parent
KEY = 'metask-jev-4b'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files(root=ROOT):
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlinks are not allowed in a freeze')
        rel=p.relative_to(root).parts
        if '__pycache__' in rel or rel[0].startswith('selftest-') or p.suffix in ('.pyc','.tmp'):
            raise ValueError('Runtime output in package: '+rel[0])
        if p.is_file() and p.name != 'MANIFEST.sha256':
            yield p

def freeze(root=ROOT):
    # A manifest cannot contain its own hash. FREEZE is hashed by MANIFEST.
    (root/'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in files(root)), encoding='utf-8',newline='\n')

def number(v, name, lo=0, hi=None):
    if type(v) not in (int,float) or not math.isfinite(v) or v < lo or (hi is not None and v > hi):
        raise ValueError('Invalid fitted field: '+name)
    return v

def finalize(fitted, sample=False):
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
    for name,value in [('config.json',raw),('calibration.json',cal)]:
        tmp=ROOT/(name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');tmp.replace(ROOT/name)
    env='LOCKED' if (ROOT/'requirements.lock').exists() else 'PENDING dependency resolution; not ready for release'
    (ROOT/'FREEZE.md').write_text(
        '# metask-jev-rain-4B\n\nStatus: '+('SAMPLE ONLY' if sample else 'PARAMETERS FITTED')+
        '\n\nFreeze time (UTC): <fill:freeze_time>\n\nEnvironment: Python 3.10; vLLM 0.31.0; torch 2.13.0+cu130; transformers and huggingface_hub from requirements.lock. '+env+
        '\n\nconfig.json SHA256: '+digest(ROOT/'config.json')+
        '\n\ncalibration.json SHA256: '+digest(ROOT/'calibration.json')+
        '\n\nprompt_sha: '+ident['prompt_sha']+
        '\n\nMANIFEST.sha256 covers every package file except itself (self-hashing is impossible). It includes this document. Run finalize again after editing any package file; fill freeze time then run python -B finalize.py --manifest-only. Keep environments, models, logs and data outside the release tree.\n',encoding='utf-8',newline='\n')
    freeze()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fitted',nargs='?',type=Path);p.add_argument('--sample',action='store_true');p.add_argument('--manifest-only',action='store_true')
    a=p.parse_args()
    if a.manifest_only: freeze();return
    if not a.fitted:p.error('fitted JSON required')
    finalize(json.loads(a.fitted.read_text(encoding='utf-8')),a.sample)
    print('Finalization complete; check FREEZE.md release status.')

if __name__=='__main__':main()
