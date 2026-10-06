"""Sequential public-data selftest. Owns and stops its GPU service processes."""
import argparse
import json
import subprocess
import sys
import tempfile
import signal
from pathlib import Path
from launch import start, stop, ROOT
from bench import load_public_tasks, PUBLIC_FILES, _read_rows, normalize_task, _require_unique

def run(*args):
    subprocess.run([sys.executable,'-B',*map(str,args)],cwd=ROOT,check=True)

def prepare_public(public, output_dir):
    public=Path(public).resolve();out=Path(output_dir).resolve()
    if out.is_relative_to(ROOT) or ROOT.is_relative_to(out):
        raise ValueError('Output directory must be outside the package')
    if out==public or public.is_relative_to(out) or out.is_relative_to(public):
        raise ValueError('Output directory must be separate from public sources')
    tasks=[]
    for filename,tier in PUBLIC_FILES:
        for row in _read_rows(public/filename):
            if 'tier' in row and row['tier']!=tier:raise ValueError('Public tier mismatch')
            tasks.append(normalize_task(row,tier=tier,require_expected=True))
    _require_unique(tasks)
    if len(tasks)!=231:raise ValueError('Expected 231 official public tasks')
    out.mkdir(parents=True,exist_ok=False)
    index=out/'public231.index.jsonl'
    with index.open('w',encoding='utf-8',newline='\n') as f:
        for task in tasks:f.write(json.dumps({k:task[k] for k in ('id','type','tier')})+'\n')
    load_public_tasks(index,public,231)
    return out,index

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--public',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--port',type=int,default=8001)
    a=p.parse_args();a.public=a.public.resolve()
    if not json.loads((ROOT/'config.json').read_text(encoding='utf-8')).get('package',{}).get('finalized'):
        raise ValueError('Finalize real fitted parameters before GPU selftest')
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    out,index=prepare_public(a.public,a.output_dir)  # validate before GPU allocation
    common=['--public',a.public,'--index',index,'--endpoint',f'http://127.0.0.1:{a.port}','--concurrency','1']
    proc=None
    try:
        proc=start(a.model,a.port,'fast_only',state_dir=out/'state',run_id='fast')
        run('bench.py',*common,'--calibrate-margin','--target-slow-share','.25','--output-prefix',out/'margin')
        run('bench.py',*common,'--engine-stats','--output-prefix',out/'fast')
        run('reconcile_usage.py','--run',out/'fast')
        stop(proc);proc=None
        # Candidate threshold is reported only; it never mutates the release config.
        for name,off in [('routed',False),('routed_off',True)]:
            proc=start(a.model,a.port,'routed',no_cache=off,state_dir=out/'state',run_id=name)
            run('bench.py',*common,'--engine-stats','--output-prefix',out/name)
            stop(proc);proc=None
        run('reconcile_usage.py','--run',out/'routed','--run-off',out/'routed_off')
        run('bench.py','--fit-calibration','--fast-run',out/'fast','--routed-run',out/'routed',
            '--model-key','metask-jev-4b','--output',out/'calibration.candidate.json')
        for name in ('fast.summary.json','routed.summary.json','routed.reconcile.json'):
            print(name);print((out/name).read_text(encoding='utf-8'))
        print('Selftest outputs written to the requested external directory.')
    finally:
        if proc:stop(proc)

if __name__=='__main__':main()
