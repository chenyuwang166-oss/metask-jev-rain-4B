"""Foreground supervisor: readiness check, print health, forward termination."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
import uuid
import urllib.request

ROOT=Path(__file__).resolve().parent

def prepare_state(state_dir=None, run_id=None):
    state=Path(state_dir).resolve() if state_dir else Path(tempfile.mkdtemp(prefix='jev4b-state-')).resolve()
    if state.is_relative_to(ROOT):raise ValueError('State directory must be outside package')
    rid=run_id or 'run-'+uuid.uuid4().hex
    if not rid or Path(rid).name!=rid or rid in ('.','..') or '/' in rid or chr(92) in rid:
        raise ValueError('run_id must be a single directory name')
    state.mkdir(parents=True,exist_ok=True)
    # Actual create/write/flush rather than a permission-bit guess.
    try:
        with tempfile.TemporaryFile(dir=state) as probe:
            probe.write(b'probe');probe.flush();os.fsync(probe.fileno())
    except OSError:
        raise ValueError('Quota state directory is not writable') from None
    if (state/rid).exists():raise ValueError('run_id already exists; choose a fresh run_id')
    return state,rid

def validate_health(health, no_cache=False):
    bad=[w for w in health.get('warnings',[]) if any(s in str(w).lower() for s in ('quota persistence','memory','calibration placeholder'))]
    if bad or health.get('prefix_caching') is not (not no_cache) or health.get('supports_slow_session') is not True:
        raise RuntimeError('Service readiness rejected: persistence, calibration or cache/session capability')

def start(model, port, mode='routed', config=None, no_cache=False, state_dir=None, run_id=None):
    if sys.version_info[:2]!=(3,10):raise ValueError('Python 3.10 required')
    for name,version in [('vllm','0.31.0'),('torch','2.13.0+cu130')]:
        if importlib.metadata.version(name).split('+')[0]!=version.split('+')[0]:raise ValueError('Environment pin mismatch: '+name+' installed '+importlib.metadata.version(name)+' pinned '+version)
    if not (ROOT/'requirements.lock').is_file():raise ValueError('Resolve and freeze requirements.lock first')
    for line in (ROOT/'requirements.lock').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(('#','--')):continue
        name,version=line.split()[0].split('==',1)
        if importlib.metadata.version(name).split('+')[0]!=version.split('+')[0]:raise ValueError('Locked dependency mismatch: '+name+' installed '+importlib.metadata.version(name)+' locked '+version)
    p=Path(model).resolve()
    for n in ('config.json','preprocessor_config.json','video_preprocessor_config.json','merges.txt','vocab.json'):
        if not (p/n).is_file():raise ValueError('Missing model file: '+n)
    cfg=Path(config or ROOT/'config.json').resolve()
    c=json.loads(cfg.read_text(encoding='utf-8'))
    if mode=='routed' and not c.get('package',{}).get('finalized'):
        raise ValueError('Real fitted parameters must be finalized before routed serving')
    state,rid=prepare_state(state_dir,run_id)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',port))
    env=dict(os.environ,VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ATTENTION_BACKEND='FLASH_ATTN',PYTHONDONTWRITEBYTECODE='1',
             VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    args=[sys.executable,'-B',str(ROOT/'server.py'),'--config',str(cfg),'--model',str(p),
          '--bind','127.0.0.1','--port',str(port),'--backend','vllm','--quant','bf16',
          '--mode',mode,'--model-key','metask-jev-4b','--state-dir',str(state),'--run-id',rid]
    if no_cache:args.append('--no-prefix-caching')
    proc=subprocess.Popen(args,cwd=ROOT,env=env,start_new_session=True)
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('Service exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                    health=json.load(r)
                    if r.status==200 and health['mode']==mode and health['model_key']=='metask-jev-4b':
                        validate_health(health,no_cache)
                        print(json.dumps(health,ensure_ascii=False),flush=True)
                        return proc
            except (OSError,ValueError,KeyError):pass
            time.sleep(1)
        raise TimeoutError('Health readiness timed out')
    except BaseException:
        stop(proc);raise

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--port',type=int,default=8000)
    ap.add_argument('--state-dir');ap.add_argument('--run-id')
    a=ap.parse_args()
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    proc=None
    try:
        proc=start(a.model,a.port,state_dir=a.state_dir,run_id=a.run_id)
        return proc.wait()
    except KeyboardInterrupt:return 130
    finally:
        if proc:stop(proc)

if __name__=='__main__':raise SystemExit(main())
