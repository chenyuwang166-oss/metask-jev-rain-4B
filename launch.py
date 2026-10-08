"""Offline foreground supervisor with isolated state and observed readiness."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
IMAGE = 'vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
VERSIONS = {'vllm': '0.31.0', 'torch': '2.13.0', 'transformers': '5.17.0',
            'tokenizers': '0.23.2', 'triton': '3.7.1'}
AUX_SHA256 = {
    'preprocessor_config.json': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
    'video_preprocessor_config.json': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
IDENTITY_SHA256 = {
    'config.json': '4d4ea499a469baaeb518473ff0e2b1e823697024639a93319219e27c9759336e',
    'tokenizer.json': '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
    'tokenizer_config.json': '66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b',
    'generation_config.json': 'afa48c3c3f7a3e873e82b5e77af288143c5b400d5df39ae4405e40e8e92d9fcb',
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
}
WEIGHT_BYTES = 9078620536
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
MODEL_FILES = frozenset((*IDENTITY_SHA256, 'model.safetensors', *AUX_SHA256))
SYMLINK_REQUIREMENT = 'state filesystem must support symbolic links'


def resolve_path(path, strict=False):
    try:
        return Path(path).resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise ValueError('Cannot resolve path ' + str(path)) from exc


def resolve_model_path(path, strict=False):
    try:
        return Path(path).resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, (FileNotFoundError, RuntimeError)) or getattr(exc, 'errno', None) == errno.ELOOP:
            raise ValueError('Model file is a dangling or looping symbolic link: ' + Path(path).name
                             + '; MODEL must contain regular files (a Hugging Face cache snapshots/<sha>'
                             + ' directory cannot be mounted alone)') from exc
        raise ValueError('Cannot resolve model path ' + str(path)) from exc


def check_python(version_info=None):
    if tuple(sys.version_info if version_info is None else version_info) < (3, 10):
        raise ValueError('Python >= 3.10 required')


def check_versions():
    installed = {}
    for name, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(name + ' missing; run in ' + IMAGE) from None
        if actual.split('+', 1)[0] != expected:
            raise ValueError(name + ' installed ' + actual + ', required ' + expected + '; run in ' + IMAGE)
        installed[name] = actual
    return installed


def external_directory(value, label):
    if value is None or not Path(value).is_absolute():
        raise ValueError(label + ' must be an absolute path')
    path = (resolve_model_path(value, strict=Path(value).is_symlink())
            if label == 'Model directory' else resolve_path(value))
    if path.is_relative_to(ROOT) or ROOT.is_relative_to(path):
        raise ValueError(label + ' must be outside and must not contain the package')
    if path.exists() and not path.is_dir():
        raise ValueError(label + ' must be a directory')
    return path


def validate_paths(state_dir, model=None):
    state = external_directory(state_dir, 'State directory')
    model_path = external_directory(model, 'Model directory') if model is not None else None
    if model_path is not None:
        if state.is_relative_to(model_path) or model_path.is_relative_to(state):
            raise ValueError('State and model directories must not contain each other')
        if not model_path.is_dir():
            raise ValueError('Model directory does not exist: ' + str(model_path))
    return state, model_path


def validate_run_id(run_id=None):
    rid = run_id if run_id is not None else datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S%fZ')
    if not isinstance(rid, str) or RUN_ID.fullmatch(rid) is None:
        raise ValueError('run_id must match ' + RUN_ID.pattern)
    return rid


def prepare_state(state_dir=None, run_id=None, model=None):
    state, _ = validate_paths(state_dir, model)
    rid = validate_run_id(run_id)
    startup = state / (rid + '-startup')
    if len(os.fsencode(startup / 'tmp')) > 70:
        raise ValueError('Startup tmp path exceeds 70 bytes; use a shorter state path or run_id')
    if (state / rid).exists():
        raise ValueError('run_id already exists; choose a fresh run_id')
    if startup.exists() or startup.is_symlink():
        raise ValueError('run_id already exists: startup state ' + str(startup) + '; use a new WORK')
    cache = resolve_path(state / 'cache')
    if not cache.is_relative_to(state):
        raise ValueError('Cache directory must remain inside state directory')
    try:
        state.mkdir(parents=True, exist_ok=True)
        startup.mkdir(exist_ok=False)
        for name in ('tmp', 'home', 'model'):
            (startup / name).mkdir(exist_ok=False)
        cache.mkdir(exist_ok=True)
        with (startup / 'write-probe').open('xb') as probe:
            probe.write(b'probe')
            probe.flush()
            os.fsync(probe.fileno())
    except OSError as exc:
        raise ValueError('State directory is not writable or run_id already exists: ' + str(state)) from exc
    return state, rid


def build_child_env(state, rid, environ=None, configure_tempdir=True):
    state = external_directory(state, 'State directory')
    startup = state / (validate_run_id(rid) + '-startup')
    if resolve_path(startup) != startup:
        raise ValueError('Startup directory must not redirect outside its assigned path')
    temp = resolve_path(startup / 'tmp')
    home = resolve_path(startup / 'home')
    if not temp.is_relative_to(startup) or not home.is_relative_to(startup):
        raise ValueError('HOME and temporary paths must remain inside startup directory')
    if len(os.fsencode(temp)) > 70:
        raise ValueError('Startup tmp path exceeds 70 bytes; use a shorter state path or run_id')
    env = dict(os.environ if environ is None else environ)
    cache_defaults = {
        'XDG_CACHE_HOME': state / 'cache', 'VLLM_CACHE_ROOT': state / 'cache' / 'vllm',
        'TRITON_CACHE_DIR': state / 'cache' / 'triton',
        'TORCHINDUCTOR_CACHE_DIR': state / 'cache' / 'inductor',
        'HF_HOME': state / 'cache' / 'hf', 'CUDA_CACHE_PATH': state / 'cache' / 'nv',
        'XDG_CONFIG_HOME': state / 'cache' / 'config',
        'XDG_DATA_HOME': state / 'cache' / 'data',
        'XDG_STATE_HOME': state / 'cache' / 'state',
        'VLLM_CONFIG_ROOT': state / 'cache' / 'config' / 'vllm',
        'TRITON_HOME': state / 'cache' / 'triton-home',
        'FLASHINFER_WORKSPACE_BASE': state / 'cache' / 'flashinfer',
        'NUMBA_CACHE_DIR': state / 'cache' / 'numba',
        'TORCH_EXTENSIONS_DIR': state / 'cache' / 'torch-extensions',
        'MPLCONFIGDIR': state / 'cache' / 'matplotlib',
    }
    for key, default in cache_defaults.items():
        env.setdefault(key, str(default))
    # These inherited overrides take precedence over the cache roots above.
    for key in (*cache_defaults, 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
                'TRANSFORMERS_CACHE', 'HF_ASSETS_CACHE', 'TORCH_HOME'):
        if key not in env:
            continue
        path = Path(env[key])
        if not path.is_absolute() or not resolve_path(path).is_relative_to(state):
            raise ValueError(key + ' must be an absolute path inside state directory')
        env[key] = str(resolve_path(path))
    env.update(VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_NO_USAGE_STATS='1',
               DO_NOT_TRACK='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', VLLM_HOST_IP='127.0.0.1',
               HOME=str(home), TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp),
               VLLM_RPC_BASE_PATH=str(temp), JEV_STARTUP_DIR=str(startup))
    # vLLM 0.31 ignores this legacy variable; only the explicit LLM argument is used.
    env.pop('VLLM_ATTENTION_BACKEND', None)
    env.pop('PYTHONPYCACHEPREFIX', None)
    if configure_tempdir:
        tempfile.tempdir = str(temp)
    return env


def _regular_model_file(path, allow_symlink=False):
    if path.is_symlink():
        resolve_model_path(path, strict=True)
        if not allow_symlink:
            raise ValueError('Model file must be a regular file, not a symbolic link: ' + path.name)
    if not path.is_file():
        raise ValueError('Missing or non-regular model file: ' + path.name)


def _verify_hash(path, expected, allow_symlink=False):
    _regular_model_file(path, allow_symlink)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Model file SHA256 mismatch: ' + path.name)


def _check_model_files(model, allow_symlink=False):
    for source in model.iterdir():
        name = source.name
        if name in MODEL_FILES:
            continue
        lowered = name.lower()
        if (lowered.endswith(('.safetensors', '.bin', 'processor_config.json'))
                or lowered in ('model.safetensors.index.json', 'added_tokens.json', 'special_tokens_map.json')):
            raise ValueError('Unexpected model loading or tokenizer file: ' + name)
    for name, expected in AUX_SHA256.items():
        _verify_hash(ROOT / 'model_aux' / name, expected)
    for name, expected in IDENTITY_SHA256.items():
        _verify_hash(model / name, expected, allow_symlink)
    weights = model / 'model.safetensors'
    _regular_model_file(weights, allow_symlink)
    if weights.stat().st_size != WEIGHT_BYTES:
        raise ValueError('Missing or wrong size model file: model.safetensors')
    present = {name for name in AUX_SHA256 if (model / name).exists() or (model / name).is_symlink()}
    if present and len(present) != len(AUX_SHA256):
        missing = sorted(set(AUX_SHA256) - present)
        raise ValueError('Partial auxiliary model files; missing: ' + ', '.join(missing))
    if present:
        for name, expected in AUX_SHA256.items():
            _verify_hash(model / name, expected, allow_symlink)
    return bool(present)


def prepare_model(model, startup):
    model = external_directory(model, 'Model directory')
    startup = external_directory(startup, 'Startup directory')
    has_aux = _check_model_files(model)
    view = startup / 'model'
    if resolve_path(view) != view or not view.is_dir():
        raise ValueError('Model view must be the prepared startup/model directory')
    if any(view.iterdir()):
        raise ValueError('Model view must be empty before preparation')
    for name in sorted(MODEL_FILES):
        source = (ROOT / 'model_aux' if name in AUX_SHA256 and not has_aux else model) / name
        target = resolve_path(source, strict=True)
        try:
            (view / name).symlink_to(target)
        except (OSError, RuntimeError) as exc:
            raise ValueError('Cannot create model view link ' + name + '; ' + SYMLINK_REQUIREMENT) from exc
    return view


def verify_model_view(model):
    if {p.name for p in model.iterdir()} != MODEL_FILES:
        raise ValueError('Prepared model view must contain exactly the model file whitelist')
    if not all((model / name).is_symlink() for name in MODEL_FILES):
        raise ValueError('Prepared model view must contain symbolic links only')
    _check_model_files(model, allow_symlink=True)
    return model


def configure_server_runtime(model, state_dir, run_id=None, startup_attempt=None):
    """Apply the same explicit state boundary to every server start."""
    state = external_directory(state_dir, 'State directory')
    model = external_directory(model, 'Model directory')
    rid = validate_run_id(run_id)
    if startup_attempt is not None:
        if startup_attempt not in (1, 2, 3):
            raise ValueError('Invalid startup attempt')
        startup = state / (rid + '-startup')
        if resolve_path(startup) != startup or not startup.is_dir():
            raise ValueError('Supervised startup directory must match state directory and run_id')
        if model != startup / 'model':
            validate_paths(state, model)
        env = build_child_env(state, rid)
        expected_log = startup / f'attempt-{startup_attempt}.log'
        if resolve_path(os.environ.get('JEV_ATTENTION_LOG', '')) != expected_log:
            raise ValueError('Attention log must match the current startup attempt')
        quota_state = startup / f'state-{startup_attempt}'
        if resolve_path(quota_state) != quota_state:
            raise ValueError('Attempt state must not redirect outside startup directory')
    else:
        env = build_child_env(state, rid, configure_tempdir=False)
        if 'JEV_ATTENTION_LOG' in env:
            log = Path(env['JEV_ATTENTION_LOG'])
            if not log.is_absolute() or not resolve_path(log).is_relative_to(state):
                raise ValueError('Attention log must be inside state directory')
            env['JEV_ATTENTION_LOG'] = str(resolve_path(log))
        state, rid = prepare_state(state, rid, model=model)
        startup = state / (rid + '-startup')
        tempfile.tempdir = env['TMPDIR']
        quota_state = state
    os.environ.clear()
    os.environ.update(env)
    view = verify_model_view(model) if model == startup / 'model' else prepare_model(model, startup)
    return view, quota_state, rid


def validate_health(health, no_cache=False):
    if (health.get('warnings', []) or health.get('status') != 'ok'
            or health.get('prefix_caching') is not (not no_cache)
            or health.get('supports_slow_session') is not True):
        raise RuntimeError('Service readiness rejected: health warnings or cache/session capability; health='
                           + json.dumps(health)[:1500])


def start(model, port=8000, mode='routed', config=None, no_cache=False,
          state_dir=None, run_id=None, bind='127.0.0.1'):
    check_python()
    state, model = validate_paths(state_dir, model)
    check_versions()
    if bind not in ('127.0.0.1', '0.0.0.0'):
        raise ValueError('bind must be 127.0.0.1 or 0.0.0.0')
    cfg = resolve_path(config or ROOT / 'config.json')
    c = json.loads(cfg.read_text(encoding='utf-8'))
    if mode == 'routed' and not c.get('package', {}).get('finalized'):
        raise ValueError('Real fitted parameters must be finalized before routed serving')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((bind, port))
    rid = validate_run_id(run_id)
    env = build_child_env(state, rid, configure_tempdir=False)
    state, rid = prepare_state(state, rid, model=model)
    startup = state / (rid + '-startup')
    tempfile.tempdir = env['TMPDIR']
    view = prepare_model(model, startup)
    args = [sys.executable, '-B', str(ROOT / 'server.py'), '--config', str(cfg), '--model', str(view),
            '--bind', bind, '--port', str(port), '--backend', 'vllm', '--quant', 'bf16',
            '--mode', mode, '--model-key', 'metask-jev-4b', '--state-dir', str(state), '--run-id', rid]
    if no_cache:
        args.append('--no-prefix-caching')
    return start_attempts(args, env, state, rid, port, mode, no_cache)


@contextmanager
def attempt_events(path):
    events = path.open('x', encoding='utf-8', newline='\n')
    try:
        yield events
    finally:
        try:
            events.close()
        except OSError:
            pass


def start_attempts(args, env, state, rid, port, mode='routed', no_cache=False, timeout=1800):
    state = external_directory(state, 'State directory')
    rid = validate_run_id(rid)
    log_dir = state / (rid + '-startup')
    env = build_child_env(state, rid, env, configure_tempdir=False)
    if not log_dir.exists():
        prepare_state(state, rid)
    if resolve_path(log_dir) != log_dir:
        raise ValueError('Startup directory must not redirect outside its assigned path')
    tempfile.tempdir = env['TMPDIR']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with attempt_events(log_dir / 'attempts.jsonl') as events:
        def event(attempt, backend, result):
            record = json.dumps(dict(attempt=attempt, requested_backend=backend or 'auto', result=result))
            try:
                events.write(record + '\n')
                events.flush()
            except OSError:
                pass
            try:
                print(record, flush=True)
            except OSError:
                pass
        for attempt, backend in enumerate((None, 'FLASH_ATTN', 'TRITON_ATTN'), 1):
            child_env = dict(env)
            child_env.pop('JEV_ATTENTION_BACKEND', None)
            if backend:
                child_env['JEV_ATTENTION_BACKEND'] = backend
            log_path = log_dir / f'attempt-{attempt}.log'
            child_env['JEV_ATTENTION_LOG'] = str(log_path)
            child_args = list(args)
            child_args[child_args.index('--state-dir') + 1] = str(state)
            child_args.extend(['--startup-attempt', str(attempt)])
            event(attempt, backend, 'starting')
            proc = None
            try:
                with log_path.open('xb') as out:
                    proc = subprocess.Popen(child_args, cwd=ROOT, env=child_env, stdout=out,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    logs = log_path.read_text(encoding='utf-8', errors='replace')
                    if 'EngineCore failed to start' in logs:
                        event(attempt, backend, 'failed: EngineCore failed to start')
                        break
                    if proc.poll() is not None:
                        event(attempt, backend, 'failed: exited before readiness')
                        break
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                            health = json.load(response)
                            ready = (response.status == 200 and health['mode'] == mode
                                     and health['model_key'] == 'metask-jev-4b')
                    except (OSError, ValueError, KeyError):
                        ready = False
                    if ready:
                        if (proc.poll() is not None or 'EngineCore failed to start' in
                                log_path.read_text(encoding='utf-8', errors='replace')):
                            event(attempt, backend, 'failed: engine stopped during readiness')
                            break
                        validate_health(health, no_cache)
                        event(attempt, backend, 'ready')
                        print(json.dumps(health, ensure_ascii=False), flush=True)
                        return proc
                    time.sleep(1)
                else:
                    raise TimeoutError('Health readiness timed out; log: ' + str(log_path))
            except BaseException as exc:
                try:
                    if proc is not None:
                        stop(proc)
                finally:
                    event(attempt, backend, 'aborted: ' + type(exc).__name__ + '; log: ' + str(log_path))
                if isinstance(exc, RuntimeError):
                    raise RuntimeError(str(exc) + '; log: ' + str(log_path)) from exc
                raise
            stop(proc)
    raise RuntimeError('All three attention backend attempts failed before readiness; logs: ' + str(log_dir))


@contextmanager
def cleanup_signals():
    """Finish child cleanup, then deliver any terminal signal received meanwhile."""
    saved = {}
    pending = []
    def remember(number, _frame):
        pending.append(number)
    try:
        if threading.current_thread() is threading.main_thread():
            for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
                if hasattr(signal, name):
                    number = getattr(signal, name)
                    saved[number] = signal.signal(number, remember)
        yield
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)
        if pending:
            raise KeyboardInterrupt


def stop(proc):
    with cleanup_signals():
        _stop_process_group(proc)


def _stop_process_group(proc):
    if os.name == 'nt':
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # Reap descendants even after the group leader exits or wait is interrupted.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--bind', choices=('127.0.0.1', '0.0.0.0'), default='127.0.0.1')
    ap.add_argument('--state-dir', required=True)
    ap.add_argument('--run-id')
    a = ap.parse_args(argv)
    def interrupted(*_):
        raise KeyboardInterrupt
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)
    proc = None
    try:
        proc = start(a.model, a.port, state_dir=a.state_dir, run_id=a.run_id, bind=a.bind)
        return proc.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if proc is not None:
            stop(proc)


if __name__ == '__main__':
    raise SystemExit(main())
