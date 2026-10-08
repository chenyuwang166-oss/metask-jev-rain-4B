"""Offline attention tests; retained temporary evidence is printed on completion."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import urllib.request

import attention
import launch

OUTPUT_DIR = None

FAKE_SERVER = r"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from attention import attention_health
backend=os.environ.get('JEV_ATTENTION_BACKEND')
if os.environ['VLLM_USE_FLASHINFER_SAMPLER']!='0':
    raise RuntimeError('sampler environment was not enforced')
if backend is None:
    print('auto failed',flush=True)
    sys.exit(1)
if backend=='FLASH_ATTN':
    print('EngineCore failed to start',flush=True)
    time.sleep(30)
    sys.exit(1)
print('Using AttentionBackendEnum.TRITON_ATTN backend.',flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        body=json.dumps(dict(status='ok',mode='routed',model_key='metask-jev-4b',
            prefix_caching=True,supports_slow_session=True,warnings=[],
            info=attention_health())).encode()
        self.send_response(200);self.end_headers();self.wfile.write(body)
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


def fake_stop(proc):
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_env():
    env = dict(os.environ)
    for key in ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'HF_HOME', 'CUDA_CACHE_PATH', 'HF_HUB_CACHE',
                'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_ASSETS_CACHE', 'TORCH_HOME',
                'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME', 'VLLM_CONFIG_ROOT',
                'TRITON_HOME', 'FLASHINFER_WORKSPACE_BASE', 'NUMBA_CACHE_DIR',
                'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR'):
        env.pop(key, None)
    return env


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_tempdir = tempfile.tempdir
        # Keep genuine socket path length checks active even for long evidence paths.
        short_root = '/tmp' if os.name == 'posix' else tempfile.gettempdir()
        cls.root = Path(tempfile.mkdtemp(prefix='ja-', dir=short_root)).resolve()
        if len(os.fsencode(cls.root / '99' / 'e-startup' / 'tmp')) > 70:
            raise unittest.SkipTest('Startup fixtures require a short temporary directory (target platform: Linux)')
        if OUTPUT_DIR is not None:
            with (OUTPUT_DIR / 'runtime-evidence.txt').open('x', encoding='utf-8') as out:
                out.write(str(cls.root) + '\n')
        cls.sequence = 0

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir = cls.original_tempdir
        print('Attention test evidence: ' + str(cls.root), flush=True)

    def setUp(self):
        type(self).sequence += 1
        self.temp = self.root / str(self.sequence)
        self.temp.mkdir()
        self.addCleanup(setattr, tempfile, 'tempdir', self.original_tempdir)

    def free_port(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            return probe.getsockname()[1]

    def test_two_failures_then_ready(self):
        port = self.free_port()
        args = [sys.executable, '-B', '-c', FAKE_SERVER, str(port), '--state-dir', str(self.temp), '--run-id', 'f']
        calls = []
        popen = subprocess.Popen
        def spawn(args, **kwargs):
            self.assertIs(kwargs['start_new_session'], True)
            proc = popen(args, **kwargs)
            calls.append((args, kwargs['env'], proc))
            return proc
        env = test_env()
        env.update({'VLLM_ATTENTION_BACKEND': 'IGNORED', 'JEV_ATTENTION_BACKEND': 'IGNORED',
                    'VLLM_USE_FLASHINFER_SAMPLER': '1', 'http_proxy': 'http://127.0.0.1:1',
                    'HTTP_PROXY': 'http://127.0.0.1:1', 'NO_PROXY': '', 'no_proxy': ''})
        try:
            with patch('launch.subprocess.Popen', side_effect=spawn), patch('launch.stop', side_effect=fake_stop), patch.dict(os.environ, {'http_proxy': 'http://127.0.0.1:1', 'HTTP_PROXY': 'http://127.0.0.1:1', 'NO_PROXY': '', 'no_proxy': ''}), contextlib.redirect_stdout(io.StringIO()):
                launch.start_attempts(args, env, self.temp, 'f', port, timeout=15)
            self.assertEqual([e.get('JEV_ATTENTION_BACKEND') for _, e, _ in calls], [None, 'FLASH_ATTN', 'TRITON_ATTN'])
            self.assertTrue(all('VLLM_ATTENTION_BACKEND' not in e for _, e, _ in calls))
            self.assertTrue(all(e['VLLM_USE_FLASHINFER_SAMPLER'] == '0' for _, e, _ in calls))
            self.assertEqual([a[a.index('--startup-attempt')+1] for a, _, _ in calls], ['1', '2', '3'])
            self.assertTrue(all(pr.poll() is not None for _, _, pr in calls[:2]))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{port}/health') as response:
                health = json.load(response)
            self.assertEqual(health['info']['attention_backend'], 'TRITON_ATTN')
            records = [json.loads(line) for line in (self.temp/'f-startup'/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in records], ['starting', 'failed: exited before readiness', 'starting', 'failed: EngineCore failed to start', 'starting', 'ready'])
        finally:
            for _, _, proc in calls:
                fake_stop(proc)

    def test_auto_ready_does_not_retry(self):
        port = self.free_port()
        code = FAKE_SERVER.replace('if backend is None:', 'if False:').replace("if backend=='FLASH_ATTN':", 'if False:').replace('Using AttentionBackendEnum.TRITON_ATTN backend.', 'Using FLASHINFER attention backend')
        args = [sys.executable, '-B', '-c', code, str(port), '--state-dir', str(self.temp)]
        proc = None
        try:
            with patch('launch.stop', side_effect=fake_stop), contextlib.redirect_stdout(io.StringIO()):
                proc = launch.start_attempts(args, test_env(), self.temp, 'a', port, timeout=15)
            records = [json.loads(line) for line in (self.temp/'a-startup'/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in records], ['starting', 'ready'])
            self.assertEqual(records[-1]['requested_backend'], 'auto')
        finally:
            if proc is not None:
                fake_stop(proc)

    def test_log_formats_and_unknown(self):
        cases = [('Using FLASHINFER attention backend', ['FLASHINFER']),
                 ('Using AttentionBackendEnum.FLASH_ATTN backend.', ['FLASH_ATTN']),
                 ('Using AttentionBackendEnum.TRITON_ATTN backend.', ['TRITON_ATTN']),
                 ('Using FLASH attention backend\nUsing TRITON_ATTN attention backend', ['FLASH', 'TRITON_ATTN']),
                 ('Using CUBLAS backend', []), ('', [])]
        for i, (text, names) in enumerate(cases):
            log = self.temp / f'engine-{i}.log'
            with log.open('x', encoding='utf-8') as out:
                out.write(text)
            with patch.dict(os.environ, {'JEV_ATTENTION_LOG': str(log)}, clear=True):
                actual = attention.attention_health()
            self.assertEqual(actual['attention_backends'], names)
            self.assertEqual(actual['attention_backend'], names[-1] if names else None)
            self.assertEqual(actual['attention_backend_requested'], 'auto')
        with patch.dict(os.environ, {'JEV_ATTENTION_BACKEND': 'FLASH_ATTN'}, clear=True):
            actual = attention.attention_health()
            self.assertEqual(actual['attention_backend_requested'], 'FLASH_ATTN')
            self.assertIsNone(actual['attention_backend'])

    def test_exhaustion_and_timeout(self):
        for timed_out in (False, True):
            state = self.temp / str(int(timed_out))
            proc = Mock()
            proc.poll.return_value = 1
            with patch('launch.subprocess.Popen', return_value=proc) as spawn, patch('launch.stop') as stop, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(TimeoutError if timed_out else RuntimeError) as raised:
                    launch.start_attempts(['fake', '--state-dir', str(state)], {}, state, 'f', 8000, timeout=0 if timed_out else 1)
                self.assertIn(str(state/'f-startup'), str(raised.exception))
                self.assertEqual(spawn.call_count, 1 if timed_out else 3)
                self.assertEqual(stop.call_count, spawn.call_count)

    def test_unhealthy_ready_is_rejected_and_reaped(self):
        proc = Mock()
        proc.poll.return_value = None
        health = dict(status='ok', mode='routed', model_key='metask-jev-4b', warnings=['bad'], prefix_caching=True, supports_slow_session=True)
        response = io.StringIO(json.dumps(health))
        response.status = 200
        opener = Mock()
        opener.open.return_value = response
        with patch('launch.subprocess.Popen', return_value=proc), patch('launch.stop') as stop, patch('launch.urllib.request.build_opener', return_value=opener), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'readiness rejected'):
                launch.start_attempts(['fake', '--state-dir', str(self.temp)], {}, self.temp, 'f', 8000, timeout=1)
            stop.assert_called_once_with(proc)

    def test_environment_stays_inside_explicit_state(self):
        launch.prepare_state(self.temp, 'e')
        legacy = 'VLLM_ATTENTION_BACKEND'
        env = launch.build_child_env(self.temp, 'e', {legacy: 'FLASH_ATTN', 'PYTHONPYCACHEPREFIX': str(self.root)})
        self.assertNotIn(legacy, env)
        self.assertNotIn('PYTHONPYCACHEPREFIX', env)
        for key in ('HOME', 'TMPDIR', 'TEMP', 'TMP', 'VLLM_RPC_BASE_PATH', 'HF_HOME', 'XDG_CACHE_HOME'):
            self.assertTrue(Path(env[key]).is_relative_to(self.temp))
        for key in ('HF_HOME', 'XDG_CACHE_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.build_child_env(self.temp, 'e', {key: str(self.root)})

    def test_supervised_state_cannot_expand_explicit_root(self):
        launch.prepare_state(self.temp, 's')
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        with patch.dict(os.environ, {'JEV_STARTUP_DIR': str(self.root), 'JEV_ATTENTION_LOG': str(self.root/'attempt-1.log')}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Attention log'):
                launch.configure_server_runtime(model, self.temp, 's', 1)

    def test_llm_attention_argument_and_runtime_reporting(self):
        import backends
        import core
        config = core.load_config()
        config['model']['path'] = str(self.temp)
        fake_engine = SimpleNamespace(
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(quantization=None),
                vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))),
            get_tokenizer=lambda: SimpleNamespace(get_vocab=lambda: {}))
        fake_llm = Mock(return_value=fake_engine)
        fake_vllm = SimpleNamespace(LLM=fake_llm, SamplingParams=lambda **kwargs: kwargs)
        fake_platforms = SimpleNamespace(current_platform=SimpleNamespace(get_device_name=lambda: 'Test GPU'))
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda='13.0'),
            cuda=SimpleNamespace(is_available=Mock(side_effect=AssertionError('must not initialize CUDA')),
                                 get_device_name=Mock(side_effect=AssertionError('must not initialize CUDA'))))
        for requested in (None, 'FLASH_ATTN', 'TRITON_ATTN'):
            env = {} if requested is None else {'JEV_ATTENTION_BACKEND': requested}
            with patch.dict(sys.modules, {'vllm': fake_vllm, 'vllm.platforms': fake_platforms, 'torch': fake_torch}), patch.dict(os.environ, env, clear=True), patch('backends.generation_audit', return_value={}), patch('backends.importlib.metadata.version', side_effect=lambda name: launch.VERSIONS[name]):
                backend = backends.VLLMBackend(config)
            passed = fake_llm.call_args.kwargs
            if requested is None:
                self.assertNotIn('attention_backend', passed)
            else:
                self.assertEqual(passed['attention_backend'], requested)
            self.assertEqual(backend.runtime['cuda'], '13.0')
            self.assertEqual(backend.runtime['gpu'], 'Test GPU')
            self.assertEqual(backend.runtime['vllm'], '0.31.0')

    def test_process_group_reaped_after_leader_exit(self):
        proc = Mock(pid=12345)
        proc.poll.return_value = 0
        with patch('launch.os.name', 'posix'), patch('launch.os.killpg', create=True) as killpg, patch('launch.signal.SIGKILL', 9, create=True):
            launch.stop(proc)
        self.assertEqual([call.args for call in killpg.call_args_list],
                         [(12345, launch.signal.SIGTERM), (12345, 9)])

    def test_cleanup_defers_repeated_signals_and_restores_handlers(self):
        proc = Mock(pid=12345)
        original = {getattr(launch.signal, name): launch.signal.getsignal(getattr(launch.signal, name))
                    for name in ('SIGTERM', 'SIGINT', 'SIGHUP') if hasattr(launch.signal, name)}
        def wait(**kwargs):
            for number in original:
                handler = launch.signal.getsignal(number)
                self.assertTrue(callable(handler))
                handler(number, None)
        proc.wait.side_effect = wait
        with patch('launch.os.name', 'posix'), patch('launch.os.killpg', create=True) as killpg, patch('launch.signal.SIGKILL', 9, create=True):
            with self.assertRaises(KeyboardInterrupt):
                launch.stop(proc)
        self.assertEqual(killpg.call_args_list[-1].args, (12345, 9))
        for number, handler in original.items():
            self.assertEqual(launch.signal.getsignal(number), handler)

    def test_interrupted_wait_still_reaps_descendants(self):
        proc = Mock(pid=12345)
        proc.wait.side_effect = [KeyboardInterrupt, 0]
        with patch('launch.os.name', 'posix'), patch('launch.os.killpg', create=True) as killpg, patch('launch.signal.SIGKILL', 9, create=True):
            with self.assertRaises(KeyboardInterrupt):
                launch.stop(proc)
        self.assertEqual(killpg.call_args_list[-1].args, (12345, 9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    args, remaining = parser.parse_known_args()
    if args.output_dir:
        OUTPUT_DIR = launch.external_directory(args.output_dir, 'Test output directory')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[sys.argv[0], *remaining])
