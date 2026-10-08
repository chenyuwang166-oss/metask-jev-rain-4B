"""CPU-only checks for item failures, fatal engine failures and HTTP latching."""
from contextlib import contextmanager
import argparse
import http.client
import json
from pathlib import Path
import tempfile
import threading
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import core
import quota
from backends import BackendError, new_meter
from server import QuietHTTPServer, make_handler


PAYLOAD = {'state': 'synthetic', 'questions': {'q': {'type': 'boolean', 'instructions': 'Is this true?'}}}
OUTPUT = None
EngineDeadError = type('EngineDeadError', (RuntimeError,), {})
VLLMServerError = type('VLLMServerError', (RuntimeError,), {'__module__': 'vllm.exceptions'})
EngineGenerateError = type('EngineGenerateError', (VLLMServerError,), {'__module__': 'vllm.exceptions'})


def wrapped(exc):
    error = BackendError('synthetic wrapped backend failure')
    error.__cause__ = exc
    return error


class FailedBackend:
    def __init__(self, error):
        self.error, self.calls = error, 0

    def prompt_ids(self, *args):
        self.calls += 1
        raise self.error

    def slow_session(self, *args):
        raise AssertionError('unexpected unpatched slow readout')


class ServiceFailureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if OUTPUT is None:
            cls.root = Path(tempfile.mkdtemp(prefix='j4-fail-')).resolve()
        else:
            cls.root = core.external_output(OUTPUT, create_parent=False)
            cls.root.mkdir(parents=True, exist_ok=False)
        cls.sequence = 0

    @classmethod
    def tearDownClass(cls):
        print('Service failure evidence retained: ' + str(cls.root))

    def service(self, error=None, *, slow=False):
        type(self).sequence += 1
        config = core.load_config()
        config['routing'].update(state_dir=str(self.root), run_id='case-' + str(self.sequence), mode='fast_only')
        if slow:
            config['routing'].update(mode='uniform', quota=1, fuse=1)
        backend = FailedBackend(error or RuntimeError('synthetic item failure'))
        return core.Service(config, core.load_json(core.ROOT / 'calibration.json'), backend, quota.RuntimeCounter(config))

    def start_http(self, service):
        httpd = QuietHTTPServer(('127.0.0.1', 0), make_handler(service))
        worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        worker.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(worker.join, 3)
        self.addCleanup(httpd.shutdown)
        return httpd.server_port

    def request(self, port, method='POST', path='/v1/systemone', payload=PAYLOAD):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        try:
            body = json.dumps(payload) if method == 'POST' else None
            connection.request(method, path, body, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    @contextmanager
    def fast_success(self):
        length = {key: False for key in ('truncated', 'character_precompressed', 'protected_content_exceeds_budget', 'physical_truncated')}
        length['original_prompt_tokens'] = 1
        with patch('core.prepare_task', return_value=(None, (None, {}, [1], [1]), length)), \
             patch('core.fast_attempt_with_retry', return_value=({'no': 0., 'yes': -1.}, 0, None)):
            yield

    def assert_failure(self, actual, code):
        self.assertEqual(actual, (code, {'error': 'service failure' if code == 503 else 'item could not be processed'}))
        self.assertNotIn('answers', actual[1])
        self.assertNotIn('usage', actual[1])

    def test_exception_graph_and_vllm_subclasses(self):
        fatal = wrapped(EngineDeadError('engine stopped'))
        branch = RuntimeError('wrapper with both links')
        branch.__cause__ = RuntimeError('recoverable cause')
        branch.__context__ = fatal
        fatal.__context__ = branch
        server_subclass = type('WorkerFailure', (VLLMServerError,), {})
        generate_subclass = type('ItemGenerationFailure', (EngineGenerateError,), {})
        for error in (fatal, branch, server_subclass('fatal'), core.ServiceFailure('fatal')):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(core.unrecoverable_engine_error(error))
        nonfatal = RuntimeError('cycle')
        nonfatal.__cause__ = nonfatal
        for error in (nonfatal, EngineGenerateError('single item'), generate_subclass('single item'),
                      type('VLLMServerError', (RuntimeError,), {})('unrelated class')):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(core.unrecoverable_engine_error(error))

    def test_engine_death_returns_503_and_health_error(self):
        for error in (wrapped(EngineDeadError('engine stopped')), wrapped(VLLMServerError('fatal'))):
            with self.subTest(error=type(error.__cause__).__name__):
                service = self.service(error)
                port = self.start_http(service)
                self.assert_failure(self.request(port), 503)
                status, health = self.request(port, 'GET', '/health')
                self.assertEqual(status, 200)
                self.assertEqual(health['status'], 'error')
                self.assertTrue(service.dead)
                calls = service.backend.calls
                self.assert_failure(self.request(port), 503)
                self.assert_failure(self.request(port, path='/unknown', payload={}), 503)
                self.assert_failure(self.request(port, payload={}), 503)
                self.assertEqual(service.backend.calls, calls)

    def test_third_consecutive_item_failure_latches_503(self):
        service = self.service()
        port = self.start_http(service)
        for expected in (422, 422, 503, 503):
            self.assert_failure(self.request(port), expected)
        self.assertEqual(service.backend.calls, 3)
        self.assertEqual(service.item_failure_streak, core.ITEM_FAILURE_LIMIT)
        self.assertEqual(service.health()['status'], 'error')

    def test_vllm_item_generation_error_stays_422(self):
        service = self.service(wrapped(EngineGenerateError('single item')))
        port = self.start_http(service)
        self.assert_failure(self.request(port), 422)
        self.assertFalse(service.dead)
        self.assertNotEqual(service.health()['status'], 'error')

    def test_success_resets_item_failure_streak(self):
        service = self.service()
        port = self.start_http(service)
        for _ in range(2):
            self.assert_failure(self.request(port), 422)
        with self.fast_success():
            self.assertEqual(self.request(port)[0], 200)
        self.assertEqual(service.item_failure_streak, 0)
        for expected in (422, 422, 503):
            self.assert_failure(self.request(port), expected)

    def test_success_within_multi_item_request_resets_streak(self):
        service = self.service()
        for _ in range(2):
            with self.assertRaises(core.ItemFailure):
                service.answer(PAYLOAD)
        with self.fast_success():
            answer, detail = service._one_impl(core.normalize_request(PAYLOAD, 8)[0][1])
        payload = {'state': 'synthetic', 'questions': dict(PAYLOAD['questions'], q2=PAYLOAD['questions']['q'])}
        with patch.object(service, '_one_impl', side_effect=[(answer, detail), core.ItemFailure('item')]), \
             self.assertRaises(core.ItemFailure):
            service.answer(payload)
        self.assertEqual(service.item_failure_streak, 1)
        self.assertFalse(service.dead)

    def test_fast_readout_failure_is_not_downgraded(self):
        service = self.service()
        port = self.start_http(service)
        with self.fast_success(), patch('core.fast_attempt_with_retry', side_effect=wrapped(EngineDeadError('dead'))):
            self.assert_failure(self.request(port), 503)
        self.assertTrue(service.dead)

    def test_fatal_oom_message_is_not_retried(self):
        fatal = EngineDeadError('out of memory')
        with patch('core.fast_readout', side_effect=fatal) as readout, patch('core.release_failed_session') as release:
            with self.assertRaises(EngineDeadError):
                core.fast_attempt_with_retry(object(), {}, {}, [], [], {}, new_meter())
        self.assertEqual(readout.call_count, 1)
        release.assert_not_called()

    def test_slow_engine_death_is_not_returned_as_fast_success(self):
        service = self.service(slow=True)
        port = self.start_http(service)
        with self.fast_success(), patch('core.slow_readout', side_effect=wrapped(EngineDeadError('dead'))):
            self.assert_failure(self.request(port), 503)
        self.assertTrue(service.dead)
        self.assertEqual(service.health()['status'], 'error')

    def test_recoverable_slow_failure_keeps_existing_fallback(self):
        service = self.service(slow=True)
        port = self.start_http(service)
        with self.fast_success(), patch('core.slow_readout', side_effect=RuntimeError('recoverable')):
            status, response = self.request(port)
        self.assertEqual(status, 200)
        self.assertEqual(response['answers']['q']['warning'], 'slow_failed: fast answer kept')
        self.assertIn('q', response['partial_errors'])
        self.assertFalse(service.dead)

    def test_slow_timeout_keeps_estimated_fast_fallback(self):
        service = self.service(slow=True)
        port = self.start_http(service)
        with self.fast_success(), patch('core.slow_readout', side_effect=TimeoutError('item timeout')):
            status, response = self.request(port)
        self.assertEqual(status, 200)
        self.assertEqual(response['answers']['q']['warning'], 'slow_timeout: fast answer kept')
        self.assertTrue(response['usage']['usage_estimated'])
        self.assertFalse(service.dead)

    def test_fatal_draining_worker_latches_after_response_timeout(self):
        service = self.service(slow=True)
        service.config['generation']['wall_timeout_seconds'] = 0.02
        entered, release, marked = threading.Event(), threading.Event(), threading.Event()
        mark_dead = service._mark_dead

        def delayed_failure(*args):
            entered.set()
            if not release.wait(3):
                raise AssertionError('test worker was not released')
            raise wrapped(EngineDeadError('late fatal engine failure'))

        def record_death():
            mark_dead()
            marked.set()

        try:
            with self.fast_success(), patch('core.slow_readout', side_effect=delayed_failure), \
                 patch.object(service, '_mark_dead', side_effect=record_death):
                response = service.answer(PAYLOAD)
                self.assertTrue(entered.is_set())
                self.assertTrue(response['usage']['usage_estimated'])
                release.set()
                self.assertTrue(marked.wait(3))
            self.assertEqual(service.health()['status'], 'error')
            with self.assertRaises(core.ServiceFailure):
                service.answer(PAYLOAD)
        finally:
            release.set()

    def test_engine_dead_flag_blocks_before_inference(self):
        service = self.service()
        service.backend.engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(resources=SimpleNamespace(engine_dead=True))))
        port = self.start_http(service)
        self.assertEqual(self.request(port, 'GET', '/health')[1]['status'], 'error')
        self.assert_failure(self.request(port, path='/unknown', payload={}), 503)
        self.assertEqual(service.backend.calls, 0)

    def test_dead_post_expect_continue_replies_without_body(self):
        service = self.service()
        service.dead = True
        port = self.start_http(service)
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        try:
            connection.putrequest('POST', '/unknown')
            connection.putheader('Expect', '100-continue')
            connection.putheader('Content-Length', '999999999')
            connection.endheaders()
            response = connection.getresponse()
            self.assert_failure((response.status, json.loads(response.read())), 503)
        finally:
            connection.close()

    def test_http_workers_are_daemon_threads(self):
        self.assertTrue(QuietHTTPServer.daemon_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args, remaining = parser.parse_known_args()
    OUTPUT = args.output_dir
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
