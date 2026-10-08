"""CPU regressions for startup, package identity, HTTP failures and output paths."""
import argparse
import ast
import copy
import errno
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

import bench
import core
import launch
import quota
import reconcile_usage
from finalize import freeze
from package_files import package_files
from selftest import prepare_public

OUTPUT = None
PAYLOAD = {'state':'synthetic test','questions':{'q':{'type':'boolean','instructions':'Is this true?'}}}

def write(path, data):
    path=Path(path)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(data)

class OfflineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_tempdir=tempfile.tempdir
        if OUTPUT is None:
            cls.root=Path(tempfile.mkdtemp(prefix='j4-')).resolve()
        else:
            cls.root=core.external_output(OUTPUT,create_parent=False)
            cls.root.mkdir(parents=True,exist_ok=False)
        # The Unix socket length guard also runs on Windows. Keep the state
        # fixtures short, independently of a caller's longer evidence path.
        short_root='/tmp' if os.name=='posix' else tempfile.gettempdir()
        cls.state_root=Path(tempfile.mkdtemp(prefix='js-',dir=short_root)).resolve()
        if len(os.fsencode(cls.state_root/'99'/'e-startup'/'tmp'))>70:
            raise unittest.SkipTest('Startup fixtures require a short temporary directory (target platform: Linux)')
        cls.sequence=0

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir=cls.original_tempdir
        print('Regression evidence retained: '+str(cls.root))
        print('Short state fixtures retained: '+str(cls.state_root))

    def setUp(self):
        type(self).sequence+=1
        self.temp=self.root/str(self.sequence);self.temp.mkdir()
        self.state=self.state_root/str(self.sequence)
        self.addCleanup(setattr,tempfile,'tempdir',self.original_tempdir)

    def test_python_gate(self):
        launch.check_python((3,12));launch.check_python((3,10))
        with self.assertRaisesRegex(ValueError,'Python >= 3.10'):launch.check_python((3,9))

    def test_versions_exact_and_local_suffix(self):
        versions=dict(launch.VERSIONS,torch='2.13.0+cu130')
        with patch('launch.importlib.metadata.version',side_effect=versions.__getitem__):
            self.assertEqual(launch.check_versions(),versions)
        versions['vllm']='0.31.1'
        with patch('launch.importlib.metadata.version',side_effect=versions.__getitem__),self.assertRaisesRegex(ValueError,launch.IMAGE):
            launch.check_versions()
        with patch('launch.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError),self.assertRaisesRegex(ValueError,'missing; run in'):
            launch.check_versions()

    def test_health_contract(self):
        good=dict(status='ok',prefix_caching=True,supports_slow_session=True,warnings=[])
        launch.validate_health(good)
        launch.validate_health(dict(good,prefix_caching=False),no_cache=True)
        for change in ({'status':'warning'},{'warnings':['unexpected']},{'prefix_caching':False},{'supports_slow_session':False}):
            with self.subTest(change=change),self.assertRaises(RuntimeError):launch.validate_health(dict(good,**change))
        for field in ('status','prefix_caching','supports_slow_session'):
            value=dict(good);del value[field]
            with self.assertRaises(RuntimeError):launch.validate_health(value)

    def test_paths_and_run_ids(self):
        model=self.temp/'m';model.mkdir()
        for state,source in ((None,model),('relative',model),(self.state,'relative'),(launch.ROOT,model),(launch.ROOT.parent,model),(model/'state',model),(self.temp,model)):
            with self.subTest(state=state,source=source),self.assertRaises(ValueError):launch.validate_paths(state,source)
        for rid in ('','..','a/b','a'+chr(92)+'b','-bad','x'*65):
            with self.subTest(rid=rid),self.assertRaises(ValueError):launch.validate_run_id(rid)
        self.assertRegex(launch.validate_run_id(),r'^run-[0-9]{8}T[0-9]{12}Z$')

    def test_state_creation_reuse_and_length(self):
        state,rid=launch.prepare_state(self.state,'s')
        self.assertEqual(state,self.state);self.assertEqual(rid,'s')
        for name in ('tmp','home','model'):self.assertTrue((state/'s-startup'/name).is_dir())
        with self.assertRaisesRegex(ValueError,'run_id already exists.*use a new WORK'):launch.prepare_state(self.state,'s')
        with self.assertRaisesRegex(ValueError,'70 bytes'):launch.prepare_state(self.state/('long'*25),'s')
        with patch('launch.Path.mkdir',side_effect=PermissionError),self.assertRaises(ValueError):launch.prepare_state(self.state,'u')

    def test_environment_and_cache_boundary(self):
        launch.prepare_state(self.state,'e')
        legacy='VLLM_ATTENTION_BACKEND'
        env=launch.build_child_env(self.state,'e',{legacy:'bad'})
        self.assertNotIn(legacy,env)
        self.assertEqual(env['VLLM_HOST_IP'],'127.0.0.1')
        for key in ('HOME','TMPDIR','XDG_CACHE_HOME','HF_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','VLLM_RPC_BASE_PATH'):
            self.assertTrue(Path(env[key]).resolve().is_relative_to(self.state))
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','DO_NOT_TRACK','VLLM_NO_USAGE_STATS','PYTHONDONTWRITEBYTECODE','PYTHONUNBUFFERED'):
            self.assertEqual(env[key],'1')
        self.assertEqual(tempfile.tempdir,env['TMPDIR'])
        with self.assertRaises(ValueError):launch.build_child_env(self.state,'e',{'HF_HOME':str(self.temp)})
        for key in ('XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','VLLM_CONFIG_ROOT',
                    'TRITON_HOME','FLASHINFER_WORKSPACE_BASE','NUMBA_CACHE_DIR',
                    'TORCH_EXTENSIONS_DIR','MPLCONFIGDIR'):
            with self.subTest(key=key):
                self.assertTrue(Path(env[key]).is_relative_to(self.state))
                with self.assertRaisesRegex(ValueError,key):
                    launch.build_child_env(self.state,'e',{key:str(self.temp)})

    def test_environment_rejection_precedes_state_creation(self):
        model=self.temp/'m';model.mkdir()
        original_tempdir=tempfile.tempdir
        keys=('XDG_CACHE_HOME','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
              'HF_HOME','CUDA_CACHE_PATH','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME',
              'VLLM_CONFIG_ROOT','TRITON_HOME','FLASHINFER_WORKSPACE_BASE','NUMBA_CACHE_DIR',
              'TORCH_EXTENSIONS_DIR','MPLCONFIGDIR','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE',
              'TRANSFORMERS_CACHE','HF_ASSETS_CACHE','TORCH_HOME')
        for key in keys:
            env={key:str(self.temp)}
            calls=(lambda:launch.start(model,0,state_dir=self.state,run_id='e'),
                   lambda:launch.configure_server_runtime(model,self.state,'e'),
                   lambda:launch.start_attempts(['fake','--state-dir',str(self.state)],env,self.state,'e',8000))
            for entrypoint,call in enumerate(calls):
                with self.subTest(key=key,entrypoint=entrypoint),patch.dict(os.environ,env,clear=True),patch('launch.check_versions'),patch('launch.prepare_state') as prepare,patch('launch.subprocess.Popen') as spawn:
                    with self.assertRaisesRegex(ValueError,key):call()
                    prepare.assert_not_called();spawn.assert_not_called()
                self.assertFalse(self.state.exists())
                self.assertEqual(tempfile.tempdir,original_tempdir)
        with patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(self.temp/'outside.log')},clear=True),patch('launch.prepare_state') as prepare:
            with self.assertRaisesRegex(ValueError,'Attention log'):launch.configure_server_runtime(model,self.state,'e')
            prepare.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_supervised_quota_states_are_independent(self):
        model=self.temp/'m';model.mkdir()
        launch.prepare_state(self.state,'a',model)
        observed=[]
        for attempt in (1,2,3):
            log=self.state/'a-startup'/f'attempt-{attempt}.log'
            with patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(log)},clear=True),patch('launch.prepare_model',return_value=model):
                returned,state,rid=launch.configure_server_runtime(model,self.state,'a',attempt)
            self.assertEqual(returned,model);self.assertEqual(rid,'a')
            self.assertTrue(state.is_relative_to(self.state));observed.append(state)
        self.assertEqual(len(set(observed)),3)

    def fake_model(self,with_aux=False):
        model=self.temp/'m';model.mkdir()
        identity={}
        for name in launch.IDENTITY_SHA256:
            text='fixture-'+name;write(model/name,text);identity[name]=hashlib.sha256(text.encode()).hexdigest()
        write(model/'model.safetensors','tiny')
        if with_aux:
            for name in launch.AUX_SHA256:
                with (model/name).open('xb') as stream:stream.write((launch.ROOT/'model_aux'/name).read_bytes())
        startup=self.temp/'startup';startup.mkdir();(startup/'model').mkdir()
        return model,startup,identity

    def test_model_view_and_immutable_source(self):
        model,startup,identity=self.fake_model()
        write(model/'.hidden','ignored');(model/'subdir').mkdir()
        for name in ('README.md','install.sh','selftest.sh','serve.py','serve.sh','temperature.json','.gitattributes'):
            write(model/name,'ignored')
        (model/'eval').mkdir()
        before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in model.iterdir()}
        directory_mtime=model.stat().st_mtime_ns
        link_plan=None
        try:
            with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4):
                view=launch.prepare_model(model,startup)
        except ValueError as exc:
            if os.name=='nt' and getattr(exc.__cause__,'winerror',None)==1314:
                link_plan={}
                def link(path,target,*args,**kwargs):link_plan[path]=Path(target)
                with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),patch.object(Path,'symlink_to',link):
                    view=launch.prepare_model(model,startup)
                print('STUB: Windows lacks symlink privilege; verified target plan and immutable source. Real symlinks require Linux validation.')
            else:raise
        self.assertEqual(view,startup/'model')
        self.assertEqual(before,{p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in model.iterdir()})
        self.assertEqual(directory_mtime,model.stat().st_mtime_ns)
        for name in identity:
            if link_plan is None:
                self.assertTrue((view/name).is_symlink());self.assertEqual((view/name).resolve(),model/name)
            else:self.assertEqual(link_plan[view/name],model/name)
        for name in launch.AUX_SHA256:
            actual=(view/name).resolve() if link_plan is None else link_plan[view/name]
            self.assertEqual(actual,(launch.ROOT/'model_aux'/name).resolve())
        self.assertFalse((view/'.hidden').exists());self.assertFalse((view/'subdir').exists())
        observed=set(link_plan) if link_plan is not None else set(view.iterdir())
        self.assertEqual({p.name for p in observed},launch.MODEL_FILES)

    def test_model_with_all_aux_still_uses_whitelist(self):
        model,startup,identity=self.fake_model(with_aux=True)
        write(model/'serve.py','ignored')
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),patch.object(Path,'symlink_to') as link:
            view=launch.prepare_model(model,startup)
        self.assertEqual(view,startup/'model')
        self.assertEqual({call.args[0] for call in link.call_args_list},{model/name for name in launch.MODEL_FILES})
        with (model/'vocab.json').open('ab') as stream:stream.write(b'wrong')
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),self.assertRaisesRegex(ValueError,'vocab.json'):
            launch.prepare_model(model,startup)

    def test_model_view_link_plan_without_os_privileges(self):
        model,startup,identity=self.fake_model()
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in model.iterdir()}
        links={}
        def link(path,target,*args,**kwargs):
            self.assertNotIn(path,links);links[path]=Path(target)
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),patch.object(Path,'symlink_to',link):
            view=launch.prepare_model(model,startup)
        expected={view/name:model/name for name in (*identity,'model.safetensors')}
        expected.update({view/name:(launch.ROOT/'model_aux'/name).resolve() for name in launch.AUX_SHA256})
        self.assertEqual(links,expected)
        self.assertTrue(all(target.is_absolute() and target.is_file() for target in links.values()))
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in model.iterdir()})

    def test_aux_and_identity_rejections(self):
        model,startup,identity=self.fake_model()
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4):
            with patch.dict(launch.AUX_SHA256,{'vocab.json':'0'*64}),self.assertRaisesRegex(ValueError,'vocab.json'):
                launch.prepare_model(model,startup)
            write(model/'vocab.json','wrong')
            with self.assertRaisesRegex(ValueError,'Partial auxiliary'):launch.prepare_model(model,startup)
        with patch('launch.WEIGHT_BYTES',4),self.assertRaisesRegex(ValueError,'config.json'):launch.prepare_model(model,startup)

    def test_model_weight_size_and_extra_loading_files(self):
        model,startup,identity=self.fake_model()
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',5),self.assertRaisesRegex(ValueError,'wrong size.*model.safetensors'):
            launch.prepare_model(model,startup)
        original_iterdir=Path.iterdir
        for name in ('extra.safetensors','pytorch_model.bin','model.safetensors.index.json',
                     'added_tokens.json','special_tokens_map.json','processor_config.json',
                     'audio_processor_config.json'):
            def entries(path):
                return iter([*original_iterdir(path),path/name]) if path==model else original_iterdir(path)
            with self.subTest(name=name),patch.object(Path,'iterdir',entries),self.assertRaisesRegex(ValueError,name):
                launch.prepare_model(model,startup)
        write(model/'extra.safetensors','unexpected weights')
        with self.assertRaisesRegex(ValueError,'Unexpected.*extra.safetensors'):
            launch.prepare_model(model,startup)

    def test_dangling_and_loop_links_and_symlink_errors(self):
        model,startup,identity=self.fake_model()
        original_resolve=Path.resolve;original_is_symlink=Path.is_symlink
        broken=model/'tokenizer.json'
        message=('Model file is a dangling or looping symbolic link: tokenizer.json; MODEL must contain '
                 'regular files (a Hugging Face cache snapshots/<sha> directory cannot be mounted alone)')
        for failure in (FileNotFoundError('dangling link'),RuntimeError('symlink loop'),OSError(errno.ELOOP,'symlink loop')):
            def resolve(path,*args,**kwargs):
                if path==broken:raise failure
                return original_resolve(path,*args,**kwargs)
            def is_symlink(path):return path==broken or original_is_symlink(path)
            with self.subTest(failure=type(failure).__name__),patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch.object(Path,'is_symlink',is_symlink),patch.object(Path,'resolve',resolve):
                with self.assertRaises(ValueError) as raised:
                    launch.prepare_model(model,startup)
                self.assertEqual(str(raised.exception),message)
        with patch.object(Path,'is_symlink',return_value=True),patch.object(Path,'resolve',side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(ValueError,'Cannot resolve model path') as raised:
                launch._regular_model_file(broken)
            self.assertNotIn('dangling',str(raised.exception))
        for failure in (OSError('symbolic links unavailable'),RuntimeError('symbolic link failure')):
            with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),patch.object(Path,'symlink_to',side_effect=failure),self.assertRaisesRegex(ValueError,'state filesystem must support symbolic links'):
                launch.prepare_model(model,startup)

    def test_model_directory_link_errors_name_the_model(self):
        model=self.temp/'model-link'
        original_resolve=Path.resolve;original_is_symlink=Path.is_symlink
        for failure in (FileNotFoundError('dangling'),RuntimeError('loop'),OSError(errno.ELOOP,'loop')):
            def resolve(path,*args,**kwargs):
                if path==model:raise failure
                return original_resolve(path,*args,**kwargs)
            def is_symlink(path):return path==model or original_is_symlink(path)
            with self.subTest(failure=type(failure).__name__),patch.object(Path,'is_symlink',is_symlink),patch.object(Path,'resolve',resolve):
                with self.assertRaisesRegex(ValueError,'Model file is a dangling or looping symbolic link: model-link') as raised:
                    launch.external_directory(model,'Model directory')
                self.assertNotIn('state filesystem',str(raised.exception))
        with patch.object(Path,'resolve',side_effect=RuntimeError('loop')):
            with self.assertRaisesRegex(ValueError,'Cannot resolve path') as raised:launch.resolve_path(self.state)
            self.assertNotIn('symbolic links',str(raised.exception))

    def test_unrelated_dangling_and_loop_links_are_ignored(self):
        model,startup,identity=self.fake_model()
        original_iterdir=Path.iterdir;original_resolve=Path.resolve
        unrelated={model/'README.md',model/'serve.py'}
        def entries(path):
            return iter([*original_iterdir(path),*unrelated]) if path==model else original_iterdir(path)
        def resolve(path,*args,**kwargs):
            if path in unrelated:raise RuntimeError('unrelated link must not be resolved')
            return original_resolve(path,*args,**kwargs)
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4),patch.object(Path,'iterdir',entries),patch.object(Path,'resolve',resolve),patch.object(Path,'symlink_to') as links:
            launch.prepare_model(model,startup)
        self.assertEqual(links.call_count,10)

    def test_native_links_and_supervised_view_validation(self):
        model,startup,identity=self.fake_model()
        try:
            (model/'README.md').symlink_to(model/'absent')
        except OSError as exc:
            if os.name=='nt' and getattr(exc,'winerror',None)==1314:
                self.skipTest('Native symbolic links require Linux or Windows symlink privilege; syscall-error tests still run')
            raise
        (model/'serve.py').symlink_to(model/'serve.py')
        with patch.dict(launch.IDENTITY_SHA256,identity,clear=True),patch('launch.WEIGHT_BYTES',4):
            view=launch.prepare_model(model,startup)
            self.assertEqual(launch.verify_model_view(view),view)
            for case,target_name in (('dangling','absent'),('loop','tokenizer.json'),('valid','tokenizer-copy.json')):
                source=self.temp/case;source.mkdir()
                for name in (*identity,'model.safetensors'):
                    if name!='tokenizer.json':
                        with (source/name).open('xb') as stream:stream.write((model/name).read_bytes())
                if case=='valid':
                    with (source/target_name).open('xb') as stream:stream.write((model/'tokenizer.json').read_bytes())
                (source/'tokenizer.json').symlink_to(source/target_name)
                expected=('regular file, not a symbolic link' if case=='valid'
                          else 'dangling or looping symbolic link')
                with self.subTest(case=case),self.assertRaisesRegex(ValueError,expected+'.*tokenizer.json'):
                    launch.prepare_model(source,startup)
            for case,target in (('dangling-directory',self.temp/'absent'),('loop-directory',self.temp/'loop-directory')):
                source=self.temp/case;source.symlink_to(target,target_is_directory=True)
                with self.subTest(case=case),self.assertRaisesRegex(ValueError,'dangling or looping symbolic link: '+case):
                    launch.external_directory(source,'Model directory')
            write(view/'README.md','unexpected')
            with self.assertRaisesRegex(ValueError,'exactly the model file whitelist'):
                launch.verify_model_view(view)

    def test_start_wires_gates_and_fixed_inference_arguments(self):
        model=self.temp/'m';model.mkdir()
        view=self.state/'w-startup'/'model'
        with patch('launch.check_python') as py,patch('launch.check_versions') as versions,patch('launch.prepare_model',return_value=view),patch('launch.start_attempts',return_value='process') as attempts,patch.dict(os.environ,{},clear=True):
            self.assertEqual(launch.start(model,0,state_dir=self.state,run_id='w'),'process')
        py.assert_called_once_with();versions.assert_called_once_with()
        args=attempts.call_args.args[0]
        for flag,value in (('--model',str(view)),('--model-key','metask-jev-4b'),('--quant','bf16'),('--backend','vllm'),('--state-dir',str(self.state)),('--run-id','w')):
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertEqual(inspect.signature(launch.start_attempts).parameters['timeout'].default,1800)

    def test_dependency_and_occupied_port_fail_before_state_or_children(self):
        model=self.temp/'m';model.mkdir()
        with patch('launch.check_python',side_effect=ValueError('unsupported Python')),patch('launch.check_versions') as versions,patch('launch.prepare_state') as state,patch('launch.start_attempts') as child:
            with self.assertRaisesRegex(ValueError,'unsupported Python'):launch.start(model,state_dir=self.state,run_id='d')
            versions.assert_not_called();state.assert_not_called();child.assert_not_called()
        with patch('launch.check_versions',side_effect=ValueError('missing dependency')),patch('launch.prepare_state') as state,patch('launch.start_attempts') as child:
            with self.assertRaisesRegex(ValueError,'missing dependency'):launch.start(model,state_dir=self.state,run_id='d')
            state.assert_not_called();child.assert_not_called()
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0));occupied.listen()
            with patch('launch.check_versions'),patch('launch.prepare_state') as state,patch('launch.start_attempts') as child,patch('launch.stop') as stop:
                with self.assertRaises(OSError):launch.start(model,occupied.getsockname()[1],state_dir=self.state,run_id='p')
                state.assert_not_called();child.assert_not_called();stop.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_main_registers_all_shutdown_signals(self):
        process=Mock();process.wait.return_value=0
        with patch('launch.signal.signal') as register,patch('launch.start',return_value=process),patch('launch.stop') as stop:
            self.assertEqual(launch.main(['--model',str(self.temp),'--state-dir',str(self.state),'--run-id','s']),0)
        self.assertEqual({call.args[0] for call in register.call_args_list},
                         {getattr(launch.signal,name) for name in ('SIGTERM','SIGINT','SIGHUP') if hasattr(launch.signal,name)})
        stop.assert_called_once_with(process)

    def test_interrupted_start_cleans_before_best_effort_events(self):
        original_open=Path.open
        for index,failed_operation in enumerate(('print','write','flush','close')):
            timeline=[];proc=Mock();proc.poll.side_effect=KeyboardInterrupt
            events=Mock()
            def event_write(record):
                if 'aborted' in record:timeline.append('aborted-write')
                if failed_operation=='write':raise OSError('event write failed')
            def event_print(record,**kwargs):
                if 'aborted' in record:
                    timeline.append('aborted-print')
                    if failed_operation=='print':raise OSError('terminal closed')
            def open_file(path,*args,**kwargs):
                return events if path.name=='attempts.jsonl' else original_open(path,*args,**kwargs)
            events.write.side_effect=event_write
            if failed_operation in ('flush','close'):
                getattr(events,failed_operation).side_effect=OSError('event stream failed')
            with self.subTest(failed_operation=failed_operation),patch.object(Path,'open',open_file),patch('launch.subprocess.Popen',return_value=proc),patch('launch.stop',side_effect=lambda _:timeline.append('stop')) as stop,patch('builtins.print',side_effect=event_print):
                with self.assertRaises(KeyboardInterrupt):
                    launch.start_attempts(['fake','--state-dir',str(self.state)],{},self.state,'e'+str(index),8000,timeout=1)
                stop.assert_called_once_with(proc)
            self.assertLess(timeline.index('stop'),timeline.index('aborted-write'))
            self.assertLess(timeline.index('stop'),timeline.index('aborted-print'))
            events.close.assert_called_once_with()

    def test_event_record_is_attempted_when_cleanup_is_interrupted(self):
        proc=Mock();proc.poll.side_effect=KeyboardInterrupt
        timeline=[]
        def stop(_):
            timeline.append('stop')
            raise KeyboardInterrupt
        def report(record,**kwargs):
            if 'aborted' in record:
                timeline.append('aborted')
                raise OSError('terminal closed')
        with patch('launch.subprocess.Popen',return_value=proc),patch('launch.stop',side_effect=stop),patch('builtins.print',side_effect=report):
            with self.assertRaises(KeyboardInterrupt):
                launch.start_attempts(['fake','--state-dir',str(self.state)],{},self.state,'e',8000,timeout=1)
        self.assertEqual(timeline,['stop','aborted'])

    def test_process_cleanup_waits_are_bounded(self):
        for platform in ('nt','posix'):
            proc=Mock(pid=12345);proc.poll.return_value=None
            proc.wait.side_effect=[subprocess.TimeoutExpired('fake',30),subprocess.TimeoutExpired('fake',60)]
            with self.subTest(platform=platform),patch('launch.os.name',platform),patch('launch.os.killpg',create=True),patch('launch.signal.SIGKILL',9,create=True):
                with self.assertRaises(subprocess.TimeoutExpired):launch.stop(proc)
            self.assertEqual([call.kwargs for call in proc.wait.call_args_list],[{'timeout':30},{'timeout':60}])

    def test_direct_counter_is_fatal(self):
        config=copy.deepcopy(core.load_config())
        for bad in (None,'relative',str(core.ROOT),str(core.ROOT.parent)):
            config['routing'].update(state_dir=bad,run_id='r')
            with self.assertRaises(quota.QuotaStateError):quota.RuntimeCounter(config)
        config['routing'].update(state_dir=str(self.temp/'state'),run_id='r')
        with patch('quota.QuotaCounter',side_effect=PermissionError),self.assertRaises(quota.QuotaStateError):quota.RuntimeCounter(config)

    def test_external_output_guards_and_exclusive_write(self):
        for path in ('relative',core.ROOT/'inside',core.ROOT.parent):
            with self.subTest(path=path),self.assertRaises(ValueError):core.external_output(path)
        future=self.temp/'absent'/'result'
        self.assertEqual(reconcile_usage.resolve(future),future)
        self.assertFalse(future.parent.exists())
        with self.assertRaises(FileNotFoundError):bench.read_run_rows(future)
        self.assertFalse(future.parent.exists())
        for prefix in (None,'relative',core.ROOT/'inside'):
            with self.assertRaises(ValueError):bench.output_paths(prefix)
        target=self.temp/'candidate.json'
        bench.write_candidate(target,{'test':True})
        with self.assertRaises(FileExistsError):bench.write_candidate(target,{'test':False})
        self.assertEqual(json.loads(target.read_text()),{'test':True})

    def public_fixture(self):
        public=self.temp/'public';public.mkdir()
        for filename,tier,count in (('easy.jsonl','easy',48),('original.jsonl','standard',72),('hard.jsonl','hard',111)):
            rows=[{'id':tier+str(i),'state':'test context','question':{'type':'boolean','instructions':'test question'},'expected':True} for i in range(count)]
            write(public/filename,''.join(json.dumps(row)+'\n' for row in rows))
        return public

    def test_public_full_output_chain(self):
        public=self.public_fixture()
        out,index=prepare_public(public,self.temp/'out')
        rows=[json.loads(line) for line in index.read_text().splitlines()]
        self.assertEqual(len(rows),231);self.assertTrue(all(set(row)=={'id','type','tier'} for row in rows))
        prefix=out/'fast';result,summary=bench.output_paths(prefix)
        self.assertEqual(core.external_output(result),result)
        self.assertEqual(reconcile_usage.resolve(prefix),prefix)
        write(result,'{}\n');write(summary,'{}\n')
        with self.assertRaises(FileExistsError):bench.output_paths(prefix)
        with self.assertRaises(FileExistsError):prepare_public(public,out)
        for bad in (core.ROOT/'bad',public/'out',public.parent):
            with self.assertRaises(ValueError):prepare_public(public,bad)

    def test_public_count_rejected_before_output(self):
        public=self.temp/'public';public.mkdir()
        for filename in ('easy.jsonl','original.jsonl','hard.jsonl'):write(public/filename,'')
        with self.assertRaises(ValueError):prepare_public(public,self.temp/'out')
        self.assertFalse((self.temp/'out').exists())

    def test_http_failure_mapping_and_proxy_free_health(self):
        from server import make_handler,QuietHTTPServer
        class BadBackend:
            def prompt_ids(self,*args):raise RuntimeError('synthetic backend failure')
        config=core.load_config();cal=core.load_json(core.ROOT/'calibration.json')
        config['routing'].update(state_dir=str(self.temp/'state'),run_id='r')
        service=core.Service(config,cal,BadBackend(),quota.RuntimeCounter(config))
        httpd=QuietHTTPServer(('127.0.0.1',0),make_handler(service))
        worker=threading.Thread(target=httpd.serve_forever,daemon=True);worker.start()
        self.addCleanup(httpd.server_close);self.addCleanup(worker.join,3);self.addCleanup(httpd.shutdown)
        url=f'http://127.0.0.1:{httpd.server_port}'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def status(expected,body):
            request=urllib.request.Request(url+'/v1/systemone',json.dumps(PAYLOAD).encode(),{'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:opener.open(request,timeout=3)
            self.assertEqual(error.exception.code,expected)
            with error.exception as response:self.assertEqual(json.load(response),body)
        status(422,{'error':'item could not be processed'})
        with patch.object(service,'answer',side_effect=RuntimeError('synthetic service failure')):
            status(503,{'error':'service failure'})
        length={key:False for key in ('truncated','character_precompressed','protected_content_exceeds_budget','physical_truncated')}
        length['original_prompt_tokens']=1
        with patch('core.prepare_task',return_value=(None,(None,{},[1],[1]),length)),patch('core.fast_attempt_with_retry',return_value=({'no':0.,'yes':-1.},0,None)),patch.object(service.counter,'decision',side_effect=quota.QuotaStateError('synthetic persistence failure')):
            status(503,{'error':'service failure'})
        with patch.dict(os.environ,{'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','no_proxy':'','NO_PROXY':''}):
            self.assertEqual(bench.get_health(url,3)['model_key'],'metask-jev-4b')
            with patch.object(service,'answer',return_value={'test':'direct POST'}):
                self.assertEqual(bench.post_json(url,PAYLOAD,3),{'test':'direct POST'})

    def test_prompt_and_json_identity(self):
        config=core.load_config()
        self.assertEqual(core.threshold_identity(config)['prompt_sha'],'e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938')
        self.assertEqual(core.validate_threshold(config),[])
        from json_equivalence import verify
        verify()

    def test_pricing_system_mapping(self):
        row=dict(ok=True,usage_complete=True,path='fast',n_new_tokens=0,latency_seconds=.1,model='metask-jev-rain-4B',prompt_tokens=10,completion_tokens=1,type='noul',expected=True,correct=True,tier='easy')
        result=bench.summarize([row],core.load_config())
        self.assertTrue(result['price_model_matches']);self.assertIsNotNone(result['estimated_total_cost_usd'])

    def test_clone_metadata_and_manifest_exclusivity(self):
        root=self.temp/'pkg';root.mkdir()
        for directory in ('.git','__pycache__','state'):
            (root/directory).mkdir();write(root/directory/'ignored','test')
        write(root/'file.txt','test')
        self.assertEqual([p.name for p in package_files(root)],['file.txt'])
        freeze(root)
        with self.assertRaises(FileExistsError):freeze(root)
        for name in ('selftest-run','artifact.pyc','artifact.tmp'):
            folder=self.temp/name;folder.mkdir();write(folder/name,'bad')
            with self.assertRaises(ValueError):freeze(folder)
            self.assertFalse((folder/'MANIFEST.sha256').exists())

    def test_optimized_checker_and_cli_state_requirement(self):
        result=subprocess.run([sys.executable,'-B','-O',str(core.ROOT/'pkg_check.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('run without -O',result.stderr)
        for script in ('server.py','launch.py'):
            result=subprocess.run([sys.executable,'-B',str(core.ROOT/script),'--model',str(self.temp)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('--state-dir',result.stderr)

    def test_shell_interpreters(self):
        import re
        for script in core.ROOT.glob('*.sh'):
            self.assertIsNone(re.search(r'(?m)(?:^|[;\s])(?:exec\s+)?python(?:\s|$)',script.read_text()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir',type=Path)
    options,remaining=parser.parse_known_args()
    OUTPUT=options.output_dir
    unittest.main(argv=[sys.argv[0],*remaining])
