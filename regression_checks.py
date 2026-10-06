"""Offline regressions for packaging and startup guards; never imports a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch
import quota
from core import load_config
from finalize import freeze
from selftest import prepare_public


class OfflineChecks(unittest.TestCase):
    def test_health_contract(self):
        healthy={'prefix_caching':True,'supports_slow_session':True,'warnings':[]}
        launch.validate_health(healthy)
        launch.validate_health(dict(healthy,prefix_caching=False),no_cache=True)
        for changes in ({'prefix_caching':False},{'supports_slow_session':False},
                        {'warnings':['Quota persistence unavailable: memory counts, fast only']},
                        {'warnings':['calibration placeholder: refit required']}):
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                launch.validate_health(dict(healthy,**changes))
        for field in ('prefix_caching','supports_slow_session'):
            bad=dict(healthy);del bad[field]
            with self.assertRaises(RuntimeError):launch.validate_health(bad)

    def test_state_success_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            state,rid=launch.prepare_state(temp,'fresh')
            self.assertEqual(rid,'fresh');self.assertEqual(state,Path(temp).resolve())
            (state/rid).mkdir()
            with self.assertRaises(ValueError):launch.prepare_state(temp,rid)

    def test_unwritable_state(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('launch.tempfile.TemporaryFile',side_effect=PermissionError),self.assertRaises(ValueError):
                launch.prepare_state(temp,'fresh')

    def test_state_paths(self):
        with self.assertRaises(ValueError):launch.prepare_state(launch.ROOT,'fresh')
        with tempfile.TemporaryDirectory() as temp:
            for rid in ('..','a/b','a'+chr(92)+'b'):
                with self.subTest(rid=rid),self.assertRaises(ValueError):launch.prepare_state(temp,rid)

    def test_direct_counter_startup_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            config=copy.deepcopy(load_config())
            config['routing'].update(state_dir=temp,run_id='fresh')
            with patch('quota.QuotaCounter',side_effect=PermissionError),self.assertRaises(quota.QuotaStateError):
                quota.RuntimeCounter(config)

    def test_freeze_rejects_runtime_output(self):
        for name in ('selftest-run','__pycache__','artifact.pyc','artifact.tmp'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/name).touch()
                with self.assertRaises(ValueError):freeze(root)
                self.assertFalse((root/'MANIFEST.sha256').exists())

    def test_public_output_and_count_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);public=root/'public';public.mkdir()
            for filename in ('easy.jsonl','original.jsonl','hard.jsonl'):
                (public/filename).write_text('',encoding='utf-8',newline='\n')
            with self.assertRaises(ValueError):prepare_public(public,launch.ROOT/'invalid-output')
            with self.assertRaises(ValueError):prepare_public(public,public/'output')
            out=root/'output'
            with self.assertRaises(ValueError):prepare_public(public,out)
            self.assertFalse(out.exists())

    def test_public_index_contains_only_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);public=root/'public';public.mkdir()
            for filename,tier,count in (('easy.jsonl','easy',48),('original.jsonl','standard',72),('hard.jsonl','hard',111)):
                rows=[{'id':tier+str(i),'state':'test context','question':{'type':'boolean','instructions':'test question'},'expected':True} for i in range(count)]
                (public/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
            out,index=prepare_public(public,root/'output')
            rows=[json.loads(line) for line in index.read_text().splitlines()]
            self.assertEqual(len(rows),231)
            self.assertTrue(all(set(row)=={'id','type','tier'} for row in rows))
            self.assertEqual(rows[0],{'id':'easy0','type':'noul','tier':'easy'})
            self.assertEqual(rows[-1]['id'],'hard110')
            with self.assertRaises(FileExistsError):prepare_public(public,out)


if __name__=='__main__':unittest.main()
