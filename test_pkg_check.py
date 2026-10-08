"""Regression tests for release policy and package evidence checks."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

import pkg_check

# Explicit test inputs are excluded from text policy scans, never executable AST checks.
SCAN_TEST_FIXTURES = {
    'private': ['10.1.0.5', '203.0.113.5', 'x@example.com', '~/project', 'ssh host', 'Run ssh example-host', 'command = "ssh example-host"'],
    'process': ['once their logs are available', 'before publication', 'must supply', 'not yet', 'once their logs are\navailable', 'must\nsupply', 'before\npublication', 'TBD', 'TBA', 'to be added', 'to be filled', 'to be determined', 'to be confirmed', 'will be added', 'will be recorded', 'will be filled'],
    'document_process': ['结果待定。', '结果待核。', 'Accuracy was 0.857 (198/231).]]'],
    'marker': '[[FACTS-PENDING: first line\nsecond line]]',
    'deletions': ['import os; os.rmdir(p)', 'from os import removedirs as drop; drop(p)', 'Path(p).rmdir()',
                  'subprocess.run(["rm", "-rf", p])', 'subprocess.run(["find", p, "-delete"])',
                  'subprocess.run(args=["rm", "-rf", p])', 'subprocess.Popen(args=["find", p, "-delete"])', 'getattr(shutil, "rm"+"tree")(p)', 'os.replace(a,b)', 'os.remove(p)',
                  'from shutil import rmtree as drop; drop(p)', 'Path(p).replace(q)', 'Path(p).rename(q)',
                  'p.replace(q)', 'p.rename(q)', 'os.execvp("rm", ["rm", p])',
                  'os.spawnlp(0, "unlink", "unlink", p)', 'from os import execvp as run; run("rmdir", ["rmdir", p])'],
    'shell_commands': ['rm', 'rmdir', 'unlink', 'truncate', 'shred'],
    'safe_replacement': 'text.replace("a", "b")',
    'quota_cleanup': 'Path(temporary).unlink(missing_ok=True)',
    'quota_replace': 'os.replace(temporary, self.state_path)',
    'other_cleanup': 'Path(other).unlink(missing_ok=True)',
    'other_replace': 'os.replace(other, self.state_path)',
    'legacy_good': 'env.pop("VLLM_ATTENTION_BACKEND", None)',
    'legacy_bad': 'env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"',
    'renderer': 'def render(task):\n    return "before publication"\n',
    'bindings': ['YES_FORMS += ["yeah"]', 'render: object = None', 'from other import render',
                 'if True:\n    def render(task): return task', 'YES_FORMS.append("yeah")',
                 'if True:\n    import other as render', 'render = None',
                 'def render(task): return None', 'Service.warmup = None'],
    'binding_source': 'YES_FORMS = ["yes"]\ndef render(task): return task\nclass Service:\n    def warmup(self): pass\n',
    'ast_legacy_dump': "FunctionDef(name='f', args=arguments(posonlyargs=[], args=[arg(arg='x')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Return(value=Name(id='x', ctx=Load()))], decorator_list=[])",
}


class PackagePolicyTests(unittest.TestCase):
    def test_versions_do_not_hide_private_addresses(self):
        for value in SCAN_TEST_FIXTURES['private']:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                pkg_check.check_private_text(value, ('1.0', '0.5', '0.113'))
        pkg_check.check_private_text('127.0.0.1 0.0.0.0 2.13.0+cu130', ('2.13.0',))

    def test_common_deletion_forms_are_rejected(self):
        for value in SCAN_TEST_FIXTURES['deletions']:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                pkg_check.check_deletions(Path('module.py'),ast.parse(value))
        pkg_check.check_deletions(Path('module.py'),ast.parse(SCAN_TEST_FIXTURES['safe_replacement']))

    def test_shell_deletion_forms_are_rejected(self):
        for command in SCAN_TEST_FIXTURES['shell_commands']:
            for value in (command+' "$X"', '/usr/bin/'+command+' "$X"', 'true && '+command+' "$X"'):
                with self.subTest(value=value), self.assertRaises(AssertionError):
                    pkg_check.check_shell_deletions(Path('script.sh'),value)
            for invocation in ('subprocess.run', 'subprocess.Popen', 'os.execvp'):
                args='['+repr(command)+', "target"]'
                if invocation=='os.execvp':args=repr(command)+', '+args
                source=invocation+'('+args+')'
                with self.subTest(source=source), self.assertRaises(AssertionError):
                    pkg_check.check_deletions(Path('module.py'),ast.parse(source))

    def test_quota_allowlist_is_exact(self):
        for name in ('quota_cleanup','quota_replace'):
            tree=ast.parse(SCAN_TEST_FIXTURES[name])
            pkg_check.check_deletions(Path('quota.py'),tree)
            with self.assertRaises(AssertionError):
                pkg_check.check_deletions(Path('other.py'),tree)
        for name in ('other_cleanup','other_replace'):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                pkg_check.check_deletions(Path('quota.py'),ast.parse(SCAN_TEST_FIXTURES[name]))

    def test_process_phrases_in_documents_and_sources(self):
        for value in SCAN_TEST_FIXTURES['process']:
            self.assertTrue(pkg_check.release_wording_errors(Path('DISCLOSURE.md'),value))
            self.assertTrue(pkg_check.release_wording_errors(Path('module.py'),'note = '+chr(34)*3+value+chr(34)*3))
        for value in SCAN_TEST_FIXTURES['document_process']:
            with self.subTest(value=value):
                self.assertTrue(pkg_check.release_wording_errors(Path('DISCLOSURE.md'),value))

    def test_revision_and_model_links_require_full_sha(self):
        path=Path('DISCLOSURE.md')
        for size in (7,8,39):
            self.assertTrue(pkg_check.release_reference_errors(path,('ab12'*10)[:size]))
        for size in (40,64):
            self.assertEqual(pkg_check.release_reference_errors(path,('ab12'*16)[:size]),[])
        for prefix in ('https://huggingface.co/example/model/tree/',
                       'https://github.com/example/project/tree/',
                       'https://github.com/example/project/blob/'):
            for revision in ('main', 'v1.0.0', 'ab12345'):
                with self.subTest(prefix=prefix,revision=revision):
                    self.assertTrue(pkg_check.release_reference_errors(path,prefix+revision))
            self.assertEqual(pkg_check.release_reference_errors(path,prefix+'ab12'*10),[])
        self.assertEqual(pkg_check.release_reference_errors(path,'https://github.com/example/project/blob/'+'ab12'*10+'/module.py'),[])

    def test_multiline_marker_is_release_blocker(self):
        errors=pkg_check.release_wording_errors(Path('DISCLOSURE.md'),SCAN_TEST_FIXTURES['marker'])
        self.assertEqual(len(errors),1)
        self.assertIn(pkg_check.SCAN_MARKER,errors[0])

    def test_renderer_exclusion_is_narrow(self):
        source=SCAN_TEST_FIXTURES['renderer']
        self.assertEqual(pkg_check.release_wording_errors(Path('core.py'),source),[])
        self.assertTrue(pkg_check.release_wording_errors(Path('other.py'),source))

    def test_legacy_variable_only_removed_by_runtime(self):
        source=SCAN_TEST_FIXTURES['legacy_good']
        pkg_check.check_legacy_environment(Path('launch.py'),source,ast.parse(source))
        source=SCAN_TEST_FIXTURES['legacy_bad']
        with self.assertRaises(AssertionError):
            pkg_check.check_legacy_environment(Path('launch.py'),source,ast.parse(source))

    def test_ast_empty_type_parameters_are_version_neutral(self):
        from json_equivalence import canonical_ast
        current=ast.parse('def f(x): return x').body[0]
        previous=copy.deepcopy(current)
        previous._fields=tuple(field for field in previous._fields if field!='type_params')
        current._fields=(*previous._fields,'type_params')
        current.type_params=[]
        self.assertEqual(canonical_ast(previous),canonical_ast(current))
        current.type_params=[ast.Name(id='T',ctx=ast.Load())]
        self.assertNotEqual(canonical_ast(previous),canonical_ast(current))

    def test_ast_empty_fields_match_legacy_dump(self):
        from json_equivalence import canonical_ast
        node=ast.parse('def f(x): return x').body[0]
        self.assertEqual(canonical_ast(node),SCAN_TEST_FIXTURES['ast_legacy_dump'])

    def test_selected_inference_names_cannot_be_rebound(self):
        from json_equivalence import canonical_ast,check_inference_ast,inference_nodes
        source=SCAN_TEST_FIXTURES['binding_source']
        tree=ast.parse(source)
        names=('YES_FORMS','render','Service.warmup')
        nodes=inference_nodes(tree)
        expected={name:hashlib.sha256(canonical_ast(nodes[name]).encode()).hexdigest() for name in names}
        self.assertEqual(check_inference_ast(tree,expected,'module.py'),len(names))
        for mutation in SCAN_TEST_FIXTURES['bindings']:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check_inference_ast(ast.parse(source+mutation+'\n'),expected,'module.py')

    def test_helper_and_class_attribute_mutations_are_rejected(self):
        from json_equivalence import check_inference_ast,inference_nodes
        record=json.loads((pkg_check.ROOT/'inference_identity.json').read_text(encoding='utf-8'))
        mutations=(('core.py','apply_budget_tier'), ('backends.py','_max_new_tokens'),
                   ('quota.py','configure_cost'), ('backends.py','VLLMBackend.readout_output_tokens'))
        for name,key in mutations:
            tree=ast.parse((pkg_check.ROOT/name).read_text(encoding='utf-8'))
            expected=record['files'][name]
            check_inference_ast(tree,expected,name)
            node=inference_nodes(tree)[key]
            if isinstance(node,ast.FunctionDef):node.body=[ast.Return(value=ast.Constant(value=None))]
            else:node.value=ast.Constant(value=2)
            with self.subTest(name=name,key=key), self.assertRaisesRegex(ValueError,'Inference AST differs'):
                check_inference_ast(tree,expected,name)

    def test_license_and_identity_evidence_required(self):
        self.assertIn('model_aux/LICENSE',pkg_check.REQUIRED)
        self.assertIn('inference_identity.json',pkg_check.REQUIRED)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args,remaining=parser.parse_known_args()
    if args.output_dir:
        from core import external_output
        output=external_output(args.output_dir,create_parent=False)
        output.mkdir(parents=True,exist_ok=False)
        print('Package-policy evidence retained: '+str(output))
    unittest.main(argv=[__file__,*remaining])
