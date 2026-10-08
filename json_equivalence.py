"""Verify the six expected JSON edits against the original release values."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY = 'metask-jev-4b'
# Canonical JSON digests from original commit 7feb587df7fddcaa7b27b042661eb1715b99e4aa,
# after applying the expected config edits and excluding only three status texts.
EXPECTED = {
    'config.json': '587073b11f4e74ebf38656c47ad34bb67dba6421de48d6f33cf56e2eda8777d3',
    'calibration.json': '31d9b1045376a6ec045c858f7a2920498d656877ba39204a0ddfe809ed5c112f',
}

def canonical_ast(node):
    # Python 3.12 added empty type_params fields. Ignore only this empty syntax
    # metadata so the same source has the same identity on Python 3.10 and newer.
    node=copy.deepcopy(node)
    for child in ast.walk(node):
        if 'type_params' in child._fields and getattr(child,'type_params',None)==[]:
            child._fields=tuple(field for field in child._fields if field!='type_params')
    # Python 3.13 added show_empty=False; retain the earlier explicit empty-list
    # representation so the recorded identities remain interpreter-independent.
    options={'show_empty':True} if 'show_empty' in inspect.signature(ast.dump).parameters else {}
    return ast.dump(node,annotate_fields=True,include_attributes=False,**options)


def inference_nodes(tree):
    """Collect direct definitions without silently accepting duplicate bindings."""
    nodes={}
    def collect(body, prefix=''):
        for node in body:
            keys=[]
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                keys=[prefix+node.name]
            elif isinstance(node,ast.Assign):
                keys=[prefix+target.id for target in node.targets if isinstance(target,ast.Name)]
            for key in keys:
                if key in nodes:raise ValueError('Repeated inference name: '+key)
                nodes[key]=node
            if isinstance(node,ast.ClassDef):collect(node.body,prefix+node.name+'.')
    collect(tree.body)
    return nodes


def check_inference_bindings(tree, expected):
    """Reject direct rebinding or mutation of selected names outside their bodies."""
    def check(body, prefix=''):
        names={key[len(prefix):].split('.')[0] for key in expected if key.startswith(prefix)}
        for node in body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                continue
            if isinstance(node,ast.ClassDef):
                if node.name in names:check(node.body,prefix+node.name+'.')
                continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and prefix+t.id in expected for t in node.targets):
                continue  # Its complete AST is checked against the recorded digest.
            for child in ast.walk(node):
                if isinstance(child,ast.Name) and child.id in names:
                    raise ValueError('Inference name used outside its recorded definition: '+prefix+child.id)
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and child.name in names:
                    raise ValueError('Conditional inference definition: '+prefix+child.name)
                if isinstance(child,(ast.Import,ast.ImportFrom)):
                    for alias in child.names:
                        bound=alias.asname or alias.name.split('.')[0]
                        if bound=='*' or bound in names:
                            raise ValueError('Inference name replaced by import: '+prefix+bound)
    check(tree.body)


def check_inference_ast(tree, expected, name):
    nodes=inference_nodes(tree)
    check_inference_bindings(tree,expected)
    for key,digest in expected.items():
        actual=hashlib.sha256(canonical_ast(nodes[key]).encode()).hexdigest() if key in nodes else None
        if actual!=digest:raise ValueError('Inference AST differs from baseline: '+name+':'+key)
    return len(expected)


def verify_inference_identity(root):
    path=Path(root)/'inference_identity.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='480144fbdf0f1863eccdf28d6c884a7855a23e361023a60868e6df7a586aa1f4':
        raise ValueError('Inference baseline identity changed')
    record=json.loads(path.read_text(encoding='utf-8'))
    count=0
    for name,expected in record['files'].items():
        tree=ast.parse((Path(root)/name).read_text(encoding='utf-8'))
        count+=check_inference_ast(tree,expected,name)
    print('PASS: '+str(count)+' selected inference AST nodes match baseline '+record['baseline_commit'])

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def comparable(name, value, original=False):
    value = copy.deepcopy(value)
    if name == 'config.json' and original:
        del value['model']['vllm_expected_version']
        del value['generation_reference']
        value['pricing']['submission_models']['metask-jev-rain-4B'] = 'Qwen3.5-4B'
    if name == 'calibration.json':
        if not isinstance(value['status'],str) or any(not isinstance(value['models'][KEY][block]['status'],str) for block in ('fast','slow')):
            raise ValueError('Calibration status fields must remain text')
        del value['status']
        for block in ('fast', 'slow'):
            del value['models'][KEY][block]['status']
    return value

def verify(root=ROOT, baseline=None):
    verify_inference_identity(root)
    for name, expected in EXPECTED.items():
        value = json.loads((Path(root)/name).read_text(encoding='utf-8'))
        actual = canonical(comparable(name, value))
        if hashlib.sha256(actual).hexdigest() != expected:
            raise ValueError('Unexpected JSON change: '+name)
        if baseline is not None:
            old = json.loads((Path(baseline)/name).read_text(encoding='utf-8'))
            if actual != canonical(comparable(name, old, original=True)):
                raise ValueError('JSON values differ from baseline: '+name)
        print('PASS: '+name+'; every other key, value and JSON type unchanged; '+expected)
    from core import load_config, threshold_identity, validate_threshold
    config = load_config(Path(root)/'config.json')
    prompt = threshold_identity(config)['prompt_sha']
    if prompt != 'e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938':
        raise ValueError('Renderer identity changed')
    if validate_threshold(config) != []:
        raise ValueError('Threshold validation produced warnings')
    print('PASS: prompt_sha='+prompt+'; validate_threshold(load_config()) == []')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path)
    args = parser.parse_args()
    verify(baseline=args.baseline_dir)
