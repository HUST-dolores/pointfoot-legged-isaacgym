"""CLI/config contracts without importing Gym or launching remote jobs."""
import argparse
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

def main_body(script):
    tree = ast.parse((ROOT / 'legged_gym/scripts' / script).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main').body

def execute(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, '<actual ship CLI>', 'exec'), namespace)

def parse(monkeypatch, tmp_path, script, value=None):
    directory = tmp_path / 'must_not_be_created'
    args = ['--output', str(directory), '--label', 'test'] if script.startswith('train') else ['--manifest-directory', str(directory)]
    if value is not None:
        args += ['--episode-length-s', str(value)]
    monkeypatch.setattr(sys, 'argv', [script, *args])
    nodes = []
    for node in main_body(script):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'mkdir':
            break
        nodes.append(node)
    namespace = dict(argparse=argparse, Path=Path, math=math, __doc__='test')
    execute(nodes, namespace)
    assert not directory.exists()
    return namespace.get('args', namespace.get('a'))

def horizon_if(script, name):
    for node in ast.walk(ast.Module(body=main_body(script), type_ignores=[])):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Attribute) and left.attr == 'episode_length_s' and isinstance(left.value, ast.Name) and left.value.id == name:
                return node
    raise AssertionError('missing actual horizon override')

@pytest.mark.parametrize('value', [None, 60.0])
def test_train_default_and_override(monkeypatch, tmp_path, value):
    args = parse(monkeypatch, tmp_path, 'train_ship_motion.py', value)
    assert args.episode_length_s == value
    cfg = SimpleNamespace(env=SimpleNamespace(episode_length_s=20.0))
    execute([horizon_if('train_ship_motion.py', 'args')], dict(args=args, cfg=cfg))
    assert cfg.env.episode_length_s == (20.0 if value is None else value)

@pytest.mark.parametrize('value', [None, 60.0])
def test_launcher_default_and_forwarding(monkeypatch, tmp_path, value):
    args = parse(monkeypatch, tmp_path, 'launch_ship_training.py', value)
    command = []
    execute([horizon_if('launch_ship_training.py', 'a')], dict(a=args, command=command))
    assert command == ([] if value is None else ['--episode-length-s', str(value)])

@pytest.mark.parametrize('script', ['train_ship_motion.py', 'launch_ship_training.py'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_horizons_rejected_before_writes(monkeypatch, tmp_path, script, value):
    with pytest.raises(SystemExit) as error:
        parse(monkeypatch, tmp_path, script, value)
    assert error.value.code == 2
    assert not (tmp_path / 'must_not_be_created').exists()
