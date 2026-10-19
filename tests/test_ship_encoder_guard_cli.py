"""Optional encoder guard CLI must remain opt-in and reject invalid limits."""
import argparse
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]

def body(script):
    tree=ast.parse((ROOT/'legged_gym/scripts'/script).read_text())
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body

def execute(nodes,namespace):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual encoder CLI>','exec'),namespace)

def parse(monkeypatch,tmp_path,script,value):
    directory=tmp_path/'not_created'
    argv=['--output',str(directory),'--label','test'] if script.startswith('train') else ['--manifest-directory',str(directory)]
    if value is not None:argv+=['--encoder-kl-limit',str(value)]
    monkeypatch.setattr(sys,'argv',[script,*argv])
    nodes=[]
    for node in body(script):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='mkdir':break
        nodes.append(node)
    ns=dict(argparse=argparse,math=math,Path=Path,__doc__='test')
    execute(nodes,ns)
    assert not directory.exists()
    return ns.get('args',ns.get('a'))

def override(script):
    return next(n for n in ast.walk(ast.Module(body=body(script),type_ignores=[]))
                if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and isinstance(n.test.left,ast.Attribute) and n.test.left.attr=='encoder_kl_limit')

@pytest.mark.parametrize('value',[None,.005])
def test_train_default_and_override(monkeypatch,tmp_path,value):
    args=parse(monkeypatch,tmp_path,'train_ship_motion.py',value)
    train=SimpleNamespace(algorithm=SimpleNamespace(learning_rate=3e-5))
    execute([override('train_ship_motion.py')],dict(args=args,train=train))
    assert vars(train.algorithm)==({'learning_rate':3e-5} if value is None else {'learning_rate':3e-5,'encoder_kl_limit':value})

@pytest.mark.parametrize('value',[None,.005])
def test_launcher_forwards_only_explicit_limit(monkeypatch,tmp_path,value):
    args=parse(monkeypatch,tmp_path,'launch_ship_training.py',value)
    command=[];execute([override('launch_ship_training.py')],dict(a=args,command=command))
    assert command==([] if value is None else ['--encoder-kl-limit',str(value)])

@pytest.mark.parametrize('script',['train_ship_motion.py','launch_ship_training.py'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_bad_limit_rejected_before_directory_creation(monkeypatch,tmp_path,script,value):
    with pytest.raises(SystemExit) as error:parse(monkeypatch,tmp_path,script,value)
    assert error.value.code==2
    assert not (tmp_path/'not_created').exists()
