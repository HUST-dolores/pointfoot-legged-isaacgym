"""Exercise actual optional CLI/config wiring without starting Gym or SSH."""
import argparse
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def body(script):
    tree = ast.parse((ROOT/'legged_gym/scripts'/script).read_text())
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body

def execute(nodes,namespace):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual ship source>','exec'),namespace)

def parse(monkeypatch,tmp_path,script,value):
    directory=tmp_path/'not_created'
    argv=['--output',str(directory),'--label','test'] if script.startswith('train') else ['--manifest-directory',str(directory)]
    if value is not None:argv+=['--yaw-tracking-scale',str(value)]
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
    for node in ast.walk(ast.Module(body=body(script),type_ignores=[])):
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Attribute) and node.test.left.attr=='yaw_tracking_scale':
            return node
    raise AssertionError('missing actual yaw tracking override')

@pytest.mark.parametrize('value',[None,0.,2.])
def test_train_optional_override(monkeypatch,tmp_path,value):
    args=parse(monkeypatch,tmp_path,'train_ship_motion.py',value)
    cfg=SimpleNamespace(rewards=SimpleNamespace(scales=SimpleNamespace(orientation=-4.)))
    execute([override('train_ship_motion.py')],dict(args=args,cfg=cfg))
    assert vars(cfg.rewards.scales)==({'orientation':-4.} if value is None else {'orientation':-4.,'tracking_ang_vel':value})

@pytest.mark.parametrize('value',[None,0.,2.])
def test_launcher_forwarding(monkeypatch,tmp_path,value):
    args=parse(monkeypatch,tmp_path,'launch_ship_training.py',value)
    command=[]
    execute([override('launch_ship_training.py')],dict(a=args,command=command))
    assert command==([] if value is None else ['--yaw-tracking-scale',str(value)])

@pytest.mark.parametrize('script',['train_ship_motion.py','launch_ship_training.py'])
@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_bad_scale_rejected_before_writes(monkeypatch,tmp_path,script,value):
    with pytest.raises(SystemExit) as error:parse(monkeypatch,tmp_path,script,value)
    assert error.value.code==2
    assert not (tmp_path/'not_created').exists()

def test_existing_reward_tracks_measured_body_yaw_rate():
    tree=ast.parse((ROOT/'legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py').read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_reward_tracking_ang_vel')
    ns={'torch':torch};execute([function],ns)
    env=SimpleNamespace(commands=torch.zeros(3,3),base_ang_vel=torch.tensor([[0.,0.,0.],[0.,0.,.5],[0.,0.,-.5]]),cfg=SimpleNamespace(rewards=SimpleNamespace(ang_tracking_sigma=.25)))
    assert torch.allclose(ns['_reward_tracking_ang_vel'](env),torch.tensor([1.,math.exp(-1),math.exp(-1)]))
