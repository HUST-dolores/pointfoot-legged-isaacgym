"""Actual heading-rate reward and strict ship finetune contracts."""
import argparse
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
motion=module('heading_math','legged_gym/utils/ship_motion.py')
warm=module('heading_warm_start','legged_gym/utils/ship_warm_start.py')

def execute(nodes,namespace):
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual source>','exec'),namespace)

def test_upright_heading_rate_equals_world_z():
    q=torch.tensor([[0.,0.,0.,1.]]).repeat(3,1)
    omega=torch.tensor([[.3,.4,.5],[0.,0.,-.4],[.2,0.,0.]])
    torch.testing.assert_close(motion.heading_yaw_rate(q,omega),omega[:,2])

def test_pure_body_roll_is_not_false_heading_turn_when_upright_heading_axis():
    q=torch.tensor([[math.sin(.3),0.,0.,math.cos(.3)]])
    omega=torch.tensor([[.4,0.,0.]])
    torch.testing.assert_close(motion.heading_yaw_rate(q,omega),torch.zeros(1))

def test_heading_rate_matches_finite_difference_at_tilt_not_body_z():
    q=torch.tensor([[.2,.3,-.1,.9]],dtype=torch.float64);q=q/q.norm(dim=-1,keepdim=True)
    omega=torch.tensor([[.4,-.6,.7]],dtype=torch.float64)
    axis=torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    f=motion.rotate(q,axis);df=torch.cross(omega,f,dim=-1);dt=1e-7
    numerical=(torch.atan2((f+dt*df)[:,1],(f+dt*df)[:,0])-torch.atan2(f[:,1],f[:,0]))/dt
    torch.testing.assert_close(motion.heading_yaw_rate(q,omega),numerical,atol=1e-6,rtol=1e-6)
    body_z=motion.inverse_rotate(q,omega)[:,2]
    assert (motion.heading_yaw_rate(q,omega)-body_z).abs().item()>.05

def test_vertical_forward_axis_has_finite_reward_rate():
    q=torch.tensor([[0.,math.sin(math.pi/4),0.,math.cos(math.pi/4)]])
    assert torch.isfinite(motion.heading_yaw_rate(q,torch.tensor([[1.,2.,3.]]))).all()

def test_actual_reward_tracks_positive_and_negative_heading_commands():
    tree=ast.parse((ROOT/'legged_gym/envs/wheelfoot_ship/wheelfoot_ship.py').read_text())
    f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_reward_ship_yaw_rate')
    ns={'torch':torch,'math':math,'heading_yaw_rate':motion.heading_yaw_rate};execute([f],ns)
    root=torch.zeros(3,13);root[:,6]=1;root[:,12]=torch.tensor([0.,.5,-.5])
    env=SimpleNamespace(root_states=root,actor_indices=torch.arange(3),commands=torch.zeros(3,3),cfg=SimpleNamespace(rewards=SimpleNamespace(heading_yaw_tracking_sigma=.25)))
    torch.testing.assert_close(ns['_reward_ship_yaw_rate'](env),torch.tensor([1.,math.exp(-1),math.exp(-1)]))
    env.commands[:,2]=root[:,12]
    torch.testing.assert_close(ns['_reward_ship_yaw_rate'](env),torch.ones(3))

def states():
    model={'actor.0.weight':torch.randn(16,47),'critic.0.weight':torch.randn(16,54),'logstd':torch.randn(8)}
    encoder={'encoder.gru.weight_ih_l0':torch.randn(24,66),'encoder.gru.bias_ih_l0':torch.randn(24)}
    return {'model_state_dict':model,'encoder_state_dict':encoder,'optimizer_state_dict':{'unused':True}},model,encoder

def test_finetune_is_exact_weights_only_without_aliasing():
    source,model,encoder=states();m,e=warm.ship_finetune_state_dicts(source,model,encoder)
    for a,b in [(m,model),(e,encoder)]:
        for k in a:
            torch.testing.assert_close(a[k],b[k]);assert a[k].data_ptr()!=b[k].data_ptr()

@pytest.mark.parametrize('problem',['width','keys','nan','gru'])
def test_finetune_rejects_incompatible_or_corrupt_weights(problem):
    source,model,encoder=states()
    source={**source,'model_state_dict':dict(model),'encoder_state_dict':dict(encoder)}
    if problem=='width':source['model_state_dict']['actor.0.weight']=torch.zeros(16,50)
    if problem=='keys':source['model_state_dict']['extra']=torch.ones(1)
    if problem=='nan':source['model_state_dict']['logstd']=torch.full((8,),float('nan'))
    if problem=='gru':source['encoder_state_dict']['encoder.gru.weight_ih_l0']=torch.zeros(24,72)
    with pytest.raises(ValueError):warm.ship_finetune_state_dicts(source,model,encoder)

def parse(monkeypatch,tmp_path,script,extra):
    folder=tmp_path/'must_not_exist';args=['--output',str(folder),'--label','test'] if script.startswith('train') else ['--manifest-directory',str(folder)]
    monkeypatch.setattr(sys,'argv',[script,*args,*extra])
    tree=ast.parse((ROOT/'legged_gym/scripts'/script).read_text());body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body
    nodes=[]
    for n in body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='mkdir':break
        nodes.append(n)
    ns=dict(argparse=argparse,math=math,Path=Path,__doc__='test');execute(nodes,ns);assert not folder.exists();return ns.get('args',ns.get('a'))

@pytest.mark.parametrize('script',['train_ship_motion.py','launch_ship_training.py'])
@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_heading_scale_rejected_before_writes(monkeypatch,tmp_path,script,value):
    with pytest.raises(SystemExit):parse(monkeypatch,tmp_path,script,['--heading-yaw-tracking-scale',str(value)])

@pytest.mark.parametrize('script',['train_ship_motion.py','launch_ship_training.py'])
def test_legacy_and_heading_reward_cannot_be_accidentally_combined(monkeypatch,tmp_path,script):
    with pytest.raises(SystemExit):parse(monkeypatch,tmp_path,script,['--yaw-tracking-scale','2','--heading-yaw-tracking-scale','2'])

def test_finetune_rejects_mobility_mode_before_writes(monkeypatch,tmp_path):
    with pytest.raises(SystemExit):parse(monkeypatch,tmp_path,'train_ship_motion.py',['--ship-finetune','ship.pt','--locomotion-mode','roll'])

def test_launcher_finetune_and_heading_flags(monkeypatch,tmp_path):
    args=parse(monkeypatch,tmp_path,'launch_ship_training.py',['--finetune-ship-policy','/remote/model.pt','--heading-yaw-tracking-scale','2'])
    assert args.finetune_ship_policy=='/remote/model.pt' and args.heading_yaw_tracking_scale==2
