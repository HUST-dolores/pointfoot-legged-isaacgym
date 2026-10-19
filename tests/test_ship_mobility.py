"""Pure movement geometry/observation/migration tests; native Gym checked separately."""
import importlib.util
import argparse
import ast
import math
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

motion=module('ship_geometry_mobility','legged_gym/utils/ship_motion.py')
warm=module('ship_migration_mobility','legged_gym/utils/ship_warm_start.py')


def test_shuttle_is_smooth_bounded_and_returns_over_complete_period():
    time=torch.linspace(0,24,24001,dtype=torch.float64)
    speed=motion.shuttle_velocity(time,.2,8.,2.)
    assert speed[0]==0 and speed.abs().max()<=.2
    assert abs(float(speed[1]))<1e-10
    # Startup transient is excluded, not silently assumed to integrate to zero.
    assert abs(float(torch.trapz(speed[8000:16001],time[8000:16001])))<1e-9


@pytest.mark.parametrize('speed,period,ramp',[(0,8,2),(.2,0,2),(.2,8,-1),(float('nan'),8,2)])
def test_invalid_shuttle_rejected(speed,period,ramp):
    with pytest.raises(ValueError):motion.shuttle_velocity(torch.zeros(2),speed,period,ramp)


def test_rolling_has_no_clock_features_and_requests_both_contacts():
    features,contacts=motion.mobility_clock(torch.linspace(0,2,100),False,1.5,.6)
    assert not features.any() and bool((contacts==1).all())


def test_step_clock_alternates_support_and_never_requests_flight():
    time=torch.tensor([0.,.2,.7,.9])
    features,contacts=motion.mobility_clock(time,True,1.,.6)
    torch.testing.assert_close(features[:,0],torch.ones(4))
    assert contacts.tolist()==[[1,1],[1,0],[0,1],[0,1]]
    assert (contacts.sum(-1)>=1).all()


def test_obs_has_fixed_imu_slice_and_last_eight_actions():
    flat=torch.arange(28.).repeat(3,1)
    imu=torch.full((3,9),71.)
    gait=torch.full((3,3),82.)
    x=motion.mobility_observations(flat,imu,gait)
    assert x.shape==(3,40)
    torch.testing.assert_close(x[:,:20],flat[:,:20])
    torch.testing.assert_close(x[:,20:29],imu)
    torch.testing.assert_close(x[:,29:32],gait)
    torch.testing.assert_close(x[:,-8:],flat[:,20:])


def foot_setup():
    deck=torch.zeros(1,13);deck[:,6]=1
    foot=torch.zeros(1,2,13);foot[:,:,6]=1;foot[:,:,2]=.2
    return deck,foot


def test_static_wheel_clearance_uses_deck_normal():
    deck,foot=foot_setup()
    clearance,slip=motion.foot_deck_state(deck,foot,.2,.1)
    torch.testing.assert_close(clearance,torch.zeros(1,2))
    assert not slip.any()


def test_no_slip_rolling_is_not_penalized_as_center_motion():
    deck,foot=foot_setup()
    foot[:,:,7]=.2  # forward wheel-center motion
    foot[:,:,11]=2. # omega_y * radius == center speed
    _,slip=motion.foot_deck_state(deck,foot,.2,.1)
    torch.testing.assert_close(slip,torch.zeros_like(slip),atol=1e-7,rtol=0)
    foot[:,:,11]=0
    _,slip=motion.foot_deck_state(deck,foot,.2,.1)
    torch.testing.assert_close(slip[:,:,0],torch.full((1,2),.2))


def test_rigidly_comoving_contact_has_zero_slip_under_deck_rotation():
    deck,foot=foot_setup()
    deck[:,7:10]=torch.tensor([.2,-.1,.3]);deck[:,10:13]=torch.tensor([.3,.1,.2])
    foot[:,:,:3]=torch.tensor([[[.4,.3,.2],[-.2,-.4,.2]]])
    foot[:,:,7:10]=motion.point_velocity(deck[:,None,:].expand(1,2,13),foot[:,:,:3])
    foot[:,:,10:13]=deck[:,None,10:13]
    _,slip=motion.foot_deck_state(deck,foot,.2,.1)
    torch.testing.assert_close(slip,torch.zeros_like(slip),atol=1e-6,rtol=0)


def migration_states():
    model={'actor.0.weight':torch.randn(16,47),'critic.0.weight':torch.randn(16,54),'logstd':torch.randn(8)}
    enc={'encoder.gru.weight_ih_l0':torch.randn(24,66),'encoder.gru.bias_ih_l0':torch.randn(24)}
    current={**model,'actor.0.weight':torch.zeros(16,50),'critic.0.weight':torch.zeros(16,57)}
    target={**enc,'encoder.gru.weight_ih_l0':torch.zeros(24,72)}
    return {'model_state_dict':model,'encoder_state_dict':enc},current,target


def test_migration_preserves_actor_critic_and_both_gru_branches():
    source,current,target=migration_states()
    model,enc=warm.migrate_ship_mobility_checkpoint(source,current,target)
    for key in ['actor.0.weight','critic.0.weight']:
        old=source['model_state_dict'][key];x=torch.randn(5,old.shape[1]);expanded=torch.randn(5,old.shape[1]+3)
        expanded[:,:36]=x[:,:36];expanded[:,39:]=x[:,36:]
        torch.testing.assert_close(x@old.T,expanded@model[key].T)
        assert not model[key][:,36:39].any()
    key='encoder.gru.weight_ih_l0';x=torch.randn(5,66);expanded=torch.randn(5,72)
    expanded[:,:29]=x[:,:29];expanded[:,32:69]=x[:,29:]
    torch.testing.assert_close(x@source['encoder_state_dict'][key].T,expanded@enc[key].T)
    assert not enc[key][:,29:32].any() and not enc[key][:,69:72].any()


def test_migration_rejects_legacy_flat_or_wrong_width():
    source,current,target=migration_states()
    source['model_state_dict']['actor.0.weight']=torch.randn(16,38)
    with pytest.raises(ValueError):warm.migrate_ship_mobility_checkpoint(source,current,target)


def test_clearance_and_no_slip_on_tilted_moving_deck():
    deck,feet=foot_setup()
    deck[:,3]=math.sin(.2);deck[:,6]=math.cos(.2)
    deck[:,7:10]=torch.tensor([.2,-.1,.3]);deck[:,10:13]=torch.tensor([.3,.1,.2])
    top,normal=motion.surface_pose(deck,.2)
    feet[:,:,:3]=top[:,None,:]+normal[:,None,:]*torch.tensor([[[.1],[.16]]])
    feet[:,:,7:10]=motion.point_velocity(deck[:,None,:].expand(1,2,13),feet[:,:,:3])
    feet[:,:,10:13]=deck[:,None,10:13]
    clearance,slip=motion.foot_deck_state(deck,feet,.2,.1)
    torch.testing.assert_close(clearance,torch.tensor([[0.,.06]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(slip,torch.zeros_like(slip),atol=1e-6,rtol=0)


def parse_cli(monkeypatch,tmp_path,script,extra):
    directory=tmp_path/'not_created'
    argv=['--output',str(directory),'--label','test'] if script.startswith('train') else ['--manifest-directory',str(directory)]
    monkeypatch.setattr(sys,'argv',[script,*argv,*extra])
    tree=ast.parse((ROOT/'legged_gym/scripts'/script).read_text())
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main').body
    nodes=[]
    for node in body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='mkdir':break
        nodes.append(node)
    ns=dict(argparse=argparse,Path=Path,math=math,__doc__='test')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual mobility CLI>','exec'),ns)
    assert not directory.exists()
    return ns.get('args',ns.get('a'))


@pytest.mark.parametrize('extra',[
    ['--ship-warm-start','source.pt'],
    ['--locomotion-mode','roll','--warm-start','flat.pt'],
    ['--locomotion-mode','step','--warm-start','flat.pt','--ship-warm-start','ship.pt'],
])
def test_invalid_training_initialization_rejected_before_writes(monkeypatch,tmp_path,extra):
    with pytest.raises(SystemExit):parse_cli(monkeypatch,tmp_path,'train_ship_motion.py',extra)


@pytest.mark.parametrize('extra',[
    ['--locomotion-mode','roll'],
    ['--locomotion-mode','step','--initial-ship-policy','relative.pt'],
    ['--initial-ship-policy','/remote/ship.pt'],
])
def test_invalid_pair_initialization_rejected_before_writes(monkeypatch,tmp_path,extra):
    with pytest.raises(SystemExit):parse_cli(monkeypatch,tmp_path,'launch_ship_training.py',extra)


def test_launcher_requires_same_explicit_ship_source_for_mobility(monkeypatch,tmp_path):
    args=parse_cli(monkeypatch,tmp_path,'launch_ship_training.py',
        ['--locomotion-mode','step','--initial-ship-policy','/remote/common_ship.pt'])
    assert args.initial_ship_policy=='/remote/common_ship.pt'
    assert args.reference_mode=='gravity_projection'
