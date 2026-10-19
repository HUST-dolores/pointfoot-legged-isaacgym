"""Exercise the actual reset sampler without importing Isaac Gym."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import torch


def reset_method():
    source=Path(__file__).resolve().parents[1]/'legged_gym/envs/wheelfoot_ship/wheelfoot_ship.py'
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BipedWFShip')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_reset_ship')
    namespace={'torch':torch,'math':math}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['_reset_ship']


def env(fraction):
    cfg=SimpleNamespace(curriculum=False,training_schedule=True,corner_fraction=fraction,
        amplitude_min=[.1,.08,.03],amplitude_max=[.4,.2,.15],
        frequency_min=[.15,.12,.18],frequency_max=[.35,.3,.4])
    value=SimpleNamespace(cfg=SimpleNamespace(ship=cfg),init_done=False,num_envs=8,device='cpu',
        ship_time=torch.zeros(8),ship_difficulty=torch.ones(8),ship_amplitude=torch.zeros(8,3),
        ship_frequency=torch.zeros(8,3),ship_phase=torch.zeros(8,3),ship_previous_twist=torch.zeros(8,6),
        deck_imu=torch.zeros(8,9),ship_imu_history=torch.zeros(8,2,9),deck_indices=torch.arange(8),root_states=torch.zeros(8,13))
    value.training_motion_scale=lambda:torch.tensor([.5,.3,.2])
    value._deck_command=lambda:(torch.zeros(8,13),None,None)
    return value


def test_default_zero_fraction_preserves_original_sampler_and_rng_sequence():
    e=env(0.)
    torch.manual_seed(41)
    cfg=e.cfg.ship;scale=e.training_motion_scale()
    low=torch.tensor(cfg.amplitude_min);high=torch.tensor(cfg.amplitude_max)
    amp=(low+torch.rand(8,3)*(high-low))*scale
    low=torch.tensor(cfg.frequency_min);high=torch.tensor(cfg.frequency_max)
    frequency=low+torch.rand(8,3)*(high-low)
    phase=2*math.pi*torch.rand(8,3)
    torch.manual_seed(41);reset_method()(e,torch.arange(8))
    torch.testing.assert_close(e.ship_amplitude,amp,atol=0,rtol=0)
    torch.testing.assert_close(e.ship_frequency,frequency,atol=0,rtol=0)
    torch.testing.assert_close(e.ship_phase,phase,atol=0,rtol=0)


def test_corner_anchors_use_all_maxima_but_still_follow_curriculum_scale():
    e=env(.25);reset_method()(e,torch.arange(8))
    expected=torch.tensor(e.cfg.ship.amplitude_max)*e.training_motion_scale()
    torch.testing.assert_close(e.ship_amplitude[:2],expected.expand(2,-1))
    torch.testing.assert_close(e.ship_frequency[:2],torch.tensor(e.cfg.ship.frequency_max).expand(2,-1))
    assert torch.all(e.ship_amplitude[2:]<=expected)


def test_subset_reset_keeps_anchor_identity_and_other_environments_unchanged():
    e=env(.25);reset_method()(e,torch.tensor([7,0,2]))
    torch.testing.assert_close(e.ship_amplitude[0],torch.tensor(e.cfg.ship.amplitude_max)*e.training_motion_scale())
    assert torch.count_nonzero(e.ship_amplitude[[1,3,4,5,6]])==0
