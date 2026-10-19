"""Exact fixed-variance KL used by the isolated encoder-update diagnostic."""
import ast
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]

def shift_kl(before,after,sigma):
    tree=ast.parse((ROOT/'legged_gym/scripts/ship_encoder_drift_probe.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gaussian_mean_shift_kl')
    ns={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual encoder drift KL>','exec'),ns)
    return ns['gaussian_mean_shift_kl'](before,after,sigma)

def test_unchanged_means_have_exact_zero_kl():
    mean=torch.randn(16,8)
    assert torch.count_nonzero(shift_kl(mean,mean,torch.full_like(mean,.2)))==0

def test_known_shift_sums_actions_not_batch():
    before=torch.zeros(2,8)
    result=shift_kl(before,torch.full_like(before,.1),torch.full_like(before,.2))
    assert result.shape==(2,)
    torch.testing.assert_close(result,torch.ones(2))

def test_mixed_action_scales_and_symmetric_sign():
    before=torch.zeros(2,2)
    after=torch.tensor([[.2,-.4],[-.2,.4]])
    sigma=torch.tensor([[.2,.4],[.2,.4]])
    torch.testing.assert_close(shift_kl(before,after,sigma),torch.ones(2))

def test_half_noise_gives_four_times_the_kl():
    before=torch.zeros(3,8);after=torch.ones_like(before)*.1
    base=shift_kl(before,after,torch.full_like(before,.4))
    assert shift_kl(before,after,torch.full_like(before,.2)).mean().item()==pytest.approx(4*base.mean().item())
