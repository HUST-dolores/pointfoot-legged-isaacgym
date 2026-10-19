import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('ship_warm_start',Path(__file__).resolve().parents[1]/'legged_gym/utils/ship_warm_start.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def states():
    old={'actor.0.weight':torch.randn(16,38),'critic.0.weight':torch.randn(16,45),'logstd':torch.randn(8)}
    enc={'encoder.gru.weight_ih_l0':torch.randn(24,48),'encoder.gru.bias_ih_l0':torch.randn(24)}
    current={**old,'actor.0.weight':torch.zeros(16,47),'critic.0.weight':torch.zeros(16,54)}
    target={**enc,'encoder.gru.weight_ih_l0':torch.zeros(24,66)}
    return {'model_state_dict':old,'encoder_state_dict':enc},current,target


def test_old_policy_and_gru_responses_are_preserved_with_arbitrary_new_imu():
    source,current,target=states()
    model,enc=m.migrate_flat_checkpoint(source,current,target)
    for key in ['actor.0.weight','critic.0.weight']:
        old=source['model_state_dict'][key]
        x=torch.randn(5,old.shape[1])
        expanded=torch.randn(5,old.shape[1]+9)
        expanded[:,:27]=x[:,:27]
        expanded[:,36:]=x[:,27:]
        torch.testing.assert_close(x@old.T,expanded@model[key].T)
        assert torch.count_nonzero(model[key][:,27:36]) == 0
    x=torch.randn(5,48)
    expanded=torch.randn(5,66)
    expanded[:,:20]=x[:,:20]
    expanded[:,29:57]=x[:,20:]
    key='encoder.gru.weight_ih_l0'
    torch.testing.assert_close(x@source['encoder_state_dict'][key].T,expanded@enc[key].T)
    assert torch.count_nonzero(enc[key][:,20:29]) == 0
    assert torch.count_nonzero(enc[key][:,57:]) == 0


def test_incompatible_gait_policy_is_rejected():
    source,current,target=states()
    source['model_state_dict']['actor.0.weight']=torch.zeros(16,41)
    with pytest.raises(ValueError):
        m.migrate_flat_checkpoint(source,current,target)
