"""Actual PPO optional guard: rollback weights AND Adam; default unchanged."""
import copy
import pytest
import torch
from torch import nn
from legged_gym.algorithm.ppo import PPO
from legged_gym.algorithm.actor_critic import ActorCritic

class Encoder(nn.Module):
    num_output_dim=7
    is_mlp_encoder=True
    def __init__(self):
        super().__init__();self.linear=nn.Linear(1,7,bias=False)
        nn.init.zeros_(self.linear.weight)
    def encode(self,history):
        self.encoder_out=self.linear(history)
        return self.encoder_out.detach()
    def get_encoder_out(self):return self.encoder_out

class Storage:
    def __init__(self,alg):
        obs=torch.zeros(4,3);history=torch.ones(4,1);cmd=torch.zeros(4,0)
        critic=torch.zeros(4,7);critic[:,0:3]=100.
        actor_input=torch.cat((alg.encoder.encode(history),obs,cmd),-1)
        alg.actor_critic.act(actor_input);actions=torch.zeros(4,1)
        self.batch=(obs,critic,history,None,cmd,actions,
                    alg.actor_critic.evaluate(critic).detach(),torch.ones(4,1),torch.ones(4,1),
                    alg.actor_critic.get_actions_log_prob(actions).detach().unsqueeze(-1),
                    alg.actor_critic.action_mean.detach().clone(),alg.actor_critic.action_std.detach().clone())
        self.observations=obs.unsqueeze(0);self.observation_history=history.unsqueeze(0);self.commands=cmd.unsqueeze(0)
    def mini_batch_generator(self,*args):yield self.batch
    def encoder_mini_batch_generator(self,*args):yield self.batch[0],self.batch[0],self.batch[1],self.batch[2]
    def clear(self):pass

def make(limit=.005,lr=.1):
    actor=ActorCritic(10,7,1,actor_hidden_dims=[8],critic_hidden_dims=[8])
    with torch.no_grad():
        for p in actor.actor.parameters():p.zero_()
        actor.actor[0].weight[0,0]=10.
        actor.actor[-1].weight[0,0]=1.
        actor.logstd.fill_(-3.)
    alg=PPO(1,Encoder(),actor,learning_rate=0.,est_learning_rate=lr,
            extra_loss_mass_w=0.,extra_loss_com_w=0.,encoder_kl_limit=limit)
    alg.storage=Storage(alg)
    return alg

def test_default_does_not_apply_guard_or_rollback():
    alg=make(None)
    result=alg.update()
    assert alg.encoder.linear.weight[0,0]>0
    assert 'encoder_policy_kl_proposed' not in result[4]
    assert alg.extra_optimizer.param_groups[0]['lr']==.1

def test_large_step_restores_weights_and_warm_adam_moments():
    alg=make()
    # Populate actual Adam state before testing rollback.
    alg.extra_optimizer.zero_grad();alg.encoder.linear.weight.square().sum().backward();alg.extra_optimizer.step()
    before_weights=copy.deepcopy(alg.encoder.state_dict())
    before_adam=copy.deepcopy(alg.extra_optimizer.state_dict())
    metrics=alg.update()[4]
    assert metrics['encoder_update_rejected']
    assert metrics['encoder_policy_kl_proposed']>.005
    assert metrics['encoder_policy_kl_applied']==0.
    for k,v in alg.encoder.state_dict().items():torch.testing.assert_close(v,before_weights[k],rtol=0,atol=0)
    after_adam=alg.extra_optimizer.state_dict()
    for key,values in before_adam['state'].items():
        for name,value in values.items():torch.testing.assert_close(after_adam['state'][key][name],value,rtol=0,atol=0)
    assert metrics['encoder_learning_rate']==pytest.approx(.05)

def test_small_step_remains_trainable_and_reports_sampled_bound():
    alg=make(lr=1e-6)
    metrics=alg.update()[4]
    assert not metrics['encoder_update_rejected']
    assert 0.<metrics['encoder_policy_kl_applied']<=.005
    assert metrics['encoder_guard_samples']==4
    assert alg.encoder.linear.weight[0,0]>0
    assert alg.extra_optimizer.state_dict()['state']

def test_sample_count_is_bounded_and_uses_no_teacher_input():
    alg=make();alg.encoder_kl_sample_size=2
    snapshot=alg._begin_encoder_policy_guard()
    assert snapshot['mu'].shape==(2,1)
    assert 'critic' not in snapshot
    assert snapshot['observations'].shape==(2,3)

def test_parallel_limits_change_acceptance_not_policy_inputs():
    tight=make(limit=.005,lr=.0006)
    loose=make(limit=.01,lr=.0006)
    a=tight.update()[4];b=loose.update()[4]
    assert .005<a['encoder_policy_kl_proposed']<.01
    assert a['encoder_policy_kl_proposed']==pytest.approx(b['encoder_policy_kl_proposed'])
    assert a['encoder_update_rejected'] and a['encoder_policy_kl_applied']==0.
    assert not b['encoder_update_rejected'] and b['encoder_policy_kl_applied']>.005
    assert tight.storage.observations.shape==loose.storage.observations.shape

def test_reference_is_after_actor_update_not_before_it():
    alg=make(lr=0.)
    for group in alg.optimizer.param_groups:group['lr']=1e-3
    batch=list(alg.storage.batch)
    batch[5]=torch.full_like(batch[5],.1)
    batch[9]=alg.actor_critic.get_actions_log_prob(batch[5]).detach().unsqueeze(-1)
    alg.storage.batch=tuple(batch)
    before=alg.actor_critic.actor[-1].bias.detach().clone()
    metrics=alg.update()[4]
    assert not torch.equal(before,alg.actor_critic.actor[-1].bias)
    assert not metrics['encoder_update_rejected']
    assert metrics['encoder_policy_kl_applied']==0.

def test_nonfinite_proposal_is_rolled_back_without_nan_metrics():
    alg=make();before=copy.deepcopy(alg.encoder.state_dict())
    snapshot=alg._begin_encoder_policy_guard()
    with torch.no_grad():alg.encoder.linear.weight.fill_(float('nan'))
    metrics=alg._finish_encoder_policy_guard(snapshot)
    assert metrics['encoder_update_rejected']
    assert metrics['encoder_policy_kl_proposed'] is None
    assert metrics['encoder_policy_kl_applied']==0.
    for k,v in alg.encoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

@pytest.mark.parametrize('limit',[0.,-1.,float('inf'),float('nan')])
def test_bad_limits_rejected(limit):
    with pytest.raises(ValueError,match='encoder_kl_limit'):make(limit)

@pytest.mark.parametrize('sample_count',[0,-1,True,1.5])
def test_bad_sample_counts_rejected(sample_count):
    with pytest.raises(ValueError,match='encoder_kl_sample_size'):
        PPO(1,Encoder(),ActorCritic(10,7,1),encoder_kl_sample_size=sample_count)
