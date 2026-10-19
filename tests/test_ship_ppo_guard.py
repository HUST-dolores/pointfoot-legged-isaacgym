"""KL early-stop regression, using real PPO/actor with no simulator."""
import torch
from torch import nn
from legged_gym.algorithm.ppo import PPO
from legged_gym.algorithm.actor_critic import ActorCritic


class EmptyEncoder(nn.Module):
    num_output_dim=0
    def encode(self,history):return history.new_zeros((history.shape[0],0))


class Storage:
    def __init__(self,actor,reject):
        obs=torch.zeros(2,3);actions=torch.zeros(2,1)
        actor.act(obs)
        mu=actor.action_mean.detach().clone()
        if reject:mu.fill_(100.)
        self.batch=(obs,obs,torch.zeros(2,1),None,torch.zeros(2,0),actions,
                    actor.evaluate(obs).detach(),torch.ones(2,1),torch.ones(2,1),
                    actor.get_actions_log_prob(actions).detach().unsqueeze(-1),
                    mu,actor.action_std.detach().clone())
        self.cleared=False
    def mini_batch_generator(self,*args):yield self.batch
    def clear(self):self.cleared=True


def make(reject):
    actor=ActorCritic(3,3,1,actor_hidden_dims=[8],critic_hidden_dims=[8])
    alg=PPO(1,EmptyEncoder(),actor,early_stop=True,desired_kl=.01)
    alg.storage=Storage(actor,reject)
    return alg,actor


def test_kl_rejecting_all_minibatches_is_safe_and_reports_zero_updates():
    alg,actor=make(True)
    before={k:v.clone() for k,v in actor.state_dict().items()}
    losses=alg.update()
    assert losses[4]['policy_updates']==0
    assert all(value==0 for value in losses[:4])
    assert alg.storage.cleared
    for k,v in actor.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_normal_kl_minibatch_still_updates_policy():
    alg,actor=make(False)
    before={k:v.clone() for k,v in actor.state_dict().items()}
    losses=alg.update()
    assert losses[4]['policy_updates']==1
    assert alg.storage.cleared
    assert any(not torch.equal(v,before[k]) for k,v in actor.state_dict().items())
