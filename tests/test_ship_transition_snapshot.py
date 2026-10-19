"""Rollout inputs must survive an environment's in-place reset operations."""
import torch
from torch import nn
from legged_gym.algorithm.ppo import PPO
from legged_gym.algorithm.actor_critic import ActorCritic


class Encoder(nn.Module):
    num_output_dim=1
    def __init__(self):
        super().__init__();self.layer=nn.Linear(2,1,bias=False)
        with torch.no_grad():self.layer.weight.copy_(torch.tensor([[1.,2.]]))
    def encode(self,x):return self.layer(x).detach()


def setup():
    torch.manual_seed(7)
    actor=ActorCritic(3,3,1,actor_hidden_dims=[8],critic_hidden_dims=[8])
    alg=PPO(2,Encoder(),actor,critic_take_latent=False)
    alg.init_storage(2,1,[1],[3],[2],[1],[1])
    obs=torch.tensor([[.1],[.2]])
    history=torch.tensor([[.2,.3],[.4,.5]])
    command=torch.tensor([[.6],[.7]])
    critic=torch.tensor([[.8,.9],[1.,1.1]])
    return alg,actor,obs,history,command,critic


def test_transition_snapshots_inputs_and_sampled_actions_before_env_step():
    alg,actor,obs,history,command,critic=setup()
    before=[x.clone() for x in [obs,history,command,critic]]
    actions=alg.act(obs,history,command,critic);sampled=actions.clone()
    obs.fill_(10.);history.fill_(20.);command.fill_(30.);critic.fill_(40.);actions.fill_(50.)
    alg.process_env_step(torch.ones(2),torch.ones(2,dtype=torch.bool),{},next_obs=torch.zeros_like(obs))
    torch.testing.assert_close(alg.storage.observations[0],before[0])
    torch.testing.assert_close(alg.storage.observation_history[0],before[1])
    torch.testing.assert_close(alg.storage.commands[0],before[2])
    torch.testing.assert_close(alg.storage.critic_obs[0],torch.cat((before[3],before[2]),dim=-1))
    torch.testing.assert_close(alg.storage.actions[0],sampled)


def test_unchanged_policy_likelihood_roundtrip_after_history_reset():
    alg,actor,obs,history,command,critic=setup()
    actions=alg.act(obs,history,command,critic)
    history[0].zero_()  # models an environment reset_idx filling its GRU history
    alg.process_env_step(torch.ones(2),torch.tensor([True,False]),{},next_obs=torch.zeros_like(obs))
    s=alg.storage
    actor_input=torch.cat((alg.encode_for_policy(s.observation_history[0],s.observations[0]),
                           s.observations[0],s.commands[0]),dim=-1)
    actor.act(actor_input)
    torch.testing.assert_close(actor.action_mean,s.mu[0],atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(actor.get_actions_log_prob(s.actions[0]),s.actions_log_prob[0,:,0],atol=1e-7,rtol=1e-6)
