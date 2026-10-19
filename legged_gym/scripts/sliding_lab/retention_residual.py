"""Bounded motor correction on an immutable pretrained balancing actor."""

import copy
import torch
from torch import nn


class RetentionResidualActor(nn.Module):
    def __init__(self, original_actor):
        super().__init__()
        self.base = copy.deepcopy(original_actor)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.correction = nn.Sequential(nn.Linear(38, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 8))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # Six position actions are scaled by .25 rad; wheel actions by .5 rad/s in the archived PD.
        self.register_buffer("limits", torch.tensor([.8, .8, .8, 16., .8, .8, .8, 16.]))

    def forward(self, observations):
        base = self.base(observations[:, :58])
        features = torch.cat((observations[:, :3], observations[:, 7:27], observations[:, 58:]), -1).clamp(-5., 5.)
        correction = torch.tanh(self.correction(features)) * self.limits
        return base + correction * observations[:, -1:]


def residual_networks(device="cuda:0", resume=None, learning_rate=2e-5, action_std=.08):
    from .retention_train import networks
    algorithm, train, error, _ = networks(device=device, resume=None, learning_rate=learning_rate, action_std=action_std)
    original = copy.deepcopy(algorithm.actor_critic.actor)
    first = nn.Linear(58, original[0].out_features).to(device)
    with torch.no_grad():
        first.weight.copy_(original[0].weight[:, :58])
        first.bias.copy_(original[0].bias)
    original[0] = first
    algorithm.actor_critic.actor = RetentionResidualActor(original).to(device)
    algorithm.architecture = "bounded_residual"
    algorithm.optimizer = torch.optim.Adam([p for p in algorithm.actor_critic.parameters() if p.requires_grad], lr=learning_rate)
    state = None
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("architecture") != "bounded_residual":
            raise ValueError("Expected bounded_residual checkpoint")
        algorithm.actor_critic.load_state_dict(state["model_state_dict"], strict=True)
        algorithm.encoder.load_state_dict(state["encoder_state_dict"], strict=True)
        algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
    with torch.no_grad():
        test = torch.randn(32, 73, device=device)
        if not resume:
            residual_error = (algorithm.actor_critic.actor(test) - original(test[:, :58])).abs().max().item()
            assert residual_error < 1e-5
    return algorithm, train, error, state
