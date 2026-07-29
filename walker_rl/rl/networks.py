"""Actor-critic network for continuous control.

Separate trunks for policy and value (they want very different features for
locomotion), orthogonal init with the usual small gain on the policy head so
the initial policy is close to "do nothing" rather than flailing, and a
state-independent learnable log-std -- the standard, well-behaved choice for
PPO on locomotion tasks.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def mlp(sizes, activation="tanh", output_gain=0.01):
    act = ACTIVATIONS[activation]
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        is_last = i == len(sizes) - 2
        gain = output_gain if is_last else math.sqrt(2.0)
        nn.init.orthogonal_(linear.weight, gain)
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        if not is_last:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), activation="tanh",
                 log_std_init=-0.5, log_std_min=-3.0, log_std_max=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy = mlp([obs_dim, *hidden, act_dim], activation, output_gain=0.01)
        self.value = mlp([obs_dim, *hidden, 1], activation, output_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Raised externally to force exploration early, lowered to let the
        # policy sharpen once it has something worth exploiting.
        self.log_std_floor = log_std_min

    def _std(self):
        floor = max(self.log_std_min, self.log_std_floor)
        return self.log_std.clamp(floor, self.log_std_max).exp()

    def distribution(self, obs):
        mean = self.policy(obs)
        return Normal(mean, self._std().expand_as(mean))

    def forward(self, obs):
        return self.distribution(obs), self.value(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mean = self.policy(obs)
        value = self.value(obs).squeeze(-1)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device), value
        dist = Normal(mean, self._std().expand_as(mean))
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def evaluate_actions(self, obs, actions):
        dist, value = self(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def predict_values(self, obs):
        return self.value(obs).squeeze(-1)

    def current_std(self):
        return self._std().detach().cpu().numpy()
