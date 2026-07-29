"""Rollout storage with GAE(lambda).

The one subtlety worth calling out: a time-limit truncation is not a real
terminal state. If it is treated as one the agent learns that the world ends
after 24 seconds and stops caring about the future near the horizon. Truncated
steps are bootstrapped with V(s_final) before GAE runs.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device="cpu"):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, values, dones):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = rewards
        self.values[i] = values
        self.dones[i] = dones
        self.ptr += 1

    def compute_returns(self, last_values, last_dones, gamma=0.99, lam=0.95):
        adv = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (self.rewards[t] + gamma * next_values * next_non_terminal
                     - self.values[t])
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def get(self):
        n = self.num_steps * self.num_envs
        to = lambda a, shape: torch.as_tensor(a.reshape(shape), device=self.device)
        return {
            "obs": to(self.obs, (n, -1)),
            "actions": to(self.actions, (n, -1)),
            "log_probs": to(self.log_probs, (n,)),
            "advantages": to(self.advantages, (n,)),
            "returns": to(self.returns, (n,)),
            "values": to(self.values, (n,)),
        }
