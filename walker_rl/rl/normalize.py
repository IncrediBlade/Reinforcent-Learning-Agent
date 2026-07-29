"""Running observation / return normalisation.

PPO on a task like this fails far more often from badly scaled inputs and
rewards than from bad hyper-parameters, so both are normalised with running
statistics and the statistics are checkpointed alongside the weights -- a
policy restored without its normaliser is a different policy.
"""

import numpy as np


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObsNormalizer:
    def __init__(self, shape, clip=10.0, epsilon=1e-8):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.epsilon = epsilon
        self.frozen = False

    def __call__(self, obs, update=True):
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        batch = obs[None] if single else obs
        if update and not self.frozen:
            self.rms.update(batch)
        out = (batch - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        out = np.clip(out, -self.clip, self.clip).astype(np.float32)
        return out[0] if single else out

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, state):
        self.rms.load_state_dict(state)


class ReturnNormalizer:
    """Scales rewards by the std of the discounted return (VecNormalize style).

    The reward is divided, never re-centred: shifting rewards would change the
    optimal policy for a task with termination.
    """

    def __init__(self, num_envs, gamma=0.99, clip=10.0, epsilon=1e-8):
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.frozen = False

    def __call__(self, rewards, dones):
        rewards = np.asarray(rewards, dtype=np.float64)
        self.returns = self.returns * self.gamma + rewards
        if not self.frozen:
            self.rms.update(self.returns[:, None].reshape(-1))
        scaled = rewards / np.sqrt(self.rms.var + self.epsilon)
        self.returns[np.asarray(dones, dtype=bool)] = 0.0
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, state):
        self.rms.load_state_dict(state)
