"""PPO with explicit exploration scheduling.

Three separate knobs control the exploration/exploitation balance, and all
three are annealed on the same normalised training progress in [0, 1]:

  1. entropy bonus        pushes the policy to keep spreading probability mass
  2. log-std floor        a hard lower bound on action noise, so the policy
                          physically cannot collapse to a deterministic gait
                          before it has explored; released later so it can
                          sharpen and actually exploit what it found
  3. learning rate        annealed so late updates refine rather than churn

On top of that, `target_kl` early-stops an update whose policy has already
moved too far, which is what keeps a lucky batch from destroying a working
gait -- the classic failure mode on locomotion tasks.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .networks import ActorCritic


@dataclass
class PPOConfig:
    # --- rollout ---
    num_envs: int = 8
    num_steps: int = 512
    total_steps: int = 20_000_000

    # --- optimisation ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    final_lr_fraction: float = 0.1
    update_epochs: int = 10
    num_minibatches: int = 8
    max_grad_norm: float = 0.5

    # --- PPO ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    target_kl: float = 0.03
    normalize_advantage: bool = True

    # --- exploration schedule ---
    # ent_coef_end is 0, not a small positive: any residual entropy bonus is a
    # permanent upward push on log_std, and late in training there is nothing
    # left to counteract it once the policy stops improving.
    ent_coef_start: float = 0.004
    ent_coef_end: float = 0.0
    log_std_init: float = -0.5
    log_std_floor_start: float = -1.0
    log_std_floor_end: float = -2.2
    # Hard ceiling on action noise. std must stay within the action range:
    # above it the env clips, the extra noise is behaviourally invisible, and
    # the entropy bonus inflates std until the policy is pure bang-bang.
    log_std_max: float = 0.0
    exploration_anneal_fraction: float = 0.7   # done exploring by 70% of training

    # --- network ---
    hidden: tuple = (256, 256)
    activation: str = "tanh"

    # --- normalisation ---
    normalize_obs: bool = True
    normalize_reward: bool = True

    def as_dict(self):
        return asdict(self)

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return max(1, self.batch_size // self.num_minibatches)


def _lerp(a, b, t):
    return a + (b - a) * t


class PPO:
    def __init__(self, obs_dim, act_dim, cfg=None, device="cpu"):
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.device = torch.device(device)
        self.model = ActorCritic(
            obs_dim, act_dim,
            hidden=tuple(self.cfg.hidden),
            activation=self.cfg.activation,
            log_std_init=self.cfg.log_std_init,
            log_std_max=self.cfg.log_std_max,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.learning_rate, eps=1e-5)
        self.set_progress(0.0)

    # -- schedules ---------------------------------------------------------
    def set_progress(self, progress):
        """progress in [0, 1] over the whole training run."""
        cfg = self.cfg
        progress = float(np.clip(progress, 0.0, 1.0))
        self.progress = progress
        t = min(1.0, progress / max(1e-6, cfg.exploration_anneal_fraction))

        self.ent_coef = _lerp(cfg.ent_coef_start, cfg.ent_coef_end, t)
        self.model.log_std_floor = _lerp(cfg.log_std_floor_start,
                                         cfg.log_std_floor_end, t)
        if cfg.anneal_lr:
            lr = cfg.learning_rate * _lerp(1.0, cfg.final_lr_fraction, progress)
        else:
            lr = cfg.learning_rate
        self.lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        action, log_prob, value = self.model.act(obs_t, deterministic)
        return (action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy())

    @torch.no_grad()
    def value_of(self, obs):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.model.predict_values(obs_t).cpu().numpy()

    # -- learning ----------------------------------------------------------
    def update(self, batch):
        cfg = self.cfg
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["values"]

        n = obs.shape[0]
        indices = np.arange(n)
        clipfracs = []
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0, "updates": 0,
                 "early_stopped": 0.0}
        epochs_done = 0

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            epoch_kl = []
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start:start + cfg.minibatch_size]
                mb_idx = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                log_prob, entropy, value = self.model.evaluate_actions(
                    obs[mb_idx], actions[mb_idx])
                log_ratio = log_prob - old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator.
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    epoch_kl.append(approx_kl.item())

                mb_adv = advantages[mb_idx]
                if cfg.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef,
                                                 1.0 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_value_loss:
                    v_unclipped = (value - returns[mb_idx]) ** 2
                    v_clipped = old_values[mb_idx] + torch.clamp(
                        value - old_values[mb_idx], -cfg.clip_coef, cfg.clip_coef)
                    v_clipped = (v_clipped - returns[mb_idx]) ** 2
                    value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    value_loss = 0.5 * ((value - returns[mb_idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (policy_loss
                        - self.ent_coef * entropy_loss
                        + cfg.value_coef * value_loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl.item()
                stats["updates"] += 1

            epochs_done += 1
            if cfg.target_kl is not None and np.mean(epoch_kl) > cfg.target_kl:
                stats["early_stopped"] = 1.0
                break

        k = max(1, stats["updates"])
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            stats[key] /= k
        stats["clip_fraction"] = float(np.mean(clipfracs)) if clipfracs else 0.0
        stats["epochs"] = epochs_done
        stats["ent_coef"] = self.ent_coef
        stats["learning_rate"] = self.lr
        stats["action_std_mean"] = float(self.model.current_std().mean())
        stats["log_std_floor"] = self.model.log_std_floor

        # Explained variance: how much of the return signal the critic captures.
        with torch.no_grad():
            y = returns.cpu().numpy()
            pred = old_values.cpu().numpy()
            var_y = np.var(y)
            stats["explained_variance"] = float(
                1.0 - np.var(y - pred) / var_y) if var_y > 0 else 0.0
        return stats

    # -- persistence -------------------------------------------------------
    def state_dict(self):
        return {"model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg.as_dict()}

    def load_state_dict(self, state, load_optimizer=True):
        self.model.load_state_dict(state["model"])
        # A checkpoint saved under a looser ceiling can carry a log_std above
        # the current one. `clamp` in the forward pass has zero gradient once
        # saturated, so the parameter would be frozen out of range forever --
        # pull the stored value back into the valid band on load instead.
        with torch.no_grad():
            self.model.log_std.clamp_(self.model.log_std_min,
                                      self.model.log_std_max)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
