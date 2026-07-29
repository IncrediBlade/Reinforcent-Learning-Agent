"""Train the stickman with PPO.

    python -m walker_rl.train --run-name walk1 --envs 8 --steps 20000000

Everything needed to reproduce or resume a run lands under runs/<name>/:
    config.json          exact env + algorithm settings
    metrics.csv          one row per update
    checkpoints/         latest.pt, best.pt, periodic snapshots
    episodes/            recorded trajectories (actions + reward breakdown)
"""

import argparse
import csv
import json
import os
import signal
import time
from collections import deque

import numpy as np
import torch

from .env import StickmanConfig, StickmanEnv
from .rl import PPO, EpisodeRecorder, ObsNormalizer, ReturnNormalizer, RolloutBuffer
from .rl.ppo import PPOConfig
from .rl.recorder import run_episode
from .rl.vec_env import make_vec_env

EVAL_TARGETS = (4.0, 6.0, 8.0, -5.0)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.join(args.out, args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.episode_dir = os.path.join(self.run_dir, "episodes")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.episode_dir, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.set_num_threads(max(1, args.torch_threads))

        self.env_cfg = self._build_env_config(args)
        self.ppo_cfg = self._build_ppo_config(args)

        self.device = args.device
        self.global_step = 0
        self.iteration = 0
        self.best_eval = -float("inf")
        self.start_time = time.time()
        self.stop_requested = False

        self.envs = make_vec_env(args.envs, self.env_cfg, seed=args.seed,
                                 subprocess=not args.no_subproc)
        self.obs_dim = self.envs.obs_dim
        self.act_dim = self.envs.act_dim

        self.agent = PPO(self.obs_dim, self.act_dim, self.ppo_cfg, self.device)
        self.obs_norm = ObsNormalizer((self.obs_dim,)) if self.ppo_cfg.normalize_obs else None
        self.ret_norm = (ReturnNormalizer(args.envs, self.ppo_cfg.gamma)
                         if self.ppo_cfg.normalize_reward else None)
        self.buffer = RolloutBuffer(self.ppo_cfg.num_steps, args.envs,
                                    self.obs_dim, self.act_dim, self.device)

        # A private single-process env for deterministic evaluation + recording.
        eval_cfg = self._build_env_config(args)
        eval_cfg.curriculum = False
        self.eval_env = StickmanEnv(eval_cfg, seed=args.seed + 99991)
        self.recorder = EpisodeRecorder(self.eval_env)

        self.ep_returns = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.ep_success = deque(maxlen=100)
        self.ep_travel = deque(maxlen=100)

        self._init_logs()
        if args.resume:
            self.load(args.resume)

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _build_env_config(args):
        cfg = StickmanConfig()
        cfg.target_min_distance = args.target_min
        cfg.target_max_distance = args.target_max
        cfg.target_left_probability = args.left_prob
        cfg.curriculum = not args.no_curriculum
        cfg.max_episode_seconds = args.episode_seconds
        return cfg

    def _build_ppo_config(self, args):
        cfg = PPOConfig()
        cfg.num_envs = args.envs
        cfg.num_steps = args.num_steps
        cfg.total_steps = args.steps
        cfg.learning_rate = args.lr
        cfg.ent_coef_start = args.ent_start
        cfg.ent_coef_end = args.ent_end
        cfg.log_std_floor_start = args.std_floor_start
        cfg.log_std_floor_end = args.std_floor_end
        cfg.target_kl = None if args.target_kl <= 0 else args.target_kl
        return cfg

    def _init_logs(self):
        with open(os.path.join(self.run_dir, "config.json"), "w") as fh:
            json.dump({"env": self.env_cfg.as_dict(),
                       "ppo": self.ppo_cfg.as_dict(),
                       "args": vars(self.args)}, fh, indent=2, default=str)
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.metric_fields = [
            "iteration", "global_step", "wall_time", "sps",
            "ep_return", "ep_length", "success_rate", "travel",
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
            "explained_variance", "action_std", "ent_coef", "log_std_floor",
            "learning_rate", "epochs", "early_stopped", "curriculum_distance",
            "eval_return", "eval_success",
        ]
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as fh:
                csv.DictWriter(fh, self.metric_fields).writeheader()

    def _log_row(self, row):
        clean = {k: row.get(k, "") for k in self.metric_fields}
        with open(self.metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, self.metric_fields).writerow(clean)

    # -- normalisation helpers --------------------------------------------
    def _norm_obs(self, obs, update=True):
        if self.obs_norm is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_norm(obs, update=update)

    # -- rollout -----------------------------------------------------------
    def collect(self, obs, dones):
        cfg = self.ppo_cfg
        self.buffer.reset()
        for _ in range(cfg.num_steps):
            actions, log_probs, values = self.agent.act(obs)
            # Send the RAW action. The env clips it to the motor range but
            # charges for the overshoot first -- pre-clipping here would hide
            # that cost and let the policy's std drift back out of range.
            raw_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

            new_dones = np.logical_or(terminated, truncated)
            if self.ret_norm is not None:
                scaled = self.ret_norm(rewards, new_dones)
            else:
                scaled = rewards.astype(np.float32)

            # Time-limit truncation is not a real terminal state: bootstrap it.
            if truncated.any():
                idx = np.nonzero(truncated)[0]
                terminal_obs = np.stack([infos[i]["terminal_observation"] for i in idx])
                term_v = self.agent.value_of(self._norm_obs(terminal_obs, update=False))
                scaled[idx] += cfg.gamma * term_v

            self.buffer.add(obs, actions, log_probs, scaled, values, dones)

            obs = self._norm_obs(raw_obs)
            dones = new_dones.astype(np.float32)
            self.global_step += self.args.envs

            for info in infos:
                ep = info.get("episode") if info else None
                if ep is not None:
                    self.ep_returns.append(ep["r"])
                    self.ep_lengths.append(ep["l"])
                    self.ep_success.append(1.0 if ep["success"] else 0.0)
                    self.ep_travel.append(ep["travelled"])
        return obs, dones

    # -- evaluation --------------------------------------------------------
    def policy_fn(self, deterministic=True):
        def fn(raw_obs):
            obs = self._norm_obs(raw_obs, update=False)
            action, _, _ = self.agent.act(obs[None], deterministic=deterministic)
            return np.clip(action[0], -1.0, 1.0)
        return fn

    def evaluate(self, record=False):
        policy = self.policy_fn(True)
        results = []
        best_return = -float("inf")
        for i, target in enumerate(EVAL_TARGETS):
            rec = self.recorder if record else None
            out = run_episode(self.eval_env, policy, recorder=rec,
                              target_x=target, seed=20240 + i)
            results.append(out)
            # The recorder still holds this episode, so save it while it is the
            # best one seen in this evaluation sweep.
            if record and out["return"] > best_return:
                best_return = out["return"]
                self.recorder.save(
                    os.path.join(self.episode_dir, "ep_step%09d_r%+.0f.npz"
                                 % (self.global_step, out["return"])),
                    global_step=self.global_step, iteration=self.iteration,
                    success=out["success"], eval_target=target)
        return {
            "eval_return": float(np.mean([r["return"] for r in results])),
            "eval_success": float(np.mean([r["success"] for r in results])),
            "eval_detail": results,
        }

    # -- checkpointing -----------------------------------------------------
    def save(self, name="latest.pt", **extra):
        path = os.path.join(self.ckpt_dir, name)
        state = {
            "ppo": self.agent.state_dict(),
            "obs_norm": self.obs_norm.state_dict() if self.obs_norm else None,
            "ret_norm": self.ret_norm.state_dict() if self.ret_norm else None,
            "global_step": self.global_step,
            "iteration": self.iteration,
            "best_eval": self.best_eval,
            "env_cfg": self.env_cfg.as_dict(),
            "ppo_cfg": self.ppo_cfg.as_dict(),
            "curriculum_distance": float(np.mean(
                self.envs.get_attr("curriculum_distance"))),
            **extra,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(state["ppo"])
        if self.obs_norm and state.get("obs_norm"):
            self.obs_norm.load_state_dict(state["obs_norm"])
        if self.ret_norm and state.get("ret_norm"):
            self.ret_norm.load_state_dict(state["ret_norm"])
        self.global_step = state.get("global_step", 0)
        self.iteration = state.get("iteration", 0)
        self.best_eval = state.get("best_eval", -float("inf"))
        if "curriculum_distance" in state:
            self.envs.call("set_curriculum_distance", state["curriculum_distance"])
        print("resumed from %s at step %d" % (path, self.global_step))

    # -- main loop ---------------------------------------------------------
    def train(self):
        cfg = self.ppo_cfg
        args = self.args
        obs = self._norm_obs(self.envs.reset())
        dones = np.zeros(args.envs, dtype=np.float32)
        total_iterations = max(1, cfg.total_steps // cfg.batch_size)

        self._install_signal_handler()
        print("obs_dim=%d act_dim=%d  envs=%d  batch=%d  iterations=%d"
              % (self.obs_dim, self.act_dim, args.envs, cfg.batch_size,
                 total_iterations))
        print("-" * 108)

        while self.global_step < cfg.total_steps and not self.stop_requested:
            self.iteration += 1
            self.agent.set_progress(self.global_step / cfg.total_steps)
            t0 = time.time()

            obs, dones = self.collect(obs, dones)

            last_values = self.agent.value_of(obs)
            self.buffer.compute_returns(last_values, dones, cfg.gamma, cfg.gae_lambda)
            stats = self.agent.update(self.buffer.get())

            row = self._make_row(stats, time.time() - t0)

            do_eval = (self.iteration % args.eval_interval == 0)
            do_record = (self.iteration % args.record_interval == 0)
            if do_eval or do_record:
                ev = self.evaluate(record=do_record)
                row["eval_return"] = round(ev["eval_return"], 2)
                row["eval_success"] = round(ev["eval_success"], 3)
                if ev["eval_return"] > self.best_eval:
                    self.best_eval = ev["eval_return"]
                    self.save("best.pt", eval_return=ev["eval_return"])

            self._log_row(row)
            self._print_row(row, header=(self.iteration % 20 == 1))

            if self.iteration % args.save_interval == 0:
                self.save("latest.pt")
                self.save("ckpt_step%09d.pt" % self.global_step)

        self.save("latest.pt")
        print("\nfinished at step %d  (best eval return %.1f)"
              % (self.global_step, self.best_eval))
        self.envs.close()

    def _make_row(self, stats, elapsed):
        sps = self.ppo_cfg.batch_size / max(1e-9, elapsed)
        row = {
            "iteration": self.iteration,
            "global_step": self.global_step,
            "wall_time": round(time.time() - self.start_time, 1),
            "sps": int(sps),
            "ep_return": round(float(np.mean(self.ep_returns)), 2) if self.ep_returns else "",
            "ep_length": round(float(np.mean(self.ep_lengths)), 1) if self.ep_lengths else "",
            "success_rate": round(float(np.mean(self.ep_success)), 3) if self.ep_success else "",
            "travel": round(float(np.mean(self.ep_travel)), 2) if self.ep_travel else "",
            "curriculum_distance": round(float(np.mean(
                self.envs.get_attr("curriculum_distance"))), 2),
        }
        for key, name in (("policy_loss", "policy_loss"), ("value_loss", "value_loss"),
                          ("entropy", "entropy"), ("approx_kl", "approx_kl"),
                          ("clip_fraction", "clip_fraction"),
                          ("explained_variance", "explained_variance"),
                          ("action_std_mean", "action_std"),
                          ("ent_coef", "ent_coef"),
                          ("log_std_floor", "log_std_floor"),
                          ("learning_rate", "learning_rate"),
                          ("epochs", "epochs"),
                          ("early_stopped", "early_stopped")):
            row[name] = round(float(stats[key]), 6)
        return row

    @staticmethod
    def _print_row(row, header=False):
        cols = [("iter", "iteration", "%6s"), ("steps", "global_step", "%10s"),
                ("sps", "sps", "%6s"), ("return", "ep_return", "%9s"),
                ("len", "ep_length", "%7s"), ("succ", "success_rate", "%6s"),
                ("travel", "travel", "%7s"), ("curric", "curriculum_distance", "%7s"),
                ("std", "action_std", "%6s"), ("kl", "approx_kl", "%7s"),
                ("expl_var", "explained_variance", "%9s"),
                ("eval", "eval_return", "%9s")]
        if header:
            print(" ".join(fmt % name for name, _, fmt in cols))
        vals = []
        for _, key, fmt in cols:
            v = row.get(key, "")
            if isinstance(v, float):
                v = "%.3f" % v
            vals.append(fmt % v)
        # flush: stdout is block-buffered when the run is piped to a log file,
        # and a training run people watch for hours should not look hung.
        print(" ".join(vals), flush=True)

    def _install_signal_handler(self):
        def handler(signum, frame):
            if self.stop_requested:
                raise KeyboardInterrupt
            print("\ninterrupt received -- finishing this update, then saving...")
            self.stop_requested = True
        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, AttributeError):
            pass


def build_parser():
    p = argparse.ArgumentParser(description="Train the stickman walker with PPO")
    p.add_argument("--run-name", default="walk")
    p.add_argument("--out", default="runs")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20_000_000,
                   help="total environment steps")
    p.add_argument("--num-steps", type=int, default=512,
                   help="rollout length per env per update")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--no-subproc", action="store_true",
                   help="run all envs in this process (slower, easier to debug)")
    p.add_argument("--resume", default=None, help="path to a checkpoint")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=0.03,
                   help="<=0 disables KL early stopping")
    p.add_argument("--ent-start", type=float, default=0.006)
    p.add_argument("--ent-end", type=float, default=0.0005)
    p.add_argument("--std-floor-start", type=float, default=-1.0)
    p.add_argument("--std-floor-end", type=float, default=-2.2)

    p.add_argument("--target-min", type=float, default=3.0)
    p.add_argument("--target-max", type=float, default=9.0)
    p.add_argument("--left-prob", type=float, default=0.15)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--episode-seconds", type=float, default=24.0)

    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--record-interval", type=int, default=25)
    p.add_argument("--save-interval", type=int, default=25)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    trainer = Trainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nhard interrupt -- saving and exiting")
        trainer.save("latest.pt")
        trainer.envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
