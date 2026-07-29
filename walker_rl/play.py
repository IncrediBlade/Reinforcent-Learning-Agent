"""Watch the stickman: live policy, live baselines, or a recorded episode.

    python -m walker_rl.play --zero                       limp ragdoll (falls)
    python -m walker_rl.play --random                      random motor noise
    python -m walker_rl.play --checkpoint runs/walk/checkpoints/best.pt
    python -m walker_rl.play --replay runs/walk/episodes/ep_step000123456_r+412.npz

Keys: space pause, n single step, r reset, +/- speed, [ ] zoom, c contacts,
      t trail, h hud, esc quit.
"""

import argparse
import dataclasses
import glob
import os

import numpy as np
import torch

from .env import StickmanConfig, StickmanEnv, config_from_dict
from .rl import PPO, ObsNormalizer
from .rl.ppo import PPOConfig
from .rl.recorder import load_episode
from .render.viewer import Viewer, replay_state


def load_policy(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    env_cfg = config_from_dict(state["env_cfg"]) if "env_cfg" in state else StickmanConfig()

    # Only feed through fields this version of PPOConfig still knows about, so
    # an older checkpoint stays loadable.
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    raw = dict(state.get("ppo_cfg", {}))
    raw["hidden"] = tuple(raw.get("hidden", (256, 256)))
    ppo_cfg = PPOConfig(**{k: v for k, v in raw.items() if k in known})

    probe = StickmanEnv(env_cfg, seed=0)
    agent = PPO(probe.obs_dim, probe.act_dim, ppo_cfg, device)
    agent.load_state_dict(state["ppo"], load_optimizer=False)
    agent.model.eval()
    agent.set_progress(1.0)

    obs_norm = None
    if state.get("obs_norm"):
        obs_norm = ObsNormalizer((probe.obs_dim,))
        obs_norm.load_state_dict(state["obs_norm"])
        obs_norm.frozen = True

    info = {
        "global_step": state.get("global_step", 0),
        "best_eval": state.get("best_eval", float("nan")),
        "curriculum_distance": state.get("curriculum_distance", float("nan")),
    }
    return agent, obs_norm, env_cfg, probe, info


def make_policy_fn(agent, obs_norm, stochastic=False):
    def fn(obs):
        x = obs_norm(obs, update=False) if obs_norm is not None else obs
        action, _, _ = agent.act(np.asarray(x, dtype=np.float32)[None],
                                 deterministic=not stochastic)
        return np.clip(action[0], -1.0, 1.0)
    return fn


def play_live(args):
    extra_lines = []
    if args.checkpoint:
        agent, obs_norm, env_cfg, env, info = load_policy(args.checkpoint, args.device)
        policy = make_policy_fn(agent, obs_norm, args.stochastic)
        extra_lines = ["ckpt  %s" % os.path.basename(args.checkpoint),
                       "trained steps %d" % info["global_step"],
                       "mode  %s" % ("stochastic" if args.stochastic else "deterministic")]
    else:
        env_cfg = StickmanConfig()
        env_cfg.curriculum = False
        env = StickmanEnv(env_cfg, seed=args.seed)
        if args.random:
            rng = np.random.default_rng(args.seed)
            policy = lambda obs: rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            extra_lines = ["mode  random actions"]
        else:
            policy = lambda obs: np.zeros(env.act_dim, dtype=np.float32)
            extra_lines = ["mode  zero actions (limp)"]

    env.cfg.curriculum = False
    viewer = Viewer(fps=int(env.cfg.control_hz), title="Stickman RL - live")

    episode = 0
    obs = env.reset(seed=args.seed, target_x=args.target)
    viewer.reset_trail()
    step_once = False

    while not viewer.closed and episode < args.episodes:
        cmd = viewer.poll()
        if cmd == "quit":
            break
        if cmd == "reset":
            obs = env.reset(target_x=args.target)
            viewer.reset_trail()
            continue
        if cmd == "step":
            step_once = True

        if not viewer.paused or step_once:
            step_once = False
            for _ in range(viewer.steps_per_frame()):
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    tag = "REACHED" if info["success"] else "fell"
                    print("episode %d  %-8s return %8.1f  steps %4d  "
                          "target %+.1f  ended at x=%+.2f"
                          % (episode + 1, tag, info["episode"]["r"],
                             info["episode"]["l"], info["target_x"], info["x"]))
                    episode += 1
                    obs = env.reset(target_x=args.target)
                    viewer.reset_trail()
                    break

        viewer.draw(env.render_state(), extra_lines)
        viewer.tick()

    viewer.close()
    env.close()


def play_replay(args):
    path = args.replay
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            raise SystemExit("no .npz episodes in %s" % path)
        path = files[-1]
    episode = load_episode(path)
    meta = episode["meta"]
    n = len(episode["rewards"])
    print("replay %s\n  %d steps, return %.1f, target %+.2f, recorded %s"
          % (os.path.basename(path), n, meta["total_return"], meta["target_x"],
             meta.get("saved_at", "?")))

    viewer = Viewer(fps=int(meta.get("control_hz", 40)), title="Stickman RL - replay")
    extra = ["replay %s" % os.path.basename(path),
             "trained steps %s" % meta.get("global_step", "?"),
             "success %s" % meta.get("success", "?")]
    frame = 0
    step_once = False
    while not viewer.closed:
        cmd = viewer.poll()
        if cmd == "quit":
            break
        if cmd == "reset":
            frame = 0
            viewer.reset_trail()
        if cmd == "step":
            step_once = True
        if not viewer.paused or step_once:
            frame += 1 if step_once else viewer.steps_per_frame()
            step_once = False
            if frame >= n:
                if args.loop:
                    frame = 0
                    viewer.reset_trail()
                else:
                    frame = n - 1
                    viewer.paused = True
        viewer.draw(replay_state(episode, min(frame, n - 1)), extra)
        viewer.tick()
    viewer.close()


def build_parser():
    p = argparse.ArgumentParser(description="Visualise the stickman")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", help="policy checkpoint to run live")
    src.add_argument("--replay", help="recorded .npz episode (or a directory)")
    src.add_argument("--random", action="store_true", help="uniform random motors")
    src.add_argument("--zero", action="store_true", help="limp ragdoll baseline")
    p.add_argument("--target", type=float, default=None,
                   help="fixed target x (default: random each episode)")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of using its mean")
    p.add_argument("--loop", action="store_true", help="loop a replay")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.replay:
        play_replay(args)
    else:
        play_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
