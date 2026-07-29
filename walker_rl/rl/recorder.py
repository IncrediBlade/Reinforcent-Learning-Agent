"""Episode recording: what the agent did, and what it got paid for doing it.

Each recording is a single .npz holding the full trajectory -- observations,
the action vector at every control step, the reward *broken down per term*, and
the exact pose of every rigid body. That last part means a replay is a literal
playback of the recorded motion, not a re-simulation that might diverge.
"""

import json
import os
import time

import numpy as np

from ..env.stickman import JOINT_ORDER
from ..physics.shapes import Circle

COMPONENT_ORDER = ("progress", "alive", "upright", "height", "support",
                   "two_feet", "imitate", "air_time", "step_length", "slip", "alternate", "periodic", "clearance",
                   "splay", "leg_swap", "stale_lead", "foot_sep",
                   "goal", "ctrl", "rate", "bound", "limit", "spin", "fall")


def capture_geometry(env):
    """Static description of every fixture, enough to redraw any pose."""
    defs = []
    for bi, body in enumerate(env.world.bodies):
        for fx in body.fixtures:
            shape = fx.shape
            if isinstance(shape, Circle):
                defs.append({"body": bi, "name": body.name, "kind": "circle",
                             "cx": shape.p.x, "cy": shape.p.y, "r": shape.radius})
            else:
                defs.append({"body": bi, "name": body.name, "kind": "poly",
                             "verts": [[v.x, v.y] for v in shape.vertices]})
    return defs


class EpisodeRecorder:
    def __init__(self, env):
        self.env = env
        self.geometry = capture_geometry(env)
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.components = []
        self.poses = []
        self.contacts = []
        self.total = 0.0

    def record(self, obs, action, reward, components):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.components.append([float(components.get(k, 0.0)) for k in COMPONENT_ORDER])
        self.poses.append(self.env.pose())
        self.total += float(reward)

    def save(self, path, **meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            obs=np.asarray(self.obs, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            components=np.asarray(self.components, dtype=np.float32),
            component_names=np.asarray(COMPONENT_ORDER),
            joint_names=np.asarray(JOINT_ORDER),
            poses=np.asarray(self.poses, dtype=np.float32),
            geometry=np.asarray(json.dumps(self.geometry)),
            meta=np.asarray(json.dumps({
                "total_return": self.total,
                "steps": len(self.rewards),
                "target_x": float(self.env.target_x),
                "platform_half": float(self.env.cfg.platform_half_width),
                "platform_height": float(self.env.cfg.platform_height),
                "goal_half": float(self.env.cfg.goal_zone_half_width),
                "control_hz": float(self.env.cfg.control_hz),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **meta,
            })),
        )
        return path


def load_episode(path):
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    out["geometry"] = json.loads(str(out["geometry"]))
    out["meta"] = json.loads(str(out["meta"]))
    out["component_names"] = [str(s) for s in out["component_names"]]
    out["joint_names"] = [str(s) for s in out["joint_names"]]
    return out


def run_episode(env, policy, recorder=None, target_x=None, seed=None,
                max_steps=None, deterministic=True, callback=None):
    """Roll one episode. `policy(obs) -> action`; None means zero action."""
    obs = env.reset(seed=seed, target_x=target_x)
    if recorder is not None:
        recorder.reset()
    total = 0.0
    steps = 0
    limit = max_steps or env.cfg.max_episode_steps
    info = {}
    while steps < limit:
        action = (np.zeros(env.act_dim, dtype=np.float32) if policy is None
                  else policy(obs))
        next_obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if recorder is not None:
            recorder.record(obs, action, reward, info["components"])
        if callback is not None and callback(env, action, reward, info) is False:
            break
        obs = next_obs
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": steps,
        "success": bool(info.get("success", False)),
        "distance": float(info.get("distance", float("nan"))),
        "target_x": float(info.get("target_x", env.target_x)),
        "x": float(info.get("x", 0.0)),
    }
