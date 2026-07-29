"""Vectorised environments.

The physics is pure Python, so it is CPU bound and does not release the GIL --
threads would buy nothing. Subprocesses do, and on Windows they must use the
'spawn' start method, which is why the env factory is a module-level picklable
class rather than a lambda.
"""

import multiprocessing as mp
import sys
import traceback

import numpy as np

from ..env import StickmanConfig, StickmanEnv


class EnvFactory:
    """Picklable env constructor."""

    def __init__(self, cfg=None, seed=0, rank=0):
        self.cfg = cfg if cfg is not None else StickmanConfig()
        self.seed = seed
        self.rank = rank

    def __call__(self):
        return StickmanEnv(self.cfg, seed=self.seed + 1000 * self.rank)


def _slim_info(info, obs, done):
    """Only what the trainer actually reads.

    The full info dict carries a 12-term reward breakdown every step; pickling
    that for every env on every step costs more than the physics it describes.
    The viewer reads components straight off a local env, so nothing is lost.
    """
    if not done:
        return None
    return {"episode": info.get("episode"), "terminal_observation": obs}


def _worker(remote, parent_remote, factory):
    parent_remote.close()
    env = factory()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                slim = _slim_info(info, obs, done)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, terminated, truncated, slim))
            elif cmd == "reset":
                remote.send(env.reset(**(data or {})))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                setattr(env, data[0], data[1])
                remote.send(True)
            elif cmd == "call":
                name, args, kwargs = data
                remote.send(getattr(env, name)(*args, **kwargs))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError("unknown command %r" % cmd)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001 - surface worker crashes to the parent
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


class SubprocVecEnv:
    def __init__(self, factories, start_method="spawn"):
        self.num_envs = len(factories)
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in factories])
        self.processes = []
        for work_remote, remote, factory in zip(self.work_remotes, self.remotes, factories):
            proc = ctx.Process(target=_worker, args=(work_remote, remote, factory),
                               daemon=True)
            proc.start()
            self.processes.append(proc)
            work_remote.close()
        self.closed = False
        self.obs_dim = self.get_attr("obs_dim")[0]
        self.act_dim = self.get_attr("act_dim")[0]

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(("reset", kwargs))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (np.stack(obs),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(terminated, dtype=bool),
                np.asarray(truncated, dtype=bool),
                list(infos))

    def get_attr(self, name):
        for remote in self.remotes:
            remote.send(("get_attr", name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, name, value):
        for remote in self.remotes:
            remote.send(("set_attr", (name, value)))
        return [remote.recv() for remote in self.remotes]

    def call(self, name, *args, **kwargs):
        for remote in self.remotes:
            remote.send(("call", (name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.closed = True


class DummyVecEnv:
    """Single-process fallback; identical interface, easier to debug."""

    def __init__(self, factories):
        self.envs = [f() for f in factories]
        self.num_envs = len(self.envs)
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self.closed = False

    def reset(self, **kwargs):
        return np.stack([env.reset(**kwargs) for env in self.envs])

    def step(self, actions):
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            info = _slim_info(info, obs, done)
            if done:
                obs = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)
        return (np.stack(obs_list),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(terms, dtype=bool),
                np.asarray(truncs, dtype=bool),
                infos)

    def get_attr(self, name):
        return [getattr(env, name) for env in self.envs]

    def set_attr(self, name, value):
        for env in self.envs:
            setattr(env, name, value)
        return [True] * self.num_envs

    def call(self, name, *args, **kwargs):
        return [getattr(env, name)(*args, **kwargs) for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()
        self.closed = True


def make_vec_env(num_envs, cfg=None, seed=0, subprocess=True):
    factories = [EnvFactory(cfg, seed, rank) for rank in range(num_envs)]
    if subprocess and num_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)
