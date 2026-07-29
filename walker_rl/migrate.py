"""Widen a checkpoint whose observation gained new channels.

Adding the gait clock grew the observation by two, which would normally mean
retraining from scratch and throwing away everything the policy already knows
about balancing. The new channels were appended at the END of the observation,
so every existing input keeps its index and the first layer only needs two more
columns. Initialising those to zero makes the widened network compute exactly
what the original did on its first forward pass -- it starts from the old
behaviour and learns to use the clock from there.

    python -m walker_rl.migrate runs/walk/checkpoints/latest.pt -o widened.pt
"""

import argparse
import os

import numpy as np
import torch

from .env import StickmanConfig, StickmanEnv


def widen_checkpoint(src, dst=None, new_obs_dim=None, verbose=True):
    state = torch.load(src, map_location="cpu", weights_only=False)
    model = state["ppo"]["model"]

    if new_obs_dim is None:
        probe = StickmanEnv(StickmanConfig(), seed=0)
        new_obs_dim = probe.obs_dim

    old_obs_dim = model["policy.0.weight"].shape[1]
    if old_obs_dim == new_obs_dim:
        if verbose:
            print("checkpoint already has obs_dim=%d, nothing to do" % old_obs_dim)
        return src
    if old_obs_dim > new_obs_dim:
        raise ValueError("checkpoint obs_dim %d is larger than the env's %d"
                         % (old_obs_dim, new_obs_dim))

    extra = new_obs_dim - old_obs_dim
    for key in ("policy.0.weight", "value.0.weight"):
        w = model[key]
        # Zero columns => the new inputs contribute nothing until trained.
        model[key] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)

    # The observation normaliser has per-channel statistics; extend with a
    # standard normal so the untrained channels pass through unscaled.
    if state.get("obs_norm"):
        rms = state["obs_norm"]
        rms["mean"] = np.concatenate([np.asarray(rms["mean"]), np.zeros(extra)])
        rms["var"] = np.concatenate([np.asarray(rms["var"]), np.ones(extra)])

    # Adam moments are per-parameter and no longer match the widened shapes.
    # Dropping the optimiser state is safe -- it rebuilds within a few updates.
    state["ppo"].pop("optimizer", None)

    dst = dst or (os.path.splitext(src)[0] + "_widened.pt")
    torch.save(state, dst)
    if verbose:
        print("widened %s: obs_dim %d -> %d (%d new channels zero-initialised)"
              % (os.path.basename(src), old_obs_dim, new_obs_dim, extra))
        print("wrote %s  (trained steps: %s)" % (dst, state.get("global_step")))
    return dst


def main(argv=None):
    p = argparse.ArgumentParser(description="Widen a checkpoint for a larger observation")
    p.add_argument("checkpoint")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args(argv)
    widen_checkpoint(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
