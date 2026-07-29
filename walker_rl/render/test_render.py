"""Headless smoke test for the viewer.

Runs against SDL's dummy video driver so it can be executed without a display:
it will not tell you the picture looks good, but it does catch every crash in
the drawing paths, which is what usually breaks after an env change.

    python -m walker_rl.render.test_render
"""

import os
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

from ..env import StickmanConfig, StickmanEnv  # noqa: E402
from ..rl.recorder import EpisodeRecorder, load_episode, run_episode  # noqa: E402
from .viewer import Viewer, replay_state  # noqa: E402


def test_live_rendering():
    cfg = StickmanConfig()
    cfg.curriculum = False
    env = StickmanEnv(cfg, seed=0)
    env.reset(seed=0, target_x=5.0)
    viewer = Viewer(width=800, height=480, title="test")
    rng = np.random.default_rng(0)
    try:
        for _ in range(60):
            action = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            _, _, term, trunc, _ = env.step(action)
            viewer.draw(env.render_state(), ["headless test"])
            if term or trunc:
                env.reset(target_x=5.0)
    finally:
        viewer.close()


def test_record_and_replay_roundtrip():
    cfg = StickmanConfig()
    cfg.curriculum = False
    env = StickmanEnv(cfg, seed=0)
    recorder = EpisodeRecorder(env)
    rng = np.random.default_rng(1)
    policy = lambda obs: rng.uniform(-1, 1, env.act_dim).astype(np.float32)
    out = run_episode(env, policy, recorder=recorder, target_x=4.0, seed=7,
                      max_steps=80)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ep.npz")
        recorder.save(path, note="roundtrip")
        episode = load_episode(path)

    n = len(episode["rewards"])
    assert n == out["steps"], "recorded %d steps, episode ran %d" % (n, out["steps"])
    assert episode["actions"].shape == (n, env.act_dim)
    assert episode["components"].shape == (n, len(episode["component_names"]))
    assert episode["poses"].shape[0] == n
    assert abs(float(episode["rewards"].sum()) - out["return"]) < 1e-3, \
        "recorded rewards do not sum to the episode return"
    # The per-term breakdown must reconstruct the total reward exactly.
    assert np.allclose(episode["components"].sum(axis=1), episode["rewards"],
                       atol=1e-5), "reward components do not sum to the reward"
    assert episode["meta"]["note"] == "roundtrip"

    viewer = Viewer(width=800, height=480, title="replay test")
    try:
        for frame in range(0, n, max(1, n // 20)):
            viewer.draw(replay_state(episode, frame), ["replay"])
    finally:
        viewer.close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %-40s %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %-40s %r" % (fn.__name__, exc))
        else:
            print("ok    %s" % fn.__name__)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
