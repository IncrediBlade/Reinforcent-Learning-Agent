"""Top-level launcher for the stickman walker project.

    python RL_Agent.py test                     run every test suite
    python RL_Agent.py demo                     watch the limp ragdoll fall over
    python RL_Agent.py train --envs 10          train with PPO
    python RL_Agent.py play  --checkpoint ...   watch a trained policy
    python RL_Agent.py replay runs/walk/episodes --loop

Anything after the subcommand is forwarded to the underlying module, so
`python RL_Agent.py train --help` shows the full training options. The real
code lives in walker_rl/ -- see walker_rl/README.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests():
    from walker_rl.env import test_env
    from walker_rl.physics import test_physics
    from walker_rl.render import test_render

    failures = 0
    for name, module in (("physics engine", test_physics),
                         ("environment", test_env),
                         ("renderer", test_render)):
        print("\n=== %s ===" % name)
        failures += module.main()
    print("\n%s" % ("all suites passed" if not failures else "SOME SUITES FAILED"))
    return failures


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "help"

    if command == "test":
        return run_tests()
    if command == "train":
        from walker_rl.train import main as train_main
        return train_main(argv)
    if command == "play":
        from walker_rl.play import main as play_main
        return play_main(argv)
    if command == "replay":
        from walker_rl.play import main as play_main
        return play_main(["--replay", *argv] if argv else ["--replay", "runs"])
    if command == "demo":
        from walker_rl.play import main as play_main
        return play_main(["--zero", *argv])

    print(__doc__)
    return 0 if command in ("help", "-h", "--help") else 1


if __name__ == "__main__":
    raise SystemExit(main())
