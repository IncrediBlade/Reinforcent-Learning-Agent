# Stickman Walker — teaching a 2D ragdoll to balance and walk with PPO

A stickman that learns, from scratch, to **stand on two legs and walk to a
target** — on top of a rigid-body physics engine written from scratch in this
repo. No Box2D, no MuJoCo, no Gym.

Nothing about balancing is scripted. The figure spawns upright and limp; with a
zero-action policy it folds up and falls over in about a second (there is a test
asserting exactly that). Every bit of postural control has to be discovered by
the agent.

```
walker_rl/
  physics/    hand-written 2D rigid-body engine (impulse solver, joints, contacts)
  env/        the stickman, its reward function, its observations, gait reference
  rl/         PPO, vectorised envs, normalisation, episode recording
  render/     pygame viewer (live + replay)
  train.py    training entry point
  play.py     visualisation entry point
RL_Agent.py   convenience launcher (test / demo / train / play / replay)
```

Requirements: `numpy`, `torch` (CPU is fine), `pygame`.

---

## Quick start

```bash
pip install numpy torch pygame

# 1. Verify the engine and environment (63 tests, ~30 s)
python RL_Agent.py test

# 2. See the problem before training anything — a limp ragdoll falls over
python RL_Agent.py demo
python -m walker_rl.play --random

# 3. Train
python -m walker_rl.train --run-name walk --envs 10

# 4. Watch it (in a second terminal, while training runs)
python -m walker_rl.play --checkpoint runs/walk/checkpoints/latest.pt
```

Viewer keys: `space` pause · `n` step · `r` reset · `+`/`-` speed · `[` `]` zoom ·
`c` contacts · `t` trail · `h` HUD · `esc` quit.

---

## Training

```bash
python -m walker_rl.train --run-name walk --envs 10 --steps 20000000
```

Useful flags:

| flag | meaning |
|---|---|
| `--envs 10` | parallel environments (subprocesses; CPU-bound, so ≈ core count) |
| `--steps` | total environment steps (default 20M) |
| `--resume PATH` | continue from a checkpoint |
| `--run-name` | output directory under `runs/` — **always set this** |
| `--target-min/--target-max` | target distance range (m) |
| `--no-curriculum` | disable the growing-distance curriculum |
| `--ent-start/--ent-end` | entropy coefficient schedule |

Everything lands in `runs/<name>/`:

```
checkpoints/   latest.pt, best.pt, ckpt_step*.pt
metrics.csv    one row per update (return, length, success rate, KL, ...)
episodes/      recorded trajectories (.npz)
config.json    exact env + algorithm settings
```

`Ctrl-C` saves cleanly. Resume with `--resume runs/<name>/checkpoints/latest.pt`.

**Checkpoints and episodes are gitignored** — they are large and you should
generate your own. Expect balance within ~30 min on a modern CPU and a walking
gait over several hours.

To monitor a run (note `-t`: newest first, since step numbers don't sort):

```bash
ls -lat runs/walk/checkpoints/ | head -5
tail -1 runs/walk/metrics.csv
```

## Playing

```bash
# a trained policy
python -m walker_rl.play --checkpoint runs/walk/checkpoints/best.pt

# baselines, to see what the agent starts from
python -m walker_rl.play --zero        # limp ragdoll
python -m walker_rl.play --random      # random motor targets

# replay a recorded episode (exact playback, not a re-simulation)
python -m walker_rl.play --replay runs/walk/episodes --loop

# pin the target, or sample the policy instead of using its mean
python -m walker_rl.play --checkpoint runs/walk/checkpoints/best.pt --target -6 --stochastic
```

---

## The RL algorithm: PPO

**Proximal Policy Optimization** (clipped surrogate objective), on-policy,
with a Gaussian policy over continuous actions. Implemented from scratch in
[`walker_rl/rl/ppo.py`](walker_rl/rl/ppo.py).

| component | choice |
|---|---|
| advantage estimation | GAE(λ), γ = 0.99, λ = 0.95 |
| objective | clipped surrogate, ε = 0.2 |
| value loss | clipped, coefficient 0.5 |
| networks | separate policy/value trunks, 2×256 tanh, orthogonal init |
| policy head gain | 0.01 — the initial policy is near-silent, not flailing |
| action distribution | Gaussian, state-independent learnable log-std |
| optimiser | Adam, lr 3e-4 annealed to 10% |
| epochs / minibatches | 10 / 8 per update, with KL early stopping |
| rollout | 512 steps × 10 envs = 5120 per update |
| normalisation | running obs normalisation + return scaling (checkpointed) |

**Exploration vs exploitation** is controlled explicitly by three schedules
annealed on training progress, plus a safety brake:

1. **Entropy bonus** 0.004 → 0, keeping probability mass spread early.
2. **A hard log-std floor** −1.0 → −2.2, so the policy physically *cannot*
   collapse to a deterministic gait before it has explored, then is released to
   sharpen. Entropy bonuses alone often let std collapse early on locomotion.
3. **Learning-rate annealing**, so late updates refine rather than churn.
4. **KL early stopping** (target 0.03) aborts an update that has already moved
   too far — this is what stops one unlucky batch from destroying a working
   gait, the classic PPO locomotion failure.

A **hard std ceiling** (`log_std_max = 0`) is also enforced. Actions are clipped
to [−1, 1], so std above the action range is behaviourally invisible while the
entropy bonus keeps inflating it — left unchecked, std grew to 1.68 and **83% of
actions saturated**, making a smooth gait impossible to express.

**Time limits are not terminal states.** Truncated steps are bootstrapped with
`γ·V(s_final)` before GAE; without this the agent learns the world ends at 24 s
and stops caring about the future near the horizon.

---

## The environment

**11 rigid bodies** — torso (with head), two thighs, two shins, two feet, two
arms. ~1.65 m, ~72 kg, with joint torque limits in the range of real human
joints.

**Actions (8)** — joint-angle targets as an offset from the neutral standing
pose, tracked by a PD controller inside the physics substep loop:
hip L/R, knee L/R, ankle L/R, shoulder L/R.

**Observations (39)** — egocentric and **target-relative**:

```
torso        sin/cos of pitch, angular velocity, vx, vy, height        (6)
joints × 8   angle normalised to its own limits, angular velocity     (16)
contacts     left foot, right foot, "something that isn't a foot"      (3)
target       direction unit vector in the TORSO's frame,               (4)
             world-frame x direction, normalised distance
previous action                                                        (8)
gait clock   sin/cos of gait phase                                     (2)
```

The target is given as a **direction, never an absolute position** — the policy
never sees "the goal is at x = 7", only "it is that way, that far". A policy
trained with targets on the right therefore transfers to targets elsewhere.

**Reward** blends a task objective (progress toward the target, staying alive,
upright, on its feet, holding the goal) with gait-quality terms (reference
tracking, contact schedule, foot separation, step placement, slip and flight
penalties). All weights live in
[`walker_rl/env/config.py`](walker_rl/env/config.py).

**Curriculum**: targets start ~2.5 m away and the maximum distance grows as the
success rate clears 55%, up to 9 m. Targets are behind the agent 15% of the
time. Spawn pose, lean and velocity are randomised every episode.

---

## How the walking gait was actually achieved

Reward shaping alone did **not** work — the agent kept converging on a
"horse gait" with one leg permanently forward. Three structural changes fixed
it, and they are the interesting part of this project:

1. **PD position control instead of a velocity servo.** The original actuator
   set a motor direction at full joint speed and capped its torque, so the only
   expressible commands were slam-forward, limp, and slam-backward. There was
   no way to *hold an angle*, which makes the graded, phased torque a walk
   requires impossible. Switching to PD targets removed a 52% flight phase.

2. **A gait phase clock in the observation.** The policy is a feedforward MLP
   with no memory. Walking is a **limit cycle**; a fixed asymmetric stance is a
   stable **fixed point**, which is far easier to represent and to find. Without
   a phase signal, alternation is not representable at all, so no reward
   weighting can select for it.

3. **Motion imitation against a reference gait.** Hand-written terms each
   described one *symptom* of walking, competed with each other, and still left
   the motion underdetermined — notably they could not express "extend the knee
   during late swing so the foot plants ahead". A reference trajectory
   ([`gait_reference.py`](walker_rl/env/gait_reference.py), following the
   standard human gait cycle) specifies the whole coordinated motion densely, at
   every timestep. It is direction-aware, since a 2D biped cannot turn around
   and must walk backwards to reach a target behind it.

The recurring lesson: **event-based rewards do not work here.** Any term that
only pays once the target behaviour already exists (a swing landing, a leg
crossing) provides no gradient from a policy that doesn't do it yet. Every
mechanism that finally worked was *dense* — it paid for being incrementally
closer.

---

## Testing

```bash
python RL_Agent.py test
```

- **Physics (19)** — asserted against closed-form results: analytic inertia and
  the parallel-axis shift, free fall vs the exact discrete solution, resting
  penetration under the slop limit, a 4-box stack, friction vs frictionless
  momentum, restitution height ≈ e²h, pendulum energy drift < 5% over 10 s,
  joint limits under a 500 N·m motor, momentum conservation, determinism.
- **Environment (42)** — the limp ragdoll must fall; target direction is
  egocentric and mirrors correctly; reward-design invariants (reaching must
  out-earn idling, dragging must lose to stepping, walking must beat hopping,
  the schedule must forbid a flight phase); and regressions for every reward
  exploit found during development.
- **Renderer (2)** — headless draw + record/replay round-trip, verifying the
  per-term reward breakdown sums exactly to the reward.

An RL agent will exploit any bug in the environment, so the environment is
verified before anything is built on it.

---

## License

MIT
