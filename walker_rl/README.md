# Stickman Walker — learning to balance and walk from scratch

A 2D stickman that has to **learn to stand on two legs and walk to a target**,
using a physics engine and an RL agent written from scratch here.

Nothing about balancing is hard-coded. The figure spawns upright and limp; if
the policy outputs zeros it folds up and falls over in about half a second
(there is a test that asserts exactly this). Every bit of postural control has
to be discovered by the agent.

```
walker_rl/
  physics/          hand-written 2D rigid-body engine (no Box2D, no pymunk)
  env/              the stickman, its reward function, its observations
  rl/               PPO, vectorised envs, normalisation, episode recording
  render/           pygame viewer (live + replay)
  train.py          training entry point
  play.py           visualisation entry point
```

Requirements: `numpy`, `torch` (CPU is fine), `pygame`.

---

## Quick start

```bash
# 1. Check the engine and the environment are sound (30 tests, ~20 s)
python -m walker_rl.physics.test_physics
python -m walker_rl.env.test_env

# 2. See the problem before training anything -- a limp ragdoll falls over
python -m walker_rl.play --zero
python -m walker_rl.play --random

# 3. Train
python -m walker_rl.train --run-name walk --envs 10 --steps 20000000

# 4. Watch the current best policy, and replay recorded episodes
python -m walker_rl.play --checkpoint runs/walk/checkpoints/best.pt
python -m walker_rl.play --replay runs/walk/episodes --loop
```

Viewer keys: `space` pause · `n` single step · `r` reset · `+`/`-` speed ·
`[` `]` zoom · `c` contacts · `t` trail · `h` HUD · `esc` quit.

---

## 1. The physics engine (`walker_rl/physics/`)

Written from scratch so that every part of the dynamics the agent has to
master is inspectable. It is an impulse-based sequential solver in the Catto /
Box2D tradition:

| module | what it does |
|---|---|
| `math2d.py` | `Vec2`, `Rot` (stored as sin/cos), `Transform` |
| `shapes.py` | convex polygons (built through a hull, so a bad point set cannot sneak in) and circles, with **exact** analytic mass, centroid and inertia |
| `body.py` | rigid bodies, multi-fixture support, category/mask/group collision filtering |
| `collision.py` | SAT for the separating axis, Sutherland–Hodgman clipping for the contact manifold |
| `contact.py` | persistent contacts, warm-started sequential impulses, Coulomb friction, restitution |
| `joints.py` | revolute joint with an angular motor and hard angle limits |
| `world.py` | broad phase, the step loop, positional correction |

Design points that matter for RL:

- **Local-frame manifolds.** Contacts are stored as a reference face plus
  clipped incident points in body-local coordinates, not as world-space points.
  That lets the position solver *re-evaluate* penetration after bodies move
  during an iteration, which is the difference between a stable stance and a
  jittering one.
- **Warm starting.** Accumulated normal/friction impulses persist across
  frames, keyed by contact feature id. Standing still is cheap and stable.
- **Split position solving.** Penetration is corrected in a separate
  pseudo-velocity pass, so pushing bodies apart never injects real kinetic
  energy that the agent could learn to farm.
- **Modern joint limits.** Separate accumulated lower/upper impulses rather
  than the old three-state machine — much better behaved when a limb is
  slammed into its stop, which happens constantly during exploration.

### It is tested against analytic results

`python -m walker_rl.physics.test_physics` — 19 checks, including: box and
circle inertia vs closed form, the parallel-axis shift, free fall against the
*exact discrete* semi-implicit-Euler solution, resting height and penetration
under the slop limit, a 4-box stack staying stacked, friction stopping a
sliding box while a frictionless one keeps its momentum exactly, restitution
bounce height ≈ e²h, a pendulum conserving energy to <5 % over 10 s, joint
limits holding under a 500 N·m motor driven into the stop, motor torque limits
being respected, momentum conservation in a free collision, and bit-exact
determinism.

An RL agent will happily exploit an engine bug, so the engine is verified
before anything is built on it.

---

## 2. The stickman (`walker_rl/env/`)

11 rigid bodies: torso (with the head as a second fixture on the same body),
two thighs, two shins, two feet, two arms. Roughly 1.65 m and ~72 kg, and the
motor torque limits are in the range of real human joint torques — if the
motors were unrealistically strong the agent would solve the task by brute
force instead of by balancing.

Limbs do not self-collide (one shared negative filter group), which is standard
for 2D walkers; feet and torso still collide with the ground and the platform.

### Actions — 8 motors, all of them needed to balance

| index | joint | limits (rad) | max torque |
|---|---|---|---|
| 0, 3 | hip L/R | −0.80 … 1.30 | 220 N·m |
| 1, 4 | knee L/R | −2.50 … 0.05 | 180 N·m |
| 2, 5 | ankle L/R | −0.60 … 0.90 | 140 N·m |
| 6, 7 | shoulder L/R | −2.40 … 2.40 | 60 N·m |

Each action in `[-1, 1]` drives a motor BipedalWalker-style: the **sign** picks
the direction, the **magnitude** picks how much torque the motor may use. An
action of 0 leaves the joint limp. Every joint also has passive damping.

### Observations — egocentric and target-relative (37 numbers)

```
torso            sin θ, cos θ, ω, vx, vy, height above the ground under it   (6)
per joint × 8    angle normalised to its own limits, angular velocity       (16)
contacts         left foot, right foot, "something that isn't a foot"        (3)
target           direction unit vector in the TORSO's frame,                 (4)
                 world-frame x direction, normalised distance
previous action                                                              (8)
```

**The target is given as a direction, never as an absolute position.** This is
the point of the design: the policy never sees "the goal is at x = 7". It sees
"the goal is that way, that far". A policy trained with targets on the right
therefore transfers to targets on the left, at new distances, in a rearranged
environment — `test_target_direction_is_egocentric` asserts the channel
mirrors correctly, and `EVAL_TARGETS` in `train.py` deliberately includes a
left-hand target that training rarely sees.

### Reward

Per-second rates (multiplied by the control timestep), so retuning the control
frequency does not silently retune the reward:

| term | rate | purpose |
|---|---|---|
| `progress` | 6.0 × (approach speed, capped at 1.4 m/s) | go to the target |
| `alive` | +1.5 /s | do not fall |
| `upright` | +1.2 /s × `exp(−(θ/0.45)²)` | keep the torso vertical |
| `height` | +1.2 /s × gaussian on hip height | stand tall, do not crawl |
| `support` | +0.8 /s | the **feet** carry the body, nothing else touches |
| `two_feet` | +0.3 /s | both feet planted — the balance scaffold |
| `air_time` | 8.0 × (swing − 0.22 s), paid on landing | reward genuine steps |
| `slip` | −0.8 /s × planted-foot sliding speed | stop a leg being dragged |
| `alternate` | +1.2 /s single support, −1.5 /s in flight | walk, don't hop |
| `leg_swap` | +2.5 one-off when the leading hip changes | make the legs trade places |
| `goal` | +6.0 /s in the zone, +25 one-off on arrival | reach and hold |
| `ctrl`, `rate`, `bound`, `limit`, `spin` | small negatives | smooth, cheap, in-range motion |
| `fall` | −6.0 one-off | terminal |

The gait terms (`air_time`, `slip`, `alternate`, `leg_swap`) are gated on
`should_walk` — still having ground to cover — so balancing and holding
position on the platform are untouched by them.

Two things in this table are load-bearing, and both were wrong in the first
draft:

**The idle rate has to lose to the walking rate by a clear margin.** Standing
still already collects alive + upright + height + support every single second
(5.0/s here). Walking has to be worth the risk of falling over, so `progress`
at its cap is 8.4/s — roughly 1.7× the entire idle package, not merely equal to
it. Set these too close and the agent learns a very good *statue*.

**Reaching the goal deliberately does not end the episode.** If arrival
terminated, the agent would forfeit all remaining alive/upright bonus, and
idling at the spawn point for the full 24 s would out-earn walking to the
target (≈120 vs ≈89 with these weights). `test_reaching_pays_better_than_
standing_still` asserts the ordering holds across every distance and walking
speed the curriculum can produce.

### Curriculum

Targets start ~2.5 m away and the maximum distance grows by 0.35 m each time
the success rate over the last 30 episodes clears 55 %, up to 9 m. The floor
follows the ceiling down, so an early curriculum stage really is easier — note
that if `target_min_distance` is allowed to override the schedule the
curriculum silently does nothing at all, which is what
`test_curriculum_actually_shortens_early_targets` guards. Targets are on
the right 85 % of the time and on the left otherwise. Spawn pose, lean, height
and velocity are randomised every episode — and the perturbations are applied
as rigid rotations of kinematic sub-chains about their proximal joint, so every
joint constraint is still exactly satisfied at t=0 and the solver never has to
fight a broken pose.

---

## 3. The agent (`walker_rl/rl/`)

PPO with GAE(λ), separate policy and value trunks (2×256, tanh, orthogonal
init with a small policy-head gain so the initial policy is near-silent rather
than flailing), clipped value loss, gradient clipping, and running
normalisation of both observations and returns. Normaliser statistics are
checkpointed with the weights — a policy restored without its normaliser is a
different policy.

### Exploration vs exploitation

Handled explicitly, with three knobs annealed on the same normalised training
progress, plus a safety brake:

1. **Entropy bonus** `0.006 → 0.0005` — keeps probability mass spread early.
2. **Log-std floor** `−1.0 → −2.2` — a *hard lower bound* on action noise, so
   the policy physically cannot collapse to a deterministic gait before it has
   explored. Released later so it can sharpen and actually exploit what it
   found. This is the knob that matters most here: entropy bonuses alone often
   let the std collapse in the first few hundred updates on locomotion tasks.
3. **Learning rate** annealed to 10 % — late updates refine rather than churn.

All three finish annealing at 70 % of training (`exploration_anneal_fraction`),
leaving a pure exploitation phase at the end.

4. **KL early stopping** (`target_kl = 0.03`) aborts an update whose policy has
   already moved too far. This is what stops one lucky batch from destroying a
   working gait — the classic PPO locomotion failure.

Evaluation always uses the distribution **mean**, so the reported eval number
measures exploitation and is not contaminated by exploration noise.

### Time-limit handling

A 24-second timeout is not a real terminal state. Truncated steps are
bootstrapped with `γ·V(s_final)` before GAE runs; without this the agent learns
that the world ends after 24 s and stops caring about the future near the
horizon.

---

## 4. Saving what the agent did

Two independent things are saved.

**Checkpoints** — `runs/<name>/checkpoints/`: `best.pt` (best deterministic
evaluation), `latest.pt`, and periodic `ckpt_step*.pt`. Each holds the network,
the optimiser, both normalisers, the curriculum state, and the **full env and
PPO config**, so a checkpoint is reproducible on its own terms.

**Episode recordings** — `runs/<name>/episodes/ep_step*_r*.npz`, written
periodically from a deterministic evaluation. Each file is a complete
trajectory:

| field | shape | contents |
|---|---|---|
| `obs` | (T, 37) | what the agent saw |
| `actions` | (T, 8) | **the motor command at every control step** |
| `rewards` | (T,) | total reward per step |
| `components` | (T, 12) | the reward **broken down per term** |
| `poses` | (T, 3·bodies) | exact pose of every rigid body |
| `geometry`, `meta` | json | shapes, target, config, timestamp |

Because the poses are stored, a replay is a literal playback of the recorded
motion, not a re-simulation that might diverge. That means you can answer "what
did it do, and which reward term was it collecting while doing it" for any
point in training:

```python
from walker_rl.rl.recorder import load_episode
ep = load_episode("runs/walk/episodes/ep_step000500000_r+312.npz")
print(ep["actions"].shape, ep["meta"])
# which reward term dominated?
dict(zip(ep["component_names"], ep["components"].sum(0)))
```

Training also appends a row per update to `runs/<name>/metrics.csv` (return,
episode length, success rate, distance travelled, KL, clip fraction, explained
variance, current action std, curriculum distance, eval return).

---

## 5. Extending it

The observation is target-relative by construction, so **changing the
environment does not require retraining from scratch**:

```python
env.reset(target_x=-7.5)   # behind it, further than anything it trained on
```

Retargeting at runtime is just `env.target_x`; the policy sees a new direction
vector and walks the other way. To add terrain, add static bodies in
`StickmanEnv._build` — the observation already carries a "something that isn't
a foot is touching" flag and per-foot contacts, and the physics engine handles
arbitrary convex polygons.

Useful knobs, all in `env/config.py`: body geometry and densities (`BodyPlan`),
motor limits (`MotorSpec`), every reward weight, the curriculum schedule, and
`randomize_friction` / `randomize_mass` for domain randomisation (off by
default — turn them on for a policy that has to survive a changed world).
