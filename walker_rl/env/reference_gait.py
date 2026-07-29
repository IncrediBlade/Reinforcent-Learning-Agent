"""A hand-written walking gait, used as a control experiment and a reference.

The point of this module is to answer a question that reward shaping cannot:
*is a human-style walk physically achievable by this body at all?* If a
deliberately-scripted gait cannot walk here, then no reward function will ever
produce one and the body, motors or physics are at fault. If it can, the task
is sound and the problem is purely one of learning.

The gait is an open-loop central-pattern-generator: joint targets are a
function of phase only, with no feedback except a PD controller tracking those
targets through the same motors the RL agent uses. Nothing here is available to
the agent -- it is a yardstick, and optionally an imitation reference.

    python -m walker_rl.env.reference_gait          # search + report
"""

import math

import numpy as np

from .config import StickmanConfig
from .stickman import JOINT_ORDER, StickmanEnv


class GaitParams:
    __slots__ = ("frequency", "hip_amp", "knee_flex", "duty", "hip_bias",
                 "ankle_gain", "arm_amp", "kp", "kd", "lean",
                 "kp_pitch", "kd_pitch", "k_place", "target_speed")

    def __init__(self, frequency=1.3, hip_amp=0.50, knee_flex=1.10, duty=0.60,
                 hip_bias=0.05, ankle_gain=0.5, arm_amp=0.35, kp=6.0, kd=0.35,
                 lean=0.0, kp_pitch=0.0, kd_pitch=0.0, k_place=0.0,
                 target_speed=0.6):
        # Balance feedback. Open-loop walking is unstable for any biped, so
        # without these the experiment tests the controller, not the body.
        self.kp_pitch = kp_pitch      # torso pitch -> hip bias
        self.kd_pitch = kd_pitch      # torso pitch rate -> hip bias
        self.k_place = k_place        # Raibert-style velocity foot placement
        self.target_speed = target_speed
        self.frequency = frequency
        self.hip_amp = hip_amp
        self.knee_flex = knee_flex
        self.duty = duty
        self.hip_bias = hip_bias
        self.ankle_gain = ankle_gain
        self.arm_amp = arm_amp
        self.kp = kp
        self.kd = kd
        self.lean = lean

    def __repr__(self):
        return ("f=%.2f hip=%.2f knee=%.2f duty=%.2f bias=%.2f kp=%.1f"
                % (self.frequency, self.hip_amp, self.knee_flex, self.duty,
                   self.hip_bias, self.kp))


def leg_targets(p, g):
    """Target (hip, knee, ankle) for one leg at its own phase p in [0, 1).

    p = 0 is the start of stance, with the leg reaching forward at heel strike;
    the hip then rotates back as the body passes over it. Swing begins at
    `duty`, where the knee flexes to shorten the leg so it can pass the stance
    leg without catching the ground -- the flexion is what actually lets the
    trailing leg come through.
    """
    hip = g.hip_amp * math.cos(2.0 * math.pi * p) + g.hip_bias
    if p < g.duty:
        knee = -0.08                      # near-straight, carrying the body
    else:
        s = (p - g.duty) / max(1e-6, 1.0 - g.duty)
        knee = -g.knee_flex * math.sin(math.pi * s)
    ankle = -g.ankle_gain * (hip + knee)  # keep the sole roughly level
    return hip, knee, ankle


def reference_pose(phase, g):
    """Target angle for all 8 joints, in JOINT_ORDER."""
    pl = phase % 1.0
    pr = (phase + 0.5) % 1.0
    hl, kl, al = leg_targets(pl, g)
    hr, kr, ar = leg_targets(pr, g)
    # Arms counter-swing against the opposite leg, as a human's do.
    return np.array([hl, kl, al, hr, kr, ar,
                     -g.arm_amp * math.cos(2.0 * math.pi * pl),
                     -g.arm_amp * math.cos(2.0 * math.pi * pr)],
                    dtype=np.float32)


class ScriptedWalker:
    """Drives the env's motors to follow `reference_pose` with a PD law."""

    def __init__(self, env, params=None):
        self.env = env
        self.g = params or GaitParams()
        self.phase = 0.0

    def reset(self, phase=0.0):
        self.phase = phase

    def action(self):
        env = self.env
        g = self.g
        target = reference_pose(self.phase, g)

        # --- balance feedback ----------------------------------------------
        torso = env.man.torso
        # Positive torso angle is a backward lean, so correcting it means
        # driving the hips the other way.
        pitch_bias = -(g.kp_pitch * torso.a + g.kd_pitch * torso.w)
        target[0] += pitch_bias
        target[3] += pitch_bias

        # Raibert stepping: the swing foot reaches further ahead the faster the
        # body is travelling, which is what actually arrests a fall forward.
        place = g.k_place * (torso.v.x - g.target_speed)
        if (self.phase % 1.0) >= g.duty:
            target[0] += place
        if ((self.phase + 0.5) % 1.0) >= g.duty:
            target[3] += place

        if env.cfg.actuation == "pd":
            # The action IS a joint-angle target here, so the gait's targets go
            # straight through -- no outer P-loop, and no bang-bang.
            action = np.clip(target / env.cfg.pd_action_scale, -1.0, 1.0)
        else:
            action = np.zeros(len(JOINT_ORDER), dtype=np.float32)
            for i, joint in enumerate(env.man.joint_list):
                err = target[i] - joint.joint_angle
                action[i] = np.clip(g.kp * err - g.kd * joint.joint_speed, -1.0, 1.0)
        self.phase = (self.phase + g.frequency * env.cfg.control_dt) % 1.0
        return action.astype(np.float32)


def evaluate(params, seconds=12.0, target_x=20.0, seed=0, cfg=None):
    """Run the scripted gait and measure what it actually achieves."""
    cfg = cfg or StickmanConfig()
    cfg.curriculum = False
    cfg.max_episode_seconds = seconds
    # The scripted gait is not trying to solve the RL task; give it room.
    env = StickmanEnv(cfg, seed=seed)
    env.reset(seed=seed, target_x=target_x)
    walker = ScriptedWalker(env, params)

    x0 = env.man.torso.c.x
    contacts, feet_x, feet_y = [], [], []
    steps = int(seconds * cfg.control_hz)
    fell = False
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(walker.action())
        fl, fr, _, _ = env._contact_flags()
        contacts.append((fl, fr))
        feet_x.append((env.man.foot_l.c.x, env.man.foot_r.c.x))
        feet_y.append((env.man.foot_l.c.y, env.man.foot_r.c.y))
        if terminated:
            fell = True
            break
        if truncated:
            break

    n = len(contacts)
    C = np.array(contacts)
    FX = np.array(feet_x)
    dist = env.man.torso.c.x - x0
    duration = n / cfg.control_hz
    sep = FX[:, 0] - FX[:, 1]
    crossings = int((np.diff(np.sign(sep)) != 0).sum()) if n > 1 else 0
    fl, fr = (C[:, 0], C[:, 1]) if n else (np.array([]), np.array([]))
    return {
        "fell": fell,
        "survived": duration,
        "distance": float(dist),
        "speed": float(dist / duration) if duration > 0 else 0.0,
        "crossings": crossings,
        "flight": float(((~fl) & (~fr)).mean()) if n else 1.0,
        "single": float((fl ^ fr).mean()) if n else 0.0,
        "steps": n,
    }


def search(seconds=10.0, verbose=True):
    """Coarse sweep for a parameter set that actually walks."""
    best = None
    results = []
    for freq in (1.0, 1.3, 1.6):
        for hip in (0.35, 0.50, 0.65):
            for knee in (0.8, 1.2, 1.6):
                for kp in (4.0, 8.0):
                    g = GaitParams(frequency=freq, hip_amp=hip, knee_flex=knee,
                                   kp=kp)
                    r = evaluate(g, seconds=seconds)
                    r["params"] = g
                    results.append(r)
                    # Rank by distance actually covered without falling.
                    score = r["distance"] - (5.0 if r["fell"] else 0.0)
                    if best is None or score > best[0]:
                        best = (score, r)
                        if verbose:
                            print("  %-52s -> %5.2f m  %s crossings=%d"
                                  % (g, r["distance"],
                                     "FELL at %.1fs" % r["survived"] if r["fell"]
                                     else "upright %.1fs" % r["survived"],
                                     r["crossings"]))
    return best[1], results


def search_with_feedback(seconds=10.0, verbose=True):
    """Second stage: the same gait plus the balance feedback any biped needs."""
    best = None
    results = []
    for kp_pitch in (1.0, 2.5, 4.0):
        for kd_pitch in (0.1, 0.4):
            for k_place in (0.0, 0.2, 0.4):
                for hip in (0.40, 0.55):
                    for freq in (1.1, 1.4):
                        g = GaitParams(frequency=freq, hip_amp=hip,
                                       knee_flex=1.1, kp=8.0, kd=0.3,
                                       kp_pitch=kp_pitch, kd_pitch=kd_pitch,
                                       k_place=k_place)
                        r = evaluate(g, seconds=seconds)
                        r["params"] = g
                        results.append(r)
                        score = r["distance"] - (5.0 if r["fell"] else 0.0)
                        if best is None or score > best[0]:
                            best = (score, r)
                            if verbose:
                                print("  pitch=%.1f/%.1f place=%.1f %-28s "
                                      "-> %5.2f m  %s cross=%d"
                                      % (kp_pitch, kd_pitch, k_place, g,
                                         r["distance"],
                                         "FELL %.1fs" % r["survived"] if r["fell"]
                                         else "UPRIGHT %.1fs" % r["survived"],
                                         r["crossings"]))
    return best[1], results


def main():
    print("Control experiment: can this body walk when told exactly how?\n")
    print("stage 1 - open loop (no balance feedback):")
    best, results = search()
    print("  %d of %d stayed upright" % (len([r for r in results if not r['fell']]),
                                         len(results)))

    print("\nstage 2 - with torso-pitch feedback and Raibert foot placement:")
    best_fb, results_fb = search_with_feedback()
    upright_fb = [r for r in results_fb if not r["fell"]]
    print("\n%d of %d stayed upright for the full run" % (len(upright_fb),
                                                          len(results_fb)))
    print("\nbest: %r" % (best_fb["params"],))
    for k in ("distance", "speed", "survived", "crossings", "single", "flight"):
        print("   %-10s %s" % (k, round(best_fb[k], 3)))

    best = best_fb
    results = results_fb

    # A scripted CPG with two hand-tuned gains failing to balance says little
    # about the body -- balancing is the hard part and is what RL is for. What
    # the experiment can settle is whether the ACTUATION permits a gait at all:
    # alternating leg crossings with no flight phase means the mechanism works.
    if best["crossings"] >= 2 and best["flight"] < 0.05:
        print("\nVERDICT: the actuation supports a real gait -- %d leg crossings"
              " with no flight phase." % best["crossings"])
        print("         It falls after %.1fs, but that is hand-tuned balance"
              " gains, not the body." % best["survived"])
    elif best["flight"] > 0.2:
        print("\nVERDICT: the gait BOUNCES (%.0f%% flight). The actuation cannot"
              " hold a joint angle;" % (100 * best["flight"]))
        print("         it is bang-bang, so no smooth gait is expressible.")
    elif best["crossings"] < 2:
        print("\nVERDICT: it moves, but the legs never pass each other.")
    else:
        print("\nVERDICT: the body CAN walk. %.2f m at %.2f m/s with %d leg"
              " crossings." % (best["distance"], best["speed"], best["crossings"]))
        print("         The task is sound -- the failure is in learning, and")
        print("         this gait can serve as an imitation reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
