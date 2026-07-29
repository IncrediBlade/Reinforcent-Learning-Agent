"""A reference human walking trajectory, as a function of gait phase.

This exists because specifying a walk through hand-written reward terms did not
work. Each term captured one symptom -- swing time, foot placement, leg
crossing, flight, splay -- and they competed with each other while still
leaving the actual motion underdetermined. The failure the terms could never
express was the important one: the knee would swing forward and then the foot
would retract instead of planting ahead, because nothing said what the leg
should be doing at each *instant* of the swing, only what should be true at
touchdown.

A reference trajectory says all of it at once, densely, at every timestep. The
angles below follow the standard human gait cycle (Perry, *Gait Analysis*),
with phase 0 at heel strike:

    0.00-0.10  loading      hip flexed forward, knee near straight, weight on
    0.10-0.30  midstance    hip extends as the body passes over the foot
    0.30-0.50  terminal     hip fully extended behind, ankle pushes off
    0.50-0.60  pre-swing    knee breaks and flexes fast, foot leaves
    0.60-0.75  early swing  knee at maximum flexion so the leg can pass the
                            stance leg without catching the ground
    0.75-0.90  mid swing    hip drives forward, knee begins extending
    0.90-1.00  late swing   KNEE EXTENDS to reach the foot out in front,
                            ready for the next heel strike

That final quarter is the part the agent never discovered on its own, and the
part the user observed missing.
"""

import math

import numpy as np


def leg_reference(p, cfg, direction=1.0):
    """Reference (hip, knee, ankle) for one leg at its own phase p in [0, 1).

    `direction` is +1 to walk toward +x and -1 toward -x. A 2D biped cannot
    turn around, so a target behind it has to be reached by walking backwards,
    and a reference that only describes forward walking would pay nothing for
    doing the right thing -- making "walk forwards regardless of where the
    target is" the rational policy.

    Backwards walking is NOT the mirror image: mirroring would negate the knee,
    and the knee only bends one way. Only the hip sweep reverses, so the leg
    reaches BACK to plant and the body then travels back over it, while the
    knee still flexes to clear the ground during swing.
    """
    duty = cfg.gait_duty

    # Hip sweeps forward-to-back through stance and back to front through
    # swing: a single smooth oscillation, reaching out at heel strike.
    hip = direction * (cfg.ref_hip_amp * math.cos(2.0 * math.pi * p)
                       + cfg.ref_hip_bias)

    if p < duty:
        # Stance: the leg carries the body, so it stays near straight. A small
        # flexion early absorbs the landing.
        s = p / max(1e-6, duty)
        knee = cfg.ref_knee_stance - cfg.ref_knee_dip * math.sin(math.pi * s)
    else:
        # Swing: flex hard early to shorten the leg and clear the ground, then
        # EXTEND through the second half so the foot reaches out in front.
        # sin(pi*s) peaks at mid-swing and returns to zero at touchdown, which
        # is what plants the foot forward instead of letting it swing back.
        s = (p - duty) / max(1e-6, 1.0 - duty)
        knee = -cfg.ref_knee_flex * math.sin(math.pi * s ** cfg.ref_swing_skew)

    # Keep the sole roughly level with the ground through stance.
    ankle = -cfg.ref_ankle_gain * (hip + knee)
    return hip, knee, ankle


def reference_joint_targets(phase, cfg, direction=1.0):
    """Reference angle for all 8 joints at `phase`, in JOINT_ORDER."""
    pl = phase % 1.0
    pr = (phase + 0.5) % 1.0
    hl, kl, al = leg_reference(pl, cfg, direction)
    hr, kr, ar = leg_reference(pr, cfg, direction)
    # Arms counter-swing against the opposite leg, as a human's do.
    return np.array([hl, kl, al, hr, kr, ar,
                     -direction * cfg.ref_arm_amp * math.cos(2.0 * math.pi * pl),
                     -direction * cfg.ref_arm_amp * math.cos(2.0 * math.pi * pr)],
                    dtype=np.float64)


def reference_foot_offsets(phase, cfg):
    """Foot offsets from the hip, in the DIRECTION-OF-TRAVEL frame.

    Deliberately always computed as if walking forwards: the caller compares
    these against a foot separation that has already been multiplied by the
    travel direction, so applying the direction here too would cancel out.
    """
    plan = cfg.plan
    thigh = 2.0 * plan.thigh[1]
    shin = 2.0 * plan.shin[1]
    out = []
    q = reference_joint_targets(phase, cfg)
    for hip, knee in ((q[0], q[1]), (q[3], q[4])):
        # Planar 2-link chain hanging from the hip.
        x = thigh * math.sin(hip) + shin * math.sin(hip + knee)
        out.append(x)
    return out[0], out[1]
