"""Sanity checks for the stickman environment.

The important one is `test_zero_action_falls_over`: if a limp ragdoll stayed
upright, the balance problem would be solved by the geometry rather than by the
policy, and the whole exercise would be pointless.
"""

import math
import torch
import time

import numpy as np

from ..physics import Vec2
from .config import StickmanConfig
from .stickman import JOINT_ORDER, StickmanEnv


def _make(**kw):
    cfg = StickmanConfig(**kw)
    return StickmanEnv(cfg, seed=0)


def test_spawn_is_consistent():
    env = _make(curriculum=False)
    env.reset(seed=1, target_x=5.0)
    man = env.man
    for joint in man.joint_list:
        pa = joint.body_a.world_point(joint.local_anchor_a)
        pb = joint.body_b.world_point(joint.local_anchor_b)
        err = (pa - pb).length()
        assert err < 1e-6, "%s spawns with %.6f m of joint error" % (joint.name, err)
    assert 45.0 < man.total_mass < 110.0, \
        "implausible body mass %.1f kg" % man.total_mass


def test_zero_action_falls_over():
    """A limp figure must NOT balance by itself."""
    env = _make(curriculum=False)
    env.reset(seed=2, target_x=6.0)
    zero = np.zeros(env.act_dim, dtype=np.float32)
    steps = 0
    terminated = False
    while steps < 400 and not terminated:
        _, _, terminated, truncated, _ = env.step(zero)
        steps += 1
        if truncated:
            break
    assert terminated, "limp ragdoll never fell -- balance is not being learned"
    assert steps < 240, "limp ragdoll took %d steps to fall (too stable)" % steps


def test_observation_is_finite_and_bounded():
    env = _make()
    obs = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs)), "non-finite observation"
        assert np.abs(obs).max() < 20.0, "observation blew up: %.3f" % np.abs(obs).max()
        assert math.isfinite(r)
        if term or trunc:
            obs = env.reset()


def test_target_direction_is_egocentric():
    """Same pose, mirrored target -> mirrored direction channel."""
    env = _make(curriculum=False, init_joint_noise=0.0, init_angle_noise=0.0,
                init_velocity_noise=0.0, init_height_offset=0.0)
    obs_r = env.reset(seed=4, target_x=6.0)
    obs_l = env.reset(seed=4, target_x=-6.0)
    idx = 6 + 2 * len(JOINT_ORDER) + 3  # dir_local.x
    assert obs_r[idx] > 0.9, "direction to a right-hand target is %.3f" % obs_r[idx]
    assert obs_l[idx] < -0.9, "direction to a left-hand target is %.3f" % obs_l[idx]
    assert abs(obs_r[idx] + obs_l[idx]) < 1e-5, "direction is not symmetric"


def test_reward_prefers_approaching_target():
    env = _make(curriculum=False)
    env.reset(seed=5, target_x=8.0)
    d0 = env._distance_to_target()

    env.prev_distance = d0
    env.man.torso.c.x += 0.02
    _, comp_t = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    env.terminated = False
    env.prev_distance = d0
    env.man.torso.c.x -= 0.04
    _, comp_a = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    assert comp_t["progress"] > 0.0 > comp_a["progress"], \
        "progress reward sign is wrong: %r %r" % (comp_t["progress"], comp_a["progress"])


def test_standing_in_the_goal_zone_registers_success():
    env = _make(curriculum=False, arrival_hold_seconds=0.2)
    env.reset(seed=6, target_x=3.0)
    dx = 3.0
    for body in env.man.parts.values():
        body.set_transform(Vec2(body.xf.p.x + dx,
                                body.xf.p.y + env.cfg.platform_height + 0.02),
                           body.a)
        body.v.set_zero()
        body.w = 0.0
    for joint in env.man.joint_list:
        joint.reset_impulses()
    env.prev_distance = env._distance_to_target()

    zero = np.zeros(env.act_dim, dtype=np.float32)
    success = False
    for _ in range(60):
        _, _, term, trunc, info = env.step(zero)
        if info["success"]:
            success = True
            break
        if term or trunc:
            break
    assert success, "standing in the goal zone never registered as success"


def test_reaching_pays_better_than_standing_still():
    """The decisive incentive check for the reward design.

    Idling at the spawn point collects alive/upright/height/support forever.
    If that out-earns walking to the target and holding it, the optimal policy
    is to never move -- so assert the ordering holds at every target distance
    the curriculum can produce.
    """
    cfg = StickmanConfig()
    horizon = cfg.max_episode_seconds
    stand_rate = (cfg.w_alive + cfg.w_upright + cfg.w_height
                  + cfg.w_foot_support + cfg.w_two_feet)
    hold_rate = stand_rate + cfg.w_goal_hold

    for distance in (cfg.target_min_distance, cfg.target_max_distance):
        for speed in (0.5, 0.8, 1.0):
            v = speed * cfg.progress_speed_cap
            walk_time = distance / v
            if walk_time >= horizon:
                continue
            # Both feet are rarely planted at once while walking.
            walk_rate = (cfg.w_alive + cfg.w_upright + cfg.w_height
                         + cfg.w_foot_support + cfg.w_progress * v)
            reach = (walk_time * walk_rate + cfg.r_reach
                     + (horizon - walk_time) * hold_rate)
            idle = horizon * stand_rate
            assert reach > idle, (
                "standing still (%.1f) beats reaching a %.1f m target at "
                "%.2f m/s (%.1f)" % (idle, distance, v, reach))


def test_curriculum_actually_shortens_early_targets():
    """Regression: `target_min_distance` must not override the curriculum."""
    env = _make()
    env.set_curriculum_distance(2.5)
    early = [abs(env.sample_target()) for _ in range(300)]
    assert max(early) <= 2.5 + 1e-6, \
        "curriculum says 2.5 m but targets reach %.2f m" % max(early)

    env.set_curriculum_distance(7.0)
    late = [abs(env.sample_target()) for _ in range(300)]
    assert max(late) > 5.0, "curriculum never widens (max %.2f m)" % max(late)
    assert min(late) >= min(env.cfg.target_min_distance, 7.0) - 1e-6


def test_dragging_a_foot_is_penalized():
    """Regression for 'one leg walks, the other is towed'.

    A planted foot sliding along the ground must cost something; a planted
    foot that is actually stationary must not.
    """
    env = _make(curriculum=False)
    env.reset(seed=8, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)

    env.man.foot_l.v.x = 0.0
    env.man.foot_r.v.x = 0.0
    _, planted = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False

    env.man.foot_l.v.x = 0.0
    env.man.foot_r.v.x = 1.5          # this one is being scraped along
    _, dragging = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    assert planted["slip"] == 0.0, "a stationary planted foot was charged slip"
    assert dragging["slip"] < 0.0, "dragging a foot cost nothing"
    assert dragging["slip"] < planted["slip"]


def test_dragging_is_not_profitable_versus_stepping():
    """The decisive economics check for locomotion.

    Shuffling forward drags BOTH feet at body speed; a real step has a
    stationary stance foot. Progress pays the same either way, so unless the
    slip penalty outweighs it, dragging is simply the cheaper way to travel --
    which is exactly what happened at w_foot_slip=0.8 vs w_progress=6.0.
    """
    cfg = StickmanConfig()
    assert 2.0 * cfg.w_foot_slip > cfg.w_progress, (
        "dragging both feet nets %+.1f/s of pure profit at 1 m/s "
        "(progress %.1f vs slip %.1f)" % (
            cfg.w_progress - 2.0 * cfg.w_foot_slip,
            cfg.w_progress, 2.0 * cfg.w_foot_slip))


def _walk_lead_alternating(env, cycles=6, swap_every=0.38):
    """Feet trade places repeatedly, as in a walk. Returns summed components."""
    acc = {}
    gap = 0.25
    steps = int(round(swap_every * env.cfg.control_hz))
    for c in range(cycles):
        gap = -gap                                   # the trailing foot passes
        for _ in range(steps):
            env.man.foot_l.c.x = env.man.torso.c.x + gap
            env.man.foot_r.c.x = env.man.torso.c.x - gap
            env.man.torso.c.x += 0.02
            _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
            env.terminated = False
            for k, v in comp.items():
                acc[k] = acc.get(k, 0.0) + v
    return acc


def _walk_lead_frozen(env, cycles=6, swap_every=0.38):
    """One foot stays permanently in front -- the observed failure gait."""
    acc = {}
    steps = int(round(swap_every * env.cfg.control_hz))
    for c in range(cycles):
        for _ in range(steps):
            env.man.foot_l.c.x = env.man.torso.c.x + 0.25   # always leading
            env.man.foot_r.c.x = env.man.torso.c.x - 0.25
            env.man.torso.c.x += 0.02
            _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
            env.terminated = False
            for k, v in comp.items():
                acc[k] = acc.get(k, 0.0) + v
    return acc


def test_lead_foot_must_alternate_and_uses_foot_position():
    """The user's criterion: the back foot must come AHEAD, then roles swap.

    Measured on foot position, not hip angle -- with splayed legs the hips rock
    and score a hip-angle swap while the feet never change order.
    """
    env = _make(curriculum=False)
    env.reset(seed=40, target_x=40.0)
    env._contact_flags = lambda: (True, True, False, False)
    alt = _walk_lead_alternating(env)

    env.reset(seed=40, target_x=40.0)
    env._contact_flags = lambda: (True, True, False, False)
    frozen = _walk_lead_frozen(env)

    assert alt["leg_swap"] > 0.0, "alternating the lead foot paid nothing"
    assert frozen["leg_swap"] == 0.0, "a frozen lead foot was paid a swap"
    assert frozen["stale_lead"] < 0.0, \
        "keeping one foot permanently in front cost nothing"
    assert alt["stale_lead"] > frozen["stale_lead"]


def test_frozen_lead_cannot_be_bought_off_with_progress():
    """It must not be able to trade the penalty away for reaching the target.

    Both gaits below travel the SAME distance, so progress is identical; the
    difference has to come from alternation alone and has to be decisive.
    """
    env = _make(curriculum=False)
    env.reset(seed=41, target_x=40.0)
    env._contact_flags = lambda: (True, True, False, False)
    alt = sum(_walk_lead_alternating(env).values())

    env.reset(seed=41, target_x=40.0)
    env._contact_flags = lambda: (True, True, False, False)
    frozen = sum(_walk_lead_frozen(env).values())

    assert alt > frozen, \
        "the frozen-lead shuffle (%.1f) is not worse than walking (%.1f)" % (
            frozen, alt)
    cfg = env.cfg
    # The penalty saturates, so it bleeds without bound the longer it persists.
    assert cfg.w_stale_lead >= 0.5 * cfg.w_progress, \
        "stale-lead penalty (%.1f) is small next to progress (%.1f) and can " \
        "simply be paid for" % (cfg.w_stale_lead, cfg.w_progress)


def test_gait_reward_fades_to_zero_at_the_goal():
    """Regression for overshooting the target.

    Imitation pays for tracking a WALKING reference. Left on after arrival it
    pays the agent to keep walking through the goal, and overshooting genuinely
    out-earns stopping -- which is exactly what it did.
    """
    env = _make(curriculum=False)
    env.reset(seed=50, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)

    def imitate_at(distance):
        env.man.torso.c.x = 6.0 - distance
        env.prev_distance = env._distance_to_target()
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp["imitate"]

    far = imitate_at(4.0)
    approaching = imitate_at(0.9)
    arrived = imitate_at(0.15)

    assert far > 0.0, "imitation pays nothing while walking"
    assert 0.0 < approaching < far, "the gait reward does not taper on approach"
    assert arrived == 0.0, \
        "the walking reward is still paid at the goal (%.4f) -- it will " \
        "overshoot" % arrived


def test_reference_reverses_for_a_target_behind():
    """A 2D biped cannot turn around, so a target behind needs a BACKWARD walk.

    If the reference only describes walking forwards, then walking the wrong
    way pays 9/s and walking the right way pays little -- so "always walk
    forwards, wherever the target is" becomes the rational policy.
    """
    from .gait_reference import reference_joint_targets
    cfg = StickmanConfig()
    fwd = reference_joint_targets(0.0, cfg, +1.0)
    back = reference_joint_targets(0.0, cfg, -1.0)

    assert fwd[0] > 0.2, "forward reference does not reach the leg out"
    assert back[0] < -0.2, "backward reference does not reverse the hip sweep"
    assert np.sign(fwd[0]) != np.sign(back[0])

    # The knee bends only one way, so it must NOT be mirrored.
    for p in (0.0, 0.3, 0.7, 0.9):
        kf = reference_joint_targets(p, cfg, +1.0)[1]
        kb = reference_joint_targets(p, cfg, -1.0)[1]
        assert kb <= cfg.knee.upper and kb >= cfg.knee.lower, \
            "backward reference drives the knee outside its limits (%.2f)" % kb
        assert np.sign(kf) == np.sign(kb) or abs(kf) < 1e-9, \
            "the knee was mirrored; it only bends backwards"


def test_walking_toward_a_target_behind_is_rewarded():
    """Walking backwards TOWARD the target must beat walking forwards away."""
    env = _make(curriculum=False)
    env.reset(seed=52, target_x=-6.0)      # target behind the spawn
    env._contact_flags = lambda: (True, True, False, False)

    def imitate_moving(dx):
        env.man.torso.c.x = 0.0
        env.prev_distance = env._distance_to_target()
        env.man.torso.c.x = dx
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp["imitate"]

    toward = imitate_moving(-0.025)        # backwards, toward the target
    away = imitate_moving(+0.025)          # forwards, away from it
    assert toward > away, \
        "walking away (%.4f) pays at least as well as walking toward the " \
        "target (%.4f)" % (away, toward)
    assert away == 0.0


def test_walking_away_from_the_goal_earns_no_gait_reward():
    """Regression for overshooting straight through the target.

    Fading the gait reward by distance alone is escapable: walk through the
    goal, the distance grows again, and the full reward comes back. Measured,
    that made leaving (8.3/s) beat staying (6.7/s) and it overshot by 6 m.
    """
    env = _make(curriculum=False)
    env.reset(seed=51, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)

    def imitate_moving(from_x, dx):
        env.man.torso.c.x = from_x
        env.prev_distance = env._distance_to_target()
        env.man.torso.c.x = from_x + dx
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp["imitate"]

    approaching = imitate_moving(2.0, +0.025)    # closing on the target
    retreating = imitate_moving(9.0, +0.025)     # past it, walking away

    assert approaching > 0.0, "approaching the target earns no gait reward"
    assert retreating == 0.0, \
        "walking away from the goal still pays %.4f -- it will overshoot" \
        % retreating


def test_standing_at_the_goal_beats_walking_through_it():
    cfg = StickmanConfig()
    posture = cfg.w_alive + cfg.w_upright + cfg.w_height + cfg.w_foot_support
    standing = posture + cfg.w_goal_hold
    # Past the goal the gait reward is gated off (not approaching) AND moving
    # further away costs progress.
    walking_through = posture + 0.0 - cfg.w_progress * 0.9
    assert standing > walking_through, \
        "walking through the goal (%.2f/s) still beats standing in it (%.2f/s)" \
        % (walking_through, standing)


def test_reference_extends_the_knee_and_plants_the_foot_forward():
    """The motion the hand-written terms could never express.

    The observed failure was: the knee swings forward, then the foot retracts
    instead of planting ahead. The reference must flex the knee to clear the
    ground mid-swing, then EXTEND it through late swing so the foot lands in
    front of the stance foot.
    """
    from .gait_reference import reference_foot_offsets, reference_joint_targets
    cfg = StickmanConfig()

    knee_mid = reference_joint_targets(0.75, cfg)[1]     # mid swing
    knee_late = reference_joint_targets(0.95, cfg)[1]    # late swing
    knee_strike = reference_joint_targets(0.0, cfg)[1]   # heel strike
    assert knee_mid < -0.5, "knee does not flex to clear the ground (%.2f)" % knee_mid
    assert knee_late > knee_mid, "knee does not extend through late swing"
    assert knee_strike > -0.25, "knee is not straight at heel strike (%.2f)" % knee_strike

    # And the foot must actually end up in front at touchdown.
    fl, fr = reference_foot_offsets(0.0, cfg)
    assert fl > fr + 0.2, \
        "at heel strike the swing foot is not planted ahead (%.2f vs %.2f)" % (fl, fr)

    seps = np.array([np.subtract(*reference_foot_offsets(i / 200.0, cfg))
                     for i in range(200)])
    assert int((np.diff(np.sign(seps)) != 0).sum()) >= 2, \
        "the reference legs never cross each other"


def test_imitation_rewards_matching_the_reference():
    """Tracking the reference must beat a frozen horse stance."""
    from .gait_reference import reference_joint_targets
    env = _make(curriculum=False)
    env.reset(seed=34, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)

    class _Fake:
        def __init__(self, a):
            self.joint_angle = a
            self.joint_speed = 0.0

    real = env.man.joint_list

    def score(angles):
        env.man.joint_list = [_Fake(a) for a in angles]
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        env.man.joint_list = real
        return comp["imitate"]

    env.gait_phase = 0.3
    ref = reference_joint_targets(0.3, env.cfg)
    matching = score(ref)
    frozen = score(np.zeros(len(JOINT_ORDER)))     # a rigid neutral stance
    assert matching > frozen, \
        "tracking the reference (%.3f) does not beat a frozen stance (%.3f)" \
        % (matching, frozen)
    assert matching > 0.0


def test_gait_shaping_is_silent_while_falling():
    """Balance has to be learnable before gait is demanded.

    A figure mid-fall cannot follow a contact schedule, so charging it for
    failing one punishes it for a task it cannot yet attempt -- and it never
    survives long enough to learn the balance the gait depends on.
    """
    env = _make(curriculum=False)
    env.reset(seed=33, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)

    def gait_terms():
        env.man.torso.c.x += 0.02
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return abs(comp["periodic"]) + abs(comp["alternate"]) + abs(comp["splay"])

    upright_mag = gait_terms()

    # Now topple it: far off vertical and collapsed.
    env.man.torso.a = 1.0
    env.man.torso.c.y = 0.55 * env.cfg.plan.torso_y
    falling_mag = gait_terms()

    assert falling_mag < 0.25 * max(upright_mag, 1e-9), \
        "gait shaping still active while falling (%.4f vs upright %.4f)" \
        % (falling_mag, upright_mag)


def test_gait_clock_is_visible_and_advances():
    """Without a phase signal a memoryless policy cannot represent alternation.

    A fixed asymmetric stance is a stable fixed point; a walk is a limit cycle.
    The clock is what makes "whose turn it is" a function of the observation.
    """
    env = _make(curriculum=False)
    obs0 = env.reset(seed=30, target_x=6.0)
    assert env.cfg.use_gait_clock
    sin0, cos0 = obs0[-2], obs0[-1]
    assert abs(sin0 ** 2 + cos0 ** 2 - 1.0) < 1e-5, "clock is not a unit phasor"

    zero = np.zeros(env.act_dim, dtype=np.float32)
    quarter = int(round(0.25 / (env.cfg.gait_frequency * env.cfg.control_dt)))
    for _ in range(quarter):
        obs, *_ = env.step(zero)
    assert abs(obs[-2] - sin0) + abs(obs[-1] - cos0) > 0.5, \
        "the gait clock is not advancing"


def test_expected_contacts_alternate_and_forbid_flight():
    """The schedule must be anti-phase, with double support and no flight."""
    env = _make(curriculum=False)
    env.reset(seed=31, target_x=6.0)
    both_down = anti_phase = flight = 0
    n = 200
    for i in range(n):
        env.gait_phase = i / n
        left, right = env._expected_contacts()
        both_down += left and right
        anti_phase += left != right
        flight += (not left) and (not right)
    assert flight == 0, "the target schedule contains a flight phase"
    assert anti_phase > 0, "the target schedule never alternates"
    assert both_down > 0, "the target schedule has no double support"


def test_horse_gait_is_charged_against_the_clock():
    """A permanently-trailing leg must COST, not merely go unrewarded.

    Every earlier term only withheld reward from the horse gait, which left it
    free. Against a schedule it is wrong half of every cycle.
    """
    env = _make(curriculum=False)
    env.reset(seed=32, target_x=6.0)

    def score(contact_fn):
        env.gait_phase = 0.0
        total = 0.0
        for _ in range(120):
            left, right = env._expected_contacts()
            env._contact_flags = contact_fn(left, right)
            env.man.torso.c.x += 0.02
            _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
            env.terminated = False
            total += comp["periodic"]
            env.gait_phase = (env.gait_phase
                              + env.cfg.gait_frequency * env.cfg.control_dt) % 1.0
        return total

    following = score(lambda l, r: (lambda: (l, r, False, False)))
    horse = score(lambda l, r: (lambda: (True, True, False, False)))
    assert following > 0.0, "following the schedule paid nothing"
    assert horse < 0.0, "a permanently-planted horse stance cost nothing"
    assert following > horse


def test_step_must_land_ahead_of_the_stance_foot():
    """The physiotherapist's criterion for a walk, and the one thing
    `leg_swap` does not check.

    A bound scissors the legs too -- it just lands the swing foot BEHIND and
    hops off it again (measured median placement: -0.57 m). Landing ahead must
    pay and landing behind must cost.
    """
    env = _make(curriculum=False)
    env.reset(seed=19, target_x=6.0)

    def land_step(placement):
        """Swing the right foot, then land it `placement` m ahead of the left."""
        env.foot_air_time[:] = 0.0
        env._swing_supported = [True, True]
        env._contact_flags = lambda: (True, False, False, False)
        swing = int(round(env.cfg.min_step_air_time * 2 * env.cfg.control_hz))
        for _ in range(swing):
            env.man.torso.c.x += 0.02
            env._reward(np.zeros(env.act_dim, dtype=np.float32))
            env.terminated = False
        env.man.foot_r.c.x = env.man.foot_l.c.x + placement
        env._contact_flags = lambda: (True, True, False, False)
        env.man.torso.c.x += 0.02
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp["step_length"]

    ahead = land_step(+0.40)
    behind = land_step(-0.40)
    assert ahead > 0.0, "landing the swing foot ahead paid nothing"
    assert behind < 0.0, "landing the swing foot behind cost nothing"
    assert ahead > behind


def test_step_placement_is_scored_during_a_flight_phase():
    """Regression: the cure must not be disabled by the disease.

    Placement was gated behind `_swing_supported`, which a bounding gait never
    satisfies -- so the term written to fix bounding was switched off by
    bounding, and only the already-correct landings ever paid.
    """
    env = _make(curriculum=False)
    env.reset(seed=21, target_x=6.0)
    env.foot_air_time[:] = 0.0
    env._swing_supported = [True, True]

    # A flight phase mid-swing: BOTH feet airborne, voiding _swing_supported.
    env._contact_flags = lambda: (False, False, False, False)
    for _ in range(int(round(env.cfg.min_step_air_time * 2 * env.cfg.control_hz))):
        env.man.torso.c.x += 0.02
        env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
    assert not env._swing_supported[1], "test setup: flight did not void support"

    # The left foot plants first, so there IS a stance leg to step onto.
    env._contact_flags = lambda: (True, False, False, False)
    env.man.torso.c.x += 0.02
    env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False

    # Now the right foot lands well behind it -- this must still be charged,
    # even though the earlier flight phase voided _swing_supported.
    env.man.foot_r.c.x = env.man.foot_l.c.x - 0.40
    env._contact_flags = lambda: (True, True, False, False)
    env.man.torso.c.x += 0.02
    _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    assert comp["air_time"] == 0.0, "a flight-phase swing was paid as a step swing"
    assert comp["step_length"] < 0.0, \
        "landing behind after a flight phase went unscored (%.3f)" % comp["step_length"]


def test_simultaneous_touchdown_is_not_scored_as_a_step():
    """A bound lands both feet at once, and one of them is always in front.

    With asymmetric caps that pays out net-positive, so a bound would harvest
    placement reward. There is no stance foot in a simultaneous touchdown, so
    it is not a step and must score nothing.
    """
    env = _make(curriculum=False)
    env.reset(seed=22, target_x=6.0)
    env.foot_air_time[:] = 0.0
    env._prev_contacts = (False, False)

    env._contact_flags = lambda: (False, False, False, False)
    for _ in range(int(round(env.cfg.min_step_air_time * 2 * env.cfg.control_hz))):
        env.man.torso.c.x += 0.02
        env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False

    env.man.foot_r.c.x = env.man.foot_l.c.x - 0.40   # both land together
    env._contact_flags = lambda: (True, True, False, False)
    env.man.torso.c.x += 0.02
    _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    assert comp["step_length"] == 0.0, \
        "a bound's simultaneous touchdown harvested %+.3f of placement reward" \
        % comp["step_length"]


def test_imperfect_walking_still_beats_standing_still():
    """The placement penalty must not make not-moving the safer option."""
    cfg = StickmanConfig()
    posture = (cfg.w_alive + cfg.w_upright + cfg.w_height
               + cfg.w_foot_support + cfg.w_two_feet)
    landings_per_sec = 2.6                     # measured on the current gait
    worst_placement = cfg.w_step_length * cfg.step_length_penalty_cap * landings_per_sec
    walking = (cfg.w_alive + cfg.w_upright + cfg.w_height + cfg.w_foot_support
               + cfg.w_progress * 0.5 + cfg.w_alternate_support - worst_placement)
    assert walking > posture, (
        "walking badly (%.2f/s) is worse than standing still (%.2f/s): the "
        "placement penalty is too harsh at %.1f landings/s"
        % (walking, posture, landings_per_sec))


def test_contact_chatter_is_not_counted_as_a_step():
    """Feet bouncing in and out of contact must not farm step-length reward."""
    env = _make(curriculum=False)
    env.reset(seed=20, target_x=6.0)
    env.foot_air_time[:] = 0.0
    env._swing_supported = [True, True]

    # One control step of air -- far below min_step_air_time.
    env._contact_flags = lambda: (True, False, False, False)
    env.man.torso.c.x += 0.02
    env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False

    env.man.foot_r.c.x = env.man.foot_l.c.x + 0.40   # would pay if counted
    env._contact_flags = lambda: (True, True, False, False)
    env.man.torso.c.x += 0.02
    _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    assert comp["step_length"] == 0.0, \
        "a single-frame contact bounce was paid %.3f as a step" % comp["step_length"]


def test_gait_bonuses_require_actually_advancing():
    """Guard against 'march on the spot outside the goal zone forever'.

    Dropping the goal-hold reward removes the incentive to sprint, but it
    creates this exploit unless gait credit is tied to closing the distance.
    """
    env = _make(curriculum=False)
    env.reset(seed=17, target_x=6.0)
    env._contact_flags = lambda: (True, False, False, False)

    def gait_reward(delta_x):
        env.prev_distance = env._distance_to_target()
        env.man.torso.c.x += delta_x
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp["alternate"]

    advancing = gait_reward(+0.02)     # closing on the target
    marching = gait_reward(0.0)        # stepping, going nowhere

    assert marching == 0.0, "marching on the spot still paid gait reward"
    assert advancing > 0.0, "advancing earned no gait reward"


def test_arriving_beats_loitering_outside_the_zone():
    """Standing just outside the goal must not beat standing in it."""
    cfg = StickmanConfig()
    posture = (cfg.w_alive + cfg.w_upright + cfg.w_height
               + cfg.w_foot_support + cfg.w_two_feet)
    assert posture + cfg.w_goal_hold > posture, "goal hold is not positive"
    # Gait bonuses are gated on advancing, so loitering earns posture only.
    assert cfg.w_goal_hold > 0.0


def test_excessive_foot_lift_is_penalized():
    """A half-metre knee tuck must cost; a normal step clearance must not."""
    env = _make(curriculum=False)
    env.reset(seed=18, target_x=6.0)
    env._contact_flags = lambda: (True, False, False, False)
    rest = env.cfg.plan.foot[1]

    env.man.foot_r.c.y = rest + 0.10        # a normal walking clearance
    _, normal = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False

    env.man.foot_r.c.y = rest + 0.50        # the observed high-knee bound
    _, bound = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    assert normal["clearance"] == 0.0, \
        "a 10 cm step clearance was charged %.4f" % normal["clearance"]
    assert bound["clearance"] < 0.0, "a 50 cm knee tuck cost nothing"


def test_permanent_splay_is_penalized():
    """A scissored stance must cost; a normal stride must not."""
    env = _make(curriculum=False)
    env.reset(seed=16, target_x=6.0)
    env._contact_flags = lambda: (True, True, False, False)
    mid = env.man.foot_l.c.x

    env.man.foot_r.c.x = mid - 0.30          # feet within a normal stride
    _, normal = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False

    env.man.foot_r.c.x = mid - 0.94          # the observed permanent scissor
    _, splayed = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    assert normal["splay"] == 0.0, "a normal stride was charged %.4f" % normal["splay"]
    assert splayed["splay"] < 0.0, "a 0.94 m permanent scissor cost nothing"


def test_lifting_a_foot_pays_on_landing():
    """A genuine swing must earn what a permanently-grounded foot cannot."""
    env = _make(curriculum=False)
    env.reset(seed=9, target_x=6.0)

    # Foot never leaves the ground -> no swing credit, ever.
    env._contact_flags = lambda: (True, True, False, False)
    for _ in range(20):
        _, grounded = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
    assert grounded["air_time"] == 0.0, "a never-lifted foot earned swing credit"

    # Now let the right foot swing for a realistic step, then land. The torso
    # has to advance too -- gait credit is gated on closing on the target.
    env._contact_flags = lambda: (True, False, False, False)
    swing_steps = int(round(env.cfg.air_time_target * 1.5 * env.cfg.control_hz))
    for _ in range(swing_steps):
        env.man.torso.c.x += 0.02
        env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
    env._contact_flags = lambda: (True, True, False, False)
    env.man.torso.c.x += 0.02
    _, landed = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    assert landed["air_time"] > 0.0, "a real step earned no swing credit"


class _FakeHip:
    """Stand-in exposing just the joint_angle the gait reward reads."""

    def __init__(self, angle):
        self.joint_angle = angle


GAIT_TERMS = ("air_time", "alternate", "leg_swap")


def _score_gait(contact_sequence, lead_sequence, seed=12):
    """Total gait reward for a scripted contact/hip pattern."""
    env = _make(curriculum=False)
    env.reset(seed=seed, target_x=6.0)
    total = 0.0
    for (fl, fr), lead in zip(contact_sequence, lead_sequence):
        env._contact_flags = lambda fl=fl, fr=fr: (fl, fr, False, False)
        env.man.joints["hip_l"] = _FakeHip(0.4 * lead)
        env.man.joints["hip_r"] = _FakeHip(-0.4 * lead)
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        total += sum(comp[k] for k in GAIT_TERMS)
    return total


def _walk_pattern(cycles=4):
    """Alternating single support with brief double support, legs trading."""
    contacts, leads = [], []
    for c in range(cycles):
        for _ in range(9):              # left stance, right swinging
            contacts.append((True, False))
            leads.append(1)
        for _ in range(2):              # double support
            contacts.append((True, True))
            leads.append(1)
        for _ in range(9):              # right stance, left swinging
            contacts.append((False, True))
            leads.append(-1)
        for _ in range(2):
            contacts.append((True, True))
            leads.append(-1)
    return contacts, leads


def _hop_pattern(cycles=4):
    """Both feet together throughout, one hip permanently in front."""
    contacts, leads = [], []
    for c in range(cycles):
        for _ in range(11):             # both planted
            contacts.append((True, True))
            leads.append(1)
        for _ in range(11):             # flight phase
            contacts.append((False, False))
            leads.append(1)
    return contacts, leads


def test_out_of_range_actions_are_charged_for():
    """Regression for policy std inflating into the clipped region.

    a=1.0 and a=5.0 drive the motors identically, so if the overshoot is free
    there is no gradient holding the policy's std inside the action range and
    it drifts until every command saturates.
    """
    env = _make(curriculum=False)
    env.reset(seed=15, target_x=6.0)

    env.step(np.ones(env.act_dim, dtype=np.float32))
    in_range = env.last_components["bound"]
    env.reset(seed=15, target_x=6.0)
    env.step(np.full(env.act_dim, 5.0, dtype=np.float32))
    overshoot = env.last_components["bound"]

    assert in_range == 0.0, "an in-range action was charged %.4f" % in_range
    assert overshoot < 0.0, "a 5x out-of-range action cost nothing"


def test_action_std_ceiling_stays_within_action_range():
    """std above the action range is invisible to the env, so cap it."""
    from ..rl.ppo import PPO, PPOConfig
    cfg = PPOConfig()
    assert cfg.log_std_max <= 0.0, \
        "std ceiling exp(%.2f)=%.2f exceeds the [-1,1] action range" % (
            cfg.log_std_max, math.exp(cfg.log_std_max))

    # A checkpoint saved above the ceiling must be pulled back in, not frozen:
    # clamp() has no gradient once saturated.
    agent = PPO(8, 4, cfg, "cpu")
    with torch.no_grad():
        agent.model.log_std.fill_(1.5)
    agent.load_state_dict({"model": agent.model.state_dict()}, load_optimizer=False)
    assert float(agent.model.log_std.max()) <= cfg.log_std_max + 1e-6, \
        "loaded log_std stayed above the ceiling and cannot be trained down"


def test_walking_beats_bunny_hopping():
    """Regression for the hop attractor.

    A hop collects a naive swing bonus on BOTH feet every cycle while needing
    no coordination at all. Alternating steps must score strictly higher.
    """
    walk = _score_gait(*_walk_pattern())
    hop = _score_gait(*_hop_pattern())
    assert walk > hop, \
        "bunny hopping (%.2f) scores at least as well as walking (%.2f)" % (hop, walk)
    assert hop <= 0.0, \
        "hopping still earns a positive gait reward (%.2f)" % hop


def test_hopping_earns_no_swing_credit():
    """Both feet airborne at once is a hop, and must not pay as two swings."""
    env = _make(curriculum=False)
    env.reset(seed=13, target_x=6.0)

    env._contact_flags = lambda: (False, False)[0:2] + (False, False)
    for _ in range(10):                 # both feet in flight together
        env._contact_flags = lambda: (False, False, False, False)
        env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False

    env._contact_flags = lambda: (True, True, False, False)   # both land
    _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    assert comp["air_time"] == 0.0, \
        "a hop was paid %.3f of swing credit" % comp["air_time"]


def test_leg_swap_only_pays_when_the_lead_changes():
    """Hip angle is not the criterion -- which foot is in FRONT is.

    A splayed stance can rock its hips back and forth without the feet ever
    changing order, and that used to score a swap.
    """
    env = _make(curriculum=False)
    env.reset(seed=14, target_x=6.0)
    env._contact_flags = lambda: (True, False, False, False)

    def step_with_lead(lead):
        env.man.foot_l.c.x = env.man.torso.c.x + 0.25 * lead
        env.man.foot_r.c.x = env.man.torso.c.x - 0.25 * lead
        env.man.torso.c.x += 0.02      # gait credit requires advancing
        _, comp = env._reward(np.zeros(env.act_dim, dtype=np.float32))
        env.terminated = False
        return comp

    step_with_lead(1)
    held = step_with_lead(1)
    swapped = step_with_lead(-1)
    assert held["leg_swap"] == 0.0, "holding one foot in front paid a swap"
    assert swapped["leg_swap"] > 0.0, "the feet trading places paid nothing"

    # Rocking the hips without the feet changing order must NOT pay.
    env.man.joints["hip_l"] = _FakeHip(0.6)
    env.man.joints["hip_r"] = _FakeHip(-0.6)
    rocked = step_with_lead(-1)        # same lead foot as the previous step
    assert rocked["leg_swap"] == 0.0, "hip rocking scored a swap"


def test_goal_requires_getting_off_the_ground():
    """Regression for the straddle exploit.

    Standing in the goal zone with a trailing leg still on the ground must not
    pay the goal, otherwise that leg never has to come up onto the platform.
    """
    env = _make(curriculum=False)
    env.reset(seed=10, target_x=4.0)
    env.man.torso.set_transform(Vec2(4.0, env.man.torso.xf.p.y), 0.0)
    env._contact_flags = lambda: (True, True, False, False)

    # Front foot on the platform, trailing foot still scraping the ground.
    env._platform_support = lambda: (1, True)
    _, straddling = env._reward(np.zeros(env.act_dim, dtype=np.float32))
    env.terminated = False
    env.hold_timer = 0.0

    # Both feet carried by the platform.
    env._platform_support = lambda: (2, False)
    _, on_platform = env._reward(np.zeros(env.act_dim, dtype=np.float32))

    assert straddling["goal"] == 0.0, \
        "straddling the platform edge still paid the goal reward"
    assert on_platform["goal"] > 0.0, \
        "standing properly on the platform paid nothing"


def test_success_does_not_end_the_episode():
    cfg = StickmanConfig()
    assert not cfg.terminate_on_success, \
        "terminating on success forfeits the remaining alive bonus"


def test_episode_terminates_and_resets_cleanly():
    env = _make()
    for ep in range(5):
        env.reset(seed=100 + ep)
        rng = np.random.default_rng(ep)
        done = False
        n = 0
        info = {}
        while not done and n < env.cfg.max_episode_steps + 5:
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
            n += 1
        assert done, "episode %d never ended" % ep
        assert "episode" in info


def test_determinism_of_env():
    def run():
        env = _make(curriculum=False)
        env.reset(seed=11, target_x=5.0)
        rng = np.random.default_rng(3)
        total = 0.0
        for _ in range(120):
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
            _, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        return round(total, 9)
    assert run() == run(), "environment is not deterministic given a seed"


def benchmark(steps=2000):
    env = _make()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 0
    for _ in range(steps):
        a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        _, _, term, trunc, _ = env.step(a)
        n += 1
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    print("  %.0f control steps/s  (%.2f ms per step, %d physics substeps)"
          % (n / dt, 1000 * dt / n, env.cfg.substeps))
    print("  ~%.1f simulated seconds per wall second"
          % (n * env.cfg.control_dt / dt))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %-44s %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %-44s %r" % (fn.__name__, exc))
        else:
            print("ok    %s" % fn.__name__)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    print("\nthroughput:")
    benchmark()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
