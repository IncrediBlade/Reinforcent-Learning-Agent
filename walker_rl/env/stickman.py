"""The stickman balance-and-walk environment.

Nothing about balancing is scripted. The figure is spawned standing upright and
completely limp -- if the policy outputs zeros it folds up and falls over. Every
bit of postural control has to come out of the 8 motors.

Action space  (8,)  in [-1, 1]
    0 hip_l      1 knee_l      2 ankle_l
    3 hip_r      4 knee_r      5 ankle_r
    6 shoulder_l 7 shoulder_r
Each action is a JOINT ANGLE TARGET, as an offset from the neutral standing
pose, tracked by a PD controller inside the physics substep loop.

The original design used a BipedalWalker-style velocity servo (sign picks the
direction at full joint speed, magnitude caps the torque). That turned out to
be the single biggest obstacle in this project: it cannot hold an angle, so the
only expressible commands are slam-forward, limp and slam-backward. 82% of
learned actions saturated, the gait bounced with a 52% flight phase, and no
reward shaping could produce the graded, phased torque a walk needs. Set
`actuation="motor"` to get the old behaviour back.

Observation is egocentric and target-relative, so a policy trained with the
target on the right generalises to a target placed anywhere.
"""

import math
from collections import deque

import numpy as np

from ..physics import (DYNAMIC, STATIC, Circle, Filter, Polygon, Vec2, World,
                       clamp)
from .config import StickmanConfig
from .gait_reference import reference_foot_offsets, reference_joint_targets

JOINT_ORDER = ("hip_l", "knee_l", "ankle_l",
               "hip_r", "knee_r", "ankle_r",
               "shoulder_l", "shoulder_r")

LEG_PARTS = ("thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r")
RAGDOLL_GROUP = -7


class Stickman:
    """Builds the articulated figure inside a physics world."""

    def __init__(self, world, cfg, x=0.0, ground_y=0.0, mass_scale=1.0):
        self.cfg = cfg
        self.world = world
        plan = cfg.plan
        self.plan = plan
        f = Filter(group=RAGDOLL_GROUP)
        ms = mass_scale

        def box_body(name, half, pos, density, friction):
            body = world.create_body(DYNAMIC, Vec2(x + pos[0], ground_y + pos[1]),
                                     0.0, name)
            body.add_fixture(Polygon.box(half[0], half[1]), density=density * ms,
                             friction=friction, restitution=plan.restitution,
                             filter=Filter(group=RAGDOLL_GROUP))
            body.linear_damping = cfg.linear_damping
            body.angular_damping = cfg.angular_damping
            return body

        self.parts = {}
        # --- torso + head (one body: the neck is not actuated) -------------
        torso = world.create_body(DYNAMIC, Vec2(x, ground_y + plan.torso_y), 0.0, "torso")
        torso.add_fixture(Polygon.box(plan.torso[0], plan.torso[1]),
                          density=plan.density_torso * ms,
                          friction=plan.friction_body,
                          filter=Filter(group=RAGDOLL_GROUP))
        torso.add_fixture(Circle(plan.head_radius, Vec2(0.0, plan.head_offset)),
                          density=plan.density_head * ms,
                          friction=plan.friction_body,
                          filter=Filter(group=RAGDOLL_GROUP))
        torso.linear_damping = cfg.linear_damping
        torso.angular_damping = cfg.angular_damping
        self.parts["torso"] = torso

        for side in ("l", "r"):
            self.parts["thigh_" + side] = box_body(
                "thigh_" + side, plan.thigh,
                (0.0, plan.knee_y + plan.thigh[1]),
                plan.density_thigh, plan.friction_body)
            self.parts["shin_" + side] = box_body(
                "shin_" + side, plan.shin,
                (0.0, plan.ankle_y + plan.shin[1]),
                plan.density_shin, plan.friction_body)
            self.parts["foot_" + side] = box_body(
                "foot_" + side, plan.foot,
                (plan.foot_forward_shift, plan.foot[1]),
                plan.density_foot, plan.friction_foot)
            self.parts["arm_" + side] = box_body(
                "arm_" + side, plan.arm,
                (0.0, plan.shoulder_y - plan.arm[1]),
                plan.density_arm, plan.friction_body)

        # --- joints --------------------------------------------------------
        specs = {"hip": cfg.hip, "knee": cfg.knee, "ankle": cfg.ankle,
                 "shoulder": cfg.shoulder}
        anchors = {"hip": plan.hip_y, "knee": plan.knee_y,
                   "ankle": plan.ankle_y, "shoulder": plan.shoulder_y}
        parents = {"hip": "torso", "knee": "thigh", "ankle": "shin",
                   "shoulder": "torso"}
        children = {"hip": "thigh", "knee": "shin", "ankle": "foot",
                    "shoulder": "arm"}

        self.joints = {}
        for name in JOINT_ORDER:
            kind, side = name.rsplit("_", 1)
            spec = specs[kind]
            parent = self.parts[parents[kind]] if parents[kind] == "torso" \
                else self.parts[parents[kind] + "_" + side]
            child = self.parts[children[kind] + "_" + side]
            anchor = Vec2(x, ground_y + anchors[kind])
            joint = world.create_revolute_joint(
                parent, child, anchor, name=name,
                lower_angle=spec.lower, upper_angle=spec.upper,
                enable_motor=True, motor_speed=0.0, max_motor_torque=0.0)
            self.joints[name] = joint
        world.mark_dirty()

        self.joint_list = [self.joints[n] for n in JOINT_ORDER]
        self.motor_specs = [specs[n.rsplit("_", 1)[0]] for n in JOINT_ORDER]
        # PD actuation drives the joints with explicit torques, so the built-in
        # velocity servo must be switched off or the two fight each other.
        self._pd_targets = [0.0] * len(JOINT_ORDER)
        if cfg.actuation == "pd":
            for joint in self.joint_list:
                joint.enable_motor = False
                joint.max_motor_torque = 0.0
        self.torso = torso
        self.foot_l = self.parts["foot_l"]
        self.foot_r = self.parts["foot_r"]
        self.total_mass = sum(b.mass for b in self.parts.values())

    # -- control ------------------------------------------------------------
    def apply_action(self, action):
        if self.cfg.actuation == "pd":
            # Target is an offset from the NEUTRAL standing pose (all joints at
            # zero), not the middle of each joint's range -- mid-range would put
            # the knee at -1.2 rad, so a=0 would command a deep squat and the
            # untrained policy would start by collapsing.
            scale = self.cfg.pd_action_scale
            for i, (spec, a) in enumerate(zip(self.motor_specs, action)):
                a = clamp(float(a), -1.0, 1.0)
                self._pd_targets[i] = clamp(a * scale, spec.lower, spec.upper)
            return
        for joint, spec, a in zip(self.joint_list, self.motor_specs, action):
            a = clamp(float(a), -1.0, 1.0)
            joint.motor_speed = math.copysign(spec.max_speed, a) if a != 0.0 else 0.0
            joint.max_motor_torque = abs(a) * spec.max_torque

    def apply_joint_torques(self):
        """Called every physics substep, before the solver.

        Under PD actuation the control torque is recomputed here rather than
        once per control step, so the servo sees the current angle at each
        substep -- integrating a stale torque across 6 substeps is what makes
        a stiff PD controller oscillate.
        """
        cfg = self.cfg
        if cfg.actuation == "pd":
            for joint, spec, target in zip(self.joint_list, self.motor_specs,
                                           self._pd_targets):
                tau = (cfg.pd_kp_scale * spec.max_torque
                       * (target - joint.joint_angle)
                       - cfg.pd_kd_scale * spec.max_torque * joint.joint_speed)
                tau = clamp(tau, -spec.max_torque, spec.max_torque)
                joint.body_a.apply_torque(-tau)
                joint.body_b.apply_torque(tau)

        c = cfg.joint_damping
        if c <= 0.0:
            return
        for joint in self.joint_list:
            tau = -c * joint.joint_speed
            joint.body_a.apply_torque(-tau)
            joint.body_b.apply_torque(tau)

    # Kept for backwards compatibility with older call sites.
    apply_passive_damping = apply_joint_torques

    def _rotate_chain(self, parts, pivot, angle):
        """Rigidly rotate a set of parts about a world pivot, joints stay intact."""
        ca, sa = math.cos(angle), math.sin(angle)
        for body in parts:
            rel = body.xf.p - pivot
            new_p = Vec2(pivot.x + ca * rel.x - sa * rel.y,
                         pivot.y + sa * rel.x + ca * rel.y)
            body.set_transform(new_p, body.a + angle)

    def set_pose_noise(self, rng, cfg):
        """Perturb the spawn pose so the policy never memorises one start state.

        Perturbations are applied as rigid rotations of kinematic sub-chains
        about their proximal joint, so every joint constraint is still exactly
        satisfied at t=0 and the solver does not have to fight a broken pose.
        """
        p = self.parts
        chains = [
            ([p["thigh_l"], p["shin_l"], p["foot_l"]], self.joints["hip_l"]),
            ([p["thigh_r"], p["shin_r"], p["foot_r"]], self.joints["hip_r"]),
            ([p["shin_l"], p["foot_l"]], self.joints["knee_l"]),
            ([p["shin_r"], p["foot_r"]], self.joints["knee_r"]),
            ([p["foot_l"]], self.joints["ankle_l"]),
            ([p["foot_r"]], self.joints["ankle_r"]),
            ([p["arm_l"]], self.joints["shoulder_l"]),
            ([p["arm_r"]], self.joints["shoulder_r"]),
        ]
        for parts, joint in chains:
            pivot = joint.anchor_world
            d = float(rng.normal(0.0, cfg.init_joint_noise))
            spec_lo, spec_hi = joint.lower_angle, joint.upper_angle
            d = clamp(d, spec_lo - joint.joint_angle + 0.01,
                      spec_hi - joint.joint_angle - 0.01)
            self._rotate_chain(parts, pivot, d)

        # Lean the whole figure by rotating everything about the foot contact.
        da = float(rng.normal(0.0, cfg.init_angle_noise))
        pivot = Vec2(self.torso.xf.p.x, 0.0)
        self._rotate_chain(list(p.values()), pivot, da)

        dy = float(rng.uniform(0.0, cfg.init_height_offset))
        for body in p.values():
            body.set_transform(Vec2(body.xf.p.x, body.xf.p.y + dy), body.a)
            body.v = Vec2(float(rng.normal(0.0, cfg.init_velocity_noise)),
                          float(rng.normal(0.0, cfg.init_velocity_noise * 0.5)))
            body.w = float(rng.normal(0.0, cfg.init_velocity_noise))

        for joint in self.joint_list:
            joint.reset_impulses()


class StickmanEnv:
    """Gym-style environment (no gym dependency)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg=None, seed=None, render_mode=None):
        self.cfg = cfg if cfg is not None else StickmanConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.act_dim = len(JOINT_ORDER)
        self.ground_y = 0.0

        self._recent_success = deque(maxlen=self.cfg.curriculum_window)
        self._curr_max_distance = (self.cfg.curriculum_start_distance
                                   if self.cfg.curriculum else self.cfg.target_max_distance)

        self.world = None
        self.man = None
        self.target_x = self.cfg.target_min_distance
        self.viewer = None
        self._build()
        obs = self.reset()
        self.obs_dim = obs.shape[0]

    # -- world construction -------------------------------------------------
    def _build(self, friction_scale=1.0, mass_scale=1.0):
        cfg = self.cfg
        self.world = World(gravity=Vec2(0.0, cfg.gravity),
                           velocity_iterations=cfg.velocity_iterations,
                           position_iterations=cfg.position_iterations)
        self.ground = self.world.add_ground(half_width=400.0, top_y=self.ground_y,
                                            friction=1.0 * friction_scale)
        self.platform = None
        self.man = Stickman(self.world, cfg, x=0.0, ground_y=self.ground_y,
                            mass_scale=mass_scale)
        if friction_scale != 1.0:
            for body in self.man.parts.values():
                for fx in body.fixtures:
                    fx.friction *= friction_scale
        self._ground_bodies = {self.ground}

    def _place_platform(self):
        cfg = self.cfg
        if self.platform is not None:
            self.world.destroy_body(self.platform)
        body = self.world.create_body(
            STATIC, Vec2(self.target_x, self.ground_y + 0.5 * cfg.platform_height),
            0.0, "platform")
        body.add_fixture(Polygon.box(cfg.platform_half_width,
                                     0.5 * cfg.platform_height),
                         density=0.0, friction=1.0)
        self.platform = body
        self._ground_bodies = {self.ground, self.platform}
        self.world.mark_dirty()

    # -- target sampling ----------------------------------------------------
    def sample_target(self):
        cfg = self.cfg
        hi = self._curr_max_distance if cfg.curriculum else cfg.target_max_distance
        hi = clamp(hi, 0.5, cfg.target_max_distance)
        # The curriculum lowers the ceiling, so the floor has to follow it down;
        # otherwise `target_min_distance` silently overrides the whole schedule
        # and early episodes are no easier than late ones.
        lo = min(cfg.target_min_distance, hi)
        d = float(self.rng.uniform(lo, hi))
        if self.rng.random() < cfg.target_left_probability:
            d = -d
        return d

    def _update_curriculum(self, success):
        cfg = self.cfg
        if not cfg.curriculum:
            return
        self._recent_success.append(1.0 if success else 0.0)
        if len(self._recent_success) == self._recent_success.maxlen:
            rate = sum(self._recent_success) / len(self._recent_success)
            if rate >= cfg.curriculum_success_threshold:
                self._curr_max_distance = min(cfg.target_max_distance,
                                              self._curr_max_distance + cfg.curriculum_step)
                self._recent_success.clear()

    @property
    def curriculum_distance(self):
        return self._curr_max_distance

    def set_curriculum_distance(self, d):
        self._curr_max_distance = float(d)

    # -- gym API ------------------------------------------------------------
    def reset(self, seed=None, target_x=None):
        cfg = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        friction_scale = 1.0
        mass_scale = 1.0
        if cfg.randomize_friction > 0.0:
            friction_scale = 1.0 + float(self.rng.uniform(-cfg.randomize_friction,
                                                          cfg.randomize_friction))
        if cfg.randomize_mass > 0.0:
            mass_scale = 1.0 + float(self.rng.uniform(-cfg.randomize_mass,
                                                      cfg.randomize_mass))
        self._build(friction_scale, mass_scale)

        self.target_x = float(target_x) if target_x is not None else self.sample_target()
        self._place_platform()

        self.man.set_pose_noise(self.rng, cfg)
        for body in self.world.bodies:
            body.synchronize_fixtures()

        self.step_count = 0
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.hold_timer = 0.0
        self.foot_air_time = np.zeros(2, dtype=np.float64)
        self._swing_supported = [True, True]
        self._prev_contacts = (False, False)
        self._lead_foot = 0
        self._stale_lead = 0.0
        self.gait_phase = (float(self.rng.random())
                           if cfg.randomize_initial_phase else 0.0)
        self._action_excess = 0.0
        self.terminated = False
        self.truncated = False
        self.success = False
        self.episode_return = 0.0
        self.start_x = self.man.torso.c.x
        self.prev_distance = self._distance_to_target()
        self.last_components = {}
        return self._observation()

    def _distance_to_target(self):
        return abs(self.man.torso.c.x - self.target_x)

    def _expected_contacts(self):
        """The contact schedule a walk should follow at the current phase.

        Left leg leads the cycle, right is half a cycle out of phase. With a
        duty factor above 0.5 the two stance windows overlap, giving the double
        support a walk has and forbidding the flight phase a bound needs.
        """
        cfg = self.cfg
        duty = cfg.gait_duty
        return (self.gait_phase % 1.0) < duty, \
               ((self.gait_phase + 0.5) % 1.0) < duty

    def _support_y(self):
        """Ground height directly under the torso (platform counts as ground)."""
        cfg = self.cfg
        if abs(self.man.torso.c.x - self.target_x) <= cfg.platform_half_width:
            return self.ground_y + cfg.platform_height
        return self.ground_y

    # -- contacts -----------------------------------------------------------
    def _platform_support(self):
        """(feet resting on the goal platform, anything still touching ground).

        Kept separate from `_contact_flags`, which deliberately treats the
        platform as just more ground: for arrival we need to know precisely
        which surface is carrying the figure.
        """
        feet = (self.man.foot_l, self.man.foot_r)
        on_platform = 0
        on_ground = False
        for contact in self.world.contacts.values():
            if not contact.touching:
                continue
            ba, bb = contact.body_a, contact.body_b
            if ba is self.platform:
                part = bb
            elif bb is self.platform:
                part = ba
            elif ba is self.ground or bb is self.ground:
                on_ground = True
                continue
            else:
                continue
            if part is feet[0] or part is feet[1]:
                on_platform += 1
        return on_platform, on_ground

    def _contact_flags(self):
        foot_l = foot_r = False
        other = False
        head = False
        torso = self.man.torso
        for contact in self.world.contacts.values():
            if not contact.touching:
                continue
            ba, bb = contact.body_a, contact.body_b
            if ba in self._ground_bodies:
                part = bb
            elif bb in self._ground_bodies:
                part = ba
            else:
                continue
            if part is self.man.foot_l:
                foot_l = True
            elif part is self.man.foot_r:
                foot_r = True
            else:
                other = True
                if part is torso:
                    head = True
        return foot_l, foot_r, other, head

    # -- observation --------------------------------------------------------
    def _observation(self):
        cfg = self.cfg
        plan = cfg.plan
        torso = self.man.torso
        foot_l, foot_r, other_contact, _ = self._contact_flags()

        support_y = self._support_y()
        height = torso.c.y - support_y

        target_point = Vec2(self.target_x,
                            self.ground_y + cfg.platform_height + plan.torso_y)
        to_target = target_point - torso.c
        dist = to_target.length()
        dir_world = to_target.normalized()
        dir_local = torso.xf.q.inv_rotate(dir_world)

        obs = [
            math.sin(torso.a), math.cos(torso.a),
            clamp(torso.w / 6.0, -3.0, 3.0),
            clamp(torso.v.x / 4.0, -3.0, 3.0),
            clamp(torso.v.y / 4.0, -3.0, 3.0),
            clamp((height - plan.torso_y) / 0.5, -3.0, 3.0),
        ]
        for joint, spec in zip(self.man.joint_list, self.man.motor_specs):
            mid = 0.5 * (spec.lower + spec.upper)
            half = max(1e-6, 0.5 * (spec.upper - spec.lower))
            obs.append(clamp((joint.joint_angle - mid) / half, -1.5, 1.5))
            obs.append(clamp(joint.joint_speed / 10.0, -3.0, 3.0))
        obs.extend([
            1.0 if foot_l else 0.0,
            1.0 if foot_r else 0.0,
            1.0 if other_contact else 0.0,
            # --- target information, the part that makes this transferable ---
            dir_local.x, dir_local.y,
            dir_world.x,
            clamp(dist / 10.0, 0.0, 1.5),
        ])
        if cfg.include_prev_action:
            obs.extend(self.prev_action.tolist())
        if cfg.use_gait_clock:
            # Appended LAST so every pre-existing channel keeps its index and
            # an older checkpoint can be widened rather than retrained.
            ph = 2.0 * math.pi * self.gait_phase
            obs.extend([math.sin(ph), math.cos(ph)])
        return np.asarray(obs, dtype=np.float32)

    # -- stepping -----------------------------------------------------------
    def step(self, action):
        cfg = self.cfg
        raw = np.asarray(action, dtype=np.float32)
        # Charge for the part of the command the motors cannot use, before it
        # is clipped away -- see `w_action_bound`. Without this the policy's
        # std has nothing pushing back on it above the action range.
        excess = np.maximum(np.abs(raw) - 1.0, 0.0)
        self._action_excess = float(np.mean(excess * excess))
        action = np.clip(raw, -1.0, 1.0)
        self.man.apply_action(action)

        dt = cfg.physics_dt
        for _ in range(cfg.substeps):
            self.man.apply_joint_torques()
            self.world.step(dt)

        self.step_count += 1
        # Score against the phase the policy actually saw, then advance it, so
        # the next observation carries the phase its next action is judged on.
        reward, components = self._reward(action)
        if cfg.use_gait_clock:
            self.gait_phase = (self.gait_phase
                               + cfg.gait_frequency * cfg.control_dt) % 1.0
        self.episode_return += reward
        self.prev_action = action.copy()
        self.last_components = components

        obs = self._observation()
        terminated = self.terminated
        truncated = (not terminated) and self.step_count >= cfg.max_episode_steps
        self.truncated = truncated
        if terminated or truncated:
            self._update_curriculum(self.success)

        info = {
            "components": components,
            "distance": self._distance_to_target(),
            "target_x": self.target_x,
            "success": self.success,
            "x": float(self.man.torso.c.x),
            "episode_return": self.episode_return,
            "steps": self.step_count,
            "curriculum_distance": self._curr_max_distance,
        }
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.step_count,
                "success": self.success,
                "distance": self._distance_to_target(),
                "target_x": self.target_x,
                "travelled": float(self.man.torso.c.x - self.start_x),
            }
        return obs, reward, terminated, truncated, info

    # -- reward -------------------------------------------------------------
    def _reward(self, action):
        cfg = self.cfg
        plan = cfg.plan
        dt = cfg.control_dt
        torso = self.man.torso
        foot_l, foot_r, other_contact, head_contact = self._contact_flags()

        support_y = self._support_y()
        height = torso.c.y - support_y
        tilt = abs(torso.a)

        # --- fall detection -------------------------------------------------
        fallen = (height < cfg.fall_height_fraction * plan.torso_y
                  or tilt > cfg.fall_angle
                  or (cfg.terminate_on_head_contact and head_contact))

        # --- progress -------------------------------------------------------
        dist = self._distance_to_target()
        approach = (self.prev_distance - dist) / dt
        self.prev_distance = dist
        approach = clamp(approach, -1.5 * cfg.progress_speed_cap, cfg.progress_speed_cap)
        r_progress = cfg.w_progress * approach * dt

        # --- posture --------------------------------------------------------
        upright = math.exp(-(torso.a / cfg.upright_scale) ** 2)
        r_upright = cfg.w_upright * upright * dt
        height_err = (height - plan.torso_y) / cfg.height_scale
        height_score = math.exp(-height_err * height_err)
        r_height = cfg.w_height * height_score * dt
        r_alive = cfg.w_alive * dt if not fallen else 0.0

        # --- support: on the feet, on two of them ---------------------------
        on_feet = (foot_l or foot_r) and not other_contact
        r_support = cfg.w_foot_support * dt if on_feet else 0.0
        two_feet = foot_l and foot_r and upright > 0.6 and height_score > 0.5
        r_two_feet = cfg.w_two_feet * dt if two_feet else 0.0

        # --- gait: make a real step strictly better than a drag -------------
        # `support` above pays for "a foot is down", which a towed leg
        # satisfies just as well as a stepping one. Swing time is only earned
        # by a foot that actually leaves the ground, and a planted foot that is
        # sliding is charged for it -- so one-leg-does-everything stops paying.
        # Air time is gated on still having somewhere to go rather than on
        # current speed, otherwise it cannot bootstrap: you need to step in
        # order to move, so gating on movement would never pay the first step.
        should_walk = dist > cfg.goal_zone_half_width
        # Gait bonuses scale with how fast it is actually closing on the
        # target, so marching on the spot earns nothing. Penalties below stay
        # ungated -- a bad gait is bad wherever it happens.
        # Gait shaping is meaningless for a figure that cannot stand yet: it
        # gets punished every step for failing a rhythm it has no ability to
        # execute, which stops it ever reaching the balance that gait requires.
        # Scaling by posture makes the curriculum implicit -- learn to stand,
        # and the gait terms fade in on their own.
        posture_gate = upright * height_score
        gait_gate = clamp(approach / cfg.gait_progress_speed, 0.0, 1.0) * posture_gate
        r_air_time = 0.0
        r_slip = 0.0
        r_step_length = 0.0
        contacts = (foot_l, foot_r)
        feet = (self.man.foot_l, self.man.foot_r)
        # Sign of "forward" -- placement is only a step if it is toward the goal.
        forward = 1.0 if self.target_x >= torso.c.x else -1.0
        for i, foot_body in enumerate(feet):
            in_contact = contacts[i]
            if in_contact:
                # A swing only counts if the OTHER leg carried the body for the
                # whole of it. Both feet in the air at once is a hop, not a
                # step, and must not be paid out as two separate swings --
                # that is precisely what a naive air-time bonus rewards.
                if (self.foot_air_time[i] > 0.0 and should_walk
                        and self._swing_supported[i]):
                    swing = self.foot_air_time[i] - cfg.air_time_target
                    r_air_time += cfg.w_air_time * gait_gate * clamp(
                        swing, -cfg.air_time_target, cfg.air_time_target)

                # Where the foot LANDS is what makes it a walking step: ahead
                # of the stance foot, not behind it. Scored on EVERY genuine
                # landing, including ones that happen during a flight phase --
                # gating this behind `_swing_supported` (as air_time is, for
                # its own reasons) meant the term written to cure the bound was
                # switched off BY the bound: 52 of 63 real landings went
                # unscored and only the already-correct ones ever paid.
                # The gate is that the OTHER foot was ALREADY planted: you
                # stepped onto a supporting leg. That is weak enough to survive
                # a flight phase mid-swing (unlike `_swing_supported`, which
                # the bound switched off), but it still excludes a bound's
                # simultaneous touchdown -- where there is no stance foot at
                # all, and whichever foot is in front would otherwise collect
                # the forward bonus while its partner paid the smaller penalty.
                if (self.foot_air_time[i] >= cfg.min_step_air_time
                        and should_walk and self._prev_contacts[1 - i]):
                    placement = (foot_body.c.x - feet[1 - i].c.x) * forward
                    r_step_length += cfg.w_step_length * gait_gate * clamp(
                        placement, -cfg.step_length_penalty_cap,
                        cfg.step_length_cap)
                self.foot_air_time[i] = 0.0
                self._swing_supported[i] = True
                r_slip -= cfg.w_foot_slip * clamp(
                    abs(foot_body.v.x) - cfg.foot_slip_deadzone,
                    0.0, cfg.foot_slip_cap) * dt
            else:
                self.foot_air_time[i] += dt
                if not contacts[1 - i]:
                    self._swing_supported[i] = False
        self._prev_contacts = (foot_l, foot_r)

        # --- a walk is single support; a hop is flight plus double stance ---
        r_alternate = 0.0
        if should_walk:
            if foot_l != foot_r:
                r_alternate = cfg.w_alternate_support * gait_gate * dt
            elif not foot_l and not foot_r:
                # Flight is the signature of a bound; a walk never has one.
                r_alternate = -cfg.w_flight * dt

        # --- track the reference gait ---------------------------------------
        # The dominant gait signal. exp(-k*err) is dense and smooth: it pays
        # for being *closer* to the reference, so there is a gradient from any
        # pose, unlike the event-based terms that only fired once the target
        # behaviour already existed. It is self-gating too -- a figure that is
        # falling matches nothing and simply earns ~0 rather than a penalty.
        # How much the figure should still be walking: 1 while there is ground
        # to cover, tapering to 0 at the goal. Without this the walking
        # reference keeps paying after arrival and carrying it past the target.
        walk_intent = clamp((dist - cfg.goal_zone_half_width)
                            / max(1e-6, cfg.slowdown_distance), 0.0, 1.0)
        # ...and only while it is actually closing on the target. Without this
        # the taper is escapable: walk through the goal, watch the distance
        # grow, and the full reward returns.
        floor = cfg.imitate_idle_floor
        walk_intent *= clamp(floor + (1.0 - floor) * approach
                             / max(1e-6, cfg.gait_progress_speed), 0.0, 1.0)

        r_imitate = 0.0
        if cfg.w_imitate > 0.0:
            # Direction-aware: a target behind requires walking backwards,
            # and the reference has to describe THAT, or walking the wrong
            # way pays more than walking the right way.
            ref = reference_joint_targets(self.gait_phase, cfg, forward)
            n_track = len(JOINT_ORDER) if cfg.imitate_arms else 6
            err = 0.0
            for i in range(n_track):
                d = self.man.joint_list[i].joint_angle - ref[i]
                err += d * d
            err /= n_track
            r_imitate = (cfg.w_imitate * walk_intent
                         * math.exp(-cfg.imitate_error_scale * err) * dt)

        # --- follow the gait clock ------------------------------------------
        # Each foot is paid for being down when the schedule says stance and up
        # when it says swing. A permanently-trailing leg is wrong half of every
        # cycle, so the horse gait is charged continuously rather than merely
        # going unrewarded -- which is what every previous term did.
        r_periodic = 0.0
        if cfg.use_gait_clock and should_walk:
            expect_l, expect_r = self._expected_contacts()
            for actual, expected in ((foot_l, expect_l), (foot_r, expect_r)):
                if actual == expected:
                    r_periodic += cfg.w_periodic_contact * dt
                else:
                    r_periodic -= (cfg.w_periodic_contact
                                   * cfg.periodic_mismatch_scale * dt)

        # --- keep the swing low: a walk, not a high-knee bound --------------
        r_clearance = 0.0
        for foot_body in (self.man.foot_l, self.man.foot_r):
            support = (self.ground_y + cfg.platform_height
                       if abs(foot_body.c.x - self.target_x) <= cfg.platform_half_width
                       else self.ground_y)
            lift = foot_body.c.y - support - plan.foot[1]
            if lift > cfg.max_foot_clearance:
                r_clearance -= cfg.w_foot_clearance * (
                    lift - cfg.max_foot_clearance) * dt

        # --- a scissored stance is not a free substitute for stepping -------
        # Parking one leg in front and one behind is the most stable posture
        # available and needs no coordination at all, so without a cost the
        # trailing leg never has any reason to come through and the feet never
        # pass each other. A real stride briefly exceeds this; a permanent
        # splay sits beyond it every single step.
        separation = abs(self.man.foot_l.c.x - self.man.foot_r.c.x)
        r_splay = -cfg.w_splay * max(0.0, separation - cfg.max_stride) * dt

        # --- and the legs have to actually trade places ---------------------
        # Hopping keeps one hip permanently in front. Pay for the crossing
        # itself, so a fixed stance earns nothing no matter how far it travels.
        # Fade the remaining gait-shape terms in with posture too, for the same
        # reason. `slip` stays ungated: dragging a foot is bad in any posture.
        r_periodic *= posture_gate
        r_alternate *= posture_gate
        r_clearance *= posture_gate
        r_splay *= posture_gate

        r_leg_swap = 0.0
        r_stale_lead = 0.0
        # Which foot is physically in front, in the direction of travel.
        lead_gap = (self.man.foot_l.c.x - self.man.foot_r.c.x) * forward

        # --- dense foot-separation tracking ---------------------------------
        # The reference implies where the feet should be relative to each other
        # at every phase: ahead, level, behind, level, repeat. Rewarding that
        # continuously gives a gradient from ANY separation, which the
        # event-based swap bonus could not.
        r_foot_sep = 0.0
        if should_walk and cfg.w_foot_sep > 0.0:
            ref_l, ref_r = reference_foot_offsets(self.gait_phase, cfg)
            sep_err = lead_gap - (ref_l - ref_r)
            r_foot_sep = (cfg.w_foot_sep
                          * math.exp(-cfg.foot_sep_scale * sep_err * sep_err)
                          * posture_gate * dt)
        lead = (1 if lead_gap > cfg.lead_swap_deadband
                else (-1 if lead_gap < -cfg.lead_swap_deadband else 0))

        if should_walk:
            self._stale_lead += dt
        else:
            self._stale_lead = 0.0

        if lead != 0:
            if self._lead_foot != 0 and lead != self._lead_foot and should_walk:
                # The trailing foot has passed the other one: a real step.
                r_leg_swap = cfg.w_leg_swap * posture_gate
                self._stale_lead = 0.0
            self._lead_foot = lead

        if should_walk:
            # Ramps in after one gait cycle without a swap and saturates after
            # two, so a permanently-leading leg bleeds for as long as it lasts.
            grace = 1.0 / max(1e-6, cfg.gait_frequency)
            overdue = (self._stale_lead - grace) / grace
            r_stale_lead = (-cfg.w_stale_lead * clamp(overdue, 0.0, 1.0)
                            * posture_gate * dt)

        # --- costs -----------------------------------------------------------
        a2 = float(np.mean(action * action))
        r_ctrl = -cfg.w_ctrl_cost * a2 * dt
        da = action - self.prev_action
        r_rate = -cfg.w_action_rate * float(np.mean(da * da)) * dt
        at_limit = 0
        for joint, spec in zip(self.man.joint_list, self.man.motor_specs):
            ang = joint.joint_angle
            if ang < spec.lower + 0.04 or ang > spec.upper - 0.04:
                at_limit += 1
        r_limit = -cfg.w_joint_limit * (at_limit / len(self.man.joint_list)) * dt
        r_bound = -cfg.w_action_bound * self._action_excess * dt
        spin = clamp(abs(torso.w) / 6.0, 0.0, 1.0)
        r_spin = -cfg.w_head_stability * spin * spin * dt

        # --- goal -------------------------------------------------------------
        in_zone = abs(torso.c.x - self.target_x) <= cfg.goal_zone_half_width
        standing = on_feet and upright > 0.55 and height_score > 0.4
        if cfg.require_platform_contact:
            feet_on_platform, still_on_ground = self._platform_support()
            standing = standing and feet_on_platform >= 1 and not still_on_ground
        r_goal = 0.0
        if in_zone and standing:
            self.hold_timer += dt
            r_goal = cfg.w_goal_hold * dt
            if self.hold_timer >= cfg.arrival_hold_seconds and not self.success:
                self.success = True
                r_goal += cfg.r_reach          # one-off arrival bonus
                if cfg.terminate_on_success:
                    self.terminated = True
        else:
            self.hold_timer = max(0.0, self.hold_timer - dt)

        r_fall = 0.0
        if fallen:
            r_fall = cfg.r_fall
            self.terminated = True

        components = {
            "progress": r_progress,
            "alive": r_alive,
            "upright": r_upright,
            "height": r_height,
            "support": r_support,
            "two_feet": r_two_feet,
            "imitate": r_imitate,
            "air_time": r_air_time,
            "step_length": r_step_length,
            "slip": r_slip,
            "alternate": r_alternate,
            "periodic": r_periodic,
            "clearance": r_clearance,
            "splay": r_splay,
            "leg_swap": r_leg_swap,
            "stale_lead": r_stale_lead,
            "foot_sep": r_foot_sep,
            "goal": r_goal,
            "ctrl": r_ctrl,
            "rate": r_rate,
            "bound": r_bound,
            "limit": r_limit,
            "spin": r_spin,
            "fall": r_fall,
        }
        return float(sum(components.values())), components

    # -- introspection / rendering ------------------------------------------
    def render_state(self):
        """A pure-data snapshot the viewer (or a replay file) can consume."""
        shapes = []
        for body in self.world.bodies:
            for fx in body.fixtures:
                shape = fx.shape
                if isinstance(shape, Circle):
                    c = body.xf.apply(shape.p)
                    shapes.append({"kind": "circle", "name": body.name,
                                   "x": c.x, "y": c.y, "r": shape.radius,
                                   "angle": body.a})
                else:
                    pts = [body.xf.apply(v) for v in shape.vertices]
                    shapes.append({"kind": "poly", "name": body.name,
                                   "pts": [(p.x, p.y) for p in pts]})
        contacts = []
        for contact in self.world.contacts.values():
            if not contact.touching:
                continue
            wm = contact.world_manifold()
            for p in wm.points:
                contacts.append((p.x, p.y))
        return {
            "shapes": shapes,
            "contacts": contacts,
            "target_x": self.target_x,
            "goal_half": self.cfg.goal_zone_half_width,
            "platform_height": self.cfg.platform_height,
            "platform_half": self.cfg.platform_half_width,
            "torso": (self.man.torso.c.x, self.man.torso.c.y, self.man.torso.a),
            "components": dict(self.last_components),
            "return": self.episode_return,
            "steps": self.step_count,
            "hold": self.hold_timer,
            "action": self.prev_action.copy(),
            "time": self.world.time,
        }

    def pose(self):
        """Compact float array of every body pose -- used for exact replays."""
        out = []
        for body in self.world.bodies:
            out.extend((body.xf.p.x, body.xf.p.y, body.a))
        return np.asarray(out, dtype=np.float32)

    def set_pose(self, arr):
        for i, body in enumerate(self.world.bodies):
            body.set_transform(Vec2(float(arr[3 * i]), float(arr[3 * i + 1])),
                               float(arr[3 * i + 2]))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_env(cfg=None, seed=None, **overrides):
    cfg = cfg if cfg is not None else StickmanConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError("unknown config field %r" % k)
        setattr(cfg, k, v)
    return StickmanEnv(cfg, seed=seed)
