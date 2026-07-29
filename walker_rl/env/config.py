"""All tunable numbers for the stickman task in one place.

Geometry is in metres, masses come out of area * density (2D density is
kg/m^2), torques in N*m. The defaults describe a ~1.65 m, ~72 kg figure so the
torque limits below are comparable to real human joint torques -- that matters,
because if the motors are unrealistically strong the agent learns to solve the
task by brute force instead of by balancing.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class BodyPlan:
    # half-extents (half_width, half_length)
    torso: tuple = (0.075, 0.26)
    head_radius: float = 0.11
    head_offset: float = 0.30      # above torso centre, in torso local frame
    thigh: tuple = (0.045, 0.21)
    shin: tuple = (0.035, 0.20)
    foot: tuple = (0.11, 0.035)
    foot_forward_shift: float = 0.045   # ankle sits behind the foot centre
    arm: tuple = (0.035, 0.17)

    # 2D densities (kg/m^2)
    density_torso: float = 265.0
    density_head: float = 300.0
    density_thigh: float = 210.0
    density_shin: float = 190.0
    density_foot: float = 150.0
    density_arm: float = 190.0

    friction_foot: float = 1.2
    friction_body: float = 0.4
    restitution: float = 0.0

    @property
    def ankle_y(self):
        return 2.0 * self.foot[1]

    @property
    def knee_y(self):
        return self.ankle_y + 2.0 * self.shin[1]

    @property
    def hip_y(self):
        return self.knee_y + 2.0 * self.thigh[1]

    @property
    def torso_y(self):
        return self.hip_y + self.torso[1]

    @property
    def shoulder_y(self):
        return self.hip_y + 2.0 * self.torso[1] - 0.06

    @property
    def standing_height(self):
        """Height of the head top when standing straight."""
        return self.torso_y + self.head_offset + self.head_radius


@dataclass
class MotorSpec:
    lower: float
    upper: float
    max_torque: float
    max_speed: float


@dataclass
class StickmanConfig:
    plan: BodyPlan = field(default_factory=BodyPlan)

    # --- simulation ---
    gravity: float = -9.81
    physics_hz: float = 240.0
    control_hz: float = 40.0
    velocity_iterations: int = 8
    position_iterations: int = 3

    # --- motors: (lower limit, upper limit, max torque, max speed) ---
    hip: MotorSpec = field(default_factory=lambda: MotorSpec(-0.80, 1.30, 220.0, 7.0))
    knee: MotorSpec = field(default_factory=lambda: MotorSpec(-2.50, 0.05, 180.0, 8.0))
    ankle: MotorSpec = field(default_factory=lambda: MotorSpec(-0.60, 0.90, 140.0, 7.0))
    shoulder: MotorSpec = field(default_factory=lambda: MotorSpec(-2.40, 2.40, 60.0, 9.0))

    # --- actuation --------------------------------------------------------
    # "pd"    action = target joint angle; the env applies
    #         tau = kp*(target - angle) - kd*speed, clamped to the joint limit.
    # "motor" action sets a velocity-servo direction at FULL joint speed and
    #         caps its torque -- so the only commands available are
    #         slam-forward, limp, and slam-backward. There is no way to hold an
    #         angle, which makes graded, phased torque impossible and forces
    #         bang-bang gaits (82% of actions were saturated under it).
    # PD position targets are what legged-locomotion RL normally uses.
    actuation: str = "pd"
    pd_kp_scale: float = 1.0        # stiffness, as a fraction of max torque
    pd_kd_scale: float = 0.05       # damping, as a fraction of max torque
    pd_action_scale: float = 1.0    # rad of target offset at |action| = 1

    joint_damping: float = 0.4      # passive damping at every joint
    linear_damping: float = 0.02
    angular_damping: float = 0.05

    # --- gait clock -------------------------------------------------------
    # The policy is a feedforward MLP with no memory, so a walking limit cycle
    # has to be reconstructed from body state alone -- but a fixed asymmetric
    # stance is a stable FIXED POINT, which is far easier to represent and to
    # find. That is why it kept converging to one leg permanently forward no
    # matter how the reward was weighted: alternation was not representable,
    # so no weighting could select for it.
    #
    # An explicit phase clock in the observation makes "whose turn it is"
    # representable, and a target contact schedule makes it rewarded. This is
    # how bipedal locomotion is normally solved; shaping alone is not.
    use_gait_clock: bool = True
    gait_frequency: float = 1.3          # gait cycles per second
    gait_duty: float = 0.60              # fraction of a cycle each foot is down
    w_periodic_contact: float = 1.5      # per second per foot, for matching it
    # Because duty > 0.5, a permanently-planted foot MATCHES the schedule more
    # often than it violates it -- so symmetric scoring pays the horse gait a
    # net bonus. Violations must outweigh matches by more than duty/(1-duty).
    periodic_mismatch_scale: float = 2.5
    randomize_initial_phase: bool = True

    # --- imitation reference ----------------------------------------------
    # Hand-written gait terms each described one symptom of walking and fought
    # each other while still leaving the motion underdetermined -- notably they
    # could not express "extend the knee during late swing so the foot plants
    # ahead", which is exactly what kept failing. A reference trajectory
    # specifies the whole coordinated motion densely, every timestep.
    w_imitate: float = 9.0               # dominant gait term
    imitate_error_scale: float = 3.5     # exp(-scale * mean squared error)
    imitate_arms: bool = False           # leave arms free for balance

    ref_hip_amp: float = 0.45            # rad, hip sweep either side of neutral
    ref_hip_bias: float = 0.05
    ref_knee_stance: float = -0.06       # near straight while carrying weight
    ref_knee_dip: float = 0.15           # slight flexion absorbing the landing
    ref_knee_flex: float = 1.05          # peak flexion mid-swing (ground clear)
    ref_swing_skew: float = 0.85         # <1 flexes early, extends longer
    ref_ankle_gain: float = 0.40
    ref_arm_amp: float = 0.30

    # Imitation pays for tracking a WALKING reference, so if it stays on after
    # arrival the agent is paid to keep walking through the target -- and
    # overshooting genuinely out-earns stopping. Fade it out over the last
    # stretch so the gait reward tapers to nothing at the goal and standing
    # takes over. The taper doubles as a natural deceleration ramp.
    slowdown_distance: float = 1.20      # m over which the gait reward fades
    # Fading by DISTANCE alone is not enough: walking straight through the goal
    # makes the distance grow again and the reward comes back, so leaving
    # out-earns staying (8.3/s vs 6.7/s) and it overshot by 6 m. The gait
    # reward must also require actually closing on the target, so walking away
    # pays nothing. The floor keeps a little signal alive while standing still,
    # so a policy that cannot walk yet still has something to learn from.
    imitate_idle_floor: float = 0.20

    # --- episode ---
    max_episode_seconds: float = 24.0
    include_prev_action: bool = True

    # --- target ---
    target_min_distance: float = 2.0
    target_max_distance: float = 9.0
    target_left_probability: float = 0.15
    goal_zone_half_width: float = 0.30      # the "square" the agent must stand in
    platform_half_width: float = 0.55
    platform_height: float = 0.08
    arrival_hold_seconds: float = 1.0       # must stay in the zone this long
    # Torso-x alone is not "arrived": it lets the figure straddle the platform
    # edge with its trailing leg still scraping the ground behind it and
    # collect the goal reward anyway, so that leg never has to come up.
    # Arrival requires the platform to be carrying it.
    require_platform_contact: bool = True
    # Reaching the goal must not end the episode. If it did, arriving would
    # forfeit the remaining alive/upright bonuses and standing still at the
    # spawn point would out-earn walking all the way to the target.
    terminate_on_success: bool = False

    # --- curriculum (target distance grows with success rate) ---
    curriculum: bool = True
    curriculum_start_distance: float = 2.5
    curriculum_step: float = 0.35
    curriculum_success_threshold: float = 0.55
    curriculum_window: int = 30

    # --- initial state randomisation ---
    init_joint_noise: float = 0.06
    init_angle_noise: float = 0.03
    init_velocity_noise: float = 0.12
    init_height_offset: float = 0.02

    # --- domain randomisation (off by default) ---
    randomize_friction: float = 0.0     # +/- fraction
    randomize_mass: float = 0.0         # +/- fraction

    # --- termination ---
    fall_height_fraction: float = 0.62   # torso centre below this * nominal -> fall
    fall_angle: float = 1.15             # |torso tilt| above this -> fall
    terminate_on_head_contact: bool = True

    # --- reward weights, all expressed per second ---
    # Standing still already collects alive + upright + height + support every
    # second. For walking to be worth the risk of falling, the progress term
    # has to clearly outweigh that whole posture package rather than merely
    # match it -- at the cap, progress alone is ~1.7x the idle rate.
    w_progress: float = 6.0
    progress_speed_cap: float = 1.0      # m/s, no reward for going faster
    w_alive: float = 1.5
    w_upright: float = 1.2
    upright_scale: float = 0.45          # rad, width of the upright gaussian
    w_height: float = 1.2
    height_scale: float = 0.22           # m, width of the height gaussian
    w_foot_support: float = 0.8          # feet (and only feet) carry the body
    w_two_feet: float = 0.3              # both feet planted -> balance scaffold
    # Dragging a leg is FREE under a plain contact bonus: a scraped foot is
    # still a foot, so `support` keeps paying while one leg does all the work
    # and the other is towed along. These are the standard legged-locomotion
    # pair that make a real step strictly better than a drag -- swing time is
    # only earned by a foot that genuinely leaves the ground, and a loaded foot
    # that is sliding gets charged for it.
    w_air_time: float = 2.0              # paid on landing, scaled by swing time
    air_time_target: float = 0.22        # s, the swing duration of a good step
    # Slip has to outweigh what dragging buys. Progress pays w_progress * v;
    # a drag moves BOTH feet at v while a real step has a stationary stance
    # foot, so the penalty must satisfy 2 * w_foot_slip > w_progress or
    # shuffling forward is simply profitable. It was 0.8 against a progress
    # weight of 6.0, which is why the figure dragged for 8 m and paid 7 points.
    w_foot_slip: float = 4.0             # penalty for a planted foot sliding
    foot_slip_deadzone: float = 0.05     # m/s, ignore contact-solver jitter
    foot_slip_cap: float = 2.0           # m/s, ceiling on the slip penalty

    # A permanently scissored stance -- one leg parked forward, one back -- is
    # maximally stable and lets the figure shuffle along without ever lifting a
    # foot or letting the legs pass each other. Charge for separation beyond a
    # normal stride so the trailing leg has to come through.
    w_splay: float = 2.0
    max_stride: float = 0.55             # m between the feet before it costs

    # A bunny hop satisfies a naive swing-time reward TWICE per cycle -- both
    # feet leave and land together, so both collect -- and it needs no
    # inter-leg coordination at all, which makes it a strong attractor. What
    # separates walking from hopping is single support: a walk always has
    # exactly one foot carrying the body and never has a flight phase, and its
    # legs trade places every cycle.
    w_alternate_support: float = 0.5     # exactly one foot down, while walking
    # The dense counter-pressure to bounding. step_length is a large carrot,
    # but it only pays once the figure ALREADY steps onto a planted foot, so
    # it cannot by itself guide a bounding gait out of its basin. Flight is
    # penalised every step the bound occurs, which is a gradient it can follow
    # from where it currently is.
    w_flight: float = 4.0               # penalty for BOTH feet airborne
    # --- lead-foot alternation -------------------------------------------
    # This is what walking IS: the trailing foot comes through, lands AHEAD of
    # the other, and becomes the lead; then the other does the same. Measured
    # on FOOT POSITION, not hip angle -- with the legs splayed the hips rock
    # back and forth and score a hip-angle "swap" while the feet never change
    # order, which is exactly the gait that kept surviving every reward I
    # wrote.
    w_leg_swap: float = 2.5              # one-off when the LEAD FOOT changes
    lead_swap_deadband: float = 0.08     # m of separation before a lead counts
    leg_swap_split: float = 0.15         # (unused; kept for old configs)
    # A one-off bonus can simply be forgone -- which is what it did, trading
    # the bonus for a faster splayed shuffle. A penalty that GROWS the longer
    # the same foot stays in front cannot be traded away: holding one leg
    # forward bleeds continuously and without bound.
    w_stale_lead: float = 4.0

    # The swap bonus and the stale penalty are both EVENTS: they pay nothing
    # until the feet actually cross. Measured on the failing gait the lead gap
    # was -0.50 m on average and peaked at +0.02, never once reaching the 0.08
    # deadband -- so neither term ever fired and neither could point the way.
    #
    # This one is dense. The reference gait implies a foot separation at every
    # phase (+-0.75 m, swapping sign twice a cycle), and this rewards matching
    # it continuously. From a frozen 0.5 m lead there is a smooth gradient to
    # follow all the way to a proper alternating stride.
    w_foot_sep: float = 6.0
    foot_sep_scale: float = 2.5          # exp(-scale * separation error^2)

    # What actually defines a walking step: the swing foot is PLACED AHEAD of
    # the stance foot. Crossing the hips is not enough -- a bound scissors the
    # legs too, then lands the foot behind and hops off it again. Measured on
    # the hopping gait: median placement -0.57 m, i.e. consistently behind.
    w_step_length: float = 2.0
    step_length_cap: float = 0.60        # m of forward placement credited
    # Deliberately asymmetric: landing behind is charged far less than landing
    # ahead is paid. A symmetric penalty at the observed landing rate would
    # make standing still safer than walking imperfectly.
    step_length_penalty_cap: float = 0.20
    min_step_air_time: float = 0.10      # s; below this it is contact chatter

    # Human walking clears the ground by 5-15 cm. A half-metre knee tuck is
    # exactly what a swing-time bonus buys when nothing charges for altitude.
    w_foot_clearance: float = 1.0
    max_foot_clearance: float = 0.18     # m above the resting foot height

    # Gait bonuses are earned only while genuinely closing on the target.
    # Without this, dropping the goal-hold reward just creates a new exploit:
    # march on the spot forever just outside the zone and collect gait reward
    # without ever arriving.
    gait_progress_speed: float = 0.5     # m/s needed for full gait credit
    w_ctrl_cost: float = 0.18
    w_action_rate: float = 0.12
    # Actions outside [-1, 1] get clipped by the motors, so without an explicit
    # cost they are behaviourally free: a=1.2 and a=5.0 do exactly the same
    # thing. That leaves no gradient pressure on the policy's std in the
    # clipped region, and the entropy bonus then inflates it without bound
    # until every command sits on the rail and only bang-bang gaits survive.
    w_action_bound: float = 3.0
    w_joint_limit: float = 0.25
    w_head_stability: float = 0.3        # discourage thrashing the torso
    # one-off (not per-second) rewards
    r_fall: float = -6.0
    r_reach: float = 25.0
    # Idling in the goal zone must not out-earn travelling well. At 6.0/s the
    # optimal policy was: sprint the whole distance in 4 s using whatever gait
    # is fastest, then bank 20 s of hold reward -- which left the gait terms
    # active for only 16% of the episode and unable to influence anything.
    w_goal_hold: float = 2.0             # per second, while standing in the zone

    def as_dict(self):
        return asdict(self)

    @property
    def substeps(self):
        n = int(round(self.physics_hz / self.control_hz))
        return max(1, n)

    @property
    def control_dt(self):
        return 1.0 / self.control_hz

    @property
    def physics_dt(self):
        return 1.0 / (self.control_hz * self.substeps)

    @property
    def max_episode_steps(self):
        return int(round(self.max_episode_seconds * self.control_hz))


def config_from_dict(d):
    """Rebuild a config from `as_dict()` -- used when restoring a checkpoint.

    The environment geometry and motor limits are part of the trained policy's
    contract, so they travel with the weights rather than being re-read from
    whatever the defaults happen to be today.
    """
    d = dict(d)
    plan = BodyPlan(**d.pop("plan"))
    motors = {k: MotorSpec(**d.pop(k)) for k in ("hip", "knee", "ankle", "shoulder")}
    return StickmanConfig(plan=plan, **motors, **d)
