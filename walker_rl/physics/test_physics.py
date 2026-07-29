"""Correctness checks for the physics engine.

Run directly (`python -m walker_rl.physics.test_physics`) or under pytest.
These assert against analytic results, not against "it looked fine" -- an RL
agent will happily exploit an engine bug, so the engine has to be right first.
"""

import math

from .body import DYNAMIC, STATIC
from .math2d import Vec2
from .shapes import Circle, Polygon
from .world import World

TOL = 1e-6


def _approx(a, b, tol):
    assert abs(a - b) <= tol, "expected %.9g ~= %.9g (tol %.3g)" % (a, b, tol)


# ---------------------------------------------------------------------------
def test_box_mass_properties():
    hw, hh, density = 0.5, 1.5, 3.0
    md = Polygon.box(hw, hh).compute_mass(density)
    area = (2 * hw) * (2 * hh)
    _approx(md.mass, density * area, 1e-9)
    _approx(md.center.x, 0.0, 1e-9)
    _approx(md.center.y, 0.0, 1e-9)
    expected_i = md.mass * ((2 * hw) ** 2 + (2 * hh) ** 2) / 12.0
    _approx(md.inertia, expected_i, 1e-9)


def test_circle_mass_properties():
    r, density = 0.7, 2.5
    md = Circle(r).compute_mass(density)
    _approx(md.mass, density * math.pi * r * r, 1e-9)
    _approx(md.inertia, 0.5 * md.mass * r * r, 1e-9)


def test_offset_box_parallel_axis():
    """A box whose centroid is offset must report inertia about its own COM."""
    hw, hh, d = 0.3, 0.8, 2.0
    offset = Vec2(1.7, -0.9)
    world = World()
    body = world.create_body(DYNAMIC, Vec2(0, 0))
    body.add_fixture(Polygon.box(hw, hh, offset), density=d)
    mass = d * (2 * hw) * (2 * hh)
    _approx(body.mass, mass, 1e-9)
    _approx(body.local_center.x, offset.x, 1e-9)
    _approx(body.local_center.y, offset.y, 1e-9)
    _approx(body.inertia, mass * ((2 * hw) ** 2 + (2 * hh) ** 2) / 12.0, 1e-7)


# ---------------------------------------------------------------------------
def test_free_fall_matches_analytic():
    world = World(gravity=Vec2(0.0, -10.0))
    body = world.create_body(DYNAMIC, Vec2(0.0, 100.0))
    body.add_fixture(Circle(0.2), density=1.0)

    dt = 1.0 / 240.0
    n = 240
    for _ in range(n):
        world.step(dt)

    t = n * dt
    # Semi-implicit Euler drops by g*dt^2*(n(n+1)/2), which is the exact
    # discrete solution -- assert against that, not the continuous one.
    expected_drop = 10.0 * dt * dt * (n * (n + 1) / 2.0)
    _approx(100.0 - body.c.y, expected_drop, 1e-9)
    _approx(body.v.y, -10.0 * t, 1e-9)


def test_static_body_ignores_gravity():
    world = World()
    ground = world.add_ground()
    for _ in range(100):
        world.step(1 / 120.0)
    _approx(ground.c.y, ground.c.y, 0.0)
    _approx(ground.v.length(), 0.0, 0.0)


# ---------------------------------------------------------------------------
def test_box_rests_on_ground():
    world = World(gravity=Vec2(0.0, -9.81))
    world.add_ground(friction=0.9)
    hh = 0.5
    body = world.create_body(DYNAMIC, Vec2(0.0, 3.0))
    body.add_fixture(Polygon.box(0.5, hh), density=1.0, friction=0.9)

    for _ in range(600):
        world.step(1 / 120.0)

    penetration = hh - body.c.y
    assert -0.02 < penetration < 0.02, "resting height off by %.4f" % penetration
    assert body.v.length() < 0.02, "box still moving: %r" % (body.v,)
    assert abs(body.w) < 0.02
    assert abs(body.a) < 0.02, "box tipped: %.4f" % body.a


def test_stack_is_stable():
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=10,
                  position_iterations=4)
    world.add_ground(friction=0.9)
    hh = 0.25
    boxes = []
    for i in range(4):
        b = world.create_body(DYNAMIC, Vec2(0.0, hh + 2 * hh * i + 0.01))
        b.add_fixture(Polygon.box(0.4, hh), density=1.0, friction=0.9)
        boxes.append(b)

    for _ in range(900):
        world.step(1 / 120.0)

    for i, b in enumerate(boxes):
        expected = hh + 2 * hh * i
        assert abs(b.c.y - expected) < 0.05, \
            "box %d settled at %.4f, expected ~%.4f" % (i, b.c.y, expected)
        assert abs(b.c.x) < 0.05, "box %d drifted to x=%.4f" % (i, b.c.x)
        assert b.v.length() < 0.05, "box %d jittering: %r" % (i, b.v)


def test_friction_stops_sliding_box():
    world = World(gravity=Vec2(0.0, -10.0))
    world.add_ground(friction=1.0)
    body = world.create_body(DYNAMIC, Vec2(0.0, 0.5))
    body.add_fixture(Polygon.box(0.5, 0.5), density=1.0, friction=1.0)
    body.v = Vec2(5.0, 0.0)

    for _ in range(400):
        world.step(1 / 120.0)
    assert abs(body.v.x) < 0.05, "friction failed to stop the box: vx=%.4f" % body.v.x


def test_frictionless_box_keeps_sliding():
    world = World(gravity=Vec2(0.0, -10.0))
    world.add_ground(friction=0.0)
    body = world.create_body(DYNAMIC, Vec2(0.0, 0.5))
    body.add_fixture(Polygon.box(0.5, 0.5), density=1.0, friction=0.0)
    body.v = Vec2(5.0, 0.0)

    for _ in range(240):
        world.step(1 / 120.0)
    _approx(body.v.x, 5.0, 1e-3)


def test_restitution_bounce_height():
    world = World(gravity=Vec2(0.0, -10.0), velocity_iterations=10)
    world.add_ground(friction=0.0)
    e = 0.8
    ball = world.create_body(DYNAMIC, Vec2(0.0, 2.0))
    ball.add_fixture(Circle(0.2), density=1.0, friction=0.0, restitution=e)

    peak = 0.0
    bounced = False
    for _ in range(1200):
        world.step(1 / 240.0)
        if ball.v.y > 0.0:
            bounced = True
        if bounced and ball.v.y <= 0.0 and peak == 0.0:
            peak = ball.c.y

    drop_height = 2.0 - 0.2
    expected = e * e * drop_height + 0.2
    assert abs(peak - expected) < 0.12, \
        "bounce peak %.3f, expected ~%.3f" % (peak, expected)


# ---------------------------------------------------------------------------
def test_revolute_joint_holds_anchor():
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=10,
                  position_iterations=4)
    anchor = Vec2(0.0, 5.0)
    static = world.create_body(STATIC, anchor)
    static.add_fixture(Circle(0.05), density=0.0)
    arm = world.create_body(DYNAMIC, Vec2(1.0, 5.0))
    arm.add_fixture(Polygon.box(1.0, 0.05), density=2.0)
    joint = world.create_revolute_joint(static, arm, anchor)

    max_error = 0.0
    for _ in range(2400):
        world.step(1 / 240.0)
        pa = static.world_point(joint.local_anchor_a)
        pb = arm.world_point(joint.local_anchor_b)
        max_error = max(max_error, (pa - pb).length())

    assert max_error < 1e-3, "joint drifted by %.6f m" % max_error


def test_pendulum_conserves_energy():
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=12,
                  position_iterations=4)
    anchor = Vec2(0.0, 5.0)
    static = world.create_body(STATIC, anchor)
    arm = world.create_body(DYNAMIC, Vec2(1.0, 5.0))
    arm.add_fixture(Polygon.box(1.0, 0.05), density=2.0)
    world.create_revolute_joint(static, arm, anchor)

    e0 = world.total_energy()
    for _ in range(2400):  # 10 s
        world.step(1 / 240.0)
    e1 = world.total_energy()
    drift = abs(e1 - e0) / max(abs(e0), 1.0)
    assert drift < 0.05, "energy drifted %.2f%% over 10 s" % (100 * drift)


def test_joint_limits_are_respected():
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=10,
                  position_iterations=4)
    anchor = Vec2(0.0, 5.0)
    static = world.create_body(STATIC, anchor)
    arm = world.create_body(DYNAMIC, Vec2(1.0, 5.0))
    arm.add_fixture(Polygon.box(1.0, 0.05), density=2.0)
    lo, hi = -0.4, 0.4
    joint = world.create_revolute_joint(
        static, arm, anchor, lower_angle=lo, upper_angle=hi,
        enable_motor=True, motor_speed=-8.0, max_motor_torque=500.0)

    worst_low = 0.0
    worst_high = 0.0
    for i in range(1200):
        if i == 600:
            joint.set_motor(8.0, 500.0)  # slam it the other way
        world.step(1 / 240.0)
        worst_low = min(worst_low, joint.joint_angle - lo)
        worst_high = max(worst_high, joint.joint_angle - hi)

    assert worst_low > -0.05, "under-shot lower limit by %.4f rad" % (-worst_low)
    assert worst_high < 0.05, "over-shot upper limit by %.4f rad" % worst_high


def test_motor_reaches_target_speed():
    world = World(gravity=Vec2(0.0, 0.0), velocity_iterations=10)
    anchor = Vec2(0.0, 0.0)
    static = world.create_body(STATIC, anchor)
    wheel = world.create_body(DYNAMIC, anchor)
    wheel.add_fixture(Circle(0.5), density=1.0)
    joint = world.create_revolute_joint(
        static, wheel, anchor, enable_motor=True, motor_speed=4.0,
        max_motor_torque=100.0)

    for _ in range(240):
        world.step(1 / 240.0)
    _approx(joint.joint_speed, 4.0, 1e-3)


def test_motor_torque_limit_is_enforced():
    """With a weak motor against gravity the arm must fail to lift."""
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=10)
    anchor = Vec2(0.0, 5.0)
    static = world.create_body(STATIC, anchor)
    arm = world.create_body(DYNAMIC, Vec2(1.0, 5.0))
    arm.add_fixture(Polygon.box(1.0, 0.1), density=5.0)
    world.create_revolute_joint(static, arm, anchor, enable_motor=True,
                                motor_speed=5.0, max_motor_torque=1.0)
    for _ in range(600):
        world.step(1 / 240.0)
    assert arm.c.y < 5.0, "weak motor lifted the arm to y=%.3f" % arm.c.y


# ---------------------------------------------------------------------------
def test_chain_does_not_explode():
    """A 6-link chain hit by gravity should stay finite and bounded."""
    world = World(gravity=Vec2(0.0, -9.81), velocity_iterations=10,
                  position_iterations=4)
    anchor = Vec2(0.0, 8.0)
    prev = world.create_body(STATIC, anchor)
    x = 0.0
    for i in range(6):
        link = world.create_body(DYNAMIC, Vec2(x + 0.3, 8.0))
        link.add_fixture(Polygon.box(0.3, 0.05), density=2.0)
        world.create_revolute_joint(prev, link, Vec2(x, 8.0))
        prev = link
        x += 0.6

    for _ in range(2400):
        world.step(1 / 240.0)
        for b in world.bodies:
            assert b.c.is_finite(), "NaN/inf in chain simulation"
            assert b.v.length() < 100.0, "chain blew up: %r" % (b.v,)


def test_momentum_conserved_in_free_collision():
    """Two circles colliding in zero gravity conserve linear momentum."""
    world = World(gravity=Vec2(0.0, 0.0), velocity_iterations=12)
    a = world.create_body(DYNAMIC, Vec2(-1.0, 0.0))
    a.add_fixture(Circle(0.25), density=1.0, restitution=1.0, friction=0.0)
    b = world.create_body(DYNAMIC, Vec2(1.0, 0.0))
    b.add_fixture(Circle(0.25), density=2.0, restitution=1.0, friction=0.0)
    a.v = Vec2(3.0, 0.0)
    b.v = Vec2(-1.0, 0.0)

    p0 = a.mass * a.v.x + b.mass * b.v.x
    for _ in range(600):
        world.step(1 / 240.0)
    p1 = a.mass * a.v.x + b.mass * b.v.x
    _approx(p1, p0, 1e-4)
    assert a.v.x < b.v.x, "circles failed to separate after impact"


def test_contact_filtering_by_group():
    from .body import Filter
    world = World(gravity=Vec2(0.0, -9.81))
    world.add_ground()
    group = Filter(group=-3)
    a = world.create_body(DYNAMIC, Vec2(0.0, 1.0))
    a.add_fixture(Polygon.box(0.5, 0.5), density=1.0, filter=Filter(group=-3))
    b = world.create_body(DYNAMIC, Vec2(0.2, 1.4))
    b.add_fixture(Polygon.box(0.5, 0.5), density=1.0, filter=Filter(group=-3))

    for _ in range(600):
        world.step(1 / 120.0)
    # Both must fall through each other and land on the ground.
    assert abs(a.c.y - 0.5) < 0.05 and abs(b.c.y - 0.5) < 0.05, \
        "grouped bodies collided: %.3f %.3f" % (a.c.y, b.c.y)


def test_determinism():
    def run():
        world = World(gravity=Vec2(0.0, -9.81))
        world.add_ground(friction=0.8)
        vals = []
        for i in range(5):
            b = world.create_body(DYNAMIC, Vec2(0.11 * i, 1.0 + 0.7 * i), 0.1 * i)
            b.add_fixture(Polygon.box(0.3, 0.2), density=1.0, friction=0.8)
        for _ in range(400):
            world.step(1 / 120.0)
        for b in world.bodies:
            vals.append((b.c.x, b.c.y, b.a))
        return vals

    assert run() == run(), "simulation is not deterministic"


# ---------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %-42s %s" % (fn.__name__, exc))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print("ERROR %-42s %r" % (fn.__name__, exc))
        else:
            print("ok    %s" % fn.__name__)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
