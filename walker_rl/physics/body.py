"""Rigid bodies and fixtures."""

import math

from .math2d import Rot, Transform, Vec2, cross_sv, vec_cross
from .shapes import AABB

STATIC = 0
KINEMATIC = 1
DYNAMIC = 2


class Filter:
    """Collision filtering, Box2D style.

    Two fixtures with the same non-zero `group` always collide (group > 0) or
    never collide (group < 0), regardless of category/mask. The ragdoll uses a
    single negative group so limbs pass through each other but still hit the
    world.
    """

    __slots__ = ("category", "mask", "group")

    def __init__(self, category=0x0001, mask=0xFFFF, group=0):
        self.category = category
        self.mask = mask
        self.group = group


def should_collide(fa, fb):
    a, b = fa.filter, fb.filter
    if a.group == b.group and a.group != 0:
        return a.group > 0
    return (a.mask & b.category) != 0 and (b.mask & a.category) != 0


class Fixture:
    __slots__ = ("shape", "density", "friction", "restitution", "filter", "body",
                 "aabb", "user_data", "index")

    def __init__(self, shape, density=1.0, friction=0.6, restitution=0.0,
                 filter=None, user_data=None):
        self.shape = shape
        self.density = density
        self.friction = friction
        self.restitution = restitution
        self.filter = filter if filter is not None else Filter()
        self.body = None
        self.aabb = AABB()
        self.user_data = user_data
        self.index = 0

    def synchronize(self):
        self.aabb = self.shape.compute_aabb(self.body.xf)


class Body:
    __slots__ = ("type", "xf", "c", "local_center", "a", "v", "w",
                 "force", "torque", "mass", "inv_mass", "inertia", "inv_inertia",
                 "linear_damping", "angular_damping", "gravity_scale",
                 "fixtures", "world", "name", "user_data", "fixed_rotation",
                 "island_awake", "sleep_time", "allow_sleep", "index")

    def __init__(self, body_type=DYNAMIC, position=None, angle=0.0, name=""):
        self.type = body_type
        self.a = float(angle)
        self.xf = Transform(position if position is not None else Vec2(), Rot(angle))
        self.local_center = Vec2()
        self.c = self.xf.p.copy()
        self.v = Vec2()
        self.w = 0.0
        self.force = Vec2()
        self.torque = 0.0
        self.mass = 0.0
        self.inv_mass = 0.0
        self.inertia = 0.0
        self.inv_inertia = 0.0
        self.linear_damping = 0.0
        self.angular_damping = 0.0
        self.gravity_scale = 1.0
        self.fixed_rotation = False
        self.fixtures = []
        self.world = None
        self.name = name
        self.user_data = None
        self.index = 0

    # -- construction -----------------------------------------------------
    def add_fixture(self, shape, density=1.0, friction=0.6, restitution=0.0,
                    filter=None, user_data=None):
        f = Fixture(shape, density, friction, restitution, filter, user_data)
        f.body = self
        f.index = len(self.fixtures)
        self.fixtures.append(f)
        self.reset_mass_data()
        f.synchronize()
        return f

    def reset_mass_data(self):
        self.mass = 0.0
        self.inv_mass = 0.0
        self.inertia = 0.0
        self.inv_inertia = 0.0
        self.local_center = Vec2()

        if self.type != DYNAMIC:
            self.c = self.xf.apply(self.local_center)
            return

        center = Vec2()
        inertia_origin = 0.0
        for f in self.fixtures:
            if f.density <= 0.0:
                continue
            md = f.shape.compute_mass(f.density)
            self.mass += md.mass
            center += md.center * md.mass
            inertia_origin += md.inertia

        if self.mass > 0.0:
            self.inv_mass = 1.0 / self.mass
            center *= self.inv_mass
        else:
            # A dynamic body must have mass; fall back to unit mass at origin.
            self.mass = 1.0
            self.inv_mass = 1.0

        if inertia_origin > 0.0 and not self.fixed_rotation:
            # Shift from the local origin to the centre of mass.
            self.inertia = inertia_origin - self.mass * center.length_sq()
            if self.inertia > 0.0:
                self.inv_inertia = 1.0 / self.inertia
            else:
                self.inertia = 0.0

        old_center = self.c
        self.local_center = center
        self.c = self.xf.apply(self.local_center)
        # Preserve the velocity of the *old* centre of mass point.
        self.v += cross_sv(self.w, self.c - old_center)

    # -- transforms -------------------------------------------------------
    def set_transform(self, position, angle):
        self.xf.p = position.copy()
        self.a = float(angle)
        self.xf.q.set(angle)
        self.c = self.xf.apply(self.local_center)
        self.synchronize_fixtures()

    def synchronize_transform(self):
        self.xf.q.s = math.sin(self.a)
        self.xf.q.c = math.cos(self.a)
        q = self.xf.q
        lc = self.local_center
        self.xf.p = Vec2(
            self.c.x - (q.c * lc.x - q.s * lc.y),
            self.c.y - (q.s * lc.x + q.c * lc.y),
        )

    def synchronize_fixtures(self):
        for f in self.fixtures:
            f.synchronize()

    @property
    def position(self):
        return self.xf.p

    @property
    def angle(self):
        return self.a

    def world_point(self, local_point):
        return self.xf.apply(local_point)

    def local_point(self, world_point):
        return self.xf.inv_apply(world_point)

    def world_vector(self, local_vector):
        return self.xf.q.rotate(local_vector)

    def local_vector(self, world_vector):
        return self.xf.q.inv_rotate(world_vector)

    def linear_velocity_from_world_point(self, world_point):
        return self.v + cross_sv(self.w, world_point - self.c)

    # -- forces -----------------------------------------------------------
    def apply_force(self, force, point=None):
        if self.type != DYNAMIC:
            return
        self.force += force
        if point is not None:
            self.torque += vec_cross(point - self.c, force)

    def apply_torque(self, torque):
        if self.type == DYNAMIC:
            self.torque += torque

    def apply_linear_impulse(self, impulse, point=None):
        if self.type != DYNAMIC:
            return
        self.v += impulse * self.inv_mass
        if point is not None:
            self.w += self.inv_inertia * vec_cross(point - self.c, impulse)

    def apply_angular_impulse(self, impulse):
        if self.type == DYNAMIC:
            self.w += self.inv_inertia * impulse

    def set_velocity(self, v, w=0.0):
        self.v = v.copy()
        self.w = w

    def kinetic_energy(self):
        return 0.5 * (self.mass * self.v.length_sq() + self.inertia * self.w * self.w)

    def compute_aabb(self):
        if not self.fixtures:
            p = self.xf.p
            return AABB(p.x, p.y, p.x, p.y)
        box = self.fixtures[0].shape.compute_aabb(self.xf)
        out = AABB(box.lower_x, box.lower_y, box.upper_x, box.upper_y)
        for f in self.fixtures[1:]:
            out.combine(f.shape.compute_aabb(self.xf))
        return out

    def __repr__(self):
        return "<Body %s type=%d pos=(%.3f, %.3f) angle=%.3f>" % (
            self.name or "?", self.type, self.xf.p.x, self.xf.p.y, self.a)
