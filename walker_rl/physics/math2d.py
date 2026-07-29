"""Minimal 2D math primitives for the physics engine.

Everything is plain Python floats -- no numpy -- because the engine touches
these objects millions of times per training run and numpy scalar overhead is
far worse than tuple/attribute access at this size.
"""

import math

EPSILON = 1e-9


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    # -- construction -----------------------------------------------------
    def copy(self):
        return Vec2(self.x, self.y)

    def set(self, x, y):
        self.x = x
        self.y = y
        return self

    def set_zero(self):
        self.x = 0.0
        self.y = 0.0
        return self

    # -- operators --------------------------------------------------------
    def __add__(self, o):
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, s):
        return Vec2(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec2(self.x / s, self.y / s)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __iadd__(self, o):
        self.x += o.x
        self.y += o.y
        return self

    def __isub__(self, o):
        self.x -= o.x
        self.y -= o.y
        return self

    def __imul__(self, s):
        self.x *= s
        self.y *= s
        return self

    def __eq__(self, o):
        return isinstance(o, Vec2) and self.x == o.x and self.y == o.y

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return "Vec2(%.6g, %.6g)" % (self.x, self.y)

    # -- geometry ---------------------------------------------------------
    def dot(self, o):
        return self.x * o.x + self.y * o.y

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def length_sq(self):
        return self.x * self.x + self.y * self.y

    def normalize(self):
        """In-place normalize, returns the previous length."""
        ln = math.sqrt(self.x * self.x + self.y * self.y)
        if ln < EPSILON:
            return 0.0
        inv = 1.0 / ln
        self.x *= inv
        self.y *= inv
        return ln

    def normalized(self):
        ln = math.sqrt(self.x * self.x + self.y * self.y)
        if ln < EPSILON:
            return Vec2(0.0, 0.0)
        inv = 1.0 / ln
        return Vec2(self.x * inv, self.y * inv)

    def skew(self):
        """Perpendicular vector: cross(1, self)."""
        return Vec2(-self.y, self.x)

    def is_finite(self):
        return math.isfinite(self.x) and math.isfinite(self.y)


def vec_cross(a, b):
    """Scalar z-component of the 3D cross product of two planar vectors."""
    return a.x * b.y - a.y * b.x


def cross_vs(v, s):
    """cross(v, s) where s is a scalar (z-axis) -> Vec2."""
    return Vec2(s * v.y, -s * v.x)


def cross_sv(s, v):
    """cross(s, v) where s is a scalar (z-axis) -> Vec2."""
    return Vec2(-s * v.y, s * v.x)


def dot(a, b):
    return a.x * b.x + a.y * b.y


def distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def clamp_vec(v, lo, hi):
    return Vec2(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y))


class Rot:
    """A 2D rotation stored as (sin, cos) -- avoids repeated trig calls."""

    __slots__ = ("s", "c")

    def __init__(self, angle=0.0):
        self.s = math.sin(angle)
        self.c = math.cos(angle)

    def set(self, angle):
        self.s = math.sin(angle)
        self.c = math.cos(angle)
        return self

    def set_identity(self):
        self.s = 0.0
        self.c = 1.0
        return self

    def angle(self):
        return math.atan2(self.s, self.c)

    def copy(self):
        r = Rot.__new__(Rot)
        r.s = self.s
        r.c = self.c
        return r

    def rotate(self, v):
        return Vec2(self.c * v.x - self.s * v.y, self.s * v.x + self.c * v.y)

    def inv_rotate(self, v):
        return Vec2(self.c * v.x + self.s * v.y, -self.s * v.x + self.c * v.y)

    def x_axis(self):
        return Vec2(self.c, self.s)

    def y_axis(self):
        return Vec2(-self.s, self.c)


class Transform:
    """Rigid transform: world = q * local + p."""

    __slots__ = ("p", "q")

    def __init__(self, p=None, q=None):
        self.p = p.copy() if p is not None else Vec2()
        self.q = q.copy() if q is not None else Rot()

    def set_identity(self):
        self.p.set_zero()
        self.q.set_identity()
        return self

    def copy(self):
        return Transform(self.p, self.q)

    def apply(self, v):
        """local -> world"""
        q = self.q
        return Vec2(
            q.c * v.x - q.s * v.y + self.p.x,
            q.s * v.x + q.c * v.y + self.p.y,
        )

    def inv_apply(self, v):
        """world -> local"""
        px = v.x - self.p.x
        py = v.y - self.p.y
        q = self.q
        return Vec2(q.c * px + q.s * py, -q.s * px + q.c * py)


def normalize_angle(a):
    """Wrap an angle into [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
