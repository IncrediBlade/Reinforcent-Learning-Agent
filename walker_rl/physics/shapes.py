"""Collision shapes and their mass properties.

Two shape types are supported: convex polygons (built with a convex hull so
callers cannot accidentally feed in a non-convex or badly wound point set) and
circles. That is enough for a ragdoll on a ground plane, and keeps the
narrow-phase small enough to stay correct.
"""

import math

from .math2d import EPSILON, Rot, Transform, Vec2, vec_cross

POLYGON = 0
CIRCLE = 1

MAX_POLYGON_VERTICES = 8


class MassData:
    __slots__ = ("mass", "center", "inertia")

    def __init__(self, mass=0.0, center=None, inertia=0.0):
        self.mass = mass
        self.center = center if center is not None else Vec2()
        self.inertia = inertia  # about the origin of the shape's local frame


class AABB:
    __slots__ = ("lower_x", "lower_y", "upper_x", "upper_y")

    def __init__(self, lx=0.0, ly=0.0, ux=0.0, uy=0.0):
        self.lower_x = lx
        self.lower_y = ly
        self.upper_x = ux
        self.upper_y = uy

    def overlaps(self, o, margin=0.0):
        if o.lower_x - margin > self.upper_x or o.upper_x + margin < self.lower_x:
            return False
        if o.lower_y - margin > self.upper_y or o.upper_y + margin < self.lower_y:
            return False
        return True

    def combine(self, o):
        self.lower_x = min(self.lower_x, o.lower_x)
        self.lower_y = min(self.lower_y, o.lower_y)
        self.upper_x = max(self.upper_x, o.upper_x)
        self.upper_y = max(self.upper_y, o.upper_y)


class Shape:
    type = -1
    radius = 0.0

    def compute_aabb(self, xf):
        raise NotImplementedError

    def compute_mass(self, density):
        raise NotImplementedError


class Circle(Shape):
    type = CIRCLE
    __slots__ = ("p", "radius")

    def __init__(self, radius, center=None):
        self.radius = float(radius)
        self.p = center.copy() if center is not None else Vec2()

    def compute_aabb(self, xf):
        c = xf.apply(self.p)
        r = self.radius
        return AABB(c.x - r, c.y - r, c.x + r, c.y + r)

    def compute_mass(self, density):
        r = self.radius
        mass = density * math.pi * r * r
        center = self.p.copy()
        # Second moment about the shape-local origin (parallel axis included).
        inertia = mass * (0.5 * r * r + self.p.length_sq())
        return MassData(mass, center, inertia)

    def support_local(self, d):
        return self.p


class Polygon(Shape):
    type = POLYGON
    __slots__ = ("vertices", "normals", "centroid", "count", "radius")

    def __init__(self, points):
        hull = _convex_hull(points)
        if len(hull) < 3:
            raise ValueError("polygon needs at least 3 non-degenerate points")
        if len(hull) > MAX_POLYGON_VERTICES:
            raise ValueError("polygon has too many vertices (max %d)" % MAX_POLYGON_VERTICES)
        self.vertices = hull
        self.count = len(hull)
        self.radius = 0.0
        normals = []
        for i in range(self.count):
            v1 = hull[i]
            v2 = hull[(i + 1) % self.count]
            edge = v2 - v1
            if edge.length_sq() <= EPSILON * EPSILON:
                raise ValueError("degenerate polygon edge")
            n = Vec2(edge.y, -edge.x)  # outward for CCW winding
            n.normalize()
            normals.append(n)
        self.normals = normals
        self.centroid = _polygon_centroid(hull)

    # -- factories --------------------------------------------------------
    @staticmethod
    def box(half_width, half_height, center=None, angle=0.0):
        hw, hh = float(half_width), float(half_height)
        pts = [Vec2(-hw, -hh), Vec2(hw, -hh), Vec2(hw, hh), Vec2(-hw, hh)]
        if angle != 0.0 or center is not None:
            rot = Rot(angle)
            off = center if center is not None else Vec2()
            xf = Transform(off, rot)
            pts = [xf.apply(p) for p in pts]
        return Polygon(pts)

    def compute_aabb(self, xf):
        v0 = xf.apply(self.vertices[0])
        lx = ux = v0.x
        ly = uy = v0.y
        for i in range(1, self.count):
            v = xf.apply(self.vertices[i])
            if v.x < lx:
                lx = v.x
            elif v.x > ux:
                ux = v.x
            if v.y < ly:
                ly = v.y
            elif v.y > uy:
                uy = v.y
        return AABB(lx, ly, ux, uy)

    def compute_mass(self, density):
        """Exact polygon mass properties by triangle decomposition."""
        area = 0.0
        cx = 0.0
        cy = 0.0
        second_moment = 0.0
        ref = self.vertices[0]
        inv3 = 1.0 / 3.0
        for i in range(1, self.count - 1):
            e1 = self.vertices[i] - ref
            e2 = self.vertices[i + 1] - ref
            d = vec_cross(e1, e2)
            tri_area = 0.5 * d
            area += tri_area
            # Triangle centroid relative to ref.
            gx = inv3 * (e1.x + e2.x)
            gy = inv3 * (e1.y + e2.y)
            cx += tri_area * gx
            cy += tri_area * gy
            intx2 = e1.x * e1.x + e2.x * e1.x + e2.x * e2.x
            inty2 = e1.y * e1.y + e2.y * e1.y + e2.y * e2.y
            second_moment += (0.25 * inv3 * d) * (intx2 + inty2)

        if area <= EPSILON:
            raise ValueError("polygon has ~zero area")

        mass = density * area
        cx /= area
        cy /= area
        center = Vec2(ref.x + cx, ref.y + cy)
        # second_moment is about `ref`; shift to the shape-local origin.
        inertia = density * second_moment
        inertia += mass * (center.length_sq() - Vec2(cx, cy).length_sq())
        return MassData(mass, center, inertia)


def _convex_hull(points):
    """Andrew's monotone chain, returning CCW vertices with collinear points removed."""
    pts = sorted({(round(p.x, 12), round(p.y, 12)) for p in points})
    if len(pts) < 3:
        return [Vec2(x, y) for x, y in pts]

    def cross_o(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross_o(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross_o(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [Vec2(x, y) for x, y in hull]


def _polygon_centroid(verts):
    n = len(verts)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        cr = vec_cross(p1, p2)
        area += cr
        cx += (p1.x + p2.x) * cr
        cy += (p1.y + p2.y) * cr
    area *= 0.5
    if abs(area) < EPSILON:
        return Vec2(verts[0].x, verts[0].y)
    factor = 1.0 / (6.0 * area)
    return Vec2(cx * factor, cy * factor)
