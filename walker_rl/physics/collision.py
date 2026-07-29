"""Narrow-phase collision detection.

Manifolds are produced in *local* coordinates (a reference face plus clipped
incident points) rather than as world-space points. That is what lets the
position solver re-evaluate penetration depth after bodies move during an
iteration, which is the difference between a stable stack and a jittery one.
"""

from .math2d import EPSILON, Rot, Transform, Vec2
from .shapes import CIRCLE, POLYGON

# Manifold types
MANIFOLD_CIRCLES = 0
MANIFOLD_FACE_A = 1
MANIFOLD_FACE_B = 2

LINEAR_SLOP = 0.005          # allowed overlap, prevents jitter
MAX_LINEAR_CORRECTION = 0.2  # per position-iteration clamp
BAUMGARTE = 0.2              # position correction rate
VELOCITY_THRESHOLD = 1.0     # below this closing speed, restitution is dropped


class ManifoldPoint:
    __slots__ = ("local_point", "normal_impulse", "tangent_impulse", "id")

    def __init__(self, local_point=None, point_id=0):
        self.local_point = local_point if local_point is not None else Vec2()
        self.normal_impulse = 0.0
        self.tangent_impulse = 0.0
        self.id = point_id


class Manifold:
    __slots__ = ("points", "local_normal", "local_point", "type")

    def __init__(self):
        self.points = []
        self.local_normal = Vec2()
        self.local_point = Vec2()
        self.type = MANIFOLD_CIRCLES

    @property
    def count(self):
        return len(self.points)


class WorldManifold:
    """World-space view of a manifold, recomputed from current transforms."""

    __slots__ = ("normal", "points", "separations")

    def __init__(self):
        self.normal = Vec2()
        self.points = []
        self.separations = []

    def initialize(self, manifold, xf_a, radius_a, xf_b, radius_b):
        self.points = []
        self.separations = []
        if not manifold.points:
            return

        if manifold.type == MANIFOLD_CIRCLES:
            pa = xf_a.apply(manifold.local_point)
            pb = xf_b.apply(manifold.points[0].local_point)
            d = pb - pa
            if d.length_sq() > EPSILON * EPSILON:
                normal = d.normalized()
            else:
                normal = Vec2(0.0, 1.0)
            self.normal = normal
            ca = pa + normal * radius_a
            cb = pb - normal * radius_b
            self.points.append((ca + cb) * 0.5)
            self.separations.append((cb - ca).dot(normal))

        elif manifold.type == MANIFOLD_FACE_A:
            normal = xf_a.q.rotate(manifold.local_normal)
            plane_point = xf_a.apply(manifold.local_point)
            self.normal = normal
            for mp in manifold.points:
                clip_point = xf_b.apply(mp.local_point)
                s = (clip_point - plane_point).dot(normal)
                ca = clip_point + normal * (radius_a - s)
                cb = clip_point - normal * radius_b
                self.points.append((ca + cb) * 0.5)
                self.separations.append((cb - ca).dot(normal))

        else:  # MANIFOLD_FACE_B
            normal = xf_b.q.rotate(manifold.local_normal)
            plane_point = xf_b.apply(manifold.local_point)
            for mp in manifold.points:
                clip_point = xf_a.apply(mp.local_point)
                s = (clip_point - plane_point).dot(normal)
                cb = clip_point + normal * (radius_b - s)
                ca = clip_point - normal * radius_a
                self.points.append((ca + cb) * 0.5)
                self.separations.append((ca - cb).dot(normal))
            # Manifold normal points from A to B by convention.
            self.normal = -normal


class ClipVertex:
    __slots__ = ("v", "id")

    def __init__(self, v, point_id):
        self.v = v
        self.id = point_id


def _make_id(index_a, index_b, type_a, type_b):
    return (index_a & 0xFF) | ((index_b & 0xFF) << 8) | ((type_a & 0xF) << 16) | ((type_b & 0xF) << 20)


def _mul_t_transform(a, b):
    """Return a^-1 * b."""
    q = Rot.__new__(Rot)
    q.s = a.q.c * b.q.s - a.q.s * b.q.c
    q.c = a.q.c * b.q.c + a.q.s * b.q.s
    p = a.q.inv_rotate(b.p - a.p)
    t = Transform.__new__(Transform)
    t.p = p
    t.q = q
    return t


def clip_segment_to_line(v_in, normal, offset, vertex_index_a):
    """Sutherland-Hodgman clip of a 2-point segment against a half-plane."""
    v_out = []
    d0 = normal.dot(v_in[0].v) - offset
    d1 = normal.dot(v_in[1].v) - offset
    if d0 <= 0.0:
        v_out.append(v_in[0])
    if d1 <= 0.0:
        v_out.append(v_in[1])
    if d0 * d1 < 0.0:
        interp = d0 / (d0 - d1)
        v = v_in[0].v + (v_in[1].v - v_in[0].v) * interp
        # The intersection lands on vertex_index_a of poly1 and edge of poly2.
        new_id = _make_id(vertex_index_a, v_in[0].id >> 8 & 0xFF, 0, 1)
        v_out.append(ClipVertex(v, new_id))
    return v_out[:2]


# ---------------------------------------------------------------------------
# circle vs circle
# ---------------------------------------------------------------------------
def collide_circles(circle_a, xf_a, circle_b, xf_b):
    m = Manifold()
    pa = xf_a.apply(circle_a.p)
    pb = xf_b.apply(circle_b.p)
    d = pb - pa
    radius = circle_a.radius + circle_b.radius
    if d.length_sq() > radius * radius:
        return m
    m.type = MANIFOLD_CIRCLES
    m.local_point = circle_a.p.copy()
    m.local_normal.set_zero()
    m.points.append(ManifoldPoint(circle_b.p.copy(), 0))
    return m


# ---------------------------------------------------------------------------
# polygon vs circle
# ---------------------------------------------------------------------------
def collide_polygon_circle(poly_a, xf_a, circle_b, xf_b):
    m = Manifold()
    c_world = xf_b.apply(circle_b.p)
    c_local = xf_a.inv_apply(c_world)
    radius = poly_a.radius + circle_b.radius

    # Deepest penetrating face.
    normal_index = 0
    separation = -1e30
    for i in range(poly_a.count):
        s = poly_a.normals[i].dot(c_local - poly_a.vertices[i])
        if s > radius:
            return m
        if s > separation:
            separation = s
            normal_index = i

    v1 = poly_a.vertices[normal_index]
    v2 = poly_a.vertices[(normal_index + 1) % poly_a.count]

    if separation < EPSILON:
        # Circle centre is inside the polygon.
        m.type = MANIFOLD_FACE_A
        m.local_normal = poly_a.normals[normal_index].copy()
        m.local_point = (v1 + v2) * 0.5
        m.points.append(ManifoldPoint(circle_b.p.copy(), 0))
        return m

    # Voronoi region test.
    u1 = (c_local - v1).dot(v2 - v1)
    u2 = (c_local - v2).dot(v1 - v2)
    if u1 <= 0.0:
        if (c_local - v1).length_sq() > radius * radius:
            return m
        m.type = MANIFOLD_FACE_A
        m.local_normal = (c_local - v1).normalized()
        m.local_point = v1.copy()
    elif u2 <= 0.0:
        if (c_local - v2).length_sq() > radius * radius:
            return m
        m.type = MANIFOLD_FACE_A
        m.local_normal = (c_local - v2).normalized()
        m.local_point = v2.copy()
    else:
        face_center = (v1 + v2) * 0.5
        s = (c_local - face_center).dot(poly_a.normals[normal_index])
        if s > radius:
            return m
        m.type = MANIFOLD_FACE_A
        m.local_normal = poly_a.normals[normal_index].copy()
        m.local_point = face_center

    m.points.append(ManifoldPoint(circle_b.p.copy(), 0))
    return m


def collide_circle_polygon(circle_a, xf_a, poly_b, xf_b):
    """Wrapper keeping A/B order: produces a FACE_B manifold."""
    m = collide_polygon_circle(poly_b, xf_b, circle_a, xf_a)
    if m.points:
        m.type = MANIFOLD_FACE_B
    return m


# ---------------------------------------------------------------------------
# polygon vs polygon
# ---------------------------------------------------------------------------
def _find_max_separation(poly1, xf1, poly2, xf2):
    xf = _mul_t_transform(xf2, xf1)  # poly1 frame -> poly2 frame
    best_index = 0
    max_separation = -1e30
    verts2 = poly2.vertices
    count2 = poly2.count
    for i in range(poly1.count):
        n = xf.q.rotate(poly1.normals[i])
        v1 = xf.apply(poly1.vertices[i])
        si = 1e30
        for j in range(count2):
            sij = n.x * (verts2[j].x - v1.x) + n.y * (verts2[j].y - v1.y)
            if sij < si:
                si = sij
        if si > max_separation:
            max_separation = si
            best_index = i
    return best_index, max_separation


def _find_incident_edge(poly1, xf1, edge1, poly2, xf2):
    normal1 = xf2.q.inv_rotate(xf1.q.rotate(poly1.normals[edge1]))
    index = 0
    min_dot = 1e30
    for i in range(poly2.count):
        d = normal1.dot(poly2.normals[i])
        if d < min_dot:
            min_dot = d
            index = i
    i1 = index
    i2 = (i1 + 1) % poly2.count
    return [
        ClipVertex(xf2.apply(poly2.vertices[i1]), _make_id(edge1, i1, 0, 1)),
        ClipVertex(xf2.apply(poly2.vertices[i2]), _make_id(edge1, i2, 0, 1)),
    ]


def collide_polygons(poly_a, xf_a, poly_b, xf_b):
    m = Manifold()
    total_radius = poly_a.radius + poly_b.radius

    edge_a, separation_a = _find_max_separation(poly_a, xf_a, poly_b, xf_b)
    if separation_a > total_radius:
        return m
    edge_b, separation_b = _find_max_separation(poly_b, xf_b, poly_a, xf_a)
    if separation_b > total_radius:
        return m

    # Prefer face A unless B is clearly better (hysteresis avoids flip-flopping
    # between reference faces on nearly-parallel contacts).
    if separation_b > 0.98 * separation_a + 0.001:
        poly1, xf1, poly2, xf2, edge1 = poly_b, xf_b, poly_a, xf_a, edge_b
        m.type = MANIFOLD_FACE_B
        flip = True
    else:
        poly1, xf1, poly2, xf2, edge1 = poly_a, xf_a, poly_b, xf_b, edge_a
        m.type = MANIFOLD_FACE_A
        flip = False

    incident = _find_incident_edge(poly1, xf1, edge1, poly2, xf2)

    iv1 = edge1
    iv2 = (edge1 + 1) % poly1.count
    v11 = poly1.vertices[iv1]
    v12 = poly1.vertices[iv2]

    local_tangent = (v12 - v11)
    local_tangent.normalize()
    local_normal = Vec2(local_tangent.y, -local_tangent.x)
    plane_point = (v11 + v12) * 0.5

    tangent = xf1.q.rotate(local_tangent)
    normal = Vec2(tangent.y, -tangent.x)

    v11w = xf1.apply(v11)
    v12w = xf1.apply(v12)

    front_offset = normal.dot(v11w)
    side_offset1 = -tangent.dot(v11w) + total_radius
    side_offset2 = tangent.dot(v12w) + total_radius

    clip1 = clip_segment_to_line(incident, -tangent, side_offset1, iv1)
    if len(clip1) < 2:
        return m
    clip2 = clip_segment_to_line(clip1, tangent, side_offset2, iv2)
    if len(clip2) < 2:
        return m

    m.local_normal = local_normal
    m.local_point = plane_point

    for cv in clip2:
        separation = normal.dot(cv.v) - front_offset
        if separation <= total_radius:
            mp = ManifoldPoint(xf2.inv_apply(cv.v), cv.id)
            if flip:
                # Swap the stored feature indices so warm-start ids stay stable.
                mp.id = _make_id((cv.id >> 8) & 0xFF, cv.id & 0xFF, 1, 0)
            m.points.append(mp)

    return m


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def collide(shape_a, xf_a, shape_b, xf_b):
    ta, tb = shape_a.type, shape_b.type
    if ta == POLYGON and tb == POLYGON:
        return collide_polygons(shape_a, xf_a, shape_b, xf_b)
    if ta == POLYGON and tb == CIRCLE:
        return collide_polygon_circle(shape_a, xf_a, shape_b, xf_b)
    if ta == CIRCLE and tb == POLYGON:
        return collide_circle_polygon(shape_a, xf_a, shape_b, xf_b)
    return collide_circles(shape_a, xf_a, shape_b, xf_b)
