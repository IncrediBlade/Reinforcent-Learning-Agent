"""Persistent contacts and the sequential-impulse contact solver.

The solver is the standard Catto formulation: warm-started accumulated
impulses, friction solved before the normal row, restitution folded into a
velocity bias, and a separate pseudo-position pass so penetration recovery
never injects energy into the velocity state.
"""

import math

from .collision import (BAUMGARTE, LINEAR_SLOP, MANIFOLD_CIRCLES,
                        MANIFOLD_FACE_A, MAX_LINEAR_CORRECTION,
                        VELOCITY_THRESHOLD, WorldManifold, collide)
from .math2d import EPSILON, Rot, Transform, Vec2, cross_sv, vec_cross

MAX_POSITION_CORRECTION = 3.0 * LINEAR_SLOP


class Contact:
    __slots__ = ("fixture_a", "fixture_b", "manifold", "friction", "restitution",
                 "touching", "enabled", "tangent_speed", "key")

    def __init__(self, fixture_a, fixture_b, key):
        self.fixture_a = fixture_a
        self.fixture_b = fixture_b
        self.manifold = None
        self.friction = math.sqrt(fixture_a.friction * fixture_b.friction)
        self.restitution = max(fixture_a.restitution, fixture_b.restitution)
        self.touching = False
        self.enabled = True
        self.tangent_speed = 0.0
        self.key = key

    @property
    def body_a(self):
        return self.fixture_a.body

    @property
    def body_b(self):
        return self.fixture_b.body

    def update(self):
        """Recompute the manifold, carrying accumulated impulses across frames."""
        old = self.manifold
        body_a = self.fixture_a.body
        body_b = self.fixture_b.body
        new = collide(self.fixture_a.shape, body_a.xf,
                      self.fixture_b.shape, body_b.xf)

        if old is not None and new.points:
            # Warm-start transfer keyed on contact feature ids.
            for mp in new.points:
                for omp in old.points:
                    if omp.id == mp.id:
                        mp.normal_impulse = omp.normal_impulse
                        mp.tangent_impulse = omp.tangent_impulse
                        break

        self.manifold = new
        was_touching = self.touching
        self.touching = len(new.points) > 0
        return was_touching, self.touching

    def world_manifold(self):
        wm = WorldManifold()
        wm.initialize(self.manifold,
                      self.fixture_a.body.xf, self.fixture_a.shape.radius,
                      self.fixture_b.body.xf, self.fixture_b.shape.radius)
        return wm

    def total_normal_impulse(self):
        if self.manifold is None:
            return 0.0
        return sum(p.normal_impulse for p in self.manifold.points)


class _VCPoint:
    __slots__ = ("r_a", "r_b", "normal_impulse", "tangent_impulse",
                 "normal_mass", "tangent_mass", "velocity_bias")

    def __init__(self):
        self.r_a = Vec2()
        self.r_b = Vec2()
        self.normal_impulse = 0.0
        self.tangent_impulse = 0.0
        self.normal_mass = 0.0
        self.tangent_mass = 0.0
        self.velocity_bias = 0.0


class _VelocityConstraint:
    __slots__ = ("points", "normal", "tangent", "friction", "restitution",
                 "body_a", "body_b", "contact")

    def __init__(self):
        self.points = []
        self.normal = Vec2()
        self.tangent = Vec2()
        self.friction = 0.0
        self.restitution = 0.0
        self.body_a = None
        self.body_b = None
        self.contact = None


class _PositionConstraint:
    __slots__ = ("local_points", "local_normal", "local_point", "type",
                 "radius_a", "radius_b", "body_a", "body_b")

    def __init__(self):
        self.local_points = []
        self.local_normal = Vec2()
        self.local_point = Vec2()
        self.type = MANIFOLD_CIRCLES
        self.radius_a = 0.0
        self.radius_b = 0.0
        self.body_a = None
        self.body_b = None


def _position_solver_manifold(pc, xf_a, xf_b, index):
    """Recompute (normal, point, separation) for one manifold point."""
    if pc.type == MANIFOLD_CIRCLES:
        pa = xf_a.apply(pc.local_point)
        pb = xf_b.apply(pc.local_points[0])
        d = pb - pa
        normal = d.normalized() if d.length_sq() > EPSILON * EPSILON else Vec2(0.0, 1.0)
        point = (pa + pb) * 0.5
        separation = d.dot(normal) - pc.radius_a - pc.radius_b
        return normal, point, separation

    if pc.type == MANIFOLD_FACE_A:
        normal = xf_a.q.rotate(pc.local_normal)
        plane_point = xf_a.apply(pc.local_point)
        clip_point = xf_b.apply(pc.local_points[index])
        separation = (clip_point - plane_point).dot(normal) - pc.radius_a - pc.radius_b
        return normal, clip_point, separation

    normal = xf_b.q.rotate(pc.local_normal)
    plane_point = xf_b.apply(pc.local_point)
    clip_point = xf_a.apply(pc.local_points[index])
    separation = (clip_point - plane_point).dot(normal) - pc.radius_a - pc.radius_b
    return -normal, clip_point, separation


class ContactSolver:
    """Builds and solves the contact constraint set for one step."""

    __slots__ = ("velocity_constraints", "position_constraints", "dt", "warm_start")

    def __init__(self, contacts, dt, warm_start=True):
        self.dt = dt
        self.warm_start = warm_start
        self.velocity_constraints = []
        self.position_constraints = []

        for contact in contacts:
            manifold = contact.manifold
            if manifold is None or not manifold.points:
                continue
            body_a = contact.fixture_a.body
            body_b = contact.fixture_b.body

            vc = _VelocityConstraint()
            vc.contact = contact
            vc.body_a = body_a
            vc.body_b = body_b
            vc.friction = contact.friction
            vc.restitution = contact.restitution
            self.velocity_constraints.append(vc)

            pc = _PositionConstraint()
            pc.body_a = body_a
            pc.body_b = body_b
            pc.local_normal = manifold.local_normal.copy()
            pc.local_point = manifold.local_point.copy()
            pc.type = manifold.type
            pc.radius_a = contact.fixture_a.shape.radius
            pc.radius_b = contact.fixture_b.shape.radius
            pc.local_points = [mp.local_point.copy() for mp in manifold.points]
            self.position_constraints.append(pc)

            for mp in manifold.points:
                p = _VCPoint()
                p.normal_impulse = mp.normal_impulse if warm_start else 0.0
                p.tangent_impulse = mp.tangent_impulse if warm_start else 0.0
                vc.points.append(p)

    # -- setup ------------------------------------------------------------
    def initialize(self):
        for vc in self.velocity_constraints:
            contact = vc.contact
            body_a, body_b = vc.body_a, vc.body_b
            wm = contact.world_manifold()
            vc.normal = wm.normal
            vc.tangent = Vec2(wm.normal.y, -wm.normal.x)

            m_a, m_b = body_a.inv_mass, body_b.inv_mass
            i_a, i_b = body_a.inv_inertia, body_b.inv_inertia
            normal = vc.normal
            tangent = vc.tangent

            for j, p in enumerate(vc.points):
                point = wm.points[j]
                p.r_a = point - body_a.c
                p.r_b = point - body_b.c

                rn_a = vec_cross(p.r_a, normal)
                rn_b = vec_cross(p.r_b, normal)
                k_normal = m_a + m_b + i_a * rn_a * rn_a + i_b * rn_b * rn_b
                p.normal_mass = 1.0 / k_normal if k_normal > 0.0 else 0.0

                rt_a = vec_cross(p.r_a, tangent)
                rt_b = vec_cross(p.r_b, tangent)
                k_tangent = m_a + m_b + i_a * rt_a * rt_a + i_b * rt_b * rt_b
                p.tangent_mass = 1.0 / k_tangent if k_tangent > 0.0 else 0.0

                # Restitution: only for genuine impacts, not resting contact.
                dv = (body_b.v + cross_sv(body_b.w, p.r_b)
                      - body_a.v - cross_sv(body_a.w, p.r_a))
                vn = dv.dot(normal)
                p.velocity_bias = 0.0
                if vn < -VELOCITY_THRESHOLD:
                    p.velocity_bias = -vc.restitution * vn

    def warm_start_impulses(self):
        if not self.warm_start:
            return
        for vc in self.velocity_constraints:
            body_a, body_b = vc.body_a, vc.body_b
            normal, tangent = vc.normal, vc.tangent
            for p in vc.points:
                impulse = normal * p.normal_impulse + tangent * p.tangent_impulse
                body_a.v -= impulse * body_a.inv_mass
                body_a.w -= body_a.inv_inertia * vec_cross(p.r_a, impulse)
                body_b.v += impulse * body_b.inv_mass
                body_b.w += body_b.inv_inertia * vec_cross(p.r_b, impulse)

    # -- velocity ---------------------------------------------------------
    def solve_velocity_constraints(self):
        for vc in self.velocity_constraints:
            body_a, body_b = vc.body_a, vc.body_b
            m_a, m_b = body_a.inv_mass, body_b.inv_mass
            i_a, i_b = body_a.inv_inertia, body_b.inv_inertia
            normal, tangent = vc.normal, vc.tangent
            friction = vc.friction

            va, vb = body_a.v, body_b.v
            tx, ty = tangent.x, tangent.y
            nx, ny = normal.x, normal.y

            # Friction first: it uses last iteration's normal impulse as bound,
            # which converges better than the other order.
            for p in vc.points:
                ra, rb = p.r_a, p.r_b
                dvx = (vb.x - body_b.w * rb.y) - (va.x - body_a.w * ra.y)
                dvy = (vb.y + body_b.w * rb.x) - (va.y + body_a.w * ra.x)
                vt = dvx * tx + dvy * ty
                lam = p.tangent_mass * (-vt)
                max_friction = friction * p.normal_impulse
                new_impulse = p.tangent_impulse + lam
                if new_impulse < -max_friction:
                    new_impulse = -max_friction
                elif new_impulse > max_friction:
                    new_impulse = max_friction
                lam = new_impulse - p.tangent_impulse
                p.tangent_impulse = new_impulse

                px = tx * lam
                py = ty * lam
                va.x -= m_a * px
                va.y -= m_a * py
                body_a.w -= i_a * (ra.x * py - ra.y * px)
                vb.x += m_b * px
                vb.y += m_b * py
                body_b.w += i_b * (rb.x * py - rb.y * px)

            for p in vc.points:
                ra, rb = p.r_a, p.r_b
                dvx = (vb.x - body_b.w * rb.y) - (va.x - body_a.w * ra.y)
                dvy = (vb.y + body_b.w * rb.x) - (va.y + body_a.w * ra.x)
                vn = dvx * nx + dvy * ny
                lam = -p.normal_mass * (vn - p.velocity_bias)
                new_impulse = p.normal_impulse + lam
                if new_impulse < 0.0:
                    new_impulse = 0.0
                lam = new_impulse - p.normal_impulse
                p.normal_impulse = new_impulse

                px = nx * lam
                py = ny * lam
                va.x -= m_a * px
                va.y -= m_a * py
                body_a.w -= i_a * (ra.x * py - ra.y * px)
                vb.x += m_b * px
                vb.y += m_b * py
                body_b.w += i_b * (rb.x * py - rb.y * px)

    def store_impulses(self):
        for vc in self.velocity_constraints:
            manifold = vc.contact.manifold
            for j, p in enumerate(vc.points):
                manifold.points[j].normal_impulse = p.normal_impulse
                manifold.points[j].tangent_impulse = p.tangent_impulse

    # -- position ---------------------------------------------------------
    def solve_position_constraints(self):
        min_separation = 0.0
        for pc in self.position_constraints:
            body_a, body_b = pc.body_a, pc.body_b
            m_a, i_a = body_a.inv_mass, body_a.inv_inertia
            m_b, i_b = body_b.inv_mass, body_b.inv_inertia
            if m_a == 0.0 and m_b == 0.0:
                continue

            for j in range(len(pc.local_points)):
                xf_a = _body_transform(body_a)
                xf_b = _body_transform(body_b)
                normal, point, separation = _position_solver_manifold(pc, xf_a, xf_b, j)
                r_a = point - body_a.c
                r_b = point - body_b.c
                if separation < min_separation:
                    min_separation = separation

                c = BAUMGARTE * (separation + LINEAR_SLOP)
                if c < -MAX_LINEAR_CORRECTION:
                    c = -MAX_LINEAR_CORRECTION
                elif c > 0.0:
                    c = 0.0

                rn_a = vec_cross(r_a, normal)
                rn_b = vec_cross(r_b, normal)
                k = m_a + m_b + i_a * rn_a * rn_a + i_b * rn_b * rn_b
                impulse = -c / k if k > 0.0 else 0.0
                p = normal * impulse

                body_a.c -= p * m_a
                body_a.a -= i_a * vec_cross(r_a, p)
                body_b.c += p * m_b
                body_b.a += i_b * vec_cross(r_b, p)
                body_a.synchronize_transform()
                body_b.synchronize_transform()

        return min_separation >= -MAX_POSITION_CORRECTION


def _body_transform(body):
    q = Rot.__new__(Rot)
    q.s = math.sin(body.a)
    q.c = math.cos(body.a)
    lc = body.local_center
    t = Transform.__new__(Transform)
    t.q = q
    t.p = Vec2(body.c.x - (q.c * lc.x - q.s * lc.y),
               body.c.y - (q.s * lc.x + q.c * lc.y))
    return t
