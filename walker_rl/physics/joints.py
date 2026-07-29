"""Joints. Only what a ragdoll needs: a revolute joint with an angular motor
and hard angle limits.

The limit formulation is the modern Box2D one (separate accumulated lower and
upper impulses) rather than the old three-state machine -- it does not need a
limit-state hysteresis hack and behaves far better when a limb is driven into
its stop, which happens constantly during RL exploration.
"""

import math

from .collision import LINEAR_SLOP, MAX_LINEAR_CORRECTION
from .math2d import Rot, Vec2, clamp, vec_cross

ANGULAR_SLOP = 2.0 / 180.0 * math.pi


def _solve22(k11, k12, k22, bx, by):
    det = k11 * k22 - k12 * k12
    if det != 0.0:
        det = 1.0 / det
    return Vec2(det * (k22 * bx - k12 * by), det * (k11 * by - k12 * bx))


class Joint:
    def initialize_velocity(self, dt, warm_start=True):
        raise NotImplementedError

    def solve_velocity(self, dt):
        raise NotImplementedError

    def solve_position(self):
        return True


class RevoluteJoint(Joint):
    __slots__ = ("body_a", "body_b", "local_anchor_a", "local_anchor_b",
                 "reference_angle", "enable_motor", "motor_speed",
                 "max_motor_torque", "motor_impulse", "enable_limit",
                 "lower_angle", "upper_angle", "lower_impulse", "upper_impulse",
                 "impulse", "r_a", "r_b", "m11", "m12", "m22", "axial_mass",
                 "name", "collide_connected")

    def __init__(self, body_a, body_b, anchor_world, name="",
                 lower_angle=None, upper_angle=None,
                 enable_motor=False, motor_speed=0.0, max_motor_torque=0.0,
                 reference_angle=None):
        self.body_a = body_a
        self.body_b = body_b
        self.local_anchor_a = body_a.local_point(anchor_world)
        self.local_anchor_b = body_b.local_point(anchor_world)
        self.reference_angle = (body_b.a - body_a.a) if reference_angle is None else reference_angle
        self.name = name
        self.collide_connected = False

        self.enable_limit = lower_angle is not None and upper_angle is not None
        self.lower_angle = lower_angle if lower_angle is not None else 0.0
        self.upper_angle = upper_angle if upper_angle is not None else 0.0

        self.enable_motor = enable_motor
        self.motor_speed = motor_speed
        self.max_motor_torque = max_motor_torque

        self.impulse = Vec2()
        self.motor_impulse = 0.0
        self.lower_impulse = 0.0
        self.upper_impulse = 0.0

        self.r_a = Vec2()
        self.r_b = Vec2()
        # Inverse of the 2x2 point-constraint effective mass, precomputed once
        # per step because the velocity loop runs it 8-12 times.
        self.m11 = self.m12 = self.m22 = 0.0
        self.axial_mass = 0.0

    # -- queries ----------------------------------------------------------
    @property
    def joint_angle(self):
        return self.body_b.a - self.body_a.a - self.reference_angle

    @property
    def joint_speed(self):
        return self.body_b.w - self.body_a.w

    @property
    def anchor_world(self):
        return self.body_a.world_point(self.local_anchor_a)

    def motor_torque(self, inv_dt):
        return self.motor_impulse * inv_dt

    def set_motor(self, speed, max_torque):
        self.motor_speed = speed
        self.max_motor_torque = max_torque

    def reset_impulses(self):
        self.impulse.set_zero()
        self.motor_impulse = 0.0
        self.lower_impulse = 0.0
        self.upper_impulse = 0.0

    # -- solver -----------------------------------------------------------
    def initialize_velocity(self, dt, warm_start=True):
        a, b = self.body_a, self.body_b
        qa = Rot(a.a)
        qb = Rot(b.a)
        self.r_a = qa.rotate(self.local_anchor_a - a.local_center)
        self.r_b = qb.rotate(self.local_anchor_b - b.local_center)

        m_a, m_b = a.inv_mass, b.inv_mass
        i_a, i_b = a.inv_inertia, b.inv_inertia

        ra, rb = self.r_a, self.r_b
        k11 = m_a + m_b + i_a * ra.y * ra.y + i_b * rb.y * rb.y
        k12 = -i_a * ra.x * ra.y - i_b * rb.x * rb.y
        k22 = m_a + m_b + i_a * ra.x * ra.x + i_b * rb.x * rb.x
        det = k11 * k22 - k12 * k12
        inv_det = 1.0 / det if det != 0.0 else 0.0
        self.m11 = k22 * inv_det
        self.m12 = -k12 * inv_det
        self.m22 = k11 * inv_det

        k = i_a + i_b
        self.axial_mass = 1.0 / k if k > 0.0 else 0.0

        if not self.enable_motor:
            self.motor_impulse = 0.0
        if not self.enable_limit:
            self.lower_impulse = 0.0
            self.upper_impulse = 0.0

        if warm_start:
            axial = self.motor_impulse + self.lower_impulse - self.upper_impulse
            p = self.impulse
            a.v -= p * m_a
            a.w -= i_a * (vec_cross(ra, p) + axial)
            b.v += p * m_b
            b.w += i_b * (vec_cross(rb, p) + axial)
        else:
            self.reset_impulses()

    def solve_velocity(self, dt):
        # Written out in scalars rather than Vec2 ops: this is the innermost
        # loop of the whole simulation (joints x iterations x substeps) and
        # temporary vector allocation dominated the profile.
        a, b = self.body_a, self.body_b
        m_a, m_b = a.inv_mass, b.inv_mass
        i_a, i_b = a.inv_inertia, b.inv_inertia
        aw = a.w
        bw = b.w
        axial_mass = self.axial_mass

        # --- motor -------------------------------------------------------
        if self.enable_motor:
            impulse = -axial_mass * (bw - aw - self.motor_speed)
            old = self.motor_impulse
            max_impulse = dt * self.max_motor_torque
            new = old + impulse
            if new < -max_impulse:
                new = -max_impulse
            elif new > max_impulse:
                new = max_impulse
            self.motor_impulse = new
            impulse = new - old
            aw -= i_a * impulse
            bw += i_b * impulse

        # --- limits ------------------------------------------------------
        if self.enable_limit:
            inv_dt = 1.0 / dt if dt > 0.0 else 0.0
            angle = b.a - a.a - self.reference_angle

            c = angle - self.lower_angle
            bias = c * inv_dt if c > 0.0 else 0.0
            impulse = -axial_mass * (bw - aw + bias)
            old = self.lower_impulse
            new = old + impulse
            if new < 0.0:
                new = 0.0
            self.lower_impulse = new
            impulse = new - old
            aw -= i_a * impulse
            bw += i_b * impulse

            c = self.upper_angle - angle
            bias = c * inv_dt if c > 0.0 else 0.0
            impulse = -axial_mass * (aw - bw + bias)
            old = self.upper_impulse
            new = old + impulse
            if new < 0.0:
                new = 0.0
            self.upper_impulse = new
            impulse = new - old
            aw += i_a * impulse
            bw -= i_b * impulse

        # --- point-to-point ----------------------------------------------
        ra, rb = self.r_a, self.r_b
        rax, ray = ra.x, ra.y
        rbx, rby = rb.x, rb.y
        av, bv = a.v, b.v
        cdx = (bv.x - bw * rby) - (av.x - aw * ray)
        cdy = (bv.y + bw * rbx) - (av.y + aw * rax)

        px = -(self.m11 * cdx + self.m12 * cdy)
        py = -(self.m12 * cdx + self.m22 * cdy)

        self.impulse.x += px
        self.impulse.y += py

        av.x -= m_a * px
        av.y -= m_a * py
        a.w = aw - i_a * (rax * py - ray * px)
        bv.x += m_b * px
        bv.y += m_b * py
        b.w = bw + i_b * (rbx * py - rby * px)

    def solve_position(self):
        a, b = self.body_a, self.body_b
        m_a, m_b = a.inv_mass, b.inv_mass
        i_a, i_b = a.inv_inertia, b.inv_inertia
        angular_error = 0.0

        # --- angular limits ---------------------------------------------
        if self.enable_limit:
            angle = b.a - a.a - self.reference_angle
            c = 0.0
            if self.upper_angle - self.lower_angle < 2.0 * ANGULAR_SLOP:
                c = clamp(angle - self.lower_angle, -MAX_LINEAR_CORRECTION,
                          MAX_LINEAR_CORRECTION)
            elif angle <= self.lower_angle:
                c = clamp(angle - self.lower_angle + ANGULAR_SLOP,
                          -MAX_LINEAR_CORRECTION, 0.0)
            elif angle >= self.upper_angle:
                c = clamp(angle - self.upper_angle - ANGULAR_SLOP,
                          0.0, MAX_LINEAR_CORRECTION)

            if c != 0.0:
                limit_impulse = -self.axial_mass * c
                a.a -= i_a * limit_impulse
                b.a += i_b * limit_impulse
                angular_error = abs(c)
                a.synchronize_transform()
                b.synchronize_transform()

        # --- point constraint --------------------------------------------
        qa = Rot(a.a)
        qb = Rot(b.a)
        ra = qa.rotate(self.local_anchor_a - a.local_center)
        rb = qb.rotate(self.local_anchor_b - b.local_center)
        c_vec = (b.c + rb) - (a.c + ra)
        position_error = c_vec.length()

        k11 = m_a + m_b + i_a * ra.y * ra.y + i_b * rb.y * rb.y
        k12 = -i_a * ra.x * ra.y - i_b * rb.x * rb.y
        k22 = m_a + m_b + i_a * ra.x * ra.x + i_b * rb.x * rb.x
        impulse = -_solve22(k11, k12, k22, c_vec.x, c_vec.y)

        a.c -= impulse * m_a
        a.a -= i_a * vec_cross(ra, impulse)
        b.c += impulse * m_b
        b.a += i_b * vec_cross(rb, impulse)
        a.synchronize_transform()
        b.synchronize_transform()

        return position_error <= LINEAR_SLOP and angular_error <= ANGULAR_SLOP
