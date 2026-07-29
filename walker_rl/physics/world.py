"""The simulation world: broad-phase, contact bookkeeping and the step loop."""

import math

from .body import DYNAMIC, STATIC, Body, should_collide
from .contact import Contact, ContactSolver
from .joints import RevoluteJoint
from .math2d import Vec2
from .shapes import Polygon

MAX_TRANSLATION = 2.0          # metres per step
MAX_ROTATION = 0.5 * math.pi   # radians per step


class World:
    def __init__(self, gravity=None, velocity_iterations=8, position_iterations=3,
                 warm_starting=True, allow_contact_margin=0.02):
        self.gravity = gravity if gravity is not None else Vec2(0.0, -9.81)
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.warm_starting = warm_starting
        self.contact_margin = allow_contact_margin

        self.bodies = []
        self.joints = []
        self.contacts = {}

        self._pairs_dirty = True
        self._candidate_pairs = []
        self._body_counter = 0

        self.step_count = 0
        self.time = 0.0

    # -- construction -----------------------------------------------------
    def create_body(self, body_type=DYNAMIC, position=None, angle=0.0, name=""):
        body = Body(body_type, position, angle, name)
        body.world = self
        body.index = self._body_counter
        self._body_counter += 1
        self.bodies.append(body)
        self._pairs_dirty = True
        return body

    def destroy_body(self, body):
        if body in self.bodies:
            self.bodies.remove(body)
        self.joints = [j for j in self.joints
                       if j.body_a is not body and j.body_b is not body]
        self.contacts = {k: c for k, c in self.contacts.items()
                         if c.body_a is not body and c.body_b is not body}
        self._pairs_dirty = True

    def create_revolute_joint(self, body_a, body_b, anchor_world, **kwargs):
        joint = RevoluteJoint(body_a, body_b, anchor_world, **kwargs)
        self.joints.append(joint)
        return joint

    def clear(self):
        self.bodies = []
        self.joints = []
        self.contacts = {}
        self._pairs_dirty = True
        self.step_count = 0
        self.time = 0.0

    def mark_dirty(self):
        self._pairs_dirty = True

    # -- broad phase ------------------------------------------------------
    def _rebuild_candidate_pairs(self):
        """Filter-eligible fixture pairs, computed once instead of per step.

        Bodies here are a fixed set for the lifetime of an episode, and the
        collision filter is static, so the only per-step work left is the AABB
        overlap test.
        """
        pairs = []
        fixtures = []
        for body in self.bodies:
            for f in body.fixtures:
                fixtures.append(f)
        n = len(fixtures)
        for i in range(n):
            fa = fixtures[i]
            ba = fa.body
            for j in range(i + 1, n):
                fb = fixtures[j]
                bb = fb.body
                if ba is bb:
                    continue
                if ba.type != DYNAMIC and bb.type != DYNAMIC:
                    continue
                if not should_collide(fa, fb):
                    continue
                if self._jointed_and_not_colliding(ba, bb):
                    continue
                pairs.append((fa, fb))
        self._candidate_pairs = pairs
        self._pairs_dirty = False

    def _jointed_and_not_colliding(self, ba, bb):
        for j in self.joints:
            if j.collide_connected:
                continue
            if (j.body_a is ba and j.body_b is bb) or (j.body_a is bb and j.body_b is ba):
                return True
        return False

    def _update_contacts(self):
        if self._pairs_dirty:
            self._rebuild_candidate_pairs()

        margin = self.contact_margin
        alive = set()
        for fa, fb in self._candidate_pairs:
            if not fa.aabb.overlaps(fb.aabb, margin):
                continue
            key = (fa.body.index, fa.index, fb.body.index, fb.index)
            contact = self.contacts.get(key)
            if contact is None:
                contact = Contact(fa, fb, key)
                self.contacts[key] = contact
            alive.add(key)

        # Drop contacts whose AABBs separated (this also frees warm-start data,
        # which is what we want -- stale impulses would fire on re-contact).
        for key in list(self.contacts.keys()):
            if key not in alive:
                del self.contacts[key]

        touching = []
        for contact in self.contacts.values():
            contact.update()
            if contact.touching and contact.enabled:
                touching.append(contact)
        return touching

    # -- stepping ---------------------------------------------------------
    def step(self, dt):
        if dt <= 0.0:
            return

        for body in self.bodies:
            body.synchronize_fixtures()

        touching = self._update_contacts()

        gx, gy = self.gravity.x, self.gravity.y
        # 1. integrate velocities
        for body in self.bodies:
            if body.type != DYNAMIC:
                continue
            body.v.x += dt * (body.gravity_scale * gx + body.inv_mass * body.force.x)
            body.v.y += dt * (body.gravity_scale * gy + body.inv_mass * body.force.y)
            body.w += dt * body.inv_inertia * body.torque
            # Exponential damping, unconditionally stable form.
            body.v *= 1.0 / (1.0 + dt * body.linear_damping)
            body.w *= 1.0 / (1.0 + dt * body.angular_damping)

        # 2. solve velocity constraints
        solver = ContactSolver(touching, dt, self.warm_starting)
        solver.initialize()
        solver.warm_start_impulses()
        for joint in self.joints:
            joint.initialize_velocity(dt, self.warm_starting)

        for _ in range(self.velocity_iterations):
            for joint in self.joints:
                joint.solve_velocity(dt)
            solver.solve_velocity_constraints()

        solver.store_impulses()

        # 3. integrate positions (with a safety clamp against tunnelling)
        for body in self.bodies:
            if body.type == STATIC:
                continue
            translation = body.v * dt
            if translation.length_sq() > MAX_TRANSLATION * MAX_TRANSLATION:
                body.v *= MAX_TRANSLATION / translation.length()
            rotation = dt * body.w
            if rotation * rotation > MAX_ROTATION * MAX_ROTATION:
                body.w *= MAX_ROTATION / abs(rotation)
            body.c += body.v * dt
            body.a += dt * body.w
            body.synchronize_transform()

        # 4. positional correction
        for _ in range(self.position_iterations):
            contacts_ok = solver.solve_position_constraints()
            joints_ok = True
            for joint in self.joints:
                joints_ok = joint.solve_position() and joints_ok
            if contacts_ok and joints_ok:
                break

        for body in self.bodies:
            body.synchronize_transform()
            body.force.set_zero()
            body.torque = 0.0

        self.step_count += 1
        self.time += dt

    # -- queries ----------------------------------------------------------
    def contacts_of(self, body):
        out = []
        for contact in self.contacts.values():
            if not contact.touching:
                continue
            if contact.body_a is body or contact.body_b is body:
                out.append(contact)
        return out

    def is_touching(self, body_a, body_b):
        for contact in self.contacts.values():
            if not contact.touching:
                continue
            if ((contact.body_a is body_a and contact.body_b is body_b) or
                    (contact.body_a is body_b and contact.body_b is body_a)):
                return True
        return False

    def total_energy(self):
        e = 0.0
        for body in self.bodies:
            if body.type != DYNAMIC:
                continue
            e += body.kinetic_energy()
            e -= body.mass * (self.gravity.x * body.c.x + self.gravity.y * body.c.y)
        return e

    def add_ground(self, half_width=200.0, top_y=0.0, thickness=2.0,
                   friction=1.0, name="ground"):
        body = self.create_body(STATIC, Vec2(0.0, top_y - thickness), 0.0, name)
        body.add_fixture(Polygon.box(half_width, thickness), density=0.0,
                         friction=friction, restitution=0.0)
        return body
