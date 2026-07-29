"""A small, self-contained 2D rigid-body physics engine.

Written from scratch (no Box2D / pymunk dependency) so that every part of the
dynamics the RL agent has to master is inspectable and tunable:

    math2d      vectors, rotations, transforms
    shapes      convex polygons + circles, exact mass properties
    body        rigid bodies, fixtures, collision filtering
    collision   SAT + Sutherland-Hodgman clipping -> local-frame manifolds
    contact     persistent contacts, warm-started sequential impulse solver
    joints      revolute joint with angular motor and hard limits
    world       broad-phase, step loop, positional correction
"""

from .body import DYNAMIC, KINEMATIC, STATIC, Body, Filter, Fixture
from .collision import LINEAR_SLOP, Manifold, WorldManifold, collide
from .contact import Contact, ContactSolver
from .joints import RevoluteJoint
from .math2d import (Rot, Transform, Vec2, clamp, cross_sv, cross_vs, distance,
                     normalize_angle, vec_cross)
from .shapes import AABB, Circle, MassData, Polygon
from .world import World

__all__ = [
    "Vec2", "Rot", "Transform", "vec_cross", "cross_vs", "cross_sv", "clamp",
    "distance", "normalize_angle",
    "Circle", "Polygon", "AABB", "MassData",
    "Body", "Fixture", "Filter", "STATIC", "KINEMATIC", "DYNAMIC",
    "collide", "Manifold", "WorldManifold", "LINEAR_SLOP",
    "Contact", "ContactSolver", "RevoluteJoint", "World",
]
