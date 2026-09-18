#!/usr/bin/env python3

"""
Vectorised 2D raycasting for the training surrogate.

This is the reason training is feasible at all. SAC needs on the order
of a million environment steps; Gazebo runs at roughly real time and
cannot be parallelised across cores, so a million steps at 10 Hz is
well over a day of wall clock. The same physics, reduced to ray-versus-
axis-aligned-box and ray-versus-circle in numpy, runs several orders of
magnitude faster and vectorises across parallel environments.

The world is deliberately restricted to two primitive types:

  * axis-aligned boxes  - every maze wall is one, exactly
  * circles             - dynamic obstacles

Both have closed-form ray intersections, so a scan is pure array
arithmetic with no Python loop over beams.
"""

import numpy as np


# ============================================================
# RAY vs AXIS-ALIGNED BOXES
# ============================================================

def ray_boxes(origin, directions, boxes, max_range):
    """Distance along each ray to the nearest box.

    origin     : (2,)      ray start, shared by all rays
    directions : (K, 2)    unit vectors
    boxes      : (N, 4)    [xmin, ymin, xmax, ymax]
    returns    : (K,)      distances, max_range where nothing is hit

    Slab method. For each axis the ray enters the box's slab at t_near
    and leaves at t_far; the ray crosses the box exactly when the
    latest entry precedes the earliest exit.
    """

    if len(boxes) == 0:
        return np.full(len(directions), max_range, dtype=np.float32)

    origin = np.asarray(origin, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    b = np.asarray(boxes, dtype=np.float64)

    # A zero component would divide by zero; the resulting inf is
    # actually the correct answer (the ray never crosses that slab),
    # so suppress the warning rather than special-casing it.
    with np.errstate(divide="ignore", invalid="ignore"):

        inv = 1.0 / d[:, None, :]              # (K, 1, 2)

        lo = (b[None, :, 0:2] - origin) * inv  # (K, N, 2)
        hi = (b[None, :, 2:4] - origin) * inv  # (K, N, 2)

        t_near = np.minimum(lo, hi)
        t_far = np.maximum(lo, hi)

        # nan arises from 0/0 when the origin lies exactly on a slab
        # boundary and the ray is parallel to it. Treat as "no
        # constraint from this axis".
        t_near = np.where(np.isnan(t_near), -np.inf, t_near)
        t_far = np.where(np.isnan(t_far), np.inf, t_far)

        enter = t_near.max(axis=2)  # (K, N)
        exit_ = t_far.min(axis=2)   # (K, N)

    hit = (enter <= exit_) & (exit_ > 0.0)

    # If the origin is inside a box, enter is negative; the first
    # surface ahead is then exit_. Clamp so a ray starting inside
    # geometry still reports a sane non-negative distance.
    t = np.where(enter >= 0.0, enter, exit_)

    t = np.where(hit, t, np.inf)

    return np.minimum(t.min(axis=1), max_range).astype(np.float32)


# ============================================================
# RAY vs CIRCLES
# ============================================================

def ray_circles(origin, directions, circles, max_range):
    """Distance along each ray to the nearest circle.

    circles : (M, 3)  [cx, cy, radius]

    Substituting the ray into |p - c|^2 = r^2 gives a quadratic in t
    whose smaller positive root is the near surface.
    """

    if len(circles) == 0:
        return np.full(len(directions), max_range, dtype=np.float32)

    origin = np.asarray(origin, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    c = np.asarray(circles, dtype=np.float64)

    oc = origin[None, :] - c[:, 0:2]        # (M, 2)

    b = d[:, None, :] @ oc[None, :, :, None]  # awkward; do it plainly
    b = (d[:, None, 0] * oc[None, :, 0]
         +
         d[:, None, 1] * oc[None, :, 1])      # (K, M)

    cc = (oc[:, 0] ** 2 + oc[:, 1] ** 2 - c[:, 2] ** 2)[None, :]  # (1, M)

    disc = b * b - cc

    ok = disc >= 0.0

    sq = np.sqrt(np.where(ok, disc, 0.0))

    t_near = -b - sq
    t_far = -b + sq

    # Prefer the near root; fall back to the far root when the ray
    # starts inside the circle.
    t = np.where(t_near > 0.0, t_near, t_far)

    t = np.where(ok & (t > 0.0), t, np.inf)

    return np.minimum(t.min(axis=1), max_range).astype(np.float32)


# ============================================================
# FULL SCAN
# ============================================================

def scan(origin, yaw, n_beams, boxes, circles, max_range,
         angle_min=-np.pi, angle_span=2.0 * np.pi):
    """Simulate one LIDAR revolution from `origin` at heading `yaw`.

    Beam i is at angle_min + i * angle_span / n_beams in the ROBOT
    frame, matching the sensor block in robot.urdf.xacro (360 beams
    spanning -pi..pi, so beam index equals degrees).
    """

    angles = angle_min + np.arange(n_beams) * (angle_span / n_beams)

    world = angles + yaw

    directions = np.stack([np.cos(world), np.sin(world)], axis=1)

    r = ray_boxes(origin, directions, boxes, max_range)

    if len(circles):
        r = np.minimum(r, ray_circles(origin, directions, circles, max_range))

    return r


# ============================================================
# COLLISION
# ============================================================

def disc_hits(centre, radius, boxes, circles):
    """True when a disc overlaps any box or circle.

    Used for the robot body. Note the body disc is NOT centred on the
    scan origin - the LIDAR sits on the rear axle while the chassis
    extends forward - so callers pass the offset centre.
    """

    if len(boxes):

        b = np.asarray(boxes, dtype=np.float64)

        # Nearest point on each box to the disc centre.
        nx = np.clip(centre[0], b[:, 0], b[:, 2])
        ny = np.clip(centre[1], b[:, 1], b[:, 3])

        if np.any((nx - centre[0]) ** 2 + (ny - centre[1]) ** 2 < radius ** 2):
            return True

    if len(circles):

        c = np.asarray(circles, dtype=np.float64)

        d2 = (c[:, 0] - centre[0]) ** 2 + (c[:, 1] - centre[1]) ** 2

        if np.any(d2 < (c[:, 2] + radius) ** 2):
            return True

    return False
