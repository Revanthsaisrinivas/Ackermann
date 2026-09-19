#!/usr/bin/env python3

"""
Headless 2D bicycle-model simulator for SAC training.

WHY NOT TRAIN IN GAZEBO
-----------------------
A single cell-to-cell move takes about 7 s of wall clock in Gazebo and
it cannot be parallelised across cores. SAC needs on the order of 10^6
steps. Gazebo is for validation only; this module is where the steps
come from.

The world is restricted to two primitives, both with closed-form ray
intersections, so a scan is array arithmetic with no Python loop over
beams:

    axis-aligned boxes  - every maze wall is exactly one
    circles             - dynamic obstacles

OUT-OF-RANGE RETURNS
--------------------
Gazebo's ray sensor and this simulator may disagree on what "no return"
is - `inf` in some versions, `range_max` in others. Those two cannot
reach the policy: `sac_observation.downsample_scan` maps non-finite
values to `LIDAR_RANGE_MAX` and then clips to `LIDAR_NORM_MAX`
(3.0 < 6.0), so `inf` and `6.0` both normalise to exactly 1.0.

A third encoding is NOT harmless, and is the reason the check exists.
Some drivers report "no return" as 0.0. That clips UP to
`LIDAR_RANGE_MIN` and normalises to 0.0 - the encoding for "obstacle
against the bumper" - so every open direction would read as a wall.
`sac_evaluate.py --check-scan` measures a live /scan and fails loudly
on any sub-range_min value.

UNKNOWN MAZE
------------
The observation never contains absolute position, goal coordinates or
the wall layout - only what the LIDAR returns plus a relative goal. The
maze is regenerated every episode. That is what lets one policy drive a
layout it has never seen, including maze.world, which this module is
forbidden from generating.
"""

import math

import numpy as np

from .rl_environment import MazeEnvironment
from .robot_constants import (
    BODY_LENGTH,
    BODY_WIDTH,
    DT,
    LIDAR_ANGLE_MIN,
    LIDAR_BEAMS,
    LIDAR_NOISE_FRAC,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    MAX_STEER,
    TAU_SPEED,
    TAU_STEER,
    V_MAX,
    V_MIN,
    WALL_THICKNESS,
    WHEELBASE,
    body_polygon,
    first_order_lag,
)
from . import sac_reward
from .sac_actions import ACTION_DIM, action_to_command
from .sac_observation import (
    FRAME_STACK,
    N_SECTORS,
    OBS_DIM,
    ScanStack,
    build_observation,
    downsample_scan,
)


GOAL_RADIUS = 0.20


# ============================================================
# RAYCASTING
# ============================================================

def ray_boxes(origin, directions, boxes, max_range):
    """Nearest box hit along each ray. Slab method, vectorised."""

    if len(boxes) == 0:
        return np.full(len(directions), max_range, dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    d = np.asarray(directions, dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):

        inv = 1.0 / d[:, None, :]

        lo = (b[None, :, 0:2] - o) * inv
        hi = (b[None, :, 2:4] - o) * inv

        t_near = np.where(np.isnan(np.minimum(lo, hi)), -np.inf,
                          np.minimum(lo, hi))
        t_far = np.where(np.isnan(np.maximum(lo, hi)), np.inf,
                         np.maximum(lo, hi))

        enter = t_near.max(axis=2)
        exit_ = t_far.min(axis=2)

    hit = (enter <= exit_) & (exit_ > 0.0)

    t = np.where(enter >= 0.0, enter, exit_)
    t = np.where(hit, t, np.inf)

    return np.minimum(t.min(axis=1), max_range).astype(np.float32)


def ray_circles(origin, directions, circles, max_range):
    """Nearest circle hit along each ray."""

    if len(circles) == 0:
        return np.full(len(directions), max_range, dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    d = np.asarray(directions, dtype=np.float32)
    c = np.asarray(circles, dtype=np.float32)

    oc = o[None, :] - c[:, 0:2]

    b = d[:, None, 0] * oc[None, :, 0] + d[:, None, 1] * oc[None, :, 1]

    cc = (oc[:, 0] ** 2 + oc[:, 1] ** 2 - c[:, 2] ** 2)[None, :]

    disc = b * b - cc
    ok = disc >= 0.0

    sq = np.sqrt(np.where(ok, disc, 0.0))

    t_near = -b - sq
    t_far = -b + sq

    t = np.where(t_near > 0.0, t_near, t_far)
    t = np.where(ok & (t > 0.0), t, np.inf)

    return np.minimum(t.min(axis=1), max_range).astype(np.float32)


# Beam directions in the robot frame, computed once.
_BEAM_ANGLES = LIDAR_ANGLE_MIN + np.arange(LIDAR_BEAMS) * (
    2.0 * math.pi / LIDAR_BEAMS
)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def walls_to_boxes(blocked_edges, grid_size, cell_size,
                   thickness=WALL_THICKNESS):
    """Blocked cell-to-cell edges + arena bounds -> box list."""

    h = thickness / 2.0
    span = grid_size * cell_size

    boxes = [
        [-h, -h, span + h, h],
        [-h, span - h, span + h, span + h],
        [-h, -h, h, span + h],
        [span - h, -h, span + h, span + h],
    ]

    for edge in blocked_edges:

        (ax, ay), (bx, by) = sorted(edge)

        if ax == bx:
            y = max(ay, by) * cell_size
            boxes.append([ax * cell_size, y - h, (ax + 1) * cell_size, y + h])
        else:
            x = max(ax, bx) * cell_size
            boxes.append([x - h, ay * cell_size, x + h, (ay + 1) * cell_size])

    return np.array(boxes, dtype=np.float64)


# Radius of the disc that encloses the body rectangle. Used only as a
# BROAD PHASE - anything further than this cannot possibly touch.
_BODY_ENCLOSING_RADIUS = math.hypot(BODY_LENGTH / 2.0, BODY_WIDTH / 2.0)


def polygon_hits_boxes(poly, boxes):
    """Separating-axis test, rectangle vs axis-aligned boxes.

    The body is a rectangle, not a disc: at 0.270 x 0.222 m an enclosing
    disc has radius 0.175 m, which would forbid legal gaps a real
    chassis passes through.

    Two phases. The broad phase rejects every box further from the body
    centre than the enclosing radius, in one vectorised operation; on a
    typical step that leaves nothing and the exact test never runs. The
    narrow phase is exact SAT on whatever survives.
    """

    if len(boxes) == 0:
        return False

    centre = poly.mean(axis=0)

    nx = np.minimum(np.maximum(centre[0], boxes[:, 0]), boxes[:, 2])
    ny = np.minimum(np.maximum(centre[1], boxes[:, 1]), boxes[:, 3])

    near = (
        (nx - centre[0]) ** 2 + (ny - centre[1]) ** 2
        < _BODY_ENCLOSING_RADIUS ** 2
    )

    if not near.any():
        return False

    edge0 = poly[1] - poly[0]
    edge1 = poly[3] - poly[0]

    axes = (
        edge0 / np.linalg.norm(edge0),
        edge1 / np.linalg.norm(edge1),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )

    for b in boxes[near]:

        box_pts = np.array([
            [b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]
        ])

        separated = False

        for ax in axes:
            pa = poly @ ax
            pb = box_pts @ ax
            if pa.max() < pb.min() or pb.max() < pa.min():
                separated = True
                break

        if not separated:
            return True

    return False


def polygon_hits_circles(poly, circles):
    """Rectangle vs circles, via nearest point on each polygon edge."""

    if len(circles) == 0:
        return False

    c = np.asarray(circles, dtype=np.float64)

    # Broad phase: enclosing disc vs each circle, one vectorised test.
    centre = poly.mean(axis=0)
    reach = c[:, 2] + _BODY_ENCLOSING_RADIUS
    near = (
        (c[:, 0] - centre[0]) ** 2 + (c[:, 1] - centre[1]) ** 2 < reach ** 2
    )
    if not near.any():
        return False
    c = c[near]

    for i in range(len(poly)):

        a = poly[i]
        b = poly[(i + 1) % len(poly)]

        ab = b - a
        denom = float(ab @ ab)

        if denom <= 0.0:
            continue

        t = np.clip(((c[:, 0:2] - a) @ ab) / denom, 0.0, 1.0)

        nearest = a[None, :] + t[:, None] * ab[None, :]

        d2 = np.sum((c[:, 0:2] - nearest) ** 2, axis=1)

        if np.any(d2 < c[:, 2] ** 2):
            return True

    return False


# ============================================================
# DYNAMIC OBSTACLES
# ============================================================

class Obstacles:
    """Circular movers that reflect off walls.

    Radii are capped so a single obstacle cannot seal a corridor: the
    corridor clear width is cell_size - WALL_THICKNESS, and the robot
    needs BODY_WIDTH of it. Without that cap some episodes are
    unsolvable from the first frame, and unsolvable episodes teach the
    agent only that trying does not pay.
    """

    def __init__(self, rng, count, radius_range, speed_range,
                 grid_size, cell_size, boxes):

        self.rng = rng
        self.span = grid_size * cell_size
        self.boxes = boxes

        corridor = cell_size - WALL_THICKNESS
        max_r = max(0.04, (corridor - BODY_WIDTH - 0.08) / 2.0)

        r_lo, r_hi = radius_range
        r_hi = min(r_hi, max_r)
        r_lo = min(r_lo, r_hi)

        self.pos = np.zeros((count, 2))
        self.vel = np.zeros((count, 2))
        self.rad = np.zeros(count)

        placed = 0

        for _ in range(count):

            r = float(rng.uniform(r_lo, r_hi))

            p = None
            for _try in range(40):
                cand = rng.uniform(r + 0.06, self.span - r - 0.06, size=2)
                if not self._in_wall(cand, r):
                    p = cand
                    break

            if p is None:
                continue

            heading = rng.uniform(-math.pi, math.pi)
            speed = rng.uniform(*speed_range)

            self.pos[placed] = p
            self.rad[placed] = r
            self.vel[placed] = [speed * math.cos(heading),
                                speed * math.sin(heading)]
            placed += 1

        self.pos = self.pos[:placed]
        self.vel = self.vel[:placed]
        self.rad = self.rad[:placed]

    def _in_wall(self, centre, radius):

        if len(self.boxes) == 0:
            return False

        b = self.boxes
        nx = np.clip(centre[0], b[:, 0], b[:, 2])
        ny = np.clip(centre[1], b[:, 1], b[:, 3])

        return bool(np.any(
            (nx - centre[0]) ** 2 + (ny - centre[1]) ** 2 < radius ** 2
        ))

    def step(self, dt):

        if len(self.pos) == 0:
            return

        proposed = self.pos + self.vel * dt

        for i in range(len(proposed)):

            if self._in_wall(proposed[i], self.rad[i]):

                only_x = np.array([proposed[i, 0], self.pos[i, 1]])
                only_y = np.array([self.pos[i, 0], proposed[i, 1]])

                if not self._in_wall(only_x, self.rad[i]):
                    self.pos[i] = only_x
                    self.vel[i, 1] *= -1.0
                elif not self._in_wall(only_y, self.rad[i]):
                    self.pos[i] = only_y
                    self.vel[i, 0] *= -1.0
                else:
                    self.vel[i] *= -1.0
            else:
                self.pos[i] = proposed[i]

        for axis in (0, 1):
            low = self.pos[:, axis] < self.rad + 0.05
            high = self.pos[:, axis] > self.span - self.rad - 0.05
            self.vel[low | high, axis] *= -1.0
            self.pos[:, axis] = np.clip(
                self.pos[:, axis], self.rad + 0.05,
                self.span - self.rad - 0.05
            )

    def as_circles(self):

        if len(self.pos) == 0:
            return np.zeros((0, 3))

        return np.concatenate([self.pos, self.rad[:, None]], axis=1)


# ============================================================
# ENVIRONMENT
# ============================================================

class SacMazeEnv:
    """Gym-like API without the gym dependency.

    reset()      -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
    """

    def __init__(
        self,
        grid_range=(4, 6),
        cell_range=(1.0, 1.4),
        wall_prob_range=(0.15, 0.45),
        n_obstacles=(0, 4),
        obstacle_radius=(0.05, 0.15),
        obstacle_speed=(0.0, 0.3),
        max_goal_cells=None,
        max_steps=800,
        seed=None,
    ):

        self.grid_range = grid_range
        self.cell_range = cell_range
        self.wall_prob_range = wall_prob_range
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.obstacle_speed = obstacle_speed
        self.max_goal_cells = max_goal_cells
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM

        self.stack = ScanStack()

        self._maze = MazeEnvironment()

    # ----------------------------------------------------
    # MAZE GENERATION
    # ----------------------------------------------------

    def _generate_maze(self, grid_size, wall_prob):
        """Reuse rl_environment's connectivity-guaranteed generator.

        It rejection-samples until is_fully_connected passes, so every
        layout it returns has a route between any two cells.
        """

        self._maze.GRID_SIZE = grid_size

        return self._maze.generate_random_maze(block_probability=wall_prob)

    def _bfs(self, blocked, start, grid_size):
        """Path length in cells. TRAINING-TIME DIFFICULTY SORTING ONLY.

        This is used to pick how far away the goal should be for the
        current curriculum stage. It is NEVER part of the observation
        and NEVER runs in the control loop - the policy has no access to
        it, and wiring it in would destroy the unknown-maze property
        this whole design rests on.
        """

        from collections import deque

        dist = {start: 0}
        q = deque([start])

        while q:
            c = q.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nb = (c[0] + dx, c[1] + dy)
                if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                    continue
                if nb in dist:
                    continue
                if frozenset({c, nb}) in blocked:
                    continue
                dist[nb] = dist[c] + 1
                q.append(nb)

        return dist

    def _route_exists(self, blocked, start, goal, grid_size, cell_size,
                      obstacles):
        """Solvable AT RESET.

        Only at reset: obstacles move, so a corridor open now can close
        later. The radius cap in Obstacles keeps any single obstacle
        from sealing a corridor, but two can still coincide, so this
        check catches the rest.
        """

        from collections import deque

        circles = obstacles.as_circles()

        def edge_free(a, b):
            if frozenset({a, b}) in blocked:
                return False
            if len(circles) == 0:
                return True
            mid = np.array([
                (a[0] + b[0] + 1.0) * 0.5 * cell_size,
                (a[1] + b[1] + 1.0) * 0.5 * cell_size,
            ])
            clearance = circles[:, 2] + BODY_WIDTH / 2.0
            return not np.any(
                np.sum((circles[:, 0:2] - mid) ** 2, axis=1) < clearance ** 2
            )

        seen = {start}
        q = deque([start])

        while q:
            c = q.popleft()
            if c == goal:
                return True
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nb = (c[0] + dx, c[1] + dy)
                if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                    continue
                if nb in seen or not edge_free(c, nb):
                    continue
                seen.add(nb)
                q.append(nb)

        return False

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------

    def reset(self, seed=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for _attempt in range(20):

            grid = int(self.rng.integers(self.grid_range[0],
                                         self.grid_range[1] + 1))
            cell = float(self.rng.uniform(*self.cell_range))
            wall_p = float(self.rng.uniform(*self.wall_prob_range))

            blocked = self._generate_maze(grid, wall_p)

            boxes = walls_to_boxes(blocked, grid, cell)

            cells = [(x, y) for x in range(grid) for y in range(grid)]
            start_cell = cells[int(self.rng.integers(len(cells)))]

            dist = self._bfs(blocked, start_cell, grid)

            candidates = [c for c in cells if c != start_cell and c in dist]

            if self.max_goal_cells is not None:
                near = [c for c in candidates
                        if dist[c] <= self.max_goal_cells]
                candidates = near or candidates

            if not candidates:
                continue

            goal_cell = candidates[int(self.rng.integers(len(candidates)))]

            lo, hi = self.n_obstacles
            count = int(self.rng.integers(lo, hi + 1))

            obstacles = Obstacles(
                self.rng, count, self.obstacle_radius, self.obstacle_speed,
                grid, cell, boxes
            )

            if not self._route_exists(blocked, start_cell, goal_cell,
                                      grid, cell, obstacles):
                continue

            # Accept this episode.
            self.grid_size = grid
            self.cell_size = cell
            self.blocked = blocked
            self.boxes = boxes
            self.obstacles = obstacles
            self.start_cell = start_cell
            self.goal_cell = goal_cell
            self.goal_path_cells = dist[goal_cell]
            break

        else:
            raise RuntimeError("could not generate a solvable episode")

        self.x = (start_cell[0] + 0.5) * cell
        self.y = (start_cell[1] + 0.5) * cell
        self.yaw = float(self.rng.uniform(-math.pi, math.pi))

        self.speed = 0.0
        self.steer = 0.0

        self.goal = np.array([
            (goal_cell[0] + 0.5) * cell,
            (goal_cell[1] + 0.5) * cell,
        ])

        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.steps = 0

        self.prev_distance = float(
            math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        )

        scan = self._scan()
        self.stack.reset(downsample_scan(scan))

        return self._observe(), {
            "grid_size": grid,
            "cell_size": cell,
            "goal_path_cells": self.goal_path_cells,
            "n_obstacles": len(self.obstacles.pos),
        }

    # ----------------------------------------------------
    # SENSING
    # ----------------------------------------------------

    def _scan(self):

        world = _BEAM_ANGLES + self.yaw

        dirs = np.stack([np.cos(world), np.sin(world)], axis=1)

        origin = np.array([self.x, self.y])

        r = ray_boxes(origin, dirs, self.boxes, LIDAR_RANGE_MAX)

        circles = self.obstacles.as_circles()
        if len(circles):
            r = np.minimum(r, ray_circles(origin, dirs, circles,
                                          LIDAR_RANGE_MAX))

        # sigma = 1% of range, as specified for the A1M8.
        r = r + self.rng.normal(0.0, LIDAR_NOISE_FRAC * r)

        # np.minimum/maximum rather than np.clip: clip runs a NaN
        # deprecation check per call that shows up in the profile.
        return np.minimum(np.maximum(r, 0.0), LIDAR_RANGE_MAX)

    def _observe(self):

        return build_observation(
            self.stack.buf,
            self.goal[0] - self.x,
            self.goal[1] - self.y,
            self.yaw,
            self.prev_action,
            self.speed,
            self.steer,
        )

    def _collided(self):

        poly = body_polygon(self.x, self.y, self.yaw)

        if polygon_hits_boxes(poly, self.boxes):
            return True

        return polygon_hits_circles(poly, self.obstacles.as_circles())

    # ----------------------------------------------------
    # STEP
    # ----------------------------------------------------

    def step(self, action):

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        v_cmd, steer_cmd = action_to_command(action)

        # First-order actuator lag, so the agent learns against real
        # dynamics rather than instantaneous response.
        self.speed = first_order_lag(self.speed, v_cmd, TAU_SPEED, DT)
        self.steer = first_order_lag(self.steer, steer_cmd, TAU_STEER, DT)

        # Bicycle model about the REAR AXLE.
        self.x += self.speed * math.cos(self.yaw) * DT
        self.y += self.speed * math.sin(self.yaw) * DT
        self.yaw += (self.speed / WHEELBASE) * math.tan(self.steer) * DT
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        self.obstacles.step(DT)

        self.steps += 1

        scan = self._scan()
        self.stack.push(downsample_scan(scan))

        distance = float(
            math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        )
        min_lidar = float(np.min(scan))

        collided = self._collided()
        reached = distance < GOAL_RADIUS

        reward, terms = sac_reward.compute(
            distance, self.prev_distance, min_lidar,
            action, self.prev_action, self.speed, reached, collided
        )

        self.prev_distance = distance
        self.prev_action = action

        terminated = bool(collided or reached)
        truncated = bool(self.steps >= self.max_steps)

        info = {
            "distance": distance,
            "min_lidar": min_lidar,
            "collided": collided,
            "reached": reached,
            "goal_path_cells": self.goal_path_cells,
            "cell_size": self.cell_size,
            **terms,
        }

        return self._observe(), reward, terminated, truncated, info

    def reset_fixed(self, blocked, grid_size, cell_size, start_cell,
                    goal_cell, n_obstacles=0, obstacle_speed=(0.0, 0.0)):
        """Reset onto a SPECIFIC layout, start and goal.

        For evaluation only - notably maze.world, which reset() must
        never generate because it is the held-out validation maze.
        Exists so evaluation does not have to reach in and mutate a
        dozen attributes by hand, which silently rots whenever the
        episode state grows a new field.
        """

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.blocked = set(blocked)
        self.boxes = walls_to_boxes(self.blocked, grid_size, cell_size)

        self.start_cell = tuple(start_cell)
        self.goal_cell = tuple(goal_cell)

        dist = self._bfs(self.blocked, self.start_cell, grid_size)
        self.goal_path_cells = dist.get(self.goal_cell, 0)

        self.obstacles = Obstacles(
            self.rng, n_obstacles, self.obstacle_radius, obstacle_speed,
            grid_size, cell_size, self.boxes
        )

        self.x = (self.start_cell[0] + 0.5) * cell_size
        self.y = (self.start_cell[1] + 0.5) * cell_size
        self.yaw = float(self.rng.uniform(-math.pi, math.pi))

        self.speed = 0.0
        self.steer = 0.0

        self.goal = np.array([
            (self.goal_cell[0] + 0.5) * cell_size,
            (self.goal_cell[1] + 0.5) * cell_size,
        ])

        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.steps = 0
        self.prev_distance = float(
            math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        )

        self.stack.reset(downsample_scan(self._scan()))

        return self._observe(), {
            "grid_size": grid_size,
            "cell_size": cell_size,
            "goal_path_cells": self.goal_path_cells,
            "n_obstacles": len(self.obstacles.pos),
        }

    def set_stage(self, **kw):
        """Curriculum hook. Only the listed knobs may be changed."""

        allowed = {
            "wall_prob_range", "n_obstacles", "obstacle_speed",
            "max_goal_cells", "grid_range", "cell_range",
        }

        for k, v in kw.items():
            if k not in allowed:
                raise KeyError(f"not a curriculum knob: {k}")
            setattr(self, k, v)
