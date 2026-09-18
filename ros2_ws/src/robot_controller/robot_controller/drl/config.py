#!/usr/bin/env python3

"""
The observation/action contract, shared by the training surrogate and
the live ROS 2 node.

WHY THIS FILE EXISTS
--------------------
A policy is a function from an observation vector to an action vector.
If the surrogate builds that vector even slightly differently from the
node that later runs the policy - a different beam count, a different
clip range, goal bearing measured from a different origin - the network
receives garbage it has never seen and behaves unpredictably, with no
error anywhere. That failure is silent and extremely hard to diagnose.

This project has already been bitten by exactly that class of bug: the
old Q-table was keyed on a 9-tuple while the controller emitted a
7-tuple, so every lookup missed and the policy quietly ran on its
fallback for entire episodes while printing "Q-TABLE LOADED".

So both sides import these constants. Neither defines its own.

GEOMETRY NOTE
-------------
base_link, base_footprint and the LIDAR all sit on the REAR AXLE (see
robot.urdf.xacro). Odometry therefore reports the rear-axle pose, and
scans originate there. The chassis extends forward of that point, so
collision has to be tested against a disc centred BODY_OFFSET ahead of
the scan origin, not at the scan origin itself.
"""

import numpy as np


# ============================================================
# CONTROL RATE
# ============================================================

CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ


# ============================================================
# ACTION SPACE  (differential drive)
#
# The network emits two values in [-1, 1]; these map them to
# physical commands. v and w are independent, which is what
# makes pivot-in-place (v=0, |w|>0) reachable - the escape
# manoeuvre for the maze's five dead ends.
# ============================================================

V_MIN = 0.0
V_MAX = 0.22

W_MAX = 0.5

ACTION_DIM = 2


def action_to_command(action):
    """Map a network action in [-1, 1]^2 to (v, w) in SI units."""

    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    # v uses the half-open mapping so that -1 -> V_MIN and +1 -> V_MAX.
    v = V_MIN + (V_MAX - V_MIN) * (action[..., 0] + 1.0) * 0.5

    w = W_MAX * action[..., 1]

    return v, w


# ============================================================
# LIDAR
#
# The A1M8 publishes 360 beams. Feeding all 360 into an MLP is
# wasteful and slows training for no benefit, so they are
# reduced to N_SECTORS angular sectors.
#
# Reduction is by MINIMUM, not mean: a thin table leg spanning
# one beam inside a 10-degree sector must survive the
# downsample. Averaging would dilute it into the free space
# around it and the robot would drive through it.
# ============================================================

N_SECTORS = 36

LIDAR_RANGE_MIN = 0.15
LIDAR_RANGE_MAX = 3.0

# Consecutive scans stacked so the network can infer which way a
# dynamic obstacle is moving. A single frame is a still photo and
# cannot distinguish an object crossing left-to-right from one
# crossing right-to-left.
FRAME_STACK = 3


def downsample_scan(ranges, n_sectors=N_SECTORS):
    """360 raw beams -> n_sectors sector minima, clipped and normalised.

    `ranges` may contain inf/nan (no return); those become max range.
    Returns float32 in [0, 1] where 0 is touching and 1 is clear.
    """

    r = np.asarray(ranges, dtype=np.float32)

    r = np.where(np.isfinite(r), r, LIDAR_RANGE_MAX)

    r = np.clip(r, LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)

    n = r.shape[-1]

    if n % n_sectors != 0:
        # Resample to a multiple of n_sectors before folding.
        idx = np.linspace(0, n - 1, n_sectors * (n // n_sectors or 1))
        r = np.interp(idx, np.arange(n), r).astype(np.float32)
        n = r.shape[-1]

    r = r.reshape(*r.shape[:-1], n_sectors, n // n_sectors)

    sector = r.min(axis=-1)

    return (
        (sector - LIDAR_RANGE_MIN)
        /
        (LIDAR_RANGE_MAX - LIDAR_RANGE_MIN)
    ).astype(np.float32)


# ============================================================
# GOAL FEATURES
#
# Relative, never absolute - the same reason the tabular state
# used (dx, dy) rather than (robot_x, robot_y, goal_x, goal_y).
# A policy fed absolute coordinates memorises one map.
#
# Bearing is given as sin/cos rather than as an angle so the
# network never sees the discontinuity at +/-pi, where two
# numerically distant inputs mean nearly the same direction.
# ============================================================

GOAL_FEATURE_DIM = 3

MAX_GOAL_DISTANCE = 7.1  # diagonal of the 5x5 m arena, for normalising


def goal_features(dx, dy, yaw):
    """World-frame goal offset + robot yaw -> (distance, sin, cos)."""

    distance = float(np.hypot(dx, dy))

    bearing = np.arctan2(dy, dx) - yaw

    return np.array(
        [
            min(distance / MAX_GOAL_DISTANCE, 1.0),
            np.sin(bearing),
            np.cos(bearing),
        ],
        dtype=np.float32,
    )


# ============================================================
# OBSERVATION LAYOUT
#
#   [ stacked sector scans | goal features | previous action ]
#
# The previous action is included because SAC emits smooth
# trajectories only if it can see what it just did; without it
# the policy tends to chatter between extremes.
# ============================================================

PREV_ACTION_DIM = ACTION_DIM

OBS_DIM = (
    N_SECTORS * FRAME_STACK
    +
    GOAL_FEATURE_DIM
    +
    PREV_ACTION_DIM
)


def build_observation(scan_stack, goal_feat, prev_action):
    """Assemble the observation vector. Both sides call exactly this."""

    return np.concatenate(
        [
            np.asarray(scan_stack, dtype=np.float32).reshape(-1),
            np.asarray(goal_feat, dtype=np.float32).reshape(-1),
            np.asarray(prev_action, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


# ============================================================
# ROBOT FOOTPRINT
#
# Measured from robot.urdf.xacro. The scan origin is the rear
# axle; the chassis centre is wheelbase/2 ahead of it.
# ============================================================

WHEELBASE = 0.213
CHASSIS_LENGTH = 0.270
CHASSIS_WIDTH = 0.1205

# Distance from scan origin (rear axle) to the chassis centre.
BODY_OFFSET = WHEELBASE / 2.0

# Disc that encloses the chassis, centred at BODY_OFFSET.
# half-diagonal = hypot(0.135, 0.0603) = 0.1479
BODY_RADIUS = 0.15

# Hard stop distance for the safety layer. The nose sits
# CHASSIS_LENGTH/2 + BODY_OFFSET = 0.2415 m ahead of the scan origin,
# so a forward reading below this means the bumper is very close.
EMERGENCY_STOP_RANGE = 0.30


# ============================================================
# ARENA
# ============================================================

GRID_SIZE = 5
CELL_SIZE = 1.0
WALL_THICKNESS = 0.10
