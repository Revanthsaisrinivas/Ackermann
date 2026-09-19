#!/usr/bin/env python3

"""
THE measured hardware constants. Nothing else defines them.

SOURCE OF TRUTH
---------------
`robot.urdf.xacro` is the source of truth, because Gazebo reads it and
the simulation's behaviour follows from it. This module mirrors it for
Python, and `test/test_robot_constants.py` parses the xacro and asserts
every value here matches. That test runs under `colcon test`, so drift
fails in CI rather than silently producing a policy trained against a
robot that does not exist.

Do not "tidy" these numbers - see CLAUDE.md. They are measured hardware.

FRAME CONVENTION
----------------
The reference point is the REAR AXLE. base_link, base_footprint and the
LIDAR all sit there deliberately: with rear-wheel drive that is the
pivot centre, and odometry origin, scan origin and rotation centre must
coincide or every turn injects an error the controller cannot observe.

The chassis extends FORWARD of that point, which is why collision needs
a body polygon offset ahead of the scan origin rather than a disc
centred on it.
"""

import math


# ============================================================
# GEOMETRY (measured)
# ============================================================

WHEELBASE = 0.213          # m, rear axle to front axle
TRACK_WIDTH = 0.222        # m, left wheel to right wheel
WHEEL_RADIUS = 0.0505      # m

CHASSIS_LENGTH = 0.270     # m, aluminium plate
CHASSIS_PLATE_WIDTH = 0.1205

# Collision width is the TRACK, not the plate: the wheels are the
# widest part of the robot.
BODY_WIDTH = TRACK_WIDTH
BODY_LENGTH = CHASSIS_LENGTH

# Plate centre sits wheelbase/2 ahead of the rear axle (overhang is
# (0.270 - 0.213)/2 = 0.0285 m at each end).
BODY_CENTRE_OFFSET = WHEELBASE / 2.0            # 0.1065 m

# Nose distance ahead of the rear axle. This is what makes the swept
# radius larger than R + track/2.
NOSE_OFFSET = BODY_CENTRE_OFFSET + BODY_LENGTH / 2.0   # 0.2415 m

REAR_OFFSET = BODY_LENGTH / 2.0 - BODY_CENTRE_OFFSET   # 0.0285 m


# ============================================================
# STEERING
# ============================================================

MAX_STEER = 0.5236         # rad (30 deg) at the knuckle

# R = L / tan(delta_max)
MIN_TURN_RADIUS = WHEELBASE / math.tan(MAX_STEER)        # 0.369 m

# Outer swept radius during a minimum-radius turn. The binding corner
# is the OUTER FRONT corner of the body, not the outer wheel, so the
# nose overhang counts:
#
#   R_out = hypot(R + width/2, nose_offset)
#
# Using R + track/2 alone understates this by ~57 mm and makes tight
# corridors look passable when they are not.
SWEPT_RADIUS_OUTER = math.hypot(
    MIN_TURN_RADIUS + BODY_WIDTH / 2.0,
    NOSE_OFFSET,
)

SWEPT_RADIUS_INNER = max(0.0, MIN_TURN_RADIUS - BODY_WIDTH / 2.0)


# ============================================================
# ACTUATOR LIMITS AND DYNAMICS
# ============================================================

V_MIN = -0.25              # m/s, reverse. Required: an Ackermann robot
V_MAX = 0.60               # cannot pivot out of a dead end.

# First-order lag time constants. The agent must learn against actuator
# dynamics, not instantaneous response.
#
# TAU_STEER is also used by sac_observation.py to DERIVE the current
# steering angle from the command history, so that the 2D sim, Gazebo
# and the physical robot all produce identical observations regardless
# of whether the hardware reports joint position at all.
TAU_STEER = 0.12           # s, LD-1501MG servo
TAU_SPEED = 0.20           # s, drivetrain


# ============================================================
# LIDAR (RPLIDAR A1M8, matching the sensor block in the xacro)
# ============================================================

LIDAR_BEAMS = 360
LIDAR_ANGLE_MIN = -math.pi
LIDAR_ANGLE_MAX = math.pi
LIDAR_RANGE_MIN = 0.15
LIDAR_RANGE_MAX = 6.0      # sensor model
LIDAR_NOISE_FRAC = 0.01    # sigma = 1% of range
LIDAR_HEIGHT = 0.1505      # m above ground, clears the 0.1010 m tyres
LIDAR_UPDATE_HZ = 5.5

# Normalisation ceiling for the OBSERVATION, deliberately below the
# sensor's range ceiling.
#
# At a 1.0 m cell a wall face reads 0.45 m and an open direction 1.45 m.
# Normalised over [0.15, 6.0] those become 0.051 and 0.222 - the entire
# decision-relevant band squeezed into 5-22% of the input range.
# Normalised over [0.15, 3.0] they become 0.105 and 0.456, separation
# 0.351 against 0.171: just over twice the discrimination where it
# matters. The sensor still reports out to 6.0 m; only the scaling
# changes.
LIDAR_NORM_MAX = 3.0


# ============================================================
# CONTROL RATE
# ============================================================

CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ


# ============================================================
# ARENA (deployment maze; the sim randomises around this)
# ============================================================

DEPLOY_GRID_SIZE = 5
DEPLOY_CELL_SIZE = 1.0
WALL_THICKNESS = 0.10

# How close counts as "arrived". This lives here, not in sac_env, because
# BOTH the trainer and the deployment node have to agree on it: the
# simulator uses it to decide when to pay R_GOAL, and run_sac_policy uses
# it to decide when to stop the robot. It was duplicated as a bare 0.20
# literal in run_sac_policy, so raising it in training would have left
# the real robot still declaring success at the old distance.
GOAL_RADIUS = 0.20


def body_polygon(x, y, yaw):
    """Corners of the body rectangle, given a REAR-AXLE pose.

    Returned counter-clockwise starting from the rear-right.
    """

    import numpy as np

    cx = x + BODY_CENTRE_OFFSET * math.cos(yaw)
    cy = y + BODY_CENTRE_OFFSET * math.sin(yaw)

    hl = BODY_LENGTH / 2.0
    hw = BODY_WIDTH / 2.0

    corners = np.array([
        [-hl, -hw],
        [+hl, -hw],
        [+hl, +hw],
        [-hl, +hw],
    ])

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])

    return corners @ rot.T + np.array([cx, cy])


def min_safe_cell_size(margin=0.10):
    """Smallest cell in which a minimum-radius 90 deg turn fits.

    The corridor clear width is cell_size - WALL_THICKNESS, and the turn
    needs SWEPT_RADIUS_OUTER - SWEPT_RADIUS_INNER of band plus margin.
    """

    band = SWEPT_RADIUS_OUTER - SWEPT_RADIUS_INNER

    return band + WALL_THICKNESS + margin


def first_order_lag(current, command, tau, dt):
    """Exact discrete first-order lag.

    Exponential rather than Euler so the result is independent of dt -
    the 2D sim, Gazebo and the robot may tick at different rates, and
    the derived steering angle must agree across all three.
    """

    if tau <= 0.0:
        return command

    alpha = math.exp(-dt / tau)

    return command + (current - command) * alpha
