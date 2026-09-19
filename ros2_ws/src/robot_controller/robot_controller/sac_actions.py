#!/usr/bin/env python3

"""
Action scaling and drive-mode command conversion. One place, both sides.

The network emits two values in [-1, 1]. Everything downstream - the 2D
simulator, the Gazebo node, the real chassis - converts through here, so
a change to the speed range or the steering limit cannot land in one
consumer and not another.

DRIVE MODES
-----------
`diff`      libgazebo_ros_diff_drive, commanded by Twist. The bicycle
            steering angle is converted to the yaw rate that produces
            the same instantaneous curvature: w = v * tan(delta) / L.
            Note this is exact only because the sim integrates the same
            bicycle model; it is a kinematic equivalence, not a claim
            that the two vehicles are the same.

`ackermann` libgazebo_ros_ackermann_drive. That plugin derives its own
            steering from atan(L * w / v), so a Twist with linear.x == 0
            produces NO motion at all - the robot cannot turn while
            stationary. This is exactly why the action space allows
            reverse: an Ackermann chassis reverses out of a dead end, it
            cannot pivot out.
"""

import math

import numpy as np

from .robot_constants import MAX_STEER, V_MAX, V_MIN, WHEELBASE


ACTION_DIM = 2


def action_to_command(action):
    """[-1, 1]^2 -> (v m/s, delta rad).

    v maps affinely so -1 -> V_MIN (full reverse) and +1 -> V_MAX.
    """

    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    v = V_MIN + (V_MAX - V_MIN) * (a[..., 0] + 1.0) * 0.5

    delta = MAX_STEER * a[..., 1]

    return float(v), float(delta)


def command_to_action(v, delta):
    """(v, delta) -> [-1, 1]^2. Inverse of action_to_command."""

    v_norm = 2.0 * (float(v) - V_MIN) / (V_MAX - V_MIN) - 1.0

    return np.array(
        [
            np.clip(v_norm, -1.0, 1.0),
            np.clip(float(delta) / MAX_STEER, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def steer_to_yaw_rate(v, delta, wheelbase=WHEELBASE):
    """Bicycle steering angle -> yaw rate, for Twist publication."""

    return float(v) * math.tan(float(delta)) / wheelbase


def to_twist(v, delta):
    """(v, delta) -> (linear.x, angular.z) for the diff-drive plugin."""

    return float(v), steer_to_yaw_rate(v, delta)
