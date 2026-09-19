#!/usr/bin/env python3

"""
THE observation builder. Training and deployment both call this.

WHY ONE MODULE
--------------
A policy is a function from an observation vector to an action. If the
training simulator and the ROS 2 node build that vector even slightly
differently - a different beam count, a different clip ceiling, a
bearing measured from a different origin - the network receives input it
has never seen and behaves unpredictably, with no error raised anywhere.
The policy looks trained and drives randomly.

This project has already been bitten by exactly that failure: the old
Q-table was keyed on a 9-tuple while the controller emitted a 7-tuple,
so every lookup missed and the policy silently ran on its fallback for
entire episodes while printing "Q-TABLE LOADED / States: 2456".

So there is one builder, and `ObsSpec` is saved next to every checkpoint
so a mismatched pairing is refused at load rather than discovered on
hardware.

DERIVED STEERING ANGLE
----------------------
Element 150 is the CURRENT ACTUAL steering angle, which differs from the
commanded angle because the servo has first-order lag. Without it the
problem is not Markov: the agent knows what it last asked for but not
which way the wheels are actually pointed.

It is NOT read from the environment. The 2D sim knows it exactly, Gazebo
exposes it as a joint state, and the physical LD-1501MG may not report
position at all - three different sources with three different error
characteristics. Instead it is DERIVED here by integrating the same
deterministic lag model from the command history, so all three
environments produce identical values by construction. Gazebo's joint
state is used to validate TAU_STEER, not to feed the observation.

LAYOUT (151 dims)
-----------------
    index      size  content
    0   - 143   144  4 stacked scans x 36 sectors, OLDEST FIRST,
                     min-pooled over each 10 deg sector, clipped to
                     [LIDAR_RANGE_MIN, LIDAR_NORM_MAX], normalised [0,1]
                       scan t-3 : 0   - 35
                       scan t-2 : 36  - 71
                       scan t-1 : 72  - 107
                       scan t   : 108 - 143
    144         1    clip(goal_distance / LIDAR_NORM_MAX, 0, 1)
    145         1    sin(goal bearing in ROBOT frame)
    146         1    cos(goal bearing in ROBOT frame)
    147         1    previous commanded v,     normalised to [-1, 1]
    148         1    previous commanded delta, normalised to [-1, 1]
    149         1    current speed / V_MAX
    150         1    current DERIVED steering angle / MAX_STEER

Goal is always relative. Absolute coordinates would let the policy
memorise one maze and destroy transfer to an unseen one.
"""

from dataclasses import asdict, dataclass

import math

import numpy as np

from .robot_constants import (
    CONTROL_HZ,
    DT,
    LIDAR_BEAMS,
    LIDAR_NORM_MAX,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    MAX_STEER,
    TAU_SPEED,
    TAU_STEER,
    V_MAX,
    V_MIN,
    first_order_lag,
)


N_SECTORS = 36
FRAME_STACK = 4

SCAN_BLOCK = N_SECTORS * FRAME_STACK      # 144
GOAL_BLOCK = 3
PROPRIO_BLOCK = 4                          # prev v, prev delta, speed, steer

OBS_DIM = SCAN_BLOCK + GOAL_BLOCK + PROPRIO_BLOCK   # 151

IDX_GOAL = SCAN_BLOCK                      # 144
IDX_PREV_ACTION = IDX_GOAL + GOAL_BLOCK    # 147
IDX_SPEED = IDX_PREV_ACTION + 2            # 149
IDX_STEER = IDX_SPEED + 1                  # 150


@dataclass(frozen=True)
class ObsSpec:
    """Saved beside every checkpoint. Loading refuses on mismatch."""

    obs_dim: int = OBS_DIM
    n_sectors: int = N_SECTORS
    frame_stack: int = FRAME_STACK
    lidar_beams: int = LIDAR_BEAMS
    range_min: float = LIDAR_RANGE_MIN
    range_max: float = LIDAR_RANGE_MAX
    norm_max: float = LIDAR_NORM_MAX
    v_min: float = V_MIN
    v_max: float = V_MAX
    max_steer: float = MAX_STEER
    tau_steer: float = TAU_STEER
    tau_speed: float = TAU_SPEED
    control_hz: float = CONTROL_HZ

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def check(saved):
        """Raise if a checkpoint's spec disagrees with this build."""

        current = ObsSpec().as_dict()

        if isinstance(saved, ObsSpec):
            saved = saved.as_dict()

        diffs = {
            k: (saved.get(k), v)
            for k, v in current.items()
            if saved.get(k) != v
        }

        if diffs:
            lines = "\n".join(
                f"    {k}: checkpoint={s!r}  this build={c!r}"
                for k, (s, c) in diffs.items()
            )
            raise RuntimeError(
                "\nOBSERVATION SPEC MISMATCH\n"
                "This checkpoint was trained against a different "
                "observation builder:\n"
                f"{lines}\n"
                "\nThe policy would receive input it has never seen and "
                "behave unpredictably.\nRetrain, or check out the "
                "matching revision.\n"
            )


# ============================================================
# SCAN REDUCTION
# ============================================================

def downsample_scan(ranges, n_sectors=N_SECTORS):
    """`LIDAR_BEAMS` raw beams -> `n_sectors` normalised sector minima.

    Reduction is by MINIMUM. A table leg spanning one beam inside a
    10 degree sector must survive; averaging would dilute it into the
    free space around it and the robot would drive through it.

    Non-finite entries mean "no return", i.e. nothing within range, and
    become the range ceiling rather than zero.
    """

    r = np.asarray(ranges, dtype=np.float32)

    r = np.where(np.isfinite(r), r, LIDAR_RANGE_MAX)

    n = r.shape[-1]

    if n % n_sectors != 0:
        raise ValueError(
            f"{n} beams does not divide into {n_sectors} sectors; "
            "resample before calling."
        )

    r = r.reshape(*r.shape[:-1], n_sectors, n // n_sectors)

    sector = r.min(axis=-1)

    # Clip against the OBSERVATION ceiling, not the sensor ceiling.
    sector = np.minimum(np.maximum(sector, LIDAR_RANGE_MIN), LIDAR_NORM_MAX)

    return (
        (sector - LIDAR_RANGE_MIN) / (LIDAR_NORM_MAX - LIDAR_RANGE_MIN)
    ).astype(np.float32)


# ============================================================
# DERIVED ACTUATOR STATE
# ============================================================

class ActuatorModel:
    """Integrates the first-order lag to derive actual (v, delta).

    Owned here rather than by the environment so that the simulator,
    Gazebo and the physical robot all produce byte-identical values from
    the same command history.
    """

    def __init__(self, dt=DT):

        self.dt = dt
        self.speed = 0.0
        self.steer = 0.0

    def reset(self, speed=0.0, steer=0.0):

        self.speed = float(speed)
        self.steer = float(steer)

    def update(self, v_cmd, steer_cmd, dt=None):

        dt = self.dt if dt is None else dt

        self.speed = first_order_lag(self.speed, float(v_cmd), TAU_SPEED, dt)
        self.steer = first_order_lag(self.steer, float(steer_cmd), TAU_STEER, dt)

        return self.speed, self.steer


# ============================================================
# GOAL FEATURES
# ============================================================

def goal_features(dx, dy, yaw):
    """World-frame goal offset + heading -> (distance, sin, cos).

    Bearing is given as sin/cos rather than as an angle so the network
    never sees the wrap discontinuity at +/-pi, where two numerically
    distant inputs describe almost the same direction.
    """

    distance = math.hypot(dx, dy)

    bearing = math.atan2(dy, dx) - yaw

    return np.array(
        [
            min(distance / LIDAR_NORM_MAX, 1.0),
            math.sin(bearing),
            math.cos(bearing),
        ],
        dtype=np.float32,
    )


# ============================================================
# THE BUILDER
# ============================================================

def build_observation(scan_stack, dx, dy, yaw, prev_action, speed, steer):
    """Assemble the 151-dim observation. Both sides call exactly this.

    scan_stack  - (FRAME_STACK, N_SECTORS) already downsampled, oldest
                  first
    dx, dy      - goal offset in the WORLD frame
    yaw         - robot heading, world frame
    prev_action - previous NORMALISED action, 2 values in [-1, 1]
    speed       - current derived speed, m/s
    steer       - current derived steering angle, rad
    """

    stack = np.asarray(scan_stack, dtype=np.float32)

    if stack.shape != (FRAME_STACK, N_SECTORS):
        raise ValueError(
            f"scan_stack must be {(FRAME_STACK, N_SECTORS)}, got {stack.shape}"
        )

    obs = np.empty(OBS_DIM, dtype=np.float32)

    obs[:SCAN_BLOCK] = stack.reshape(-1)

    obs[IDX_GOAL:IDX_GOAL + GOAL_BLOCK] = goal_features(dx, dy, yaw)

    obs[IDX_PREV_ACTION:IDX_PREV_ACTION + 2] = np.clip(
        np.asarray(prev_action, dtype=np.float32), -1.0, 1.0
    )

    obs[IDX_SPEED] = np.clip(speed / V_MAX, -1.0, 1.0)

    obs[IDX_STEER] = np.clip(steer / MAX_STEER, -1.0, 1.0)

    return obs


class ScanStack:
    """Rolling FRAME_STACK buffer, oldest first.

    One frame is not Markov when obstacles move: a still image cannot
    distinguish an object crossing left-to-right from one crossing
    right-to-left, so a single-frame policy drives into whatever is
    approaching it.
    """

    def __init__(self, n=FRAME_STACK, sectors=N_SECTORS):

        self.n = n
        self.sectors = sectors
        self.buf = np.zeros((n, sectors), dtype=np.float32)

    def reset(self, first_scan):
        """Fill with the first scan so the stack is never partly zeros."""

        self.buf[:] = first_scan

        return self.buf

    def push(self, scan):

        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = scan

        return self.buf
