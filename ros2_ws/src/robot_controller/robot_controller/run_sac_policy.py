#!/usr/bin/env python3

"""
Run a trained SAC policy in Gazebo (or on the real robot).

    ros2 run robot_controller run_sac_policy --goal 4 4

Mirrors run_policy.py's structure: subscribe /scan and /odom, publish
/cmd_vel, act deterministically.

DETERMINISTIC
-------------
The mean action, never a sample. Exploration noise belongs in training;
on hardware it is just an unpredictable robot. A pilot run in this
project measured the deterministic policy at 28% success against 16.7%
for the same weights sampled stochastically.

THE SAFETY LAYER IS NOT OPTIONAL
--------------------------------
`SafetyGovernor` sits below the network and is independent of it. If the
minimum forward return drops under the hard threshold it refuses forward
motion, whatever the policy asked for. A learned policy will
occasionally do something stupid; that must not become a collision.

It deliberately does NOT simply stop dead - it zeroes forward speed
while leaving steering authority, so the robot can turn away from what
it is facing rather than sitting nose-first against a wall until the
step limit.
"""

import argparse
import math

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion

from . import sac_model_path
from .robot_constants import (
    CONTROL_HZ,
    DEPLOY_CELL_SIZE,
    LIDAR_BEAMS,
    MAX_STEER,
    V_MAX,
)
from .sac_actions import action_to_command, to_twist
from .sac_agent import SAC
from .sac_observation import (
    ActuatorModel,
    ScanStack,
    build_observation,
    downsample_scan,
)


# Hard floor, independent of the policy. Below this, no forward motion.
EMERGENCY_STOP_RANGE = 0.30

# Half-width of the forward cone the governor watches.
GOVERNOR_CONE = math.radians(35.0)


class SafetyGovernor:
    """Independent of the network. Vetoes, never commands."""

    def __init__(self, stop_range=EMERGENCY_STOP_RANGE, cone=GOVERNOR_CONE):

        self.stop_range = stop_range
        self.cone = cone
        self.vetoes = 0

    def forward_clearance(self, scan, angle_min, angle_increment):

        if scan is None or len(scan) == 0:
            return float("inf")

        idx = np.arange(len(scan))
        angles = angle_min + idx * angle_increment

        ahead = np.abs(
            np.arctan2(np.sin(angles), np.cos(angles))
        ) <= self.cone

        window = np.asarray(scan)[ahead]
        window = window[np.isfinite(window)]

        return float(window.min()) if window.size else float("inf")

    def apply(self, v, delta, clearance):

        if clearance < self.stop_range and v > 0.0:

            self.vetoes += 1

            # Keep steering authority so it can turn out.
            return 0.0, delta, True

        return v, delta, False


class SacPolicyNode(Node):

    def __init__(self, goal_cell, checkpoint, drive_mode, cell_size):

        super().__init__("sac_policy")

        self.agent = SAC.load(checkpoint)      # verifies ObsSpec
        self.get_logger().info(f"loaded {checkpoint}.pt")

        self.goal = np.array([
            (goal_cell[0] + 0.5) * cell_size,
            (goal_cell[1] + 0.5) * cell_size,
        ])

        self.drive_mode = drive_mode

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.scan = None
        self.angle_min = -math.pi
        self.angle_increment = 2.0 * math.pi / LIDAR_BEAMS

        self.x = self.y = self.yaw = None
        self.start_x = self.start_y = None

        self.stack = ScanStack()
        self.stack_ready = False

        # Derived, not measured - see sac_observation. The same lag
        # model runs in training, here, and on the real chassis, so the
        # observation is identical regardless of what the hardware
        # reports.
        self.actuator = ActuatorModel()

        self.prev_action = np.zeros(2, dtype=np.float32)

        self.governor = SafetyGovernor()

        self.steps = 0

        self.create_timer(1.0 / CONTROL_HZ, self._control)

    # ----------------------------------------------------

    def _scan_cb(self, msg):

        self.scan = np.asarray(msg.ranges, dtype=np.float32)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def _odom_cb(self, msg):

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.x, self.y = p.x, p.y

        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw

        if self.start_x is None:
            self.start_x, self.start_y = p.x, p.y

    # ----------------------------------------------------

    def _stop(self):

        self.cmd_pub.publish(Twist())

    def _control(self):

        if self.scan is None or self.x is None:
            return

        if len(self.scan) != LIDAR_BEAMS:
            self.get_logger().warn(
                f"scan has {len(self.scan)} beams, expected {LIDAR_BEAMS}; "
                "the observation builder requires the trained geometry."
            )
            self._stop()
            return

        sectors = downsample_scan(self.scan)

        if not self.stack_ready:
            self.stack.reset(sectors)
            self.stack_ready = True
        else:
            self.stack.push(sectors)

        # Odometry starts at (0,0) at spawn, so shift by the latched
        # origin to get world coordinates the goal is expressed in.
        wx = self.x - self.start_x + self.goal_origin[0]
        wy = self.y - self.start_y + self.goal_origin[1]

        dx = self.goal[0] - wx
        dy = self.goal[1] - wy

        obs = build_observation(
            self.stack.buf, dx, dy, self.yaw,
            self.prev_action, self.actuator.speed, self.actuator.steer,
        )

        action = self.agent.act(obs, deterministic=True)

        v, delta = action_to_command(action)

        clearance = self.governor.forward_clearance(
            self.scan, self.angle_min, self.angle_increment
        )

        v, delta, vetoed = self.governor.apply(v, delta, clearance)

        # Advance the derived actuator state with what was ACTUALLY
        # commanded, including any veto, or the observation drifts away
        # from the robot's real state.
        self.actuator.update(v, delta)

        self.prev_action = action

        msg = Twist()

        if self.drive_mode == "diff":
            lin, ang = to_twist(v, delta)
            msg.linear.x = lin
            msg.angular.z = ang
        else:
            msg.linear.x = v
            msg.angular.z = delta

        self.cmd_pub.publish(msg)

        self.steps += 1

        if math.hypot(dx, dy) < 0.20:
            self.get_logger().info(
                f"GOAL REACHED in {self.steps} steps "
                f"({self.governor.vetoes} safety vetoes)"
            )
            self._stop()
            rclpy.shutdown()

        if self.steps % 20 == 0:
            self.get_logger().info(
                f"d={math.hypot(dx, dy):.2f} m  v={v:+.2f}  "
                f"delta={delta:+.3f}  clear={clearance:.2f}"
                + ("  [SAFETY VETO]" if vetoed else "")
            )


def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", nargs=2, type=int, required=True)
    parser.add_argument("--start", nargs=2, type=int, default=[0, 0])
    parser.add_argument("--name", default="policy")
    parser.add_argument("--drive-mode", default="diff",
                        choices=["diff", "ackermann"])
    parser.add_argument("--cell-size", type=float, default=DEPLOY_CELL_SIZE)
    cli, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)

    node = SacPolicyNode(
        tuple(cli.goal),
        sac_model_path.checkpoint_path(cli.name),
        cli.drive_mode,
        cli.cell_size,
    )

    # World position of the start cell centre; odometry is relative to
    # wherever the robot spawned.
    node.goal_origin = np.array([
        (cli.start[0] + 0.5) * cli.cell_size,
        (cli.start[1] + 0.5) * cli.cell_size,
    ])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
