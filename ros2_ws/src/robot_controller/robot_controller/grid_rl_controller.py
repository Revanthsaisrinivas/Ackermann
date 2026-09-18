#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from tf_transformations import euler_from_quaternion


class GridRLController(Node):

    # ============================================================
    # GRID CONFIGURATION
    # ============================================================

    GRID_SIZE = 5

    START_CELL = (0, 0)

    CELL_SIZE = 1.0

    # ============================================================
    # ACTIONS
    #
    # Priority:
    #
    # EAST -> NORTH -> WEST -> SOUTH
    # ============================================================

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    ACTIONS = [
        EAST,
        NORTH,
        WEST,
        SOUTH,
    ]

    ACTION_NAMES = {
        EAST: "EAST",
        NORTH: "NORTH",
        WEST: "WEST",
        SOUTH: "SOUTH",
    }

    # ============================================================
    # CONSTRUCTOR
    # ============================================================

    def __init__(self):

        super().__init__(
            "grid_rl_controller"
        )

        # --------------------------------------------------------
        # ROS PUBLISHER
        # --------------------------------------------------------

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        # --------------------------------------------------------
        # ROS SUBSCRIBERS
        # --------------------------------------------------------

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        # --------------------------------------------------------
        # SENSOR DATA
        # --------------------------------------------------------

        self.latest_scan = None

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        # --------------------------------------------------------
        # GRID STATE
        # --------------------------------------------------------

        self.current_cell = (
            0,
            0
        )

        self.goal_cell = (
            4,
            4
        )

        # Initial physical orientation
        self.heading = self.EAST

        # --------------------------------------------------------
        # CONTROLLER SETTINGS
        # --------------------------------------------------------

        self.linear_speed = 0.15

        self.angular_speed = 0.6

        self.position_tolerance = 0.08

        # 0.08 rad is 4.6 deg, which drifts sin(0.08) = 0.080 m sideways
        # per metre driven - equal to position_tolerance, so a single
        # sloppy turn was enough to make the old move loop's exit test
        # unsatisfiable. The move loop below now steers, but leaving
        # less residual error to correct still helps.
        self.angle_tolerance = 0.03

        # Distance below which a wall is considered present.
        # A separate, tighter clearance is used for live motion so the
        # robot stops before the wall contact point instead of driving
        # straight into the obstacle while trying to finish the cell-center
        # approach.
        # At a cell centre a wall face reads 0.45 m and an open
        # direction reads 1.45 m, so the threshold belongs between
        # them with margin on both sides. 0.55 was far too close to
        # 0.45: any stop that landed off-centre pushed a real wall
        # past it and the wall was reported OPEN.
        self.wall_distance = 0.80

        # Keep at 0.35. The LIDAR now sits on the rear axle and the
        # nose is 0.2415 m ahead of it, so 0.35 leaves ~0.11 m of
        # real clearance. Lowering this drives the bumper into walls.
        self.safe_clearance = 0.35

        # --------------------------------------------------------
        # START WORLD POSITION
        # --------------------------------------------------------

        self.start_world_x = None
        self.start_world_y = None

        # --------------------------------------------------------
        # READY FLAGS
        # --------------------------------------------------------

        self.scan_ready = False

        self.odom_ready = False

        self.initialized = False

        self.get_logger().info(
            "Grid RL Controller initialized."
        )

    # ============================================================
    # SCAN CALLBACK
    # ============================================================

    def scan_callback(
        self,
        msg
    ):

        self.latest_scan = msg

        self.scan_ready = True

    # ============================================================
    # ODOM CALLBACK
    # ============================================================

    def odom_callback(
        self,
        msg
    ):

        position = (
            msg.pose.pose.position
        )

        orientation = (
            msg.pose.pose.orientation
        )

        self.current_x = (
            position.x
        )

        self.current_y = (
            position.y
        )

        quaternion = [

            orientation.x,

            orientation.y,

            orientation.z,

            orientation.w,
        ]

        (
            _,
            _,
            yaw
        ) = euler_from_quaternion(
            quaternion
        )

        self.current_yaw = (
            self.normalize_angle(
                yaw
            )
        )

        self.odom_ready = True

    # ============================================================
    # WAIT FOR SENSORS
    # ============================================================

    def wait_until_ready(self):

        self.get_logger().info(
            "Waiting for controller and sensors..."
        )

        while rclpy.ok():

            if (
                self.scan_ready
                and
                self.odom_ready
            ):

                break

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

        if not rclpy.ok():

            return False

        # Store initial world position

        if self.start_world_x is None:

            self.start_world_x = (
                self.current_x
            )

            self.start_world_y = (
                self.current_y
            )

        self.initialized = True

        self.get_logger().info(
            "Controller and sensors ready."
        )

        return True

    # ============================================================
    # COMPATIBILITY METHOD
    #
    # run_policy.py calls:
    #
    # robot.wait_for_ready()
    #
    # This method redirects to wait_until_ready().
    # ============================================================

    def wait_for_ready(self):

        return self.wait_until_ready()

    # ============================================================
    # ESTIMATE CURRENT CELL FROM ODOMETRY
    # ============================================================

    def estimate_cell_from_pose(self):

        if self.current_x is None or self.current_y is None:
            return self.current_cell

        if self.start_world_x is None or self.start_world_y is None:
            return self.current_cell

        cell_x = round(
            (self.current_x - self.start_world_x) / self.CELL_SIZE + self.START_CELL[0]
        )

        cell_y = round(
            (self.current_y - self.start_world_y) / self.CELL_SIZE + self.START_CELL[1]
        )

        estimated = (int(cell_x), int(cell_y))

        if not self.cell_in_bounds(estimated):
            return self.current_cell

        return estimated

    # ============================================================
    # SET START CELL
    # ============================================================

    def set_start_cell(self, start_x, start_y=None):

        # Supports all formats:
        #
        # set_start_cell((0, 0))
        # set_start_cell(0, 0)
        # set_start_cell((0, 0), something)
        #
        # Extract tuple coordinates first.

        if isinstance(start_x, (tuple, list)):

            x = start_x[0]
            y = start_x[1]

        else:

            x = start_x
            y = start_y

        self.current_cell = (
            int(x),
            int(y) if y is not None else 0
        )

        self.START_CELL = self.current_cell
    # ============================================================
    # SET GOAL CELL
    # ============================================================

    def set_goal_cell(self, goal_x, goal_y=None):

        # Supports:
        #
        # set_goal_cell((4, 4))
        # set_goal_cell(4, 4)

        if isinstance(goal_x, (tuple, list)):

            x = goal_x[0]
            y = goal_x[1]

        else:

            x = goal_x
            y = goal_y

        # Validate and convert goal coordinates to integers
        if x is None or y is None:
            raise ValueError(f"Invalid goal coordinates: ({x}, {y})")

        try:
            gx = int(x)
            gy = int(y)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid goal coordinates: ({x}, {y})") from e

        self.goal_cell = (gx, gy)

    # ============================================================
    # SET HEADING
    # ============================================================

    def set_heading(
        self,
        heading
    ):

        if heading not in self.ACTIONS:

            raise ValueError(
                f"Invalid heading: {heading}"
            )

        self.heading = heading

    # ============================================================
    # PRINT START INFORMATION
    # ============================================================

    def print_start_information(self):

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            f"START CELL: {self.current_cell}"
        )

        self.get_logger().info(
            "START HEADING: "
            f"{self.ACTION_NAMES[self.heading]}"
        )

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            f"GOAL CELL: {self.goal_cell}"
        )

    # ============================================================
    # NORMALIZE ANGLE
    # ============================================================

    def normalize_angle(
        self,
        angle
    ):

        while angle > math.pi:

            angle -= (
                2.0
                *
                math.pi
            )

        while angle < -math.pi:

            angle += (
                2.0
                *
                math.pi
            )

        return angle

    # ============================================================
    # ACTION TO YAW
    # ============================================================

    def action_to_yaw(
        self,
        action
    ):

        # Gazebo coordinate system:
        #
        # EAST  = +X
        # NORTH = +Y
        # WEST  = -X
        # SOUTH = -Y

        if action == self.EAST:

            return 0.0

        elif action == self.NORTH:

            return (
                math.pi
                /
                2.0
            )

        elif action == self.WEST:

            return math.pi

        elif action == self.SOUTH:

            return (
                -math.pi
                /
                2.0
            )

        raise ValueError(
            f"Invalid action: {action}"
        )

    # ============================================================
    # GET LASER RANGE AT RELATIVE ANGLE
    #
    # Samples a small WINDOW of rays around target_angle and
    # returns their median, instead of trusting a single ray.
    #
    # A single-ray read is fragile: one stray reflection off part
    # of the robot's own frame/mount, or a millimeter of timing
    # jitter right at a doorway edge, is enough to flip an actually
    # -open direction to a false "blocked" reading for that one
    # scan - and since this function feeds directly into wall
    # detection, one bad ray becomes one wrong grid-cell decision.
    # A handful of nearby rays almost always agree with each other
    # when a wall is genuinely present (a wall spans much more
    # than a fraction of a degree at this range), so the median
    # smooths out exactly this kind of single-sample glitch while
    # still catching real walls.
    # ============================================================

    def get_range_at_angle(
        self,
        target_angle,
        window_rays=5,
        window_span_rad=0.15,
    ):

        if self.latest_scan is None:

            return float("inf")

        scan = self.latest_scan

        angle_min = scan.angle_min

        angle_increment = (
            scan.angle_increment
        )

        if angle_increment == 0:

            return float("inf")

        total = len(
            scan.ranges
        )

        if total == 0:

            return float("inf")

        half_span = window_span_rad / 2.0

        steps = max(1, window_rays // 2)

        samples = []

        for offset_index in range(-steps, steps + 1):

            offset = (
                (offset_index / steps) * half_span
                if steps > 0
                else 0.0
            )

            angle = target_angle + offset

            index = int(
                round(
                    (angle - angle_min) / angle_increment
                )
            ) % total

            value = scan.ranges[index]

            if math.isinf(value) or math.isnan(value):
                continue

            samples.append(value)

        if not samples:

            return float("inf")

        samples.sort()

        mid = len(samples) // 2

        if len(samples) % 2 == 1:

            return samples[mid]

        return (samples[mid - 1] + samples[mid]) / 2.0

    # ============================================================
    # GET ABSOLUTE WALLS
    #
    # Returns:
    #
    # EAST
    # NORTH
    # WEST
    # SOUTH
    #
    # True  = BLOCKED
    # False = OPEN
    #
    # This is LIDAR-only (plus grid-boundary checks, which are
    # legitimate since the arena size is known in advance, unlike
    # the internal obstacle layout, which is not).
    # ============================================================

    def cell_in_bounds(self, cell):

        x, y = cell

        return (
            0 <= x < self.GRID_SIZE
            and
            0 <= y < self.GRID_SIZE
        )

    def get_absolute_walls(self):

        self.refresh_current_cell()

        if self.current_cell is None:
            return {
                self.EAST: True,
                self.NORTH: True,
                self.WEST: True,
                self.SOUTH: True,
            }

        walls = {
            self.EAST: False,
            self.NORTH: False,
            self.WEST: False,
            self.SOUTH: False,
        }

        x, y = self.current_cell

        if x <= 0:
            walls[self.WEST] = True
        if x >= self.GRID_SIZE - 1:
            walls[self.EAST] = True
        if y <= 0:
            walls[self.SOUTH] = True
        if y >= self.GRID_SIZE - 1:
            walls[self.NORTH] = True

        if (
            self.latest_scan is None
            or
            self.current_yaw is None
        ):
            return walls

        east_relative_angle = self.normalize_angle(0.0 - self.current_yaw)
        east_distance = self.get_range_at_angle(east_relative_angle)
        walls[self.EAST] = (east_distance < self.wall_distance) or walls[self.EAST]

        north_relative_angle = self.normalize_angle((math.pi / 2.0) - self.current_yaw)
        north_distance = self.get_range_at_angle(north_relative_angle)
        walls[self.NORTH] = (north_distance < self.wall_distance) or walls[self.NORTH]

        west_relative_angle = self.normalize_angle(math.pi - self.current_yaw)
        west_distance = self.get_range_at_angle(west_relative_angle)
        walls[self.WEST] = (west_distance < self.wall_distance) or walls[self.WEST]

        south_relative_angle = self.normalize_angle((-math.pi / 2.0) - self.current_yaw)
        south_distance = self.get_range_at_angle(south_relative_angle)
        walls[self.SOUTH] = (south_distance < self.wall_distance) or walls[self.SOUTH]

        # Do not force corner walls to be blocked here: boundary checks above already cover
        # the outer maze edges, and the live LIDAR scan should determine whether a particular
        # direction is free or blocked. Hardcoding the start cell as fully blocked breaks the
        # initial valid-action list at (0, 0).
        return walls

    # ============================================================
    # SCAN CURRENT CELL
    #
    # No physical rotation.
    # ============================================================

    def refresh_current_cell(self):

        estimated = self.estimate_cell_from_pose()

        if estimated is not None and self.cell_in_bounds(estimated):
            self.current_cell = estimated

        return self.current_cell

    def scan_current_cell(self):

        rclpy.spin_once(
            self,
            timeout_sec=0.1
        )

        self.refresh_current_cell()

        walls = (
            self.get_absolute_walls()
        )

        self.get_logger().info(
            "Sensor scan complete without physical rotation."
        )

        return walls

    # ============================================================
    # VALID ACTIONS
    #
    # Priority:
    #
    # EAST -> NORTH -> WEST -> SOUTH
    # ============================================================

    def get_valid_actions(
        self,
        walls=None
    ):

        if walls is None:

            walls = (
                self.scan_current_cell()
            )

        self.refresh_current_cell()

        valid_actions = []

        for action in self.ACTIONS:

            if walls.get(action, True):
                continue

            x, y = self.current_cell
            dx, dy = {
                self.EAST: (1, 0),
                self.NORTH: (0, 1),
                self.WEST: (-1, 0),
                self.SOUTH: (0, -1),
            }[action]

            target_x = x + dx
            target_y = y + dy

            if not (0 <= target_x < self.GRID_SIZE):
                continue

            if not (0 <= target_y < self.GRID_SIZE):
                continue

            valid_actions.append(action)

        return valid_actions

    # ============================================================
    # GET RL STATE
    #
    # EXACTLY MATCHES TRAINING STATE:
    #
    # (
    #     dx,            # goal_x - robot_x
    #     dy,            # goal_y - robot_y
    #     heading,
    #     east_wall,
    #     north_wall,
    #     west_wall,
    #     south_wall
    # )
    #
    # The robot never knows the true maze layout: robot_cell comes
    # from odometry (self.current_cell) and goal_cell is whatever
    # it was told (self.goal_cell). Only the relative offset is
    # fed to the Q-table, along with what LIDAR currently reports
    # as open/blocked. That is what lets a Q-table trained on
    # randomized mazes work in a physical maze it has never seen.
    # ============================================================

    def get_state(
        self,
        walls=None
    ):

        if walls is None:

            walls = (
                self.scan_current_cell()
            )

        robot_x = (
            self.current_cell[0]
        )

        robot_y = (
            self.current_cell[1]
        )

        goal_x = (
            self.goal_cell[0]
        )

        goal_y = (
            self.goal_cell[1]
        )

        dx = goal_x - robot_x

        dy = goal_y - robot_y

        east_wall = int(
            walls[self.EAST]
        )

        north_wall = int(
            walls[self.NORTH]
        )

        west_wall = int(
            walls[self.WEST]
        )

        south_wall = int(
            walls[self.SOUTH]
        )

        return (

            dx,

            dy,

            self.heading,

            east_wall,

            north_wall,

            west_wall,

            south_wall,

        )

    # ============================================================
    # GET CURRENT INFORMATION
    #
    # Scan only once.
    # ============================================================

    def get_current_information(self):

        walls = (
            self.scan_current_cell()
        )

        valid_actions = (
            self.get_valid_actions(
                walls
            )
        )

        state = (
            self.get_state(
                walls
            )
        )

        return (

            state,

            walls,

            valid_actions

        )

    # ============================================================
    # STOP ROBOT
    # ============================================================

    def stop_robot(self):

        msg = Twist()

        self.cmd_pub.publish(
            msg
        )

    # ============================================================
    # TURN TO ACTION
    # ============================================================

    def turn_to_action(
        self,
        action
    ):

        target_yaw = (

            self.action_to_yaw(
                action
            )

        )

        msg = Twist()

        # A hard timeout so a stuck/oscillating turn can never hang
        # this call forever - it just gives up, logs it, and moves
        # on with whatever heading it actually reached.
        turn_started_at = time.time()
        turn_timeout_seconds = 10.0

        while rclpy.ok():

            rclpy.spin_once(

                self,

                timeout_sec=0.01

            )

            if self.current_yaw is None:

                continue

            error = (

                self.normalize_angle(

                    target_yaw

                    -

                    self.current_yaw

                )

            )

            if abs(error) < self.angle_tolerance:

                break

            if (time.time() - turn_started_at) > turn_timeout_seconds:

                self.get_logger().error(
                    "TURN TIMEOUT: could not reach heading "
                    f"{self.ACTION_NAMES[action]} within "
                    f"{turn_timeout_seconds:.1f}s "
                    f"(remaining error={error:.3f} rad)"
                )

                break

            msg.linear.x = 0.0

            # Proportional approach so the robot settles inside
            # angle_tolerance instead of hunting across it.
            speed = max(
                0.15,
                min(self.angular_speed, 1.5 * abs(error))
            )

            msg.angular.z = speed if error > 0 else -speed

            self.cmd_pub.publish(
                msg
            )

        self.stop_robot()

        time.sleep(0.2)

    # ============================================================
    # GET WORLD POSITION FOR GRID CELL
    # ============================================================

    def get_world_position_for_cell(
        self,
        cell
    ):

        cell_x = cell[0]

        cell_y = cell[1]

        world_x = (

            self.start_world_x

            +

            (
                cell_x
                -
                self.START_CELL[0]
            )

            *

            self.CELL_SIZE

        )

        world_y = (

            self.start_world_y

            +

            (
                cell_y
                -
                self.START_CELL[1]
            )

            *

            self.CELL_SIZE

        )

        return (

            world_x,

            world_y

        )

    # ============================================================
    # MOVE TO NEXT CELL
    # ============================================================

    def move_to_next_cell(
        self,
        action
    ):

        x, y = (
            self.current_cell
        )

        # --------------------------------------------------------
        # DETERMINE NEXT CELL
        # --------------------------------------------------------

        if action == self.EAST:

            next_cell = (
                x + 1,
                y
            )

        elif action == self.NORTH:

            next_cell = (
                x,
                y + 1
            )

        elif action == self.WEST:

            next_cell = (
                x - 1,
                y
            )

        elif action == self.SOUTH:

            next_cell = (
                x,
                y - 1
            )

        else:

            raise ValueError(
                f"Invalid action: {action}"
            )

        if not self.cell_in_bounds(next_cell):
            self.get_logger().error(
                f"ILLEGAL GRID MOVE: {self.current_cell} + {self.ACTION_NAMES[action]} -> {next_cell}"
            )
            return False

        self.get_logger().info(

            f"ACTION: "

            f"{self.ACTION_NAMES[action]} "

            f"{self.current_cell} "

            f"-> "

            f"{next_cell}"

        )

        # --------------------------------------------------------
        # TURN ROBOT
        # --------------------------------------------------------

        self.turn_to_action(
            action
        )

        # --------------------------------------------------------
        # SAFETY CHECK: inspect the LIDAR scan directly in the robot frame.
        # The laser is mounted facing the robot's forward direction, so the
        # local front of the robot is always at scan angle 0.0. Checking the
        # world-east angle (0.0 - current_yaw) instead reads the robot's right
        # side and can incorrectly reject valid moves such as NORTH from the
        # start cell.
        #
        # This pre-check is a fast fail before we even start moving. It is
        # NOT sufficient by itself: the loop below re-checks this
        # continuously while driving, because clearance can change (or an
        # initial reading can be marginal) over the course of the move.
        # --------------------------------------------------------

        if self.current_yaw is not None:
            forward_distance = self.get_range_at_angle(0.0)
            if math.isfinite(forward_distance) and forward_distance < self.safe_clearance:
                self.get_logger().warn(
                    "FRONT CLEARANCE TOO SMALL: "
                    f"action={self.ACTION_NAMES[action]} "
                    f"yaw={self.current_yaw:.3f} "
                    f"forward_distance={forward_distance:.3f} m"
                )
                self.stop_robot()
                return False

        # --------------------------------------------------------
        # TARGET WORLD POSITION
        # --------------------------------------------------------

        target_x, target_y = (

            self.get_world_position_for_cell(
                next_cell
            )

        )

        msg = Twist()

        # --------------------------------------------------------
        # MOVE TO CELL CENTER
        #
        # Two safety nets that were previously missing:
        #
        # 1. Continuous forward-clearance check. The old code only
        #    checked clearance once, before starting to move. If
        #    the robot drifts into a wall (physical contact, or an
        #    odometry-computed target that lands past the real
        #    wall surface due to accumulated drift), nothing here
        #    ever told it to stop - it just kept publishing forward
        #    velocity into the wall.
        #
        # 2. A hard timeout. If the target position can never be
        #    reached (because it's physically blocked), the old
        #    loop's only exit condition (`distance < tolerance`)
        #    is never satisfied, so it hangs forever: no
        #    "ARRIVED AT CENTER" print, no further sensor scan, no
        #    goal check, ever - which is exactly the symptom of
        #    "keeps driving, crashes, nothing prints" reported.
        #    Now it gives up after a bounded time, stops, and
        #    reports failure so run_policy.py can print
        #    "MOVEMENT FAILED" and exit cleanly instead of hanging.
        # --------------------------------------------------------

        target_yaw = self.action_to_yaw(action)

        move_started_at = time.time()
        move_timeout_seconds = 20.0

        while rclpy.ok():

            rclpy.spin_once(

                self,

                timeout_sec=0.01

            )

            if (
                self.current_x is None
                or
                self.current_y is None
                or
                self.current_yaw is None
            ):

                continue

            # ------------------------------------------------------
            # Decompose the offset to the target into a component
            # ALONG the direction of travel and one ACROSS it.
            #
            #   remaining > 0  -> target still ahead
            #   remaining <= 0 -> target reached or passed
            #   lateral   > 0  -> target lies to the robot's left
            #
            # Exiting on `remaining` instead of on straight-line
            # distance is what makes this robust: once the robot
            # crosses the plane through the cell centre it stops,
            # however far off the centreline it happens to be. The
            # old Euclidean test could be missed entirely.
            # ------------------------------------------------------

            offset_x = target_x - self.current_x
            offset_y = target_y - self.current_y

            cos_t = math.cos(target_yaw)
            sin_t = math.sin(target_yaw)

            remaining = (cos_t * offset_x) + (sin_t * offset_y)
            lateral = (-sin_t * offset_x) + (cos_t * offset_y)

            if remaining < self.position_tolerance:

                break

            # Obstacle check, but only while real travel remains. The
            # LIDAR is on the rear axle and the nose is 0.2415 m ahead
            # of it, so approaching a cell whose far side is a wall
            # legitimately reads about 0.45 m.
            if remaining > 0.25:

                forward_distance = self.get_range_at_angle(0.0)

                if (
                    math.isfinite(forward_distance)
                    and
                    forward_distance < self.safe_clearance
                ):

                    self.get_logger().warn(
                        "FRONT CLEARANCE TOO SMALL DURING MOVE: "
                        f"action={self.ACTION_NAMES[action]} "
                        f"forward_distance={forward_distance:.3f} m - stopping."
                    )

                    self.stop_robot()
                    self.refresh_current_cell()
                    return False

            if (time.time() - move_started_at) > move_timeout_seconds:

                self.get_logger().error(
                    "MOVE TIMEOUT: could not reach target cell "
                    f"{next_cell} for action {self.ACTION_NAMES[action]} "
                    f"within {move_timeout_seconds:.1f}s "
                    f"(remaining={remaining:.3f} m, "
                    f"lateral={lateral:.3f} m). Treating as blocked."
                )

                self.stop_robot()
                self.refresh_current_cell()
                return False

            # ------------------------------------------------------
            # Steer while driving. Two terms:
            #   heading  - point along the travel axis
            #   lateral  - converge back onto the cell centreline
            #
            # The lateral term is what actually performs the CENTER
            # phase the controller has always advertised. Without it
            # the robot leaves every turn on a diagonal and the error
            # compounds move after move.
            # ------------------------------------------------------

            heading_error = self.normalize_angle(
                target_yaw - self.current_yaw
            )

            lateral_term = max(-0.30, min(0.30, 1.2 * lateral))

            correction = (1.5 * heading_error) + lateral_term

            msg.angular.z = max(-0.60, min(0.60, correction))

            # Decelerate on approach, and slow down further while a
            # large heading error is being corrected, so the robot
            # turns back onto line instead of arcing wide.
            speed = min(self.linear_speed, 1.5 * remaining)

            if abs(heading_error) > 0.30:
                speed = min(speed, 0.05)

            msg.linear.x = max(0.04, speed)

            self.cmd_pub.publish(
                msg
            )

        # --------------------------------------------------------
        # STOP ROBOT
        # --------------------------------------------------------

        self.stop_robot()

        time.sleep(0.2)

        # --------------------------------------------------------
        # UPDATE GRID POSITION FROM ODOMETRY, NOT ONLY FROM intent
        # --------------------------------------------------------

        old_cell = (
            self.current_cell
        )

        estimated_cell = self.estimate_cell_from_pose()
        if estimated_cell is not None:
            self.current_cell = estimated_cell
        else:
            self.current_cell = next_cell

        self.heading = (
            action
        )

        centring_error = math.hypot(
            target_x - self.current_x,
            target_y - self.current_y,
        )

        # Split out the sideways component as well. The loop exits at
        # `remaining < position_tolerance`, so it always stops up to
        # 0.08 m SHORT along the travel axis, and that undershoot alone
        # puts centring_error near 0.080 on a perfectly straight move.
        # `lateral` is the part that actually reflects steering health,
        # and it is the one that must not grow move over move.
        lateral_error = (
            -math.sin(target_yaw) * (target_x - self.current_x)
            +
            math.cos(target_yaw) * (target_y - self.current_y)
        )

        self.get_logger().info(
            f"ARRIVED AT CENTER: {old_cell} -> {self.current_cell} "
            f"(off-centre by {centring_error:.3f} m, "
            f"lateral {lateral_error:+.3f} m)"
        )

        # --------------------------------------------------------
        # SCAN NEW CELL
        # --------------------------------------------------------

        self.scan_current_cell()

        return True

    # ============================================================
    # EXECUTE ACTION
    #
    # Only moves if sensor confirms direction is OPEN.
    # ============================================================

    def execute_action(
        self,
        action,
        walls=None
    ):

        if action not in self.ACTIONS:
            self.get_logger().error(f"INVALID ACTION: {action}")
            return False

        if walls is None:
            walls = self.scan_current_cell()

        self.refresh_current_cell()

        x, y = self.current_cell
        dx, dy = {
            self.EAST: (1, 0),
            self.NORTH: (0, 1),
            self.WEST: (-1, 0),
            self.SOUTH: (0, -1),
        }[action]

        target_x = x + dx
        target_y = y + dy

        if not (0 <= target_x < self.GRID_SIZE):
            self.get_logger().error(
                f"GRID BOUNDARY BLOCKED: {self.current_cell} + {self.ACTION_NAMES[action]} -> ({target_x}, {target_y})"
            )
            return False

        if not (0 <= target_y < self.GRID_SIZE):
            self.get_logger().error(
                f"GRID BOUNDARY BLOCKED: {self.current_cell} + {self.ACTION_NAMES[action]} -> ({target_x}, {target_y})"
            )
            return False

        if walls.get(action, True):
            self.get_logger().error(
                f"BLOCKED ACTION: {self.ACTION_NAMES[action]}"
            )
            return False

        if not self.cell_in_bounds((target_x, target_y)):
            self.get_logger().error(
                f"TARGET OUT OF GRID: {self.current_cell} -> ({target_x}, {target_y})"
            )
            return False

        valid_actions = self.get_valid_actions(walls)
        if action not in valid_actions:
            self.get_logger().error(
                f"ACTION NOT VALID ACCORDING TO CELL/ODOM: {self.ACTION_NAMES[action]} at {self.current_cell}"
            )
            return False

        return self.move_to_next_cell(action)

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def destroy_node(self):

        try:

            self.stop_robot()

        except Exception:

            pass

        super().destroy_node()


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    controller = GridRLController()

    try:

        controller.wait_for_ready()

        controller.print_start_information()

        rclpy.spin(
            controller
        )

    except KeyboardInterrupt:

        pass

    finally:

        controller.stop_robot()

        controller.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()