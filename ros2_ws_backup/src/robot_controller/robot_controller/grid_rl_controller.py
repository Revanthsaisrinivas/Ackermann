#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class GridRLController(Node):

    # =========================================================
    # GLOBAL DIRECTIONS & STATES
    # =========================================================

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    DIRECTION_NAMES = {
        EAST: 'EAST',
        NORTH: 'NORTH',
        WEST: 'WEST',
        SOUTH: 'SOUTH'
    }

    DIRECTION_DELTA = {
        EAST: (1, 0),
        NORTH: (0, 1),
        WEST: (-1, 0),
        SOUTH: (0, -1)
    }

    def __init__(self):
        super().__init__('grid_rl_controller')

        # =====================================================
        # ROS PARAMETERS
        # =====================================================

        self.declare_parameter('x_cell', 0)
        self.declare_parameter('y_cell', 0)
        self.declare_parameter('heading', 0)

        self.start_x_cell = self.get_parameter('x_cell').value
        self.start_y_cell = self.get_parameter('y_cell').value
        self.configured_start_heading = self.get_parameter('heading').value

        # =====================================================
        # ROS PUBLISHER / SUBSCRIBERS
        # =====================================================

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # =====================================================
        # MAZE / ROBOT PARAMETERS
        # =====================================================

        self.cell_size = 1.0
        self.wall_threshold = 0.70

        # =====================================================
        # MOTION & CONTROL PARAMETERS
        # =====================================================

        self.forward_speed = 0.12
        self.turn_speed = 0.30

        self.position_tolerance = 0.02
        self.lateral_tolerance = 0.04
        self.angle_tolerance = math.radians(1.5)
        self.emergency_distance = 0.18

        # =====================================================
        # SENSOR DATA & TRUE SCAN BUFFERING
        # =====================================================

        self.scan_received = False
        self.odom_received = False

        self.front_distance = 10.0
        self.left_distance = 10.0
        self.right_distance = 10.0
        self.back_distance = 10.0

        self.scan_buffer = []
        self.required_scan_samples = 5

        # =====================================================
        # ODOMETRY
        # =====================================================

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.locked_yaw = 0.0

        # =====================================================
        # GRID / STATE TRACKING
        # =====================================================

        self.current_cell = (0, 0)
        self.heading = self.EAST

        self.state = 'WAITING'

        self.target_heading = None
        self.target_cell = None
        self.target_x = None
        self.target_y = None
        self.initialized = False

        self.state_timer = 0.0
        self.stabilization_duration = 0.4

        # Control Timer (20 Hz)
        self.control_timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.get_logger().info('Grid-Centric RL Motion Controller Initialized')

    # =========================================================
    # MATH & GEOMETRY HELPERS
    # =========================================================

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def direction_to_yaw(self, direction):
        if direction == self.EAST:
            return 0.0
        if direction == self.NORTH:
            return math.pi / 2.0
        if direction == self.WEST:
            return math.pi
        if direction == self.SOUTH:
            return -math.pi / 2.0
        return 0.0

    def yaw_to_direction(self, yaw):
        candidates = {
            self.EAST: 0.0,
            self.NORTH: math.pi / 2.0,
            self.WEST: math.pi,
            self.SOUTH: -math.pi / 2.0
        }
        best_direction = self.EAST
        smallest_error = float('inf')

        for direction, target_yaw in candidates.items():
            error = abs(self.normalize_angle(target_yaw - yaw))
            if error < smallest_error:
                smallest_error = error
                best_direction = direction

        return best_direction

    def get_neighbor(self, cell, direction):
        dx, dy = self.DIRECTION_DELTA[direction]
        return (cell[0] + dx, cell[1] + dy)

    def left_direction(self, direction):
        return (direction + 1) % 4

    def right_direction(self, direction):
        return (direction - 1) % 4

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    # =========================================================
    # SENSOR CALLBACKS
    # =========================================================

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_received = True

    def get_sector_distance_from_ranges(self, ranges, angle_min, angle_increment, center_angle_deg, half_width_deg=5):
        center_angle = math.radians(center_angle_deg)
        half_width = math.radians(half_width_deg)
        valid_values = []

        for index, distance in enumerate(ranges):
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            angle = angle_min + index * angle_increment
            if abs(self.normalize_angle(angle - center_angle)) <= half_width:
                valid_values.append(distance)

        if not valid_values:
            return 10.0

        valid_values.sort()
        middle = len(valid_values) // 2
        if len(valid_values) % 2 == 0:
            return (valid_values[middle - 1] + valid_values[middle]) / 2.0
        return valid_values[middle]

    def scan_callback(self, msg):
        if len(msg.ranges) == 0:
            return

        self.front_distance = self.get_sector_distance_from_ranges(
            msg.ranges, msg.angle_min, msg.angle_increment, 0
        )
        self.left_distance = self.get_sector_distance_from_ranges(
            msg.ranges, msg.angle_min, msg.angle_increment, 90
        )
        self.right_distance = self.get_sector_distance_from_ranges(
            msg.ranges, msg.angle_min, msg.angle_increment, -90
        )
        self.back_distance = self.get_sector_distance_from_ranges(
            msg.ranges, msg.angle_min, msg.angle_increment, 180
        )

        self.scan_received = True
        
        self.get_logger().info(
            f"Distances -> "
            f"Front={self.front_distance:.2f} "
            f"Left={self.left_distance:.2f} "
            f"Right={self.right_distance:.2f} "
            f"Back={self.back_distance:.2f}"
)

        if self.state in ['CENTERING', 'SCANNING']:
            self.scan_buffer.append(msg)

    def compute_averaged_scan(self):
        if not self.scan_buffer:
            return

        num_rays = len(self.scan_buffer[0].ranges)
        angle_min = self.scan_buffer[0].angle_min
        angle_increment = self.scan_buffer[0].angle_increment

        averaged_ranges = []
        for i in range(num_rays):
            ray_sum = 0.0
            ray_count = 0
            for scan in self.scan_buffer:
                if i < len(scan.ranges):
                    val = scan.ranges[i]
                    if math.isfinite(val) and val > 0.0:
                        ray_sum += val
                        ray_count += 1
            if ray_count > 0:
                averaged_ranges.append(ray_sum / ray_count)
            else:
                averaged_ranges.append(float('inf'))

        self.front_distance = self.get_sector_distance_from_ranges(
            averaged_ranges, angle_min, angle_increment, 0
        )
        self.left_distance = self.get_sector_distance_from_ranges(
            averaged_ranges, angle_min, angle_increment, 90
        )
        self.right_distance = self.get_sector_distance_from_ranges(
            averaged_ranges, angle_min, angle_increment, -90
        )
        self.back_distance = self.get_sector_distance_from_ranges(
            averaged_ranges, angle_min, angle_increment, 180
        )

        self.scan_buffer.clear()

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize_navigation(self):
        self.current_cell = (self.start_x_cell, self.start_y_cell)

        if 0 <= self.configured_start_heading <= 3:
            self.heading = self.configured_start_heading
        else:
            self.heading = self.yaw_to_direction(self.yaw)

        self.initialized = True
        self.state = 'SCANNING'
        self.state_timer = self.get_clock().now().nanoseconds / 1e9
        self.scan_buffer.clear()

        self.get_logger().info('==========================================')
        self.get_logger().info(f'START CELL: {self.current_cell}')
        self.get_logger().info(f'LOCKED HEADING: {self.DIRECTION_NAMES[self.heading]}')
        self.get_logger().info('==========================================')

    # =========================================================
    # WALL DETECTION EXPOSURE FOR RL AGENT
    # =========================================================

    def get_current_walls(self):

        front_dir = self.heading
        left_dir = self.left_direction(self.heading)
        right_dir = self.right_direction(self.heading)
        back_dir = (self.heading + 2) % 4

        self.get_logger().info(
            f"Distances : "
            f"F={self.front_distance:.3f} "
            f"L={self.left_distance:.3f} "
            f"R={self.right_distance:.3f} "
            f"B={self.back_distance:.3f}"
        )

        self.get_logger().info(
            f"Threshold = {self.wall_threshold:.2f}"
        )

        walls = {
            front_dir: self.front_distance <= self.wall_threshold,
            left_dir: self.left_distance <= self.wall_threshold,
            right_dir: self.right_distance <= self.wall_threshold,
            back_dir: self.back_distance <= self.wall_threshold
        }

        self.get_logger().info(f"Walls = {walls}")

        return walls

    # =========================================================
    # EXECUTION API CALLED BY RL POLICY SCRIPT
    # =========================================================

    def execute_action(self, direction, target_cell):
        self.target_heading = direction
        self.target_cell = target_cell
        self.target_x = target_cell[0] + 0.5
        self.target_y = target_cell[1] + 0.5

        self.state = 'TURNING'
        self.stop_robot()

    # =========================================================
    # EXECUTION STATES
    # =========================================================

    def execute_turn(self):
        target_yaw = self.direction_to_yaw(self.target_heading)
        angle_error = self.normalize_angle(target_yaw - self.yaw)

        if abs(angle_error) <= self.angle_tolerance:
            self.stop_robot()
            self.heading = self.target_heading
            self.locked_yaw = target_yaw
            self.state = 'MOVING'
            return

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.turn_speed if angle_error > 0.0 else -self.turn_speed
        self.cmd_pub.publish(cmd)

    def execute_move(self):
        if self.target_x is None or self.target_y is None:
            self.stop_robot()
            self.state = 'SCANNING'
            self.state_timer = self.get_clock().now().nanoseconds / 1e9
            self.scan_buffer.clear()
            return

        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance_to_target = math.hypot(dx, dy)

        forward_vector_x = math.cos(self.locked_yaw)
        forward_vector_y = math.sin(self.locked_yaw)
        perp_vector_x = -forward_vector_y
        perp_vector_y = forward_vector_x

        prev_center_x = self.current_cell[0] + 0.5
        prev_center_y = self.current_cell[1] + 0.5

        longitudinal_progress = (self.x - prev_center_x) * forward_vector_x + (self.y - prev_center_y) * forward_vector_y
        lateral_error = abs((self.x - prev_center_x) * perp_vector_x + (self.y - prev_center_y) * perp_vector_y)

        arrival_condition_met = (
            (distance_to_target <= self.position_tolerance or longitudinal_progress >= self.cell_size) and
            (lateral_error <= self.lateral_tolerance)
        )

        if arrival_condition_met:
            self.stop_robot()
            
            previous_cell = self.current_cell
            self.current_cell = self.target_cell
            
            self.get_logger().info(
                f"ODOM=({self.x:.3f},{self.y:.3f})  CELL={self.current_cell}"
            )

            self.get_logger().info(f'ARRIVED AT CENTER: {previous_cell} -> {self.current_cell}')

            self.target_cell = None
            self.target_x = None
            self.target_y = None

            self.state = 'CENTERING'
            self.state_timer = self.get_clock().now().nanoseconds / 1e9
            self.scan_buffer.clear()
            return

        if self.front_distance < self.emergency_distance:
            self.stop_robot()
            self.get_logger().warning(f'EMERGENCY STOP: Obstacle detected at {self.front_distance:.2f}m')
            self.target_cell = None
            self.target_x = None
            self.target_y = None
            self.state = 'SCANNING'
            self.state_timer = self.get_clock().now().nanoseconds / 1e9
            self.scan_buffer.clear()
            return

        cmd = Twist()
        heading_error = self.normalize_angle(self.locked_yaw - self.yaw)
        signed_lateral_error = (self.x - prev_center_x) * perp_vector_x + (self.y - prev_center_y) * perp_vector_y

        cmd.linear.x = self.forward_speed
        cmd.angular.z = 1.2 * heading_error - 0.8 * signed_lateral_error
        cmd.angular.z = max(-0.2, min(0.2, cmd.angular.z))

        self.cmd_pub.publish(cmd)

    # =========================================================
    # CONTROL LOOP STATE MACHINE
    # =========================================================

    def control_loop(self):
        if not self.scan_received or not self.odom_received:
            return

        if not self.initialized:
            self.initialize_navigation()
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.state == 'WAITING' or self.state == 'IDLE':
            self.stop_robot()

        elif self.state == 'TURNING':
            self.execute_turn()

        elif self.state == 'MOVING':
            self.execute_move()

        elif self.state == 'CENTERING':
            self.stop_robot()
            if current_time - self.state_timer >= self.stabilization_duration and len(self.scan_buffer) >= self.required_scan_samples:
                self.compute_averaged_scan()
                self.state = 'SCANNING'
                self.state_timer = current_time

        elif self.state == 'SCANNING':
            self.stop_robot()
            if current_time - self.state_timer >= self.stabilization_duration and len(self.scan_buffer) >= self.required_scan_samples:
                self.compute_averaged_scan()
                self.state = 'IDLE'
                self.get_logger().info('Robot centered and scanned. Awaiting policy decision...')

        elif self.state == 'FINISHED':
            self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = GridRLController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()