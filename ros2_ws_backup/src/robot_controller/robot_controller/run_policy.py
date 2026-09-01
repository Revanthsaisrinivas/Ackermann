#!/usr/bin/env python3
"""
Run a trained Q-learning policy on the maze using the ROS2 GridRLController.

This script:
1. Initializes ROS2, the MazeEnvironment, and the GridRLController node (the body).
2. Verifies initialization cell alignment against the expected start coordinate.
3. Loads the trained Q-table via QLearningAgent.
4. Caches environment configurations (like grid size) for high performance.
5. Loops with step limits, waiting for the robot to reach IDLE, evaluating the next action,
   tracking trajectory loops, executing movements with safety timeouts, and securely synchronizing state.
"""
from ament_index_python.packages import get_package_share_directory
import os
import time
import rclpy

from .rl_environment import MazeEnvironment
from .q_learning import QLearningAgent
from .grid_rl_controller import GridRLController

Q_TABLE_FILE = os.path.join(
    get_package_share_directory("robot_controller"),
    "q_table.pkl"
)
MAX_STEPS = 200


def run_policy(start=(0, 0), goal=(4, 4)):
    # Initialize ROS2 and the controller node
    if not rclpy.ok():
        rclpy.init()

    robot = GridRLController()

    # Create and explicitly reset the environment with the correct start and goal
    env = MazeEnvironment()
    env.reset(start=start, goal=goal)
    grid_size = env.GRID_SIZE

    agent = QLearningAgent(
        actions=env.ACTIONS,
        states=env.get_all_states(),
        epsilon=0.0
    )

    try:
        agent.load(Q_TABLE_FILE)
        robot.get_logger().info(f"Loaded Q-table from {Q_TABLE_FILE}")
    except Exception as e:
        robot.get_logger().error(f"Failed to load Q-table: {e}")
        robot.destroy_node()
        rclpy.shutdown()
        return

    robot.get_logger().info("========== POLICY EXECUTION ==========")
    robot.get_logger().info(f"Start : {start}")
    robot.get_logger().info(f"Goal  : {goal}")

    # Wait for sensors, localization, and initial setup to complete
    while rclpy.ok() and not robot.initialized:
        rclpy.spin_once(robot, timeout_sec=0.1)

    # Verify that the controller's initialized cell matches the requested start position
    if robot.current_cell != start:
        robot.get_logger().warning(
            f"Controller initialized at {robot.current_cell}, expected {start}"
        )

    steps = 0
    done = False
    visited_history = []

    try:
        while rclpy.ok() and not done:
            # Keep ROS2 callbacks spinning to update odometry and laser scans asynchronously
            rclpy.spin_once(robot, timeout_sec=0.05)

            # Enforce maximum step limit to prevent infinite loops during testing
            if steps >= MAX_STEPS:
                robot.get_logger().error("Maximum step limit reached.")
                robot.stop_robot()
                break

            # Wait until the robot has completed movement, centered, and scanned (IDLE state)
            if robot.state != 'IDLE':
                continue

            current_cell = robot.current_cell
            robot.get_logger().info(
    f"PATH: Step {steps}  Cell={current_cell}"
)

            # Check if goal has been reached
            if current_cell == goal:
                robot.stop_robot()
                robot.state = 'FINISHED'
                robot.get_logger().info("==========================================")
                robot.get_logger().info("GOAL REACHED BY RL POLICY!")
                robot.get_logger().info(f"Total Steps : {steps}")
                robot.get_logger().info(f"Visited unique cells : {len(set(visited_history))}")
                robot.get_logger().info("==========================================")
                done = True
                break

            # Optional Loop Detection for imperfect Q-tables
            visited_history.append(current_cell)
            if len(visited_history) > 20:
                visited_history.pop(0)
            if visited_history.count(current_cell) > 5:
                robot.get_logger().warning("Possible policy loop detected!")

            # Construct state descriptor expected by the Q-learning agent
            state = (current_cell[0], current_cell[1], goal[0], goal[1])

            # Use the same valid-action logic as the training environment
            env.reset(start=current_cell, goal=goal)

            valid_actions = env.get_valid_actions()

            robot.get_logger().info(f"Current Cell : {current_cell}")
            robot.get_logger().info(f"Heading      : {robot.DIRECTION_NAMES[robot.heading]}")
            robot.get_logger().info(
                f"Valid Action Names : {[robot.DIRECTION_NAMES[a] for a in valid_actions]}"
            )
            robot.get_logger().info(f"Valid Actions: {valid_actions}")
            robot.get_logger().info(
                f"Valid Action Names : {[robot.DIRECTION_NAMES[a] for a in valid_actions]}"
            )

            if not valid_actions:
                robot.get_logger().error("No valid actions available! Robot is trapped.")
                robot.stop_robot()
                break

            q = agent.q_table[state]
            robot.get_logger().info(f"RAW Q = {q}")

            robot.get_logger().info("Q-values:")
            for a in valid_actions:
                robot.get_logger().info(
                    f"  {robot.DIRECTION_NAMES[a]} : {q[a]:.4f}"
                )

            # ---- Step 1: log the complete decision ----
            robot.get_logger().info("=" * 50)
            robot.get_logger().info(f"State : {state}")
            robot.get_logger().info(f"Current Cell : {current_cell}")

            for a in valid_actions:
                robot.get_logger().info(
                    f"{robot.DIRECTION_NAMES[a]} : {q[a]:.6f}"
                )

            # Get the best action from the agent using the Q-table
            action = agent.best_action(state, valid_actions)
            robot.get_logger().info(
                f"PATH: Action={robot.DIRECTION_NAMES[action]}"
            )

            robot.get_logger().info(
                f"Chosen Action : {robot.DIRECTION_NAMES[action]}"
            )
            # ---- end Step 1 ----

            if action is None:
                robot.get_logger().error("Agent returned no valid action.")
                robot.stop_robot()
                break

            target_cell = robot.get_neighbor(current_cell, action)

            robot.get_logger().info(
                f"Target Cell  : {target_cell}"
            )

            steps += 1

            robot.get_logger().info("--------------------------------------")
            robot.get_logger().info(f"Step   : {steps}")
            robot.get_logger().info(f"Cell   : {current_cell} -> {target_cell}")
            robot.get_logger().info(f"Action : {robot.DIRECTION_NAMES[action]}")

            # Command the robot to execute the action
            robot.execute_action(action, target_cell)

            # Wait until the robot finishes the movement and returns to IDLE (with a 30s timeout guard)
            start_wait = time.time()
            while rclpy.ok() and robot.state != "IDLE":
                rclpy.spin_once(robot, timeout_sec=0.05)

                if time.time() - start_wait > 30:
                    robot.get_logger().error("Movement timeout.")
                    robot.stop_robot()
                    return

            # ---- Step 2: verify execution ----
            robot.get_logger().info(
                f"Expected Cell : {target_cell}"
            )
            robot.get_logger().info(
                f"Actual Cell   : {robot.current_cell}"
            )
            robot.get_logger().info(
                f"Heading After Move : {robot.DIRECTION_NAMES[robot.heading]}"
            )
            # ---- end Step 2 ----

            # Verify that the robot actually reached the intended cell
            if robot.current_cell != target_cell:
                robot.get_logger().warning(
                    f"Movement failed! Expected {target_cell}, "
                    f"but robot is at {robot.current_cell}"
                )
                robot.get_logger().warning("Replanning from current position...")
                continue

            # Keep logical environment position synchronized for debugging and analytics
            if hasattr(env, "current_position"):
                env.current_position = robot.current_cell
            elif hasattr(env, "set_position"):
                env.set_position(robot.current_cell)

    except KeyboardInterrupt:
        robot.get_logger().info("Policy execution interrupted by user.")
    finally:
        robot.stop_robot()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # Lightweight node to read parameters
    node = rclpy.create_node("policy_parameter_reader")

    node.declare_parameter("start_x", 0)
    node.declare_parameter("start_y", 0)
    node.declare_parameter("goal_x", 4)
    node.declare_parameter("goal_y", 4)

    start = (
        node.get_parameter("start_x").value,
        node.get_parameter("start_y").value
    )

    goal = (
        node.get_parameter("goal_x").value,
        node.get_parameter("goal_y").value
    )

    node.get_logger().info(f"Start : {start}")
    node.get_logger().info(f"Goal  : {goal}")

    node.destroy_node()
    rclpy.shutdown()

    run_policy(start=start, goal=goal)


if __name__ == "__main__":
    main()