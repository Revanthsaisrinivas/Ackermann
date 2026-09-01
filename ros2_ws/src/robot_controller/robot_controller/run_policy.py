#!/usr/bin/env python3

"""
Run (and keep training) a Q-learning policy on the maze using the
ROS2 GridRLController.

Responsibilities:

    Gazebo launch:
        - Starts Gazebo
        - Spawns rl_robot
        - Sets the initial world pose
        (this script does NOT do any of that -- see note below)

    GridRLController:
        - Reads /scan, /odom
        - Controls /cmd_vel
        - Executes movement
        - get_current_walls() reports what the LIDAR actually senses
          at the robot's CURRENT cell -- there is no ground-truth
          wall table anywhere in that file, so this is genuinely
          "unknown map" sensing, not a simulated stand-in for it.

    run_policy.py (this file):
        - Prompts on the keyboard for start cell/heading AND goal
          cell (previously these came from ROS parameters with silent
          defaults of (0,0)/(4,4), which is why the controller kept
          starting from 0,0 regardless of what was actually spawned)
        - Builds state the SAME way rl_environment.py's unknown-map
          MazeEnvironment does: (cell_x, cell_y, goal_x, goal_y,
          heading, front_wall, left_wall, right_wall, back_wall) --
          the wall bits come from robot.get_current_walls(), i.e. the
          real sensor, never from a simulated/ground-truth table
        - Selects actions epsilon-greedy and keeps training live
          (agent.update_q_table() every step), same as the earlier
          agreed design -- this was previously pure greedy replay
          with no learning happening at all
        - Loads/saves the same q_table.pkl train.py uses, so learning
          accumulates across simulated AND real/Gazebo runs

IMPORTANT: this script NEVER spawns or deletes the robot. Gazebo must
already be running (via gazebo.launch.py, launched separately) with
rl_robot already spawned before you run this.
"""

from ament_index_python.packages import get_package_share_directory

import os
import sys
import time

import rclpy

from .rl_environment import MazeEnvironment
from .q_learning import QLearningAgent
from .grid_rl_controller import GridRLController


# =========================================================
# CONFIGURATION
# =========================================================

Q_TABLE_FILE = os.path.join(
    get_package_share_directory("robot_controller"),
    "q_table.pkl"
)

GRID_WIDTH = MazeEnvironment.GRID_SIZE
GRID_HEIGHT = MazeEnvironment.GRID_SIZE

MAX_STEPS = 200

# Real actions take real seconds (turn + drive a full cell) --
# without a cap here a stuck controller would hang this script
# forever instead of erroring out.
STEP_TIMEOUT_SEC = 30.0
STARTUP_TIMEOUT_SEC = 30.0

HEADING_NAMES = {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH"}


# =========================================================
# KEYBOARD INPUT
# =========================================================
# Replaces the old ROS-parameter-based start_x/start_y/start_heading/
# goal_x/goal_y (which silently defaulted to (0,0) and (4,4) if never
# overridden on the command line -- the direct cause of "the
# controller keeps starting from 0,0"). Prompted here instead, the
# same way gazebo.launch.py now prompts for the spawn cell -- ENTER
# THE SAME START CELL/HEADING YOU USED WHEN LAUNCHING GAZEBO, or the
# controller's internal cell-tracking will disagree with where the
# robot actually is in the physics world.

def prompt_cell(label):
    while True:
        raw = input(f"{label} (x y, space-separated, 0-{GRID_WIDTH - 1}): ")
        parts = raw.split()
        if len(parts) != 2:
            print("  Enter two integers separated by a space, e.g. '2 2'.")
            continue
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("  Both values must be integers.")
            continue
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            return (x, y)
        print(f"  Both values must be in 0-{GRID_WIDTH - 1}.")


def prompt_heading():
    while True:
        raw = input("Start heading (0=EAST, 1=NORTH, 2=WEST, 3=SOUTH): ")
        try:
            heading = int(raw)
        except ValueError:
            print("  Enter a single integer 0-3.")
            continue
        if heading in HEADING_NAMES:
            return heading
        print("  Must be one of 0, 1, 2, 3.")


# =========================================================
# STATE / ACTION HELPERS
# (pure functions, kept separate from the rclpy plumbing below so
# they can be tested without a live ROS2/Gazebo environment)
# =========================================================

def get_walls_relative_to_heading(heading, walls_by_direction, left_dir, right_dir, back_dir):
    """
    Lay out the controller's live wall booleans as
    (front, left, right, back) relative to heading -- the exact order
    rl_environment.py's get_state() uses -- so state tuples built here
    are directly compatible with a Q-table trained in simulation.
    """
    return (
        int(walls_by_direction[heading]),
        int(walls_by_direction[left_dir]),
        int(walls_by_direction[right_dir]),
        int(walls_by_direction[back_dir]),
    )


def get_valid_actions(cell, walls_by_direction, direction_delta):
    """
    Same semantics as MazeEnvironment.get_discovered_valid_actions():
    a direction is available if it's in-bounds AND the controller's
    live scan didn't just mark it blocked. The real robot fully
    (re)senses its current cell every time it's standing in it (see
    GridRLController.get_current_walls()), so this is never
    "cheating" -- it's the discovered-map model, just backed by a
    real sensor instead of a simulated one.
    """
    actions = []
    for action, (dx, dy) in direction_delta.items():
        if walls_by_direction.get(action, False):
            continue
        nx, ny = cell[0] + dx, cell[1] + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            actions.append(action)
    return actions


def build_state(cell, goal, heading, front, left, right, back):
    return (cell[0], cell[1], goal[0], goal[1], heading, front, left, right, back)


def read_walls(robot):
    """Returns (walls_by_direction dict, front, left, right, back)."""
    walls_by_direction = robot.get_current_walls()
    left_dir = robot.left_direction(robot.heading)
    right_dir = robot.right_direction(robot.heading)
    back_dir = (robot.heading + 2) % 4
    front, left, right, back = get_walls_relative_to_heading(
        robot.heading, walls_by_direction, left_dir, right_dir, back_dir
    )
    return walls_by_direction, front, left, right, back


def wait_until(robot, predicate, timeout_sec, description):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        rclpy.spin_once(robot, timeout_sec=0.1)
        if predicate(robot):
            return True
    robot.get_logger().warning(f"Timed out waiting for: {description}")
    return False


# =========================================================
# POLICY
# =========================================================

def run_policy(start, goal, start_heading, agent):
    """
    IMPORTANT: GridRLController DOES NOT spawn the robot.
    gazebo.launch.py must already have spawned rl_robot.
    """

    if not rclpy.ok():
        rclpy.init()

    robot = GridRLController(
        start_x=start[0],
        start_y=start[1],
        start_heading=start_heading
    )

    robot.get_logger().info("========== POLICY EXECUTION ==========")
    robot.get_logger().info(f"Start   : {start}")
    robot.get_logger().info(f"Goal    : {goal}")
    robot.get_logger().info(
        f"Heading : {HEADING_NAMES.get(start_heading, 'UNKNOWN')}"
    )

    try:
        print("Waiting for the controller to finish its startup scan...")
        ready = wait_until(
            robot,
            lambda r: r.initialized and r.state == "IDLE",
            STARTUP_TIMEOUT_SEC,
            "controller startup (scan_received/odom_received/initial scan)"
        )
        if not ready:
            print(
                "Controller never reached IDLE after startup -- check that "
                "Gazebo is actually running with rl_robot already spawned "
                "and /scan, /odom are publishing "
                "(ros2 topic hz /scan, ros2 topic hz /odom)."
            )
            return

        if robot.current_cell != start:
            robot.get_logger().warning(
                f"Controller initialized at {robot.current_cell}, "
                f"expected {start} -- this means the cell you entered here "
                f"doesn't match what Gazebo was actually launched with."
            )
        else:
            robot.get_logger().info(
                f"Controller correctly initialized at {robot.current_cell}"
            )

        total_reward = 0.0
        reached_goal = False

        for step_num in range(1, MAX_STEPS + 1):

            cell_before = robot.current_cell
            walls_dict, front, left, right, back = read_walls(robot)
            state = build_state(cell_before, goal, robot.heading, front, left, right, back)
            valid_actions = get_valid_actions(cell_before, walls_dict, robot.DIRECTION_DELTA)

            robot.get_logger().info(f"PATH: Step {step_num}  Cell={cell_before}")

            if cell_before == goal:
                robot.stop_robot()
                robot.state = 'FINISHED'
                robot.get_logger().info("=" * 42)
                robot.get_logger().info("GOAL REACHED!")
                robot.get_logger().info(f"Total steps: {step_num - 1}")
                robot.get_logger().info("=" * 42)
                reached_goal = True
                break

            if not valid_actions:
                robot.get_logger().error(
                    f"No valid actions from {cell_before} -- stuck."
                )
                break

            action = agent.choose_action(state, valid_actions)
            target_cell = robot.get_neighbor(cell_before, action)

            robot.get_logger().info(
                f"Step {step_num}: {cell_before} -> "
                f"{HEADING_NAMES[action]} -> {target_cell}"
            )

            robot.execute_action(action, target_cell)

            reached_idle = wait_until(
                robot,
                lambda r: r.state == "IDLE",
                STEP_TIMEOUT_SEC,
                f"action {HEADING_NAMES[action]} to reach IDLE"
            )

            if not reached_idle:
                robot.get_logger().error(
                    "Controller did not return to IDLE in time; "
                    "aborting episode."
                )
                break

            moved = robot.current_cell == target_cell
            just_reached_goal = robot.current_cell == goal

            if not moved:
                # Mirrors MazeEnvironment.INVALID_MOVE_PENALTY: the
                # controller's own emergency-stop logic
                # (front_distance < emergency_distance) aborted the
                # move before reaching target_cell.
                reward = MazeEnvironment.INVALID_MOVE_PENALTY
                robot.get_logger().warning(
                    f"Blocked before reaching {target_cell}, reward {reward}"
                )
            elif just_reached_goal:
                reward = MazeEnvironment.GOAL_REWARD
                robot.get_logger().info(f"Reached goal! reward {reward}")
            else:
                reward = MazeEnvironment.STEP_PENALTY

            done = just_reached_goal or (step_num == MAX_STEPS)

            next_cell = robot.current_cell
            next_walls_dict, nf, nl, nr, nb = read_walls(robot)
            next_state = build_state(next_cell, goal, robot.heading, nf, nl, nr, nb)
            next_valid_actions = get_valid_actions(next_cell, next_walls_dict, robot.DIRECTION_DELTA)

            agent.update_q_table(state, action, reward, next_state, next_valid_actions, done)

            total_reward += reward

            if just_reached_goal:
                reached_goal = True
                break

        agent.decay_epsilon()

        print("\n" + "=" * 50)
        print("Reached goal" if reached_goal else "Episode ended without reaching goal")
        print(f"Total reward: {total_reward}")
        print(f"Epsilon after this episode: {agent.epsilon:.3f}")
        print("=" * 50)

    except KeyboardInterrupt:
        robot.get_logger().info("Policy execution interrupted by user.")

    finally:
        # Only stop the robot. DO NOT delete rl_robot.
        robot.stop_robot()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("Live Q-learning episode against the real/Gazebo robot")
    print("=" * 60)
    print(
        "Make sure Gazebo is ALREADY running (in another terminal) with\n"
        "rl_robot already spawned at the SAME start cell/heading you're\n"
        "about to enter here.\n"
    )

    start = prompt_cell("Start cell")
    start_heading = prompt_heading()
    goal = prompt_cell("Goal cell")

    if goal == start:
        print("Goal cell must differ from the start cell. Exiting.")
        sys.exit(1)

    agent = QLearningAgent(
        actions=MazeEnvironment.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    try:
        agent.load(Q_TABLE_FILE)
        print(
            f"Loaded existing Q-table ({len(agent.q_table)} states, "
            f"epsilon={agent.epsilon:.3f}) from {Q_TABLE_FILE}"
        )
    except Exception as e:
        print(f"No usable Q-table at {Q_TABLE_FILE} ({e}); starting fresh.")

    run_policy(start, goal, start_heading, agent)

    agent.save(Q_TABLE_FILE)
    print(f"Q-table saved ({len(agent.q_table)} states) to {Q_TABLE_FILE}")


if __name__ == '__main__':
    main()
