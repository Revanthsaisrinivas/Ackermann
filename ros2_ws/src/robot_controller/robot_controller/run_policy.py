#!/usr/bin/env python3
"""
Run the trained Q-Learning policy in Gazebo (or on the real robot).

State format (7 values, maze-independent):

(
    dx,            # goal_x - robot_x
    dy,            # goal_y - robot_y
    heading,
    east_wall,
    north_wall,
    west_wall,
    south_wall
)

Action priority when Q-values are equal:

EAST -> NORTH -> WEST -> SOUTH

IMPORTANT: this script never has access to a "true" maze layout.
The robot only knows its own estimated cell (from odometry), the
goal cell it was told, and what its LIDAR currently sees. That is
what makes this work in an environment nobody trained on directly
(e.g. an instructor's chairs) as long as the Q-table was trained
with domain randomization (see train.py).
"""

import argparse
import os

import rclpy

from .rl_environment import MazeEnvironment
from .q_learning import QLearningAgent, STATE_FORMAT
from .grid_rl_controller import GridRLController
from .maze_memory import MazeMemory
from .q_table_path import (
    ENV_VAR,
    describe_q_table_source,
    q_table_path,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Resolved once, in q_table_path.py, and shared with train.py.
# This used to search three candidate locations in an order that
# did not match where train.py saved, so the policy could silently
# load a stale table. There is now exactly one file. See
# q_table_path.py.
Q_TABLE_FILE = q_table_path()

MAX_STEPS = 100

# Names of the 7 state elements, in order. Used only to check that a
# loaded Q-table is keyed the same way the controller asks questions.
EXPECTED_STATE_SHAPE = (
    "dx",
    "dy",
    "heading",
    "east",
    "north",
    "west",
    "south",
)


# ============================================================
# ACTION PRIORITY
#
# EAST -> NORTH -> WEST -> SOUTH
# ============================================================

ACTION_PRIORITY = [
    MazeEnvironment.EAST,
    MazeEnvironment.NORTH,
    MazeEnvironment.WEST,
    MazeEnvironment.SOUTH,
]


# ============================================================
# ACTION NAMES
# ============================================================

ACTION_NAMES = {
    MazeEnvironment.EAST: "EAST",
    MazeEnvironment.NORTH: "NORTH",
    MazeEnvironment.WEST: "WEST",
    MazeEnvironment.SOUTH: "SOUTH",
}


def action_name(action):

    return ACTION_NAMES.get(
        action,
        f"UNKNOWN({action})"
    )


# ============================================================
# LIDAR-ONLY FALLBACK
#
# Used only when the current state was never seen during
# training. This must NOT use any knowledge of the true maze
# layout (the real maze is unknown), so it works purely off:
#
#   - dx, dy   -> relative goal direction (already known: the
#                 goal cell was given to the robot, and the
#                 robot's own cell comes from odometry)
#   - valid_actions -> directions LIDAR currently reports open
#
# Strategy: prefer whichever open direction reduces the
# distance to the goal, tie-broken EAST -> NORTH -> WEST -> SOUTH.
# If no open direction reduces distance (goal is "behind" a
# wall), fall back to plain priority order among open directions.
# ============================================================

def lidar_greedy_fallback(dx, dy, valid_actions):

    preferred = set()

    if dx > 0:
        preferred.add(MazeEnvironment.EAST)
    elif dx < 0:
        preferred.add(MazeEnvironment.WEST)

    if dy > 0:
        preferred.add(MazeEnvironment.NORTH)
    elif dy < 0:
        preferred.add(MazeEnvironment.SOUTH)

    for action in ACTION_PRIORITY:

        if action in preferred and action in valid_actions:

            return action

    # Nothing moves us closer right now (e.g. goal direction is
    # walled off) - just take the highest-priority open direction.
    for action in ACTION_PRIORITY:

        if action in valid_actions:

            return action

    return None


# ============================================================
# LOAD Q-TABLE
# ============================================================

def load_agent():

    if not os.path.isfile(Q_TABLE_FILE):

        raise FileNotFoundError(
            "\nQ-table not found:\n"
            f"{Q_TABLE_FILE}\n"
            "\nThis is the only location that is consulted - there is\n"
            "no fallback search any more. Either train one:\n"
            "    python3 -m robot_controller.train\n"
            f"or point {ENV_VAR} at an existing table:\n"
            f"    export {ENV_VAR}=/path/to/q_table.pkl\n"
        )

    agent = QLearningAgent(

        actions=MazeEnvironment.ACTIONS,

        learning_rate=0.1,

        discount_factor=0.95,

        epsilon=0.0,

        epsilon_decay=1.0,

        epsilon_min=0.0,
    )

    agent.load(Q_TABLE_FILE)

    # Absolutely disable exploration.
    agent.epsilon = 0.0

    if not agent.q_table:

        raise RuntimeError(
            "Q-table is empty."
        )

    # ========================================================
    # STATE-FORMAT GUARD
    #
    # A table keyed on a different state encoding is not empty
    # and not undertrained - every single lookup just misses,
    # and the policy quietly runs on its fallback for the whole
    # episode while still printing "Q-TABLE LOADED / States:
    # 2456". That is exactly what this workspace was doing:
    # the saved table used the old 9-tuple absolute encoding
    # while the controller emits the 7-tuple relative one.
    # Fail loudly instead.
    # ========================================================

    sample_state = next(iter(agent.q_table))

    if (
        agent.state_format is not None
        and
        agent.state_format != STATE_FORMAT
    ):

        raise RuntimeError(
            "\nQ-TABLE STATE FORMAT MISMATCH\n"
            f"  file     : {Q_TABLE_FILE}\n"
            f"  saved as : {agent.state_format}\n"
            f"  expected : {STATE_FORMAT}\n"
            f"  key seen : {sample_state} ({len(sample_state)} elements)\n"
            "\nThis table was trained with a different state encoding, so\n"
            "every lookup would miss and the policy would silently run on\n"
            "the exploration fallback alone. Retrain:\n"
            "    python3 -m robot_controller.train\n"
        )

    if len(sample_state) != len(EXPECTED_STATE_SHAPE):

        raise RuntimeError(
            "\nQ-TABLE STATE ARITY MISMATCH\n"
            f"  file     : {Q_TABLE_FILE}\n"
            f"  key seen : {sample_state} ({len(sample_state)} elements)\n"
            f"  expected : {len(EXPECTED_STATE_SHAPE)} elements "
            f"{EXPECTED_STATE_SHAPE}\n"
            "\nRetrain:\n"
            "    python3 -m robot_controller.train\n"
        )

    print()

    print("=" * 60)
    print("Q-TABLE LOADED")
    print("=" * 60)

    print(
        f"File: {describe_q_table_source(Q_TABLE_FILE)}"
    )

    print(
        f"States: {len(agent.q_table)}"
    )

    print()

    print(
        "Mode: Q-TABLE + LIDAR-ONLY FALLBACK"
    )

    print(
        "Tie priority:"
    )

    print(
        "EAST -> NORTH -> WEST -> SOUTH"
    )

    print("=" * 60)

    return agent


# ============================================================
# SELECT BEST ACTION
#
# Only valid/open actions (as reported by LIDAR right now) are
# considered. If Q-values are equal:
#
# EAST -> NORTH -> WEST -> SOUTH
# ============================================================

def select_policy_action(
    agent,
    state,
    valid_actions
):

    if not valid_actions:

        return None

    # Ensure priority order.
    ordered_actions = [

        action

        for action in ACTION_PRIORITY

        if action in valid_actions

    ]

    if not ordered_actions:

        return None

    # ========================================================
    # STATE EXISTS
    # ========================================================

    if state in agent.q_table:

        q_values = agent.q_table[state]

        print()
        print("Q-values for this state:")
        for action in ACTION_PRIORITY:
            if action in valid_actions:
                print(f"  {action_name(action)} = {q_values[action]:.12f}")

        best_action = ordered_actions[0]
        best_value = q_values[best_action]
        tolerance = 1e-9

        for action in ordered_actions[1:]:

            q_value = q_values[action]

            # Equal values retain the earlier priority order.
            if abs(q_value - best_value) <= tolerance:
                continue

            if q_value > best_value + tolerance:

                best_value = q_value
                best_action = action

        print(f"Selected action by Q-value rule: {action_name(best_action)}")

        # Explicit tie detection for debugging.
        same_value_actions = []
        for action in ordered_actions:
            if abs(q_values[action] - best_value) <= tolerance:
                same_value_actions.append(action)

        if len(same_value_actions) > 1:
            print("Tie detected among actions: " + ", ".join(action_name(a) for a in same_value_actions))
            print("Priority order kept the earlier one: " + action_name(best_action))

        return best_action

    # ========================================================
    # STATE NOT FOUND -> LIDAR-ONLY GREEDY FALLBACK
    #
    # We deliberately do NOT consult the training maze here.
    # The real maze is unknown, so the only information we are
    # allowed to use is the relative goal direction and what
    # LIDAR currently reports as open.
    # ========================================================

    print()

    print("=" * 60)

    print(
        "WARNING: STATE NOT FOUND IN Q-TABLE"
    )

    print("=" * 60)

    print(
        f"State: {state}"
    )

    print()

    dx, dy = state[0], state[1]

    fallback_action = lidar_greedy_fallback(dx, dy, valid_actions)

    if fallback_action is not None:
        print(
            "Using LIDAR-only greedy fallback (toward goal direction, "
            "tie-break EAST -> NORTH -> WEST -> SOUTH):"
        )
        print(
            f"{action_name(fallback_action)}"
        )
        print()
        return fallback_action

    print(
        "Using priority fallback:"
    )

    print(
        "EAST -> NORTH -> WEST -> SOUTH"
    )

    print()

    return ordered_actions[0]


# ============================================================
# PRINT SENSOR WALLS
# ============================================================

def print_sensor_walls(walls):

    print(
        "Sensor walls:"
    )

    for action in ACTION_PRIORITY:

        if walls[action]:

            status = "BLOCKED"

        else:

            status = "OPEN"

        print(

            f"  {action_name(action)}: "

            f"{status}"

        )


# ============================================================
# EXECUTE POLICY
# ============================================================

def execute_policy(

    start,

    goal,

    start_heading,

    agent

):

    robot = None

    total_reward = 0

    try:

        # ====================================================
        # INITIALIZE ROS
        # ====================================================

        rclpy.init()

        # ====================================================
        # CREATE CONTROLLER
        # ====================================================

        robot = GridRLController()

        # ====================================================
        # WAIT FOR SENSORS
        # ====================================================

        print()

        print("=" * 60)
        print("UNKNOWN MAZE RL NAVIGATION")
        print("=" * 60)

        print(
            f"Start: {start}"
        )

        print(
            f"Goal: {goal}"
        )

        print(
            "Initial orientation: "
            f"{action_name(start_heading)}"
        )

        print()

        print(
            "The initial heading is physical orientation only."
        )

        print(
            "Movement is selected only from "
            "sensor-confirmed OPEN directions."
        )

        print()

        print(
            "Controller movement sequence:"
        )

        print(
            "CENTER -> TURN -> MOVE -> CENTER -> SCAN"
        )

        print()

        print(
            "Waiting for controller and sensors..."
        )

        # ====================================================
        # WAIT FOR CONTROLLER
        # ====================================================

        if not robot.wait_for_ready():

            raise RuntimeError(

                "Controller and sensor initialization timed out."

            )

        # ====================================================
        # SET START CELL
        # ====================================================

        robot.set_start_cell(start)
        robot.set_heading(start_heading)

        # ====================================================
        # SET GOAL
        # ====================================================

        robot.set_goal_cell(goal)

        # ====================================================
        # INITIAL SENSOR SCAN
        # ====================================================

        robot.scan_current_cell()

        print()

        print(
            "Controller and sensors ready."
        )

        # ====================================================
        # MAIN POLICY LOOP
        # ====================================================

        memory = MazeMemory(MazeEnvironment.GRID_SIZE)

        for step_number in range(

            1,

            MAX_STEPS + 1

        ):

            # =================================================
            # CURRENT CELL
            # =================================================

            current_cell = robot.current_cell

            # =================================================
            # GOAL CHECK
            # =================================================

            if current_cell == goal:

                print()

                print("=" * 60)

                print(
                    "GOAL REACHED"
                )

                print("=" * 60)

                print(
                    f"Steps: {step_number - 1}"
                )

                print(
                    f"Final cell: {current_cell}"
                )

                print(
                    f"Total reward: {total_reward}"
                )

                robot.stop_robot()

                return True

            # =================================================
            # SCAN CURRENT CELL
            #
            # Sensor information is refreshed before choosing
            # every action.
            # =================================================

            walls = robot.scan_current_cell()

            if walls is None:

                raise RuntimeError(

                    "Sensor scan failed."

                )

            # =================================================
            # VALID ACTIONS
            #
            # IMPORTANT: pass the SAME `walls` dict taken above.
            # If this is called with no argument, the controller
            # scans LIDAR again internally, moments after the
            # first scan - two independent readings that can
            # disagree with each other (and with `walls` above)
            # if the robot hasn't fully settled. That mismatch is
            # what caused "Sensor walls: NORTH BLOCKED" to appear
            # next to "Valid actions: [... NORTH ...]" in earlier
            # runs. One scan per step, reused everywhere below.
            # =================================================

            valid_actions = robot.get_valid_actions(walls)

            if not valid_actions:

                print()

                print(
                    "NO VALID ACTIONS AVAILABLE"
                )

                print(
                    f"Current cell: {current_cell}"
                )

                robot.stop_robot()

                return False

            # =================================================
            # GET CURRENT STATE
            #
            # (dx, dy, heading, east, north, west, south)
            #
            # Same `walls` reused again here, for the same reason
            # as above - one scan, one consistent picture of the
            # cell, for the whole step.
            # =================================================

            state = robot.get_state(walls)

            memory.observe(current_cell, walls)

            # =================================================
            # SELECT ACTION FROM Q-TABLE
            # =================================================

            # Consult the Q-table only when it actually knows this
            # state. When it doesn't, hand memory a None and let the
            # frontier/backtrack rule decide, instead of the old
            # memoryless greedy fallback that livelocked on dead ends.
            if state in agent.q_table:
                q_action = select_policy_action(
                    agent,
                    state,
                    valid_actions,
                )
            else:
                q_action = None

            action, reason = memory.choose(
                current_cell,
                goal,
                valid_actions,
                q_action,
            )

            if action is None:

                raise RuntimeError(
                    "No open direction available."
                )

            # =================================================
            # PRINT STEP INFORMATION
            # =================================================

            print()

            print(
                f"STEP {step_number}"
            )

            print()

            print(
                f"Cell: {current_cell}"
            )

            print(
                f"Heading: "
                f"{action_name(robot.heading)}"
            )

            print(
                f"State: {state}"
            )

            print()

            print_sensor_walls(

                walls

            )

            print()

            print(

                "Valid actions: "

                +

                str([

                    action_name(action)

                    for action

                    in valid_actions

                ])

            )

            print()

            print(
                f"Action: "
                f"{action_name(action)}  [{reason}]"
            )

            print(
                f"Memory: {memory.summary()}"
            )

            # =================================================
            # SAVE OLD CELL
            # =================================================

            old_cell = robot.current_cell

            # =================================================
            # EXECUTE ACTION
            #
            # The controller internally performs:
            #
            # TURN
            # ->
            # MOVE ONE CELL
            # ->
            # UPDATE CELL
            # ->
            # SCAN NEW CELL
            # =================================================

            success = robot.execute_action(

                action,

                walls

            )

            # =================================================
            # MOVEMENT FAILURE
            # =================================================

            if not success:

                print()

                print("=" * 60)

                print(
                    "MOVEMENT FAILED"
                )

                print("=" * 60)

                print(
                    f"Action: {action_name(action)}"
                )

                print(
                    f"Cell: {old_cell}"
                )

                reward = (

                    MazeEnvironment

                    .INVALID_MOVE_PENALTY

                )

                total_reward += reward

                print(
                    f"Reward: {reward}"
                )

                robot.stop_robot()

                return False

            # =================================================
            # NEW CELL
            # =================================================

            new_cell = robot.current_cell

            # =================================================
            # SAFETY CHECK
            # =================================================

            if new_cell == old_cell:

                print()

                print(
                    "Movement failed."
                )

                print(
                    "Robot did not enter "
                    "a new cell."
                )

                reward = (

                    MazeEnvironment

                    .INVALID_MOVE_PENALTY

                )

                total_reward += reward

                print(
                    f"Reward: {reward}"
                )

                robot.stop_robot()

                return False

            # =================================================
            # GOAL REACHED
            # =================================================

            if new_cell == goal:

                reward = (

                    MazeEnvironment

                    .GOAL_REWARD

                )

                total_reward += reward

                print()

                print(
                    f"New cell: {new_cell}"
                )

                print(
                    f"Reward: {reward}"
                )

                print()

                print("=" * 60)

                print(
                    "GOAL REACHED"
                )

                print("=" * 60)

                print(
                    f"Steps: {step_number}"
                )

                print(
                    f"Final cell: {new_cell}"
                )

                print(
                    f"Total reward: {total_reward}"
                )

                robot.stop_robot()

                return True

            # =================================================
            # NORMAL STEP
            # =================================================

            reward = (

                MazeEnvironment

                .STEP_PENALTY

            )

            total_reward += reward

            print()

            print(
                f"New cell: {new_cell}"
            )

            print(
                f"Reward: {reward}"
            )

        # ====================================================
        # MAX STEPS
        # ====================================================

        print()

        print("=" * 60)

        print(
            "MAXIMUM STEPS REACHED"
        )

        print("=" * 60)

        robot.stop_robot()

        return False

    except KeyboardInterrupt:

        print()

        print(
            "Navigation interrupted."
        )

        if robot is not None:

            robot.stop_robot()

        return False

    finally:

        if robot is not None:

            try:

                robot.stop_robot()

            except Exception:

                pass

            try:

                robot.destroy_node()

            except Exception:

                pass

        if rclpy.ok():

            rclpy.shutdown()


# ============================================================
# CLI PARSING
# ============================================================

def parse_heading(value):

    heading_map = {
        "EAST": MazeEnvironment.EAST,
        "NORTH": MazeEnvironment.NORTH,
        "WEST": MazeEnvironment.WEST,
        "SOUTH": MazeEnvironment.SOUTH,
    }

    if isinstance(value, str):
        key = value.strip().upper()
        if key in heading_map:
            return heading_map[key]
        try:
            value = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Heading must be one of EAST, NORTH, WEST, SOUTH or an integer 0..3."
            ) from exc

    if value not in MazeEnvironment.ACTIONS:
        raise argparse.ArgumentTypeError(
            f"Invalid heading: {value}. Valid headings are {MazeEnvironment.ACTIONS}."
        )

    return int(value)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run the trained RL policy in an unknown maze (Gazebo or real robot)."
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        required=True,
        help="Grid start position as X Y.",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        required=True,
        help="Grid goal position as X Y.",
    )
    parser.add_argument(
        "--heading",
        type=parse_heading,
        default=MazeEnvironment.EAST,
        help="Initial heading: EAST, NORTH, WEST, SOUTH or 0, 1, 2, 3.",
    )

    args = parser.parse_args()

    start = tuple(args.start)
    goal = tuple(args.goal)
    start_heading = args.heading

    if start == goal:
        parser.error("Start and goal positions must be different.")

    if not (0 <= start[0] < MazeEnvironment.GRID_SIZE and 0 <= start[1] < MazeEnvironment.GRID_SIZE):
        parser.error(f"Start position {start} is outside the {MazeEnvironment.GRID_SIZE}x{MazeEnvironment.GRID_SIZE} grid.")

    if not (0 <= goal[0] < MazeEnvironment.GRID_SIZE and 0 <= goal[1] < MazeEnvironment.GRID_SIZE):
        parser.error(f"Goal position {goal} is outside the {MazeEnvironment.GRID_SIZE}x{MazeEnvironment.GRID_SIZE} grid.")

    # ========================================================
    # LOAD Q-TABLE
    # ========================================================

    agent = load_agent()

    # ========================================================
    # EXECUTE POLICY
    # ========================================================

    success = execute_policy(

        start=start,

        goal=goal,

        start_heading=start_heading,

        agent=agent,

    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    if success:

        print("=" * 60)

        print(
            "NAVIGATION COMPLETED SUCCESSFULLY"
        )

        print("=" * 60)

    else:

        print("=" * 60)

        print(
            "NAVIGATION FAILED"
        )

        print("=" * 60)


if __name__ == "__main__":

    main()