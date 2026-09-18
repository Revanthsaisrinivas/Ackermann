#!/usr/bin/env python3

"""
Q-Learning Training for UNKNOWN Maze Navigation.

The maze structure is never included in the RL state, and the
maze itself is randomized every episode (domain randomization).
That combination is what lets the trained Q-table generalize to
a physical maze the robot has never seen before (e.g. an
instructor placing chairs as obstacles) instead of memorizing one
specific hidden layout.

State format (7 values, all maze-independent):

(
    dx,            # goal_x - robot_x
    dy,            # goal_y - robot_y
    heading,
    east_wall,
    north_wall,
    west_wall,
    south_wall
)

Action priority for equal Q-values:

EAST -> NORTH -> WEST -> SOUTH
"""

from .rl_environment import MazeEnvironment
from .q_learning import QLearningAgent
from .q_table_path import (
    describe_q_table_source,
    ensure_q_table_dir,
    q_table_path,
)

import random


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 3000

PRINT_EVERY = 10

# Probability that any given internal cell-to-cell edge is a
# wall when a new random training maze is generated. Lower this
# if the real deployment maze tends to be quite open (few chairs),
# raise it if it tends to be tight and maze-like.
BLOCK_PROBABILITY = 0.30


# Resolved once, in q_table_path.py, and shared with run_policy.py.
# NOT next to __file__ - that is what let training and the policy
# end up on two different files. See q_table_path.py.
SAVE_FILE = q_table_path()


# ============================================================
# TRAIN
# ============================================================

def train():

    print()

    print("=" * 60)

    print(
        "UNKNOWN MAZE Q-LEARNING TRAINING (DOMAIN RANDOMIZED)"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    env = MazeEnvironment()

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = QLearningAgent(

        actions=env.ACTIONS,

        learning_rate=0.1,

        discount_factor=0.95,

        epsilon=1.0,

        epsilon_decay=0.998,

        epsilon_min=0.05,
    )

    print(
        "Grid:",
        env.GRID_SIZE,
        "x",
        env.GRID_SIZE
    )

    print(
        "Episodes:",
        EPISODES
    )

    print()

    print(
        "State format:"
    )

    print(
        "(dx, dy, heading, east, north, west, south)"
    )

    print(
        "dx = goal_x - robot_x, dy = goal_y - robot_y"
    )

    print()

    print(
        "Tie priority:"
    )

    print(
        "EAST -> NORTH -> WEST -> SOUTH"
    )

    print()

    print(
        f"A NEW random maze is generated every episode "
        f"(block probability = {BLOCK_PROBABILITY})."
    )

    print()

    # ========================================================
    # CREATE ALL TRAINING CASES
    #
    # Start
    # Goal
    # Heading
    #
    # These are enumerated once. The maze itself is what gets
    # randomized every episode, not the start/goal/heading set.
    # ========================================================

    cases = []

    for goal_x in range(
        env.GRID_SIZE
    ):

        for goal_y in range(
            env.GRID_SIZE
        ):

            goal = (
                goal_x,
                goal_y
            )

            for robot_x in range(
                env.GRID_SIZE
            ):

                for robot_y in range(
                    env.GRID_SIZE
                ):

                    start = (
                        robot_x,
                        robot_y
                    )

                    # Start cannot equal goal
                    if start == goal:

                        continue

                    # Train all possible headings
                    for heading in env.ACTIONS:

                        cases.append(

                            (
                                start,
                                goal,
                                heading
                            )
                        )

    cases_per_episode = len(
        cases
    )

    print(
        "Cases per episode:",
        cases_per_episode
    )

    print()

    print(
        "Training started..."
    )

    print()

    total_successes = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(

        1,

        EPISODES + 1
    ):

        success_this_episode = 0

        # ----------------------------------------------------
        # NEW RANDOM MAZE FOR THIS EPISODE
        #
        # This is the key change that makes the agent learn a
        # policy driven by (relative goal direction + sensed
        # walls) instead of memorizing one fixed hidden maze.
        # ----------------------------------------------------

        env.blocked_edges = env.generate_random_maze(
            block_probability=BLOCK_PROBABILITY
        )

        # Randomize training cases
        random.shuffle(
            cases
        )

        # ----------------------------------------------------
        # TRAIN EACH CASE
        # ----------------------------------------------------

        for (

            start,

            goal,

            heading

        ) in cases:

            state = env.reset(

                start=start,

                goal=goal,

                heading=heading,

                randomize_maze=False,
            )

            done = False

            reward = 0

            # ------------------------------------------------
            # EPISODE
            # ------------------------------------------------

            while not done:

                # Sensor-confirmed open directions
                valid_actions = (
                    env.get_valid_actions()
                )

                if not valid_actions:

                    break

                # Training action
                action = agent.choose_action(

                    state,

                    valid_actions
                )

                if action is None:

                    break

                # Execute action
                (

                    next_state,

                    reward,

                    done

                ) = env.step(
                    action
                )

                # Valid actions in next state
                next_valid_actions = (

                    env.get_valid_actions()
                )

                # Update Q-table
                agent.update_q_table(

                    state,

                    action,

                    reward,

                    next_state,

                    next_valid_actions,

                    done
                )

                state = next_state

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if reward == env.GOAL_REWARD:

                success_this_episode += 1

        # ====================================================
        # END EPISODE
        # ====================================================

        total_successes += (
            success_this_episode
        )

        # Reduce exploration
        agent.decay_epsilon()

        # ====================================================
        # PRINT PROGRESS
        # ====================================================

        if episode % PRINT_EVERY == 0:

            total_cases = (

                episode

                *

                cases_per_episode
            )

            success_rate = (

                total_successes

                /

                total_cases

            ) * 100

            print()

            print(
                "-" * 60
            )

            print(
                f"Episode      : {episode}"
            )

            print(
                f"Success Rate : "
                f"{success_rate:.2f}%"
            )

            print(
                f"Epsilon      : "
                f"{agent.epsilon:.4f}"
            )

            print(
                f"Q States     : "
                f"{len(agent.q_table)}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Saving Q-table..."
    )

    ensure_q_table_dir(SAVE_FILE)

    agent.save(
        SAVE_FILE
    )

    print()

    print(
        f"Episodes       : "
        f"{EPISODES}"
    )

    print(
        f"Total Successes: "
        f"{total_successes}"
    )

    print(
        f"Final Epsilon  : "
        f"{agent.epsilon:.4f}"
    )

    print(
        f"Total States   : "
        f"{len(agent.q_table)}"
    )

    print()

    print(
        "Q-table saved:"
    )

    print(
        describe_q_table_source(SAVE_FILE)
    )

    print()

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    train()


if __name__ == "__main__":

    main()