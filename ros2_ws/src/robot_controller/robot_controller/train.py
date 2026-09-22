#!/usr/bin/env python3

"""
Q-Learning training for ONE FIXED MAZE.

So the maze is now FIXED and the state is ABSOLUTE:

    (robot_x, robot_y, goal_x, goal_y, heading)

25 cells x 25 goals x 4 headings = 2500 states, every one enumerable.
Training sweeps all of them, and this script REFUSES TO SAVE a table
that does not then solve every solvable pair by a shortest path.

THE MAZE IS STILL UNKNOWN TO THE ROBOT
--------------------------------------
Absolute state does not mean the executor reads the map. It knows where
it is (odometry) and where it was told to go; it never reads
blocked_edges. The layout reaches the policy only as Q-values baked in
here. That is the whole point: the environment is unknown until it is
trained on, and known afterwards.

CONSEQUENCE: A TABLE BELONGS TO A LAYOUT
----------------------------------------
Edit worlds/maze.world and this table is wrong for it. Retrain. The
layout signature is stamped into the file and reported by run_policy so
the mismatch is visible instead of silent.

    python3 -m robot_controller.train
"""

import argparse
import collections
import random
import time

from . import maze_layout

from .q_learning import (
    QLearningAgent,
    STATE_FORMAT,
)

from .q_table_path import (
    describe_q_table_source,
    ensure_q_table_dir,
    q_table_path,
)

from .rl_environment import MazeEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

# One pass = one episode from every (start, goal, heading) triple.
MAX_PASSES = 400

# Epsilon falls linearly from 1.0 to EPSILON_MIN across this many
# passes, then stays there. Exploring starts (below) already guarantee
# every state-action pair is tried once per pass, so epsilon only has
# to vary the REST of each trajectory.
EXPLORE_PASSES = 20

EPSILON_MIN = 0.10

# Check the greedy policy this often. Training stops as soon as it is
# optimal on every solvable pair, so the pass count above is a ceiling
# rather than a target.
EVAL_EVERY = 5

# Consecutive clean evaluations required before stopping, so a single
# lucky check cannot end training early.
STABLE_EVALS = 2

SAVE_FILE = q_table_path()


# ============================================================
# MAZE FACTS (trainer-side only; never reaches the policy)
# ============================================================

def bfs_distances(env, goal):
    """Shortest-path distance from every reachable cell to `goal`."""

    distance = {goal: 0}

    queue = collections.deque([goal])

    while queue:

        cell = queue.popleft()

        for action in env.ACTION_PRIORITY:

            # Walls are symmetric, so stepping backwards out of `cell`
            # is the same test as stepping forwards into it.
            if env.is_true_wall(cell, action):
                continue

            dx, dy = env.DIRECTIONS[action]

            neighbour = (cell[0] + dx, cell[1] + dy)

            if neighbour in distance:
                continue

            distance[neighbour] = distance[cell] + 1

            queue.append(neighbour)

    return distance


def all_cells(env):

    return [
        (x, y)
        for x in range(env.GRID_SIZE)
        for y in range(env.GRID_SIZE)
    ]


def enumerate_states(env):
    """Every state this maze can present, terminal ones excluded.

    Terminal states (robot already on the goal) are never acted from, so
    they carry no policy and are left out of the table.
    """

    states = []

    for goal in all_cells(env):

        for cell in all_cells(env):

            if cell == goal:
                continue

            for heading in env.ACTIONS:

                states.append(
                    (cell[0], cell[1], goal[0], goal[1], heading)
                )

    return states


# ============================================================
# EVALUATION
#
# Follows the greedy table exactly as run_policy.py does - same
# tie-break, same restriction to directions that are actually open -
# so a clean result here is a statement about the deployed policy and
# not about some idealised version of it.
# ============================================================

def evaluate(agent, env, distances, solvable):

    reached = 0
    optimal = 0
    missing = 0

    failures = []

    for (start, goal, heading) in solvable:

        cell = start
        current_heading = heading

        steps = 0
        done = False

        # A shortest path is at most GRID_SIZE^2 steps; twice that is a
        # generous loop guard.
        limit = 2 * env.GRID_SIZE * env.GRID_SIZE

        while steps < limit:

            state = (
                cell[0], cell[1], goal[0], goal[1], current_heading
            )

            if state not in agent.q_table:
                missing += 1
                break

            valid_actions = [
                action
                for action in env.ACTION_PRIORITY
                if not env.is_true_wall(cell, action)
            ]

            action = agent.best_action(state, valid_actions)

            if action is None:
                break

            dx, dy = env.DIRECTIONS[action]

            cell = (cell[0] + dx, cell[1] + dy)

            current_heading = action

            steps += 1

            if cell == goal:
                done = True
                break

        if done:

            reached += 1

            if steps == distances[goal][start]:
                optimal += 1
            else:
                failures.append((start, goal, heading, "suboptimal", steps))

        else:
            failures.append((start, goal, heading, "no goal", steps))

    return {
        "total": len(solvable),
        "reached": reached,
        "optimal": optimal,
        "missing": missing,
        "failures": failures,
    }


# ============================================================
# TRAIN
# ============================================================

def train(max_passes=MAX_PASSES, seed=0, verbose=True):

    random.seed(seed)

    env = MazeEnvironment()

    # The revisit penalty reads episode history, which is not in the
    # state. In a fixed maze that turns a converging table into a noisy
    # one. See rl_environment.
    env.use_revisit_penalty = False

    agent = QLearningAgent(
        actions=env.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=1.0,
        epsilon_min=EPSILON_MIN,
    )

    agent.layout_signature = env.layout_signature
    agent.layout_source = str(env.layout_source)

    # --------------------------------------------------------
    # Enumerate the whole state space up front, so the saved
    # table is complete by construction rather than by luck.
    # --------------------------------------------------------

    states = enumerate_states(env)

    agent.initialize_q_table(states)

    cells = all_cells(env)

    distances = {goal: bfs_distances(env, goal) for goal in cells}

    solvable = []
    unsolvable = []

    for goal in cells:
        for start in cells:
            if start == goal:
                continue
            for heading in env.ACTIONS:
                if start in distances[goal]:
                    solvable.append((start, goal, heading))
                else:
                    unsolvable.append((start, goal, heading))

    if verbose:
        print("=" * 62)
        print("FIXED-MAZE Q-LEARNING")
        print("=" * 62)
        print(f"Maze          : {env.layout_source}")
        print(f"Signature     : {env.layout_signature}")
        print(f"Inner walls   : {len(env.blocked_edges)}")
        print(f"State format  : {STATE_FORMAT}")
        print(f"States        : {len(states)}  (all initialised)")
        print(f"Solvable pairs: {len(solvable)}")

        if unsolvable:
            print(
                f"UNSOLVABLE    : {len(unsolvable)} "
                f"(maze is not fully connected)"
            )

        print(f"Saving to     : {SAVE_FILE}")
        print()

    # --------------------------------------------------------
    # EXPLORING STARTS
    #
    # Without this the table converges to a WRONG fixed point, and it
    # does so quietly. Measured here before the fix: from (1,0) with
    # goal (0,0), WEST steps straight onto the goal and is worth +100,
    # but it sat at 52.17 while EAST sat at 79.29 and was chosen.
    #
    # The cause is visit counts, not the algorithm. A state like
    # (1,0,0,0,EAST) can only be entered by moving EAST into (1,0),
    # which from goal (0,0) cannot happen - so it is reached ONLY as an
    # episode start, once per pass. Whichever action happened to lead
    # early then got picked by the greedy rule, and as epsilon fell the
    # other action stopped being sampled and froze at its stale value.
    #
    # Forcing the first action fixes it exactly. Every state here IS
    # the start state of some (cell, goal, heading) triple, so cycling
    # the forced action over the valid ones gives every state-action
    # pair one update per pass - the textbook exploring-starts
    # condition, which is what Q-learning's convergence proof assumes.
    # --------------------------------------------------------

    episodes = []

    for (start, goal, heading) in solvable:

        for action in env.ACTION_PRIORITY:

            if env.is_true_wall(start, action):
                continue

            episodes.append((start, goal, heading, action))

    if verbose:
        print(f"Episodes/pass : {len(episodes)}  (exploring starts)")
        print()

    started = time.time()

    clean_evals = 0
    result = None

    for pass_index in range(1, max_passes + 1):

        agent.epsilon = max(
            EPSILON_MIN,
            1.0 - (pass_index - 1) / float(EXPLORE_PASSES),
        )

        random.shuffle(episodes)

        for (start, goal, heading, forced_action) in episodes:

            state = env.reset(
                start=start,
                goal=goal,
                heading=heading,
                randomize_maze=False,
            )

            for step_index in range(env.max_steps):

                valid_actions = env.get_valid_actions()

                if step_index == 0:
                    action = forced_action
                else:
                    action = agent.choose_action(state, valid_actions)

                if action is None:
                    break

                next_state, reward, done = env.step(action)

                agent.update_q_table(
                    state,
                    action,
                    reward,
                    next_state,
                    env.get_valid_actions(),
                    done,
                )

                state = next_state

                if done:
                    break

        if pass_index % EVAL_EVERY:
            continue

        result = evaluate(agent, env, distances, solvable)

        if verbose:
            print(
                f"  pass {pass_index:>4}  "
                f"eps {agent.epsilon:4.2f}  "
                f"reach {result['reached']}/{result['total']}  "
                f"optimal {result['optimal']}/{result['total']} "
                f"({result['optimal'] / result['total']:.1%})"
            )

        if result["optimal"] == result["total"]:
            clean_evals += 1
            if clean_evals >= STABLE_EVALS:
                break
        else:
            clean_evals = 0

    elapsed = time.time() - started

    if result is None:
        result = evaluate(agent, env, distances, solvable)

    if verbose:
        print()
        print("=" * 62)
        print("RESULT")
        print("=" * 62)
        print(f"Passes run    : {pass_index}")
        print(f"Wall clock    : {elapsed:.1f}s")
        print(f"Table states  : {len(agent.q_table)}")
        print(f"Reached goal  : {result['reached']}/{result['total']}")
        print(
            f"Optimal path  : {result['optimal']}/{result['total']} "
            f"({result['optimal'] / result['total']:.1%})"
        )

        if result["missing"]:
            print(f"MISSING states: {result['missing']}")

        for failure in result["failures"][:10]:
            print(f"    {failure}")

        print()

    return agent, env, result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a Q-table for the fixed maze in maze.world.",
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=MAX_PASSES,
        help="maximum sweeps over every (start, goal, heading) triple",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="save even if the policy is not optimal on every pair",
    )

    args = parser.parse_args()

    agent, env, result = train(
        max_passes=args.passes,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # A table that does not solve its own maze is worse than no
    # table: it looks trained, and the failure shows up as a
    # robot driving badly. Refuse to write one unless asked.
    # --------------------------------------------------------

    if result["optimal"] != result["total"] and not args.force:

        print("NOT SAVING.")
        print()
        print(
            f"The greedy policy is optimal on only "
            f"{result['optimal']}/{result['total']} pairs."
        )
        print("Raise --passes, or pass --force to save it anyway.")
        print()

        return 1

    ensure_q_table_dir(SAVE_FILE)

    agent.save(SAVE_FILE)

    print("Q-table saved.")
    print()
    print(f"  path      : {describe_q_table_source(SAVE_FILE)}")
    print(f"  states    : {len(agent.q_table)}")
    print(f"  format    : {STATE_FORMAT}")
    print(f"  layout    : {env.layout_signature}")
    print()
    print("This table is valid for THIS maze only.")
    print("If worlds/maze.world changes, run this again.")
    print()

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
