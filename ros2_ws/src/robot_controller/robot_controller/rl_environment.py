#!/usr/bin/env python3
"""
RL training environment for the 5x5 maze defined in maze.world.

WALL DATA SOURCE OF TRUTH
---------------------------
BLOCKED_EDGES below is derived DIRECTLY from the wall <pose>/<size>
values in maze.world. Each entry is an explicit, unambiguous pair of
adjacent cells that a wall sits between -- commented with exactly
which named wall model in maze.world it comes from, so there's no
row/col ordering convention to get backwards (that ambiguity is what
caused a real, confirmed-by-testing bug in an earlier version of this
file's wall tables -- this representation removes that failure mode
entirely rather than just fixing one instance of it).

If you ever edit a wall in maze.world, add/update the matching entry
here.
"""

import random


class MazeEnvironment:

    # =====================================================
    # Directions
    # =====================================================

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    ACTIONS = [EAST, NORTH, WEST, SOUTH]

    DELTA = {
        EAST:  (1, 0),
        NORTH: (0, 1),
        WEST:  (-1, 0),
        SOUTH: (0, -1)
    }

    GRID_SIZE = 5

    # =====================================================
    # Maze wall layout -- one frozenset pair per wall segment
    # in maze.world. Order within each pair doesn't matter
    # (frozenset), which is exactly what removes the row/col
    # ordering ambiguity that caused the earlier bug.
    # =====================================================

    BLOCKED_EDGES = {
        frozenset({(2, 4), (3, 4)}),  # wall_D1_D2  (X=3.0, row 4)
        frozenset({(3, 4), (4, 4)}),  # wall_E1_E2  (X=4.0, row 4)
        frozenset({(2, 3), (2, 4)}),  # wall_C2_D2  (Y=4.0, col 2)
        frozenset({(1, 2), (1, 3)}),  # wall_B3_D3  (Y=3.0, col 1)
        frozenset({(2, 2), (2, 3)}),  # wall_B3_D3  (Y=3.0, col 2)
        frozenset({(2, 2), (3, 2)}),  # wall_D3_D4  (X=3.0, row 2)
        frozenset({(3, 1), (4, 1)}),  # wall_E3_E5  (X=4.0, row 1)
        frozenset({(3, 2), (4, 2)}),  # wall_E3_E5  (X=4.0, row 2)
        frozenset({(2, 1), (2, 2)}),  # wall_C4_D4  (Y=2.0, col 2)
        frozenset({(0, 0), (0, 1)}),  # wall_A5_C5  (Y=1.0, col 0)
        frozenset({(1, 0), (1, 1)}),  # wall_A5_C5  (Y=1.0, col 1)
        frozenset({(3, 0), (3, 1)}),  # wall_D5_E5  (Y=1.0, col 3)
    }

    # =====================================================
    # Reward shaping constants -- kept as named constants so
    # they're easy to tune later without hunting through step().
    # =====================================================

    STEP_PENALTY = -1
    REVISIT_PENALTY = 0
    INVALID_MOVE_PENALTY = -20
    GOAL_REWARD = 100
    TIMEOUT_PENALTY = -50

    def __init__(self):

        # Instance-level copy of the class wall table -- kept as a
        # separate attribute (rather than reading the class constant
        # directly everywhere) so a future variant of this environment
        # could load a *different* maze at runtime by overwriting this
        # set after construction.
        self.blocked_edges = set(self.BLOCKED_EDGES)

        self.robot = (0, 0)
        self.goal = (4, 4)

        self.visited = set()

        self.max_steps = 100
        self.current_steps = 0

    ##########################################################
    # RESET
    ##########################################################

    def reset(self, start=None, goal=None):

        if start is None:
            start = (
                random.randint(0, self.GRID_SIZE - 1),
                random.randint(0, self.GRID_SIZE - 1)
            )

        if goal is None:
            goal = (
                random.randint(0, self.GRID_SIZE - 1),
                random.randint(0, self.GRID_SIZE - 1)
            )

        while goal == start:
            goal = (
                random.randint(0, self.GRID_SIZE - 1),
                random.randint(0, self.GRID_SIZE - 1)
            )

        self.robot = start
        self.goal = goal
        self.visited = {start}
        self.current_steps = 0

        return self.get_state()

    ##########################################################
    # STATE
    ##########################################################

    def get_state(self):

        return (
            self.robot[0],
            self.robot[1],
            self.goal[0],
            self.goal[1]
        )

    ##########################################################
    # CHECK WALL
    ##########################################################

    def is_valid_move(self, action):

        x, y = self.robot

        dx, dy = self.DELTA[action]

        nx = x + dx
        ny = y + dy

        # Outside maze
        if nx < 0 or nx >= self.GRID_SIZE:
            return False

        if ny < 0 or ny >= self.GRID_SIZE:
            return False

        # Internal walls -- a single, unambiguous lookup: is there a
        # wall on the edge between the current cell and the cell
        # we're trying to move into? frozenset means the pair order
        # never matters, which is exactly what removes the row/col
        # ordering bug that existed in the previous tuple-based
        # version of this check.
        edge = frozenset({(x, y), (nx, ny)})

        if edge in self.blocked_edges:
            return False

        return True

    ##########################################################
    # REWARD
    ##########################################################

    def compute_reward(self, action_valid, already_visited, reached_goal):
        """
        Isolated reward function -- step() only calls this, so reward
        shaping can be tuned here without touching movement logic.
        """

        if not action_valid:
            return self.INVALID_MOVE_PENALTY

        if reached_goal:
            return self.GOAL_REWARD

        reward = self.STEP_PENALTY

        if already_visited:
            reward += self.REVISIT_PENALTY

        return reward

    ##########################################################
    # STEP
    ##########################################################

    def step(self, action):

        self.current_steps += 1

        if not self.is_valid_move(action):
            reward = self.compute_reward(
                action_valid=False,
                already_visited=False,
                reached_goal=False
            )
            done = self.current_steps >= self.max_steps
            if done:
                reward = self.TIMEOUT_PENALTY

            return self.get_state(), reward, done

        x, y = self.robot
        dx, dy = self.DELTA[action]

        new_position = (x + dx, y + dy)

        already_visited = new_position in self.visited
        reached_goal = new_position == self.goal

        self.robot = new_position
        self.visited.add(new_position)

        reward = self.compute_reward(
            action_valid=True,
            already_visited=already_visited,
            reached_goal=reached_goal
        )

        done = reached_goal

        if not done and self.current_steps >= self.max_steps:
            reward = self.TIMEOUT_PENALTY
            done = True

        return self.get_state(), reward, done

    ##########################################################
    # ALL STATES (for Q-table initialization)
    ##########################################################

    def get_all_states(self):

        states = []

        for rx in range(self.GRID_SIZE):
            for ry in range(self.GRID_SIZE):
                for gx in range(self.GRID_SIZE):
                    for gy in range(self.GRID_SIZE):

                        if (rx, ry) != (gx, gy):
                            states.append(
                                (rx, ry, gx, gy)
                            )

        return states

    ##########################################################
    # VALID ACTIONS (so the agent only chooses among legal moves)
    ##########################################################

    def get_valid_actions(self):

        actions = []

        for action in self.ACTIONS:

            if self.is_valid_move(action):
                actions.append(action)

        return actions

    ##########################################################
    # RENDER
    ##########################################################

    def render(self):

        print("-------------------------")

        for y in reversed(range(self.GRID_SIZE)):

            row = ""

            for x in range(self.GRID_SIZE):

                if (x, y) == self.robot:
                    row += " R "

                elif (x, y) == self.goal:
                    row += " G "

                else:
                    row += " . "

            print(row)

        print("-------------------------")


if __name__ == "__main__":
    # Quick manual sanity check: confirm wall blocking actually works
    # against a few cells whose expected behaviour is known from
    # maze.world / localized_maze_controller.py's own validation.
    env = MazeEnvironment()
    env.reset(start=(0, 1), goal=(4, 4))
    env.render()
    print("Valid moves from (0,1):")
    for action, name in [
        (env.EAST, "EAST"), (env.NORTH, "NORTH"),
        (env.WEST, "WEST"), (env.SOUTH, "SOUTH")
    ]:
        print(f"  {name}: {env.is_valid_move(action)}")