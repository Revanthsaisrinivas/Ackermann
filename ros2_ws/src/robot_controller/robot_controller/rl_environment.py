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

TWO SEPARATE SOURCES OF TRUTH
------------------------------
This file deliberately keeps two things that must never be mixed:

  * self.blocked_edges (ground truth) -- the real walls, used ONLY by
    is_valid_move() to decide what actually happens when the robot
    tries to move, and by scan_current_cell() to simulate what the
    robot's sensors would detect. Nothing the RL agent reads directly
    touches this.

  * self.discovered_walls (agent's knowledge) -- what the robot has
    actually sensed so far. It starts empty every episode and is
    written to in exactly one place: scan_current_cell(). This is the
    only thing get_state() / get_valid_actions() are allowed to read.

step() is the boundary between the two: it asks "is this action even
choosable given what's been discovered?" (informational, via
get_valid_actions()) and, separately, "did the robot actually hit a
wall?" (decided purely from ground truth). The agent is never handed
wall information about a cell it hasn't yet stood in and scanned.
"""

import random
from collections import defaultdict


class MazeEnvironment:

    # =====================================================
    # Directions (absolute, arranged counter-clockwise:
    # EAST -> NORTH -> WEST -> SOUTH -> EAST)
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

        # Robot's discovered map -- the ONLY thing the RL-facing API
        # is allowed to read from. Populated exclusively by
        # scan_current_cell(). Must start empty every episode.
        self.discovered_walls = {}

        self.robot = (0, 0)
        self.goal = (4, 4)
        self.heading = self.EAST

        self.visited = set()

        self.max_steps = 100
        self.current_steps = 0

    ##########################################################
    # RESET
    ##########################################################

    def reset(self, start=None, goal=None, heading=None):
        self.discovered_walls = {}

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
        self.heading = heading if heading is not None else self.EAST
        self.visited = {start}
        self.current_steps = 0

        # Robot "powers on" in place and its sensors read the walls
        # of the cell it's standing in -- same mechanism as any other
        # scan, just triggered at episode start instead of mid-step.
        self.scan_current_cell()

        return self.get_state()

    ##########################################################
    # STATE
    ##########################################################

    def _relative_action(self, offset):
        """Absolute direction `offset` quarter-turns CCW from heading."""
        return (self.heading + offset) % len(self.ACTIONS)

    def _front_action(self):
        return self._relative_action(0)

    def _left_action(self):
        return self._relative_action(1)

    def _right_action(self):
        return self._relative_action(-1)

    def _back_action(self):
        return self._relative_action(2)

    def get_state(self):
        """
        State is (robot_x, robot_y, goal_x, goal_y, heading,
        front_wall, left_wall, right_wall, back_wall).

        The four wall bits come ONLY from discovered_walls (never
        from blocked_edges directly) and are relative to the robot's
        current heading, matching the directional controller's
        front/left/right/back sensor layout. If the current cell
        hasn't been scanned yet (shouldn't normally happen, since
        reset()/step() always scan on arrival) unexplored directions
        default to 0 (open), consistent with
        get_discovered_valid_actions()'s "unexplored == available"
        convention.
        """

        walls = self.discovered_walls.get(self.robot, {})

        front_wall = int(walls.get(self._front_action(), False))
        left_wall = int(walls.get(self._left_action(), False))
        right_wall = int(walls.get(self._right_action(), False))
        back_wall = int(walls.get(self._back_action(), False))

        return (
            self.robot[0],
            self.robot[1],
            self.goal[0],
            self.goal[1],
            self.heading,
            front_wall,
            left_wall,
            right_wall,
            back_wall
        )

    ##########################################################
    # CHECK WALL (ground truth -- never read by the RL agent
    # directly; only used internally by step()/scan_current_cell())
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

    def scan_current_cell(self):
        """
        Simulate the robot's laser scanner.

        Returns the wall configuration of the current cell and stores
        it in the discovered map. This is the ONLY place
        discovered_walls is ever written -- every bit of wall
        knowledge the RL side ever gets has to come through here.
        Idempotent: a cell already in discovered_walls is returned
        from cache rather than re-derived.
        """

        if self.robot in self.discovered_walls:
            return self.discovered_walls[self.robot]

        walls = {}

        for action in self.ACTIONS:
            walls[action] = not self.is_valid_move(action)

        self.discovered_walls[self.robot] = walls

        return walls

    def get_discovered_valid_actions(self):
        """
        Return actions using ONLY the robot's discovered map.

        Unknown directions are treated as unexplored and therefore
        allowed. Once a cell has been scanned, its wall information
        is remembered forever.
        """

        actions = []

        x, y = self.robot

        # If this cell has never been scanned,
        # every in-bounds direction is considered possible.
        if self.robot not in self.discovered_walls:

            for action in self.ACTIONS:

                dx, dy = self.DELTA[action]

                nx = x + dx
                ny = y + dy

                if (
                    0 <= nx < self.GRID_SIZE and
                    0 <= ny < self.GRID_SIZE
                ):
                    actions.append(action)

            return actions

        walls = self.discovered_walls[self.robot]

        for action in self.ACTIONS:

            if walls[action]:
                continue

            dx, dy = self.DELTA[action]

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < self.GRID_SIZE and
                0 <= ny < self.GRID_SIZE
            ):
                actions.append(action)

        return actions

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

        # Check the actual movement against the hidden ground-truth maze.
        # The RL agent does not see blocked_edges directly.
        if not self.is_valid_move(action):
            reward = self.compute_reward(
                action_valid=False,
                already_visited=False,
                reached_goal=False
            )

            done = self.current_steps >= self.max_steps

            if done:
                reward = self.TIMEOUT_PENALTY

            # Stay in the same cell and keep the previous heading.
            return self.get_state(), reward, done

        # Movement is valid, so now update the robot's heading.
        self.heading = action

        x, y = self.robot
        dx, dy = self.DELTA[action]

        new_position = (x + dx, y + dy)

        already_visited = new_position in self.visited
        reached_goal = new_position == self.goal

        self.robot = new_position
        self.visited.add(new_position)

        # Scan only after entering the new cell.
        self.scan_current_cell()

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
    # ALL STATES (Q-table initialization)
    ##########################################################

    def get_all_states(self):
        """
        DEPRECATED for Q-table pre-population.

        With the old (robot, goal) state, every state was known in
        advance, so a dense table could be built up front. With
        heading + discovered front/left/right/back walls in the
        state, the state space includes configurations that are
        either unreachable or simply never visited in a given run
        (e.g. "back wall known blocked" states can't occur at cells
        adjacent to the boundary the robot entered from), so
        pre-populating a dense table wastes a lot of memory for
        entries that will never be looked up.

        Prefer new_q_table() below: a defaultdict that creates
        zeroed Q-values lazily, the first time a state is actually
        seen. This method is kept only for callers that still expect
        a full state list; wall bits are enumerated as all-zero
        (i.e. "nothing discovered yet") placeholders and don't
        reflect any particular episode's discoveries.
        """

        states = []

        for rx in range(self.GRID_SIZE):
            for ry in range(self.GRID_SIZE):
                for gx in range(self.GRID_SIZE):
                    for gy in range(self.GRID_SIZE):

                        if (rx, ry) == (gx, gy):
                            continue

                        for heading in self.ACTIONS:
                            states.append(
                                (rx, ry, gx, gy, heading, 0, 0, 0, 0)
                            )

        return states

    @classmethod
    def default_q_values(cls):
        """Zeroed Q-value row, one entry per action."""
        return [0.0 for _ in cls.ACTIONS]

    @classmethod
    def new_q_table(cls):
        """
        A Q-table that initializes rows lazily as states are actually
        encountered during training, instead of enumerating the full
        (much larger, mostly-unused) discovered-wall state space up
        front. Usage: q_table[state][action].
        """
        return defaultdict(cls.default_q_values)

    ##########################################################
    # VALID ACTIONS (so the agent only chooses among legal moves)
    ##########################################################

    def get_valid_actions(self):
        """
        Actions available to the agent right now.

        This is deliberately delegated to
        get_discovered_valid_actions() rather than checking
        is_valid_move() against the full ground-truth wall map: the
        agent should only ever be offered choices consistent with
        what it has actually sensed, not the complete maze layout.
        """
        return self.get_discovered_valid_actions()

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
    # Quick manual sanity check: confirm wall blocking, heading
    # tracking, and the discovered-vs-ground-truth split all behave
    # as expected.
    HEADING_NAMES = {
        MazeEnvironment.EAST: "EAST",
        MazeEnvironment.NORTH: "NORTH",
        MazeEnvironment.WEST: "WEST",
        MazeEnvironment.SOUTH: "SOUTH",
    }

    env = MazeEnvironment()
    state = env.reset(start=(0, 1), goal=(4, 4))
    env.render()
    print(f"Initial state: {state}")
    print(f"Heading: {HEADING_NAMES[env.heading]}")
    print(f"Discovered-valid actions from (0,1): "
          f"{[HEADING_NAMES[a] for a in env.get_valid_actions()]}")

    print()
    print("Attempting to move NORTH (ground truth will decide outcome):")
    state, reward, done = env.step(env.NORTH)
    print(f"  new state: {state}, reward: {reward}, done: {done}")
    print(f"  heading now: {HEADING_NAMES[env.heading]}")