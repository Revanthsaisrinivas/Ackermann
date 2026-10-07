#!/usr/bin/env python3

import random
from collections import deque

from . import maze_layout


class MazeEnvironment:

    GRID_SIZE = 5

    # Action priority:
    # EAST -> NORTH -> WEST -> SOUTH
    ACTIONS = [0, 1, 2, 3]

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    DIRECTIONS = {
        EAST: (1, 0),
        NORTH: (0, 1),
        WEST: (-1, 0),
        SOUTH: (0, -1),
    }

    ACTION_PRIORITY = [EAST, NORTH, WEST, SOUTH]

    INVALID_MOVE_PENALTY = -10
    STEP_PENALTY = -1
    GOAL_REWARD = 100
    TIMEOUT_PENALTY = -20

    # Raised from -2. A revisit penalty this close to the plain
    # STEP_PENALTY (-1) barely discourages backtracking - in states
    # that are rarely visited during domain-randomized training,
    # Q-values for a backward move and a forward move can end up
    # within noise distance of each other (this is what produced the
    # WEST=30.2 vs NORTH=29.7 near-tie at one state in testing, even
    # though the actually-selected action, EAST, was still correct).
    # A stronger gap gives cleaner separation.
    REVISIT_PENALTY = -5

    # Potential-based reward shaping: add a small bonus/penalty each
    # step proportional to how much closer/farther the move made the
    # robot from the goal (Manhattan distance). This does not change
    # what the optimal policy is (it's a zero-sum telescoping term),
    # it just gives the agent a much clearer, denser signal that
    # moving toward the goal is good and moving away is bad, on every
    # single step rather than only at terminal states - which speeds
    # up convergence and reduces exactly the kind of close Q-value
    # ties described above. Set to 0.0 to disable.
    DISTANCE_SHAPING_WEIGHT = 0.5

    def __init__(self, blocked_edges=None):

        # ----------------------------------------------------
        # HIDDEN MAZE
        #
        # The agent does NOT directly receive blocked_edges. It only
        # receives sensor wall observations, and the executor never
        # reads this set at all.
        #
        # The layout is parsed from worlds/maze.world, so the walls the
        # trainer learns are by construction the walls Gazebo spawns.
        # It used to be a hand-written copy kept here, which nothing
        # checked against the world file.
        #
        # Pass `blocked_edges` to train on some other fixed layout, or
        # call reset(randomize_maze=True) for the old domain-randomised
        # behaviour. Fixed-maze training is now the default: see
        # train.py for why.
        # ----------------------------------------------------

        if blocked_edges is not None:

            self.blocked_edges = set(blocked_edges)

            self.layout_signature = maze_layout.signature(
                self.blocked_edges
            )

            self.layout_source = "caller-supplied"

        else:

            self.blocked_edges = maze_layout.blocked_edges()

            self.layout_signature = maze_layout.signature(
                self.blocked_edges
            )

            self.layout_source = maze_layout.world_path()

        self.robot = (0, 0)
        self.goal = (4, 4)

        self.heading = self.EAST

        self.discovered_walls = {}

        self.visited = set()

        self.max_steps = 100
        self.steps = 0

        # The revisit penalty reads self.visited, which is episode
        # history and is NOT part of the state. That makes the reward
        # non-Markov: the same state can pay -5 or -1 depending on how
        # the robot got there, and Q-learning then averages over paths
        # instead of converging. It is harmless under domain
        # randomisation, where nothing converges tightly anyway, but it
        # is exactly what puts near-ties into a fixed-maze table.
        # train.py turns it off; nothing else does.
        self.use_revisit_penalty = True

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        start=(0, 0),
        goal=(4, 4),
        heading=EAST,
        randomize_maze=False,
        block_probability=0.35,
    ):

        if randomize_maze:
            self.blocked_edges = self.generate_random_maze(
                block_probability=block_probability
            )

        self.robot = start
        self.goal = goal
        self.heading = heading

        self.discovered_walls = {}

        self.visited = {self.robot}

        self.steps = 0

        # Initial sensor scan
        self.scan_current_cell()

        return self.get_state()

    # ============================================================
    # RANDOM MAZE GENERATION (DOMAIN RANDOMIZATION)
    #
    # Generates a random set of blocked edges such that the
    # resulting maze is fully connected (every cell reachable
    # from every other cell), so any start/goal pair is solvable.
    #
    # Training against a new random maze every episode is what
    # forces the agent to learn a policy that reacts to LIDAR
    # wall readings + relative goal direction, instead of
    # memorizing one specific hidden layout.
    # ============================================================

    def all_edges(self):

        edges = set()

        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):

                if x + 1 < self.GRID_SIZE:
                    edges.add(frozenset({(x, y), (x + 1, y)}))

                if y + 1 < self.GRID_SIZE:
                    edges.add(frozenset({(x, y), (x, y + 1)}))

        return edges

    def is_fully_connected(self, blocked_edges):

        open_edges = self.all_edges() - blocked_edges

        adjacency = {}

        for edge in open_edges:

            a, b = tuple(edge)

            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        start = (0, 0)

        visited = {start}

        queue = deque([start])

        while queue:

            cell = queue.popleft()

            for neighbor in adjacency.get(cell, []):

                if neighbor not in visited:

                    visited.add(neighbor)

                    queue.append(neighbor)

        total_cells = self.GRID_SIZE * self.GRID_SIZE

        return len(visited) == total_cells

    def generate_random_maze(
        self,
        block_probability=0.35,
        max_attempts=500,
    ):

        edges = list(self.all_edges())

        for _ in range(max_attempts):

            candidate = {
                edge for edge in edges
                if random.random() < block_probability
            }

            if self.is_fully_connected(candidate):

                return candidate

        # Fallback: fully open maze (always solvable) if we
        # somehow failed to sample a connected layout.
        return set()

    # ============================================================
    # GRID CHECK
    # ============================================================

    # ============================================================
    # MANHATTAN DISTANCE
    #
    # Used only for reward shaping in step(). Not part of the
    # agent-visible state.
    # ============================================================

    def manhattan_distance(self, a, b):

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_inside_grid(self, position):

        x, y = position

        return (
            0 <= x < self.GRID_SIZE
            and
            0 <= y < self.GRID_SIZE
        )

    # ============================================================
    # HIDDEN TRUE WALL
    #
    # Used internally by the simulator.
    # Agent never receives blocked_edges directly.
    # ============================================================

    def is_true_wall(
        self,
        position,
        action
    ):

        dx, dy = self.DIRECTIONS[action]

        next_position = (
            position[0] + dx,
            position[1] + dy,
        )

        # Grid boundary
        if not self.is_inside_grid(next_position):
            return True

        edge = frozenset({
            position,
            next_position
        })

        return edge in self.blocked_edges

    # ============================================================
    # SENSOR SCAN
    # ============================================================

    def scan_current_cell(self):

        walls = {}

        for action in self.ACTIONS:

            walls[action] = self.is_true_wall(
                self.robot,
                action
            )

        self.discovered_walls[
            self.robot
        ] = walls

        return walls

    # ============================================================
    # CURRENT SENSOR WALLS
    # ============================================================

    def get_current_walls(self):

        if self.robot not in self.discovered_walls:

            self.scan_current_cell()

        walls = self.discovered_walls[
            self.robot
        ].copy()

        x, y = self.robot

        if x <= 0:
            walls[self.WEST] = True
        if x >= self.GRID_SIZE - 1:
            walls[self.EAST] = True
        if y <= 0:
            walls[self.SOUTH] = True
        if y >= self.GRID_SIZE - 1:
            walls[self.NORTH] = True

        return walls

    def is_valid_grid_move(self, position, action):

        x, y = position
        dx, dy = self.DIRECTIONS[action]
        new_x = x + dx
        new_y = y + dy

        if not (0 <= new_x < self.GRID_SIZE):
            return False

        if not (0 <= new_y < self.GRID_SIZE):
            return False

        return True

    # ============================================================
    # VALID ACTIONS
    #
    # Returned in priority order:
    # EAST -> NORTH -> WEST -> SOUTH
    # ============================================================

    def get_valid_actions(self):

        walls = self.get_current_walls()

        valid_actions = []

        for action in self.ACTION_PRIORITY:

            if walls[action]:
                continue

            if not self.is_valid_grid_move(self.robot, action):
                continue

            valid_actions.append(action)

        return valid_actions


    def get_state(self):

        robot_x, robot_y = self.robot

        goal_x, goal_y = self.goal

        return (

            robot_x,
            robot_y,

            goal_x,
            goal_y,

            self.heading,
        )

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action):

        self.steps += 1

        # Robot heading becomes chosen movement direction
        self.heading = action

        old_position = self.robot

        # Invalid move
        if self.is_true_wall(
            self.robot,
            action
        ):

            reward = self.INVALID_MOVE_PENALTY

            done = False

        else:

            dx, dy = self.DIRECTIONS[action]

            next_position = (

                self.robot[0] + dx,
                self.robot[1] + dy,
            )

            self.robot = next_position

            # Sensor discovers new cell
            self.scan_current_cell()

            # Goal
            if self.robot == self.goal:

                reward = self.GOAL_REWARD

                done = True

            # Revisiting
            elif (
                self.use_revisit_penalty
                and
                self.robot in self.visited
            ):

                reward = self.REVISIT_PENALTY

                done = False

            # New cell
            else:

                self.visited.add(
                    self.robot
                )

                reward = self.STEP_PENALTY

                done = False

            # --------------------------------------------------
            # DISTANCE SHAPING
            #
            # Only applied on real (non-invalid-move) steps that
            # aren't already terminal, and added on top of
            # whichever base reward above already applies. A move
            # that gets 1 cell closer to the goal earns
            # +DISTANCE_SHAPING_WEIGHT; a move that gets 1 cell
            # farther (i.e. most backtracking) costs
            # -DISTANCE_SHAPING_WEIGHT, every single step - not
            # just at the end of an episode. That is what gives
            # forward-vs-backward actions a consistent, dense gap
            # to learn from instead of only separating out via the
            # sparse goal/revisit rewards.
            # --------------------------------------------------

            if self.DISTANCE_SHAPING_WEIGHT and not done:

                old_distance = self.manhattan_distance(
                    old_position,
                    self.goal
                )

                new_distance = self.manhattan_distance(
                    self.robot,
                    self.goal
                )

                reward += self.DISTANCE_SHAPING_WEIGHT * (
                    old_distance - new_distance
                )

        # Timeout
        if (
            not done
            and
            self.steps >= self.max_steps
        ):

            reward = self.TIMEOUT_PENALTY

            done = True

        return (

            self.get_state(),

            reward,

            done,
        )

    # ============================================================
    # DEBUG
    # ============================================================

    def print_discovered_walls(self):

        print()

        print("Discovered Walls:")

        for position in sorted(
            self.discovered_walls.keys()
        ):

            print(

                position,

                self.discovered_walls[
                    position
                ]
            )

    def print_current_information(self):

        print()

        print(
            "Robot Position:",
            self.robot
        )

        print(
            "Goal Position:",
            self.goal
        )

        print(
            "Heading:",
            self.heading
        )

        print(
            "Current Walls:",
            self.get_current_walls()
        )

        print(
            "Valid Actions:",
            self.get_valid_actions()
        )

        print(
            "State (robot_x, robot_y, goal_x, goal_y, heading):",
            self.get_state()
        )