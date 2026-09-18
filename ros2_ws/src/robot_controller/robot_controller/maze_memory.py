#!/usr/bin/env python3

"""
Exploration memory for UNKNOWN maze navigation.

WHY THIS EXISTS
---------------
The RL state is

    (dx, dy, heading, east, north, west, south)

which is maze-independent, and that is exactly the property that
lets one Q-table work in any layout. But it is also memoryless: it
cannot tell "I am standing in a cell I have already been in" from
"I am standing here for the first time". Two visits to the same
cell, arrived at from the same heading, are literally the same
state, so any policy keyed on it - Q-table or greedy fallback -
must choose the same action both times.

That guarantees a livelock in front of every dead end. On the maze
in maze.world, the greedy fallback loops forever between (1,2) and
(2,2), and between (1,4) and (2,4): 6 of the 24 possible start
cells never reach (4,4), they just burn all 100 steps.

Nothing in here is prior knowledge of the maze. It only records
what the LIDAR has already reported, which the robot is entitled
to remember. The layout is still discovered entirely at run time.

WHAT IT DOES
------------
Standard depth-first search with an explicit path stack:

  * prefer open directions leading to cells never visited, choosing
    among them with the Q-table action when the Q-table offered one
    and it is in that set, otherwise greedily toward the goal;
  * when every open direction leads somewhere already visited, pop
    the stack and step back to the parent cell.

DFS on a connected graph visits every reachable cell, so the goal
is always found. On maze.world this reaches (4,4) from all 24
start cells in at most 10 moves.
"""


class MazeMemory:

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    ACTION_PRIORITY = [EAST, NORTH, WEST, SOUTH]

    DELTAS = {
        EAST: (1, 0),
        NORTH: (0, 1),
        WEST: (-1, 0),
        SOUTH: (0, -1),
    }

    ACTION_NAMES = {
        EAST: "EAST",
        NORTH: "NORTH",
        WEST: "WEST",
        SOUTH: "SOUTH",
    }

    def __init__(self, grid_size=5):

        self.grid_size = grid_size

        # Cells the robot has physically occupied.
        self.visited = set()

        # Current DFS path. stack[-1] is the cell we are standing in;
        # stack[-2] is the cell we came from and will retreat to.
        self.stack = []

        # Everything LIDAR has reported, keyed by cell.
        # Kept for debugging and for print_map().
        self.discovered_walls = {}

        # Set when the last decision was a retreat, so the caller
        # can log it distinctly from a forward move.
        self.last_was_backtrack = False

    # ============================================================
    # NEIGHBOUR
    # ============================================================

    def neighbour(self, cell, action):

        dx, dy = self.DELTAS[action]

        return (cell[0] + dx, cell[1] + dy)

    # ============================================================
    # RECORD AN ARRIVAL
    #
    # Call this once per step, after the scan, before choose().
    # ============================================================

    def observe(self, cell, walls=None):

        if walls is not None:
            self.discovered_walls[cell] = dict(walls)

        if cell not in self.visited:

            self.visited.add(cell)

            self.stack.append(cell)

        elif self.stack and self.stack[-1] != cell:

            # We stepped back into an earlier cell. Unwind the stack
            # to it so the parent pointer stays correct.
            if cell in self.stack:

                while self.stack and self.stack[-1] != cell:
                    self.stack.pop()

            else:

                self.stack.append(cell)

    # ============================================================
    # CHOOSE AN ACTION
    #
    # cell           - current cell
    # goal           - goal cell
    # valid_actions  - directions LIDAR reports open RIGHT NOW
    # q_action       - what the Q-table wanted, or None if the state
    #                  was not in the table
    #
    # Returns (action, reason) so the caller can print why.
    # ============================================================

    def choose(self, cell, goal, valid_actions, q_action=None):

        self.last_was_backtrack = False

        if not valid_actions:
            return None, "no open direction"

        frontier = [
            action for action in valid_actions
            if self.neighbour(cell, action) not in self.visited
        ]

        # --------------------------------------------------------
        # FORWARD: there is somewhere new to go
        # --------------------------------------------------------

        if frontier:

            # Respect the Q-table whenever it points at unexplored
            # ground. The table is still driving; memory only rules
            # out moves that would retread old cells.
            if q_action is not None and q_action in frontier:
                return q_action, "q-table"

            goal_dx = goal[0] - cell[0]
            goal_dy = goal[1] - cell[1]

            preferred = set()

            if goal_dx > 0:
                preferred.add(self.EAST)
            elif goal_dx < 0:
                preferred.add(self.WEST)

            if goal_dy > 0:
                preferred.add(self.NORTH)
            elif goal_dy < 0:
                preferred.add(self.SOUTH)

            for action in self.ACTION_PRIORITY:
                if action in preferred and action in frontier:
                    return action, "frontier toward goal"

            for action in self.ACTION_PRIORITY:
                if action in frontier:
                    return action, "frontier"

        # --------------------------------------------------------
        # BACKTRACK: every open direction is already explored
        # --------------------------------------------------------

        if len(self.stack) >= 2 and self.stack[-1] == cell:

            parent = self.stack[-2]

            for action in self.ACTION_PRIORITY:

                if (
                    action in valid_actions
                    and
                    self.neighbour(cell, action) == parent
                ):

                    self.stack.pop()

                    self.last_was_backtrack = True

                    return action, f"backtrack to {parent}"

        # --------------------------------------------------------
        # Stack is unusable (first step, or the robot was displaced).
        # Fall through to plain priority order rather than stopping.
        # --------------------------------------------------------

        if q_action is not None and q_action in valid_actions:
            return q_action, "q-table (no frontier, no parent)"

        for action in self.ACTION_PRIORITY:
            if action in valid_actions:
                return action, "priority fallback"

        return None, "no open direction"

    # ============================================================
    # DEBUG
    # ============================================================

    def summary(self):

        return (
            f"visited={len(self.visited)} "
            f"stack_depth={len(self.stack)} "
            f"path={self.stack}"
        )

    def print_map(self):

        print()
        print("Discovered so far (V = visited, . = unseen):")

        for y in range(self.grid_size - 1, -1, -1):

            row = []

            for x in range(self.grid_size):
                row.append("V" if (x, y) in self.visited else ".")

            print("  " + " ".join(row))
