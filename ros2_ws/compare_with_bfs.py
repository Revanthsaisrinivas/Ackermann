#!/usr/bin/env python3

import random
import pickle
from collections import deque

from robot_controller.rl_environment import MazeEnvironment

# -----------------------------
# Load Q-table
# -----------------------------
with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]

env = MazeEnvironment()

TOTAL_TESTS = 100

success = 0
loops = 0
failures = 0

optimal = 0
near_optimal = 0

total_extra = 0


# ----------------------------------------------------
# BFS
# ----------------------------------------------------
def bfs(start, goal):

    queue = deque()

    queue.append((start, 0))

    visited = {start}

    while queue:

        position, dist = queue.popleft()

        if position == goal:
            return dist

        env.reset(start=position, goal=goal)

        for action in env.get_valid_actions():

            x, y = env.robot

            dx, dy = env.DELTA[action]

            nxt = (x + dx, y + dy)

            if nxt not in visited:

                visited.add(nxt)

                queue.append((nxt, dist + 1))

    return None


# ----------------------------------------------------
# RL
# ----------------------------------------------------
def rl(start, goal):

    env.reset(start=start, goal=goal)

    visited = set()

    steps = 0

    while True:

        state = env.get_state()

        if env.robot == goal:
            return True, steps

        if state in visited:
            return False, steps

        visited.add(state)

        valid = env.get_valid_actions()

        if len(valid) == 0:
            return False, steps

        q = q_table[state]

        action = max(valid, key=lambda a: q[a])

        env.step(action)

        steps += 1

        if steps > 100:
            return False, steps


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
for _ in range(TOTAL_TESTS):

    while True:

        start = (
            random.randint(0, 4),
            random.randint(0, 4)
        )

        goal = (
            random.randint(0, 4),
            random.randint(0, 4)
        )

        if start != goal:
            break

    bfs_steps = bfs(start, goal)

    ok, rl_steps = rl(start, goal)

    if not ok:

        failures += 1

        continue

    success += 1

    extra = rl_steps - bfs_steps

    total_extra += extra

    if extra == 0:

        optimal += 1

    else:

        near_optimal += 1

print("\n====================================")
print("BFS vs RL")
print("====================================")
print("Tests           :", TOTAL_TESTS)
print("Success         :", success)
print("Failures        :", failures)
print("Optimal         :", optimal)
print("Near Optimal    :", near_optimal)

if success:

    print("Average Extra Steps :", round(total_extra / success, 3))

print("====================================")