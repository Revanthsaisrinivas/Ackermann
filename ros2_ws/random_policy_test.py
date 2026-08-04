#!/usr/bin/env python3

import random
import pickle

from robot_controller.rl_environment import MazeEnvironment

with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]

env = MazeEnvironment()

ACTION_NAMES = {
    env.EAST: "EAST",
    env.NORTH: "NORTH",
    env.WEST: "WEST",
    env.SOUTH: "SOUTH"
}

TOTAL_TESTS = 100

success = 0
loops = 0
failures = 0

total_steps = 0
max_steps = 0
min_steps = 999

for test in range(1, TOTAL_TESTS + 1):

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

    env.reset(start=start, goal=goal)

    visited = set()

    steps = 0

    while True:

        state = env.get_state()

        if env.robot == goal:
            success += 1
            total_steps += steps
            max_steps = max(max_steps, steps)
            min_steps = min(min_steps, steps)
            break

        if state in visited:
            loops += 1
            break

        visited.add(state)

        valid = env.get_valid_actions()

        if len(valid) == 0:
            failures += 1
            break

        q = q_table[state]

        action = max(valid, key=lambda a: q[a])

        env.step(action)

        steps += 1

        if steps > 100:
            failures += 1
            break

print("\n==============================")
print(" RANDOM POLICY TEST RESULTS")
print("==============================")
print("Total Tests :", TOTAL_TESTS)
print("Successes   :", success)
print("Failures    :", failures)
print("Loops       :", loops)

if success > 0:
    print("Average Steps :", round(total_steps / success, 2))
    print("Shortest Path :", min_steps)
    print("Longest Path  :", max_steps)

print("==============================")