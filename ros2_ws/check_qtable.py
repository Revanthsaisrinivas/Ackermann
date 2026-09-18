#!/usr/bin/env python3

import os
import pickle
import sys

# Reuse the package's single source of truth rather than hardcoding an
# install/ path, which was how this script ended up reporting a
# different table from the one run_policy.py actually loaded.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "robot_controller"),
)

from robot_controller.q_table_path import q_table_path  # noqa: E402

Q_TABLE_PATH = q_table_path()

EAST = 0
NORTH = 1
WEST = 2
SOUTH = 3

ACTION_NAMES = {
    EAST: "EAST",
    NORTH: "NORTH",
    WEST: "WEST",
    SOUTH: "SOUTH",
}


def load_q_table():

    with open(Q_TABLE_PATH, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "q_table" in data:
        return data["q_table"]

    return data


q_table = load_q_table()

print("\n" + "=" * 70)
print("Q-TABLE MANUAL CHECK")
print("=" * 70)
print("States:", len(q_table))
print()

states_to_check = [

    # (0,0) goal (4,4)
    (4, 4, 0, 0, 1, 1, 1),
    (4, 4, 1, 0, 1, 1, 1),
    (4, 4, 2, 0, 1, 1, 1),
    (4, 4, 3, 0, 1, 1, 1),

    # (1,0)
    (3, 4, 0, 0, 1, 1, 0),

    # (2,0)
    (2, 4, 0, 0, 0, 1, 0),

    # (2,1)
    (2, 3, 1, 1, 0, 1, 0),

    # (3,3)
    (1, 1, 0, 0, 0, 1, 1),

    # (3,4)
    (1, 0, 1, 0, 1, 1, 1),
]


for state in states_to_check:

    print("-" * 70)
    print("STATE:", state)

    if state not in q_table:
        print("STATUS: STATE NOT FOUND IN Q-TABLE")
        continue

    q_values = q_table[state]

    print()

    for action in [EAST, NORTH, WEST, SOUTH]:

        value = q_values[action]

        print(
            f"{ACTION_NAMES[action]:<6} "
            f"Q = {value:.6f}"
        )

    max_value = max(q_values)

    best_actions = [
        action
        for action in [EAST, NORTH, WEST, SOUTH]
        if q_values[action] == max_value
    ]

    print()

    print(
        "BEST:",
        " -> ".join(
            ACTION_NAMES[a]
            for a in best_actions
        )
    )

print("\n" + "=" * 70)
