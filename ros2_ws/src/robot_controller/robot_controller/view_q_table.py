#!/usr/bin/env python3

import pickle
import csv
import os

from .q_table_path import q_table_path


# Same single path train.py and run_policy.py use, so this can never
# show you a different table from the one the robot is driving on.
Q_TABLE_FILE = q_table_path()

OUTPUT_FILE = os.path.join(
    os.path.dirname(Q_TABLE_FILE),
    "q_table.csv"
)

ACTION_NAMES = [
    "EAST",
    "NORTH",
    "WEST",
    "SOUTH",
]


def extract_state_columns(state):
    """Return CSV columns and values for the current and legacy formats.

    Current is the absolute 5-tuple; the 7- and 9-element branches below
    are older encodings kept only so an archived table still opens.
    """
    if len(state) == 5:
        (
            robot_x,
            robot_y,
            goal_x,
            goal_y,
            heading,
        ) = state
        return [
            "Robot X",
            "Robot Y",
            "Goal X",
            "Goal Y",
            "Heading",
        ], [
            robot_x,
            robot_y,
            goal_x,
            goal_y,
            heading,
        ]

    if len(state) == 7:
        (
            goal_dx,
            goal_dy,
            heading,
            front_wall,
            left_wall,
            right_wall,
            back_wall,
        ) = state
        return [
            "Goal DX",
            "Goal DY",
            "Heading",
            "Front Wall",
            "Left Wall",
            "Right Wall",
            "Back Wall",
        ], [
            goal_dx,
            goal_dy,
            heading,
            front_wall,
            left_wall,
            right_wall,
            back_wall,
        ]

    if len(state) >= 9:
        (
            robot_x,
            robot_y,
            goal_x,
            goal_y,
            heading,
            east_wall,
            north_wall,
            west_wall,
            south_wall,
        ) = state[:9]
        return [
            "Robot X",
            "Robot Y",
            "Goal X",
            "Goal Y",
            "Heading",
            "East Wall",
            "North Wall",
            "West Wall",
            "South Wall",
        ], [
            robot_x,
            robot_y,
            goal_x,
            goal_y,
            heading,
            east_wall,
            north_wall,
            west_wall,
            south_wall,
        ]

    raise ValueError(
        f"Unsupported state format in Q-table: {state!r} "
        f"(len={len(state)})"
    )


def main():

    print("Loading Q-table...")

    with open(Q_TABLE_FILE, "rb") as f:
        data = pickle.load(f)

    # Handle new format
    if isinstance(data, dict) and "q_table" in data:
        q_table = data["q_table"]
    else:
        q_table = data

    if not q_table:
        raise ValueError("Q-table is empty or missing.")

    print("Total states:", len(q_table))

    print("Creating CSV...")

    first_state = next(iter(q_table.keys()))
    state_columns, _ = extract_state_columns(first_state)
    header = state_columns + [
        "Q EAST",
        "Q NORTH",
        "Q WEST",
        "Q SOUTH",
        "Best Action",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(header)

        # Sort states for easier viewing
        for state in sorted(q_table.keys()):

            q_values = q_table[state]
            _, state_values = extract_state_columns(state)

            # Find best action using priority
            # EAST -> NORTH -> WEST -> SOUTH
            best_action = 0
            best_value = q_values[0]

            for action in [0, 1, 2, 3]:
                if q_values[action] > best_value:
                    best_value = q_values[action]
                    best_action = action

            writer.writerow([
                *state_values,
                q_values[0],
                q_values[1],
                q_values[2],
                q_values[3],
                ACTION_NAMES[best_action],
            ])

    print()
    print("Q-table converted successfully!")
    print("CSV file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()