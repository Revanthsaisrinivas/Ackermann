import pickle
import csv

from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]

actions = ["EAST", "NORTH", "WEST", "SOUTH"]

with open("q_table.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Robot_X",
        "Robot_Y",
        "Goal_X",
        "Goal_Y",
        "East",
        "North",
        "West",
        "South",
        "Best_Action",
        "Best_Q"
    ])

    for state, values in q_table.items():

        rx, ry, gx, gy = state

        # Reset environment to this state
        env.reset(start=(rx, ry), goal=(gx, gy))

        # Get only VALID actions
        valid_actions = env.get_valid_actions()

        if valid_actions:
            best = max(valid_actions, key=lambda a: values[a])
            best_action = actions[best]
            best_q = values[best]
        else:
            best_action = "NONE"
            best_q = 0

        writer.writerow([
            rx,
            ry,
            gx,
            gy,
            round(values[0], 4),
            round(values[1], 4),
            round(values[2], 4),
            round(values[3], 4),
            best_action,
            round(best_q, 4)
        ])

print("Created q_table.csv successfully!")