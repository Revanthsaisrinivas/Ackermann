import pickle
import csv

# Load Q-table
with open("src/robot_controller/robot_controller/q_table.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the actual Q-table
if isinstance(data, dict) and "q_table" in data:
    q = data["q_table"]
else:
    q = data

actions = ["EAST", "NORTH", "WEST", "SOUTH"]

with open("q_table.csv", "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

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

    for state, values in q.items():

        # Skip metadata entries
        if not isinstance(state, tuple):
            continue

        rx, ry, gx, gy = state

        best_index = max(range(4), key=lambda i: values[i])

        writer.writerow([
            rx,
            ry,
            gx,
            gy,
            round(values[0], 4),
            round(values[1], 4),
            round(values[2], 4),
            round(values[3], 4),
            actions[best_index],
            round(values[best_index], 4)
        ])

print("CSV exported successfully!")
print("File saved as: q_table.csv")
