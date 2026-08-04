import pickle
import os
from ament_index_python.packages import get_package_share_directory
from robot_controller.rl_environment import MazeEnvironment

# Load Q-table
Q_TABLE = os.path.join(
    get_package_share_directory("robot_controller"),
    "q_table.pkl"
)

with open(Q_TABLE, "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]
from robot_controller.rl_environment import MazeEnvironment

# Load Q-table
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

print("="*70)
print("POLICY VALIDATION")
print("="*70)

start = tuple(map(int, input("Start (x y): ").split()))
goal = tuple(map(int, input("Goal (x y): ").split()))

env.reset(start=start, goal=goal)

visited = set()

step = 0

while True:

    state = env.get_state()

    print(f"\nStep {step}")
    print("State:", state)

    if env.robot == goal:
        print("\nGOAL REACHED!")
        break

    if state in visited:
        print("\nLOOP DETECTED!")
        break

    visited.add(state)

    valid = env.get_valid_actions()

    q = q_table[state]

    print("Q-values:")

    for a in valid:
        print(f"{ACTION_NAMES[a]:6} : {q[a]:.4f}")

    best = max(valid, key=lambda a: q[a])

    print("Chosen:", ACTION_NAMES[best])

    env.step(best)

    step += 1

    if step > 50:
        print("Too many steps.")
        break
