import pickle

gamma = 0.95

with open(
    "install/robot_controller/lib/python3.10/site-packages/robot_controller/q_table.pkl",
    "rb"
) as f:
    data = pickle.load(f)

q = data["q_table"]

current = q[(2,0,4,4)][0]      # EAST from (2,0)
future = max(q[(3,0,4,4)])     # Best action from (3,0)

expected = -1 + gamma * future

print("Current Q :", current)
print("Future max:", future)
print("Expected  :", expected)
print("Difference:", expected - current)
