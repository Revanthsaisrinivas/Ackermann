import pickle

path = "src/robot_controller/robot_controller/q_table.pkl"

with open(path, "rb") as f:
    q_table = pickle.load(f)

print("Number of states:", len(q_table))
print()

for i, (state, q_values) in enumerate(q_table.items()):
    print(f"State: {state}")
    print(f"Q-values: {q_values}")
    print("-" * 40)

    if i == 9:   # Print first 10 states only
        break
