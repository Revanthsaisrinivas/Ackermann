import pickle

with open(
    "install/robot_controller/lib/python3.10/site-packages/robot_controller/q_table.pkl",
    "rb"
) as f:
    data = pickle.load(f)

q = data["q_table"]

states = [
    (2,0,4,4),
    (3,0,4,4),
    (4,0,4,4),
    (4,1,4,4),
    (4,2,4,4),
    (4,3,4,4)
]

for s in states:
    print(s, q[s])
