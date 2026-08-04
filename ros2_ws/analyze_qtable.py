import pickle

with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]

total_states = len(q_table)
all_zero = 0
only_one_updated = 0
positive_actions = 0
negative_actions = 0
mixed_states = 0

print("=" * 70)
print("Q TABLE ANALYSIS")
print("=" * 70)

for state, values in q_table.items():

    # Ignore terminal states
    if state[0] == state[2] and state[1] == state[3]:
        continue

    positives = sum(v > 0 for v in values)
    negatives = sum(v < 0 for v in values)
    zeros = sum(v == 0 for v in values)

    if zeros == 4:
        all_zero += 1

    if positives == 1:
        only_one_updated += 1

    if positives > 1:
        positive_actions += 1

    if negatives > 0:
        negative_actions += 1

    if positives > 0 and negatives > 0:
        mixed_states += 1

print(f"Total states              : {total_states}")
print(f"All zero states           : {all_zero}")
print(f"Only one positive action  : {only_one_updated}")
print(f"Multiple positive actions : {positive_actions}")
print(f"States having negatives   : {negative_actions}")
print(f"Mixed (+ and -) states    : {mixed_states}")
