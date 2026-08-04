from collections import deque
from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

ACTION_NAMES = {
    env.EAST: "EAST",
    env.NORTH: "NORTH",
    env.WEST: "WEST",
    env.SOUTH: "SOUTH"
}


def shortest_actions(start, goal):

    queue = deque()
    queue.append((start, []))

    visited = set()
    visited.add(start)

    shortest = None
    best_paths = []

    while queue:

        cell, path = queue.popleft()

        if shortest is not None and len(path) > shortest:
            continue

        if cell == goal:
            shortest = len(path)
            best_paths.append(path)
            continue

        env.reset(start=cell, goal=goal)

        for action in env.get_valid_actions():

            dx, dy = env.DELTA[action]
            nxt = (cell[0] + dx, cell[1] + dy)

            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [action]))

    return best_paths


start = (2,0)
goal = (4,4)

paths = shortest_actions(start, goal)

print("="*60)
print("Shortest paths")
print("="*60)

for i, path in enumerate(paths,1):
    print(f"\nPath {i}")

    print(" -> ".join(ACTION_NAMES[a] for a in path))