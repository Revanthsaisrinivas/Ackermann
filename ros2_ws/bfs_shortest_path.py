from collections import deque
from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

ACTION_NAMES = {
    env.EAST: "EAST",
    env.NORTH: "NORTH",
    env.WEST: "WEST",
    env.SOUTH: "SOUTH"
}

start = (2,0)
goal = (4,4)

queue = deque()
queue.append((start, []))

visited = set()
visited.add(start)

while queue:

    cell, path = queue.popleft()

    if cell == goal:
        print("Shortest path found:")
        print(" -> ".join(ACTION_NAMES[a] for a in path))
        print("Steps:", len(path))
        break

    env.reset(start=cell, goal=goal)

    for action in env.get_valid_actions():

        dx, dy = env.DELTA[action]
        nxt = (cell[0] + dx, cell[1] + dy)

        if nxt not in visited:
            visited.add(nxt)
            queue.append((nxt, path + [action]))
