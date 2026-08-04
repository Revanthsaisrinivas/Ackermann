"""
Mesh Navigation Bot - Pathfinding + Simulation
------------------------------------------------
Models the hand-drawn mesh as a graph (6x6 grid of waypoints, columns A-F, rows 1-6).
Only the connections that were actually drawn in the mesh are valid bot paths.

Algorithm: BFS for shortest hop-count path, A* for shortest weighted distance.
Includes a matplotlib animation showing the bot moving along the computed path.

Run:
    pip install matplotlib networkx --break-system-packages
    python mesh_bot_navigation.py
"""

import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------
# 1. MESH DEFINITION (from the hand-drawn diagram)
# ---------------------------------------------------------
# Grid coordinates: columns A-F (x=0..5), rows 1-6 (y=0..5, row1 = top)
COLS = ['A', 'B', 'C', 'D', 'E', 'F']
ROWS = [1, 2, 3, 4, 5, 6]

def pos(node):
    """Return (x, y) plotting position for a node id like 'A1'."""
    col = COLS.index(node[0])
    row = int(node[1:])
    return (col, -row)  # negative y so row1 is on top when plotted

NODES = [f"{c}{r}" for r in ROWS for c in COLS]

# Edges = actual connections drawn in the mesh
EDGES = [
    # Outer border loop
    ("A1", "B1"), ("B1", "C1"), ("C1", "D1"), ("D1", "E1"), ("E1", "F1"),
    ("A1", "A2"), ("A2", "A3"), ("A3", "A4"), ("A4", "A5"), ("A5", "A6"),
    ("F1", "F2"), ("F2", "F3"), ("F3", "F4"), ("F4", "F5"), ("F5", "F6"),
    ("A6", "B6"), ("B6", "C6"), ("C6", "D6"), ("D6", "E6"), ("E6", "F6"),

    # Interior connections
    ("D1", "D2"), ("D2", "C2"),
    ("E1", "E2"),
    ("B3", "C3"), ("C3", "D3"),
    ("D3", "D4"),
    ("D4", "C4"),
    ("E3", "E4"), ("E4", "E5"),
    ("E5", "D5"),
    ("A5", "B5"), ("B5", "C5"),
]

# ---------------------------------------------------------
# 2. BUILD GRAPH (adjacency list)
# ---------------------------------------------------------
graph = {n: [] for n in NODES}
for a, b in EDGES:
    graph[a].append(b)
    graph[b].append(a)


# ---------------------------------------------------------
# 3. PATHFINDING ALGORITHMS
# ---------------------------------------------------------
def bfs_path(start, goal):
    """Shortest path by number of hops (unweighted)."""
    if start not in graph or goal not in graph:
        return None
    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return None  # no path (nodes not connected in the mesh)


def euclidean(a, b):
    ax, ay = pos(a)
    bx, by = pos(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def astar_path(start, goal):
    """Shortest path by real distance, using A* with Euclidean heuristic."""
    if start not in graph or goal not in graph:
        return None
    open_set = [(0, start)]
    came_from = {}
    g_score = {n: float('inf') for n in NODES}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in graph[current]:
            tentative = g_score[current] + euclidean(current, neighbor)
            if tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f = tentative + euclidean(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return None


# ---------------------------------------------------------
# 4. BOT NAVIGATION SIMULATION (animated)
# ---------------------------------------------------------
def animate_path(path, title="Bot Navigation"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.axis('off')

    # Draw all mesh edges (gray, background)
    for a, b in EDGES:
        ax1, ay1 = pos(a)
        ax2, ay2 = pos(b)
        ax.plot([ax1, ax2], [ay1, ay2], color='lightgray', linewidth=2, zorder=1)

    # Draw all mesh nodes
    for n in NODES:
        x, y = pos(n)
        ax.plot(x, y, 'o', color='lightgray', markersize=8, zorder=2)
        ax.text(x, y + 0.15, n, fontsize=7, ha='center', color='gray')

    # Highlight the planned path (blue)
    px = [pos(n)[0] for n in path]
    py = [pos(n)[1] for n in path]
    ax.plot(px, py, color='royalblue', linewidth=3, zorder=3)

    # Start and goal markers
    ax.plot(*pos(path[0]), 'go', markersize=14, zorder=4, label='Start')
    ax.plot(*pos(path[-1]), 'r*', markersize=18, zorder=4, label='Goal')
    ax.legend(loc='upper right')

    # Bot marker (animated)
    bot, = ax.plot([], [], 'ko', markersize=12, zorder=5)

    def init():
        bot.set_data([], [])
        return bot,

    def update(frame):
        x, y = pos(path[frame])
        bot.set_data([x], [y])
        return bot,

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), init_func=init,
        interval=600, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()
    return ani


# ---------------------------------------------------------
# 5. DEMO
# ---------------------------------------------------------
if __name__ == "__main__":
    start_node = "A1"
    goal_node = "F6"

    bfs_result = bfs_path(start_node, goal_node)
    astar_result = astar_path(start_node, goal_node)

    print(f"BFS path  ({len(bfs_result)-1} hops):  {' -> '.join(bfs_result)}")
    print(f"A* path   ({len(astar_result)-1} hops): {' -> '.join(astar_result)}")

    # Animate the A* (shortest real-distance) path
    animate_path(astar_result, title=f"Bot Path: {start_node} to {goal_node}")
