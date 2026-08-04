"""
Maze/Arena Bot Navigation
--------------------------
Correct interpretation of the hand-drawn mesh:
  - The DOTS form a 6x6 grid (columns A-F, rows 1-6).
  - The DRAWN LINES are WALLS (obstacles) - just like the walls of a physical
    arena/maze that a real LiDAR robot has to avoid.
  - The WHITE SPACE between the walls is the open floor the robot drives through.

This models it as a 5x5 grid of CELLS (the open squares between 4 neighboring
dots). Two neighboring cells are connected (bot can pass between them) UNLESS
a wall segment was drawn on the boundary between them.

Algorithm: BFS / A* over the free-space cell graph -> shortest collision-free
path -> animated bot moving smoothly through the open white space.

Run:
    pip install matplotlib
    python3 maze_bot_navigation.py
"""

import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. DOT GRID (from the hand-drawn mesh)
# ---------------------------------------------------------
COLS = ['A', 'B', 'C', 'D', 'E', 'F']   # 6 columns -> indices 0-5
ROWS = [1, 2, 3, 4, 5, 6]               # 6 rows    -> indices 0-5

def col_idx(node):
    return COLS.index(node[0])

def row_idx(node):
    return int(node[1:]) - 1

# Every WALL segment actually drawn in the mesh (dot-to-dot lines)
WALL_EDGES = [
    # Outer border (arena boundary walls)
    ("A1", "B1"), ("B1", "C1"), ("C1", "D1"), ("D1", "E1"), ("E1", "F1"),
    ("A1", "A2"), ("A2", "A3"), ("A3", "A4"), ("A4", "A5"), ("A5", "A6"),
    ("F1", "F2"), ("F2", "F3"), ("F3", "F4"), ("F4", "F5"), ("F5", "F6"),
    ("A6", "B6"), ("B6", "C6"), ("C6", "D6"), ("D6", "E6"), ("E6", "F6"),

    # Interior walls (obstacles inside the arena)
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
# 2. CONVERT WALL EDGES -> BLOCKED CELL-TO-CELL PASSAGES
# ---------------------------------------------------------
# Cells are indexed (c, r) for c in 0..4, r in 0..4
# cell(c, r) = the open square between columns c/c+1 and rows r/r+1
N_CELL_COLS = len(COLS) - 1   # 5
N_CELL_ROWS = len(ROWS) - 1   # 5

blocked_pairs = set()  # set of frozenset({(c1,r1), (c2,r2)}) that CANNOT pass through

for a, b in WALL_EDGES:
    ca, ra = col_idx(a), row_idx(a)
    cb, rb = col_idx(b), row_idx(b)

    if ra == rb:
        # Horizontal wall segment -> blocks vertical passage between the
        # cell above and the cell below, at cell-column = min(ca,cb)
        c = min(ca, cb)
        r_dot = ra
        cells = []
        if r_dot - 1 >= 0:
            cells.append((c, r_dot - 1))
        if r_dot <= N_CELL_ROWS - 1:
            cells.append((c, r_dot))
        if len(cells) == 2:
            blocked_pairs.add(frozenset(cells))

    else:
        # Vertical wall segment -> blocks horizontal passage between the
        # cell to the left and the cell to the right, at cell-row = min(ra,rb)
        r = min(ra, rb)
        c_dot = ca
        cells = []
        if c_dot - 1 >= 0:
            cells.append((c_dot - 1, r))
        if c_dot <= N_CELL_COLS - 1:
            cells.append((c_dot, r))
        if len(cells) == 2:
            blocked_pairs.add(frozenset(cells))

# ---------------------------------------------------------
# 3. BUILD FREE-SPACE GRAPH (cells the bot can actually drive through)
# ---------------------------------------------------------
def neighbors(cell):
    c, r = cell
    candidates = [(c+1, r), (c-1, r), (c, r+1), (c, r-1)]
    result = []
    for nc, nr in candidates:
        if 0 <= nc < N_CELL_COLS and 0 <= nr < N_CELL_ROWS:
            if frozenset({(c, r), (nc, nr)}) not in blocked_pairs:
                result.append((nc, nr))
    return result


# ---------------------------------------------------------
# 4. PATHFINDING (BFS + A*)
# ---------------------------------------------------------
def bfs_path(start, goal):
    visited = {start}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for nxt in neighbors(node):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return None


def astar_path(start, goal):
    def h(cell):
        return ((cell[0]-goal[0])**2 + (cell[1]-goal[1])**2) ** 0.5

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for nb in neighbors(current):
            tentative = g_score[current] + 1
            if tentative < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_set, (tentative + h(nb), nb))
    return None


# ---------------------------------------------------------
# 5. VISUALIZATION + ANIMATION
# ---------------------------------------------------------
def cell_center(cell):
    c, r = cell
    return (c + 0.5, -(r + 0.5))  # negative y so row1 is on top

def dot_pos(node):
    return (col_idx(node), -row_idx(node))

def animate_maze(path, title="Bot Navigating Free Space"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_xlim(-0.5, N_CELL_COLS + 0.5)
    ax.set_ylim(-(N_CELL_ROWS + 0.5), 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Shade the whole arena floor (white space) light gray
    ax.add_patch(patches.Rectangle((0, -N_CELL_ROWS), N_CELL_COLS, N_CELL_ROWS,
                                    facecolor='#f5f5f5', zorder=0))

    # Draw the walls (thick black lines) - exactly as in the mesh
    for a, b in WALL_EDGES:
        x1, y1 = dot_pos(a)
        x2, y2 = dot_pos(b)
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=4, zorder=3)

    # Draw dots for reference
    for c in COLS:
        for r in ROWS:
            x, y = dot_pos(f"{c}{r}")
            ax.plot(x, y, 'o', color='dimgray', markersize=4, zorder=2)

    # Draw the planned free-space path (blue line through cell centers)
    px = [cell_center(c)[0] for c in path]
    py = [cell_center(c)[1] for c in path]
    ax.plot(px, py, color='royalblue', linewidth=3, zorder=4, alpha=0.7)

    # Start / goal markers
    ax.plot(*cell_center(path[0]), 'go', markersize=16, zorder=5, label='Start')
    ax.plot(*cell_center(path[-1]), 'r*', markersize=20, zorder=5, label='Goal')
    ax.legend(loc='upper right')

    # Animated bot
    bot, = ax.plot([], [], 'ko', markersize=14, zorder=6)

    def init():
        bot.set_data([], [])
        return bot,

    def update(frame):
        x, y = cell_center(path[frame])
        bot.set_data([x], [y])
        return bot,

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), init_func=init,
        interval=500, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()
    return ani


# ---------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------
if __name__ == "__main__":
    # Pick a start and goal CELL (not dot!) - cells are (col_index, row_index), 0-4 each
    start_cell = (0, 0)   # top-left open square
    goal_cell = (4, 4)    # bottom-right open square

    path_bfs = bfs_path(start_cell, goal_cell)
    path_astar = astar_path(start_cell, goal_cell)

    print(f"BFS path  ({len(path_bfs)-1} moves):  {path_bfs}")
    print(f"A* path   ({len(path_astar)-1} moves): {path_astar}")

    animate_maze(path_astar, title=f"Bot: cell{start_cell} -> cell{goal_cell}")
