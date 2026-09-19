#!/usr/bin/env python3

"""
Benchmark the SAC policy against the grid Q-learning + DFS baseline.

    .venv-drl/bin/python -m robot_controller.sac_evaluate --mazes 100

BE HONEST ABOUT WHAT THIS SHOWS
-------------------------------
The DFS baseline solves every static maze with a guarantee - depth-first
search on a connected graph visits every reachable cell, so it cannot
fail to find the goal. SAC will not beat that on static success rate and
does not need to. Its case rests on two things the baseline cannot do:

  * react to obstacles that move within an episode
  * emit continuous (speed, steering) an Ackermann chassis can execute

If the numbers do not support that, this script should say so, and the
report should say so. A comparison that only lists the metrics where the
new method wins is not a comparison.

PATH LENGTH / OPTIMAL
---------------------
The reference is the BFS shortest cell path times the cell size - a
lower bound, since a continuous vehicle with a 0.369 m minimum turning
radius cannot drive the cell-centre polyline exactly. A ratio slightly
above 1.0 is therefore expected and correct, not a defect.
"""

import argparse
import math
import os
import time
from collections import deque

import numpy as np

from . import sac_model_path
from .maze_memory import MazeMemory
from .robot_constants import DEPLOY_CELL_SIZE, DEPLOY_GRID_SIZE
from .sac_agent import SAC
from .sac_env import SacMazeEnv, GOAL_RADIUS


# ============================================================
# BASELINE: grid DFS over discovered cells
#
# Imported from the existing maze_memory.py, unmodified. This is the
# system SAC has to justify replacing.
# ============================================================

def baseline_run(blocked, grid_size, start, goal, max_steps=200):
    """Cell-level DFS. Returns (reached, cells_travelled)."""

    mem = MazeMemory(grid_size)

    cell = start
    steps = 0

    for steps in range(1, max_steps + 1):

        if cell == goal:
            return True, steps - 1

        walls = {}
        for a, (dx, dy) in ((0, (1, 0)), (1, (0, 1)),
                            (2, (-1, 0)), (3, (0, -1))):
            nb = (cell[0] + dx, cell[1] + dy)
            inside = 0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size
            walls[a] = (not inside) or frozenset({cell, nb}) in blocked

        valid = [a for a in (0, 1, 2, 3) if not walls[a]]

        mem.observe(cell, walls)
        action, _ = mem.choose(cell, goal, valid, None)

        if action is None:
            return False, steps

        d = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[action]
        cell = (cell[0] + d[0], cell[1] + d[1])

    return cell == goal, steps


def bfs_path_cells(blocked, grid_size, start, goal):

    dist = {start: 0}
    q = deque([start])

    while q:
        c = q.popleft()
        if c == goal:
            return dist[c]
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nb = (c[0] + dx, c[1] + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue
            if nb in dist or frozenset({c, nb}) in blocked:
                continue
            dist[nb] = dist[c] + 1
            q.append(nb)

    return None


# ============================================================
# SAC ROLLOUT
# ============================================================

def sac_run(agent, env, max_steps=1200):
    """One deterministic episode. Returns a result dict."""

    obs, info = env.reset()

    travelled = 0.0
    px, py = env.x, env.y

    for step in range(1, max_steps + 1):

        action = agent.act(obs, deterministic=True)

        obs, _, term, trunc, step_info = env.step(action)

        travelled += math.hypot(env.x - px, env.y - py)
        px, py = env.x, env.y

        if step_info["reached"]:
            return dict(outcome="reached", steps=step, travelled=travelled,
                        optimal=info["goal_path_cells"] * env.cell_size,
                        seconds=step / 10.0)

        if step_info["collided"]:
            return dict(outcome="collided", steps=step, travelled=travelled,
                        optimal=info["goal_path_cells"] * env.cell_size,
                        seconds=step / 10.0)

        if trunc:
            break

    return dict(outcome="timeout", steps=max_steps, travelled=travelled,
                optimal=info["goal_path_cells"] * env.cell_size,
                seconds=max_steps / 10.0)


def summarise(results):

    n = len(results)

    reached = [r for r in results if r["outcome"] == "reached"]

    ratios = [
        r["travelled"] / r["optimal"]
        for r in reached if r["optimal"] > 1e-6
    ]

    return dict(
        success=len(reached) / n if n else 0.0,
        collision=sum(r["outcome"] == "collided" for r in results) / n,
        timeout=sum(r["outcome"] == "timeout" for r in results) / n,
        ratio=float(np.mean(ratios)) if ratios else float("nan"),
        seconds=float(np.mean([r["seconds"] for r in reached]))
        if reached else float("nan"),
    )


def row(label, s):

    def f(x, pct=False):
        if x != x:
            return "     -"
        return f"{x:6.1%}" if pct else f"{x:6.2f}"

    return (f"  {label:<26} {f(s['success'], True)} {f(s['collision'], True)} "
            f"{f(s['timeout'], True)}  {f(s['ratio'])}  {f(s['seconds'])}")


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--name", default="policy")
    p.add_argument("--mazes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    ckpt = sac_model_path.checkpoint_path(args.name)

    if not os.path.exists(ckpt + ".pt"):
        raise SystemExit(f"no checkpoint at {ckpt}.pt")

    agent = SAC.load(ckpt)

    print(f"policy: {sac_model_path.describe(args.name)}")
    print(f"held-out mazes per suite: {args.mazes}\n")

    header = (f"  {'':<26} {'success':>6} {'collis':>6} {'timeout':>6}  "
              f"{'path/opt':>6}  {'sec':>6}")

    # ---- held-out random mazes -------------------------
    suites = (
        ("static, no obstacles", dict(n_obstacles=(0, 0),
                                      obstacle_speed=(0.0, 0.0))),
        ("static clutter", dict(n_obstacles=(1, 4),
                                obstacle_speed=(0.0, 0.0))),
        ("moving obstacles", dict(n_obstacles=(1, 4),
                                  obstacle_speed=(0.05, 0.30))),
    )

    print("HELD-OUT RANDOM MAZES (never trained on these seeds)")
    print(header)

    for name, kw in suites:

        env = SacMazeEnv(seed=args.seed, **kw)

        results = [sac_run(agent, env) for _ in range(args.mazes)]

        print(row(f"SAC  {name}", summarise(results)))

    # ---- baseline on the same static distribution ------
    print()
    print("  Baseline (grid Q-learning + maze_memory DFS) is cell-level and")
    print("  static-only: it re-plans per cell and cannot react to an")
    print("  obstacle that moves mid-move, so it is scored on static mazes")
    print("  alone. On those it is exhaustive - DFS on a connected graph")
    print("  visits every reachable cell, so it cannot fail to find a goal.")

    env = SacMazeEnv(seed=args.seed, n_obstacles=(0, 0))

    ok = 0
    ratios = []

    for _ in range(args.mazes):
        env.reset()
        reached, cells = baseline_run(
            env.blocked, env.grid_size, env.start_cell, env.goal_cell
        )
        ok += reached
        opt = bfs_path_cells(env.blocked, env.grid_size,
                             env.start_cell, env.goal_cell)
        if reached and opt:
            ratios.append(cells / opt)

    print(f"\n  {'DFS baseline  static':<26} {ok / args.mazes:6.1%} "
          f"{0.0:6.1%} {1 - ok / args.mazes:6.1%}  "
          f"{np.mean(ratios) if ratios else float('nan'):6.2f}       -")

    # ---- maze.world, all 24 start cells ----------------
    print("\nmaze.world, all 24 start cells -> (4,4)   [NEVER TRAINED ON]")
    print(header)

    # maze.world's layout is rl_environment's default blocked_edges -
    # CLAUDE.md records that its 12 inner walls match exactly. Taking it
    # from there rather than a second copy keeps one source of truth.
    from .rl_environment import MazeEnvironment
    DEFAULT_BLOCKED_EDGES = MazeEnvironment().blocked_edges

    env = SacMazeEnv(seed=args.seed, n_obstacles=(0, 0))

    results = []

    for sx in range(DEPLOY_GRID_SIZE):
        for sy in range(DEPLOY_GRID_SIZE):

            if (sx, sy) == (4, 4):
                continue

            obs, _ = env.reset_fixed(
                DEFAULT_BLOCKED_EDGES, DEPLOY_GRID_SIZE, DEPLOY_CELL_SIZE,
                (sx, sy), (4, 4), n_obstacles=0,
            )


            travelled = 0.0
            px, py = env.x, env.y
            outcome = "timeout"
            steps = 1200

            for step in range(1, 1201):
                a = agent.act(obs, deterministic=True)
                obs, _, term, trunc, si = env.step(a)
                travelled += math.hypot(env.x - px, env.y - py)
                px, py = env.x, env.y
                if si["reached"]:
                    outcome, steps = "reached", step
                    break
                if si["collided"]:
                    outcome, steps = "collided", step
                    break

            results.append(dict(
                outcome=outcome, steps=steps, travelled=travelled,
                optimal=env.goal_path_cells * DEPLOY_CELL_SIZE,
                seconds=steps / 10.0,
            ))

    print(row("SAC  maze.world (24 starts)", summarise(results)))

    dfs_ok = 0
    for sx in range(DEPLOY_GRID_SIZE):
        for sy in range(DEPLOY_GRID_SIZE):
            if (sx, sy) == (4, 4):
                continue
            r, _ = baseline_run(DEFAULT_BLOCKED_EDGES, DEPLOY_GRID_SIZE,
                                (sx, sy), (4, 4))
            dfs_ok += r

    print(f"  {'DFS  maze.world (24)':<26} {dfs_ok / 24:6.1%} {0.0:6.1%} "
          f"{1 - dfs_ok / 24:6.1%}       -       -")


if __name__ == "__main__":
    main()
