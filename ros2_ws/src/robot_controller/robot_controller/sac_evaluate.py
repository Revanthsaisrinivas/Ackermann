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
from .robot_constants import (
    DEPLOY_CELL_SIZE,
    DEPLOY_GRID_SIZE,
    LIDAR_ANGLE_MIN,
    LIDAR_BEAMS,
    LIDAR_NORM_MAX,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
)
from .sac_agent import SAC
from .sac_env import SacMazeEnv, GOAL_RADIUS


# ============================================================
# --check-scan : confirm the real /scan against the trained geometry
# ============================================================

def check_scan(topic="/scan", n_msgs=20, timeout_s=20.0):
    """Compare a live /scan against what the observation builder assumes.

    WHAT THIS IS ACTUALLY LOOKING FOR
    ---------------------------------
    `downsample_scan` maps non-finite beams to LIDAR_RANGE_MAX and then
    clips into [LIDAR_RANGE_MIN, LIDAR_NORM_MAX]. So the two common
    "no return" encodings are both harmless:

        inf -> RANGE_MAX -> clipped to NORM_MAX -> normalises to 1.0
        6.0 -> 6.0       -> clipped to NORM_MAX -> normalises to 1.0

    But a driver that encodes "no return" as 0.0 - and some do - is NOT
    harmless. 0.0 clips UP to LIDAR_RANGE_MIN and normalises to 0.0,
    which is the encoding for "obstacle against the bumper". Every open
    direction would read as a wall. The policy would be correct and the
    sensor would be lying to it.

    A below-range_min reading is therefore the finding this exists to
    catch, not the inf-vs-range_max question the sac_env docstring
    originally framed it as.

    Run with Gazebo up:

        .venv-drl/bin/python -m robot_controller.sac_evaluate --check-scan
    """

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan

    from .sac_observation import downsample_scan

    rclpy.init()

    node = Node("scan_check")
    got = []

    node.create_subscription(
        LaserScan, topic, lambda m: got.append(m), 10
    )

    print(f"listening on {topic} for up to {timeout_s:.0f} s ...")

    deadline = time.time() + timeout_s

    while len(got) < n_msgs and time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    if not got:
        print(f"\nFAIL  no messages on {topic}. Is Gazebo running?")
        return 1

    print(f"received {len(got)} messages\n")

    m = got[0]
    ranges = np.concatenate([np.asarray(s.ranges, np.float64) for s in got])

    expected_inc = (2.0 * math.pi) / LIDAR_BEAMS

    checks = [
        ("beam count", len(m.ranges), LIDAR_BEAMS,
         len(m.ranges) == LIDAR_BEAMS),
        ("angle_min", m.angle_min, LIDAR_ANGLE_MIN,
         abs(m.angle_min - LIDAR_ANGLE_MIN) < 1e-3),
        ("angle_increment", m.angle_increment, expected_inc,
         abs(m.angle_increment - expected_inc) < 1e-4),
        ("range_min", m.range_min, LIDAR_RANGE_MIN,
         abs(m.range_min - LIDAR_RANGE_MIN) < 1e-2),
        ("range_max", m.range_max, LIDAR_RANGE_MAX,
         abs(m.range_max - LIDAR_RANGE_MAX) < 1e-2),
    ]

    print(f"  {'field':<18} {'measured':>12} {'expected':>12}   result")
    for label, actual, want, ok in checks:
        a = f"{actual:.5f}" if isinstance(actual, float) else str(actual)
        w = f"{want:.5f}" if isinstance(want, float) else str(want)
        print(f"  {label:<18} {a:>12} {w:>12}   {'ok' if ok else 'MISMATCH'}")

    # A per-beam increment can sit inside tolerance and still accumulate
    # into a real angular offset by the last beam, which shifts which
    # sector a return lands in. Report the cumulative drift, since that
    # is the number that decides whether it matters.
    drift = abs(m.angle_increment - expected_inc) * (len(m.ranges) - 1)

    if drift > math.radians(0.5):

        inc_359 = (2.0 * math.pi) / (len(m.ranges) - 1)

        print(f"\n  NOTE  angle_increment accumulates to {math.degrees(drift):.2f} "
              f"deg of drift by the last beam.")

        if abs(m.angle_increment - inc_359) < 1e-6:
            print(f"        The value is exactly 2*pi/{len(m.ranges) - 1}, not "
                  f"2*pi/{len(m.ranges)}: this driver treats angle_min and")
            print("        angle_max as BOTH inclusive, so the last beam "
                  "duplicates the first.")

        sector_deg = 360.0 / 36
        print(f"        Sectors are {sector_deg:.0f} deg wide, so this is "
              f"{math.degrees(drift) / sector_deg:.2f} of one sector - "
              f"below the")
        print("        min-pooling resolution, and it does not invalidate "
              "the policy.")

    # ---- how does this driver say "nothing there"? ----
    n = ranges.size
    n_inf = int(np.isinf(ranges).sum())
    n_nan = int(np.isnan(ranges).sum())
    finite = ranges[np.isfinite(ranges)]
    n_atmax = int((finite >= LIDAR_RANGE_MAX - 1e-3).sum())
    n_zero = int((finite <= 1e-6).sum())
    n_under = int(((finite > 1e-6) & (finite < LIDAR_RANGE_MIN)).sum())

    print(f"\n  out-of-range encoding, over {n} beams:")
    print(f"    inf                     {n_inf:7d}  ({n_inf / n:5.1%})")
    print(f"    nan                     {n_nan:7d}  ({n_nan / n:5.1%})")
    print(f"    >= range_max            {n_atmax:7d}  ({n_atmax / n:5.1%})")
    print(f"    exactly 0.0             {n_zero:7d}  ({n_zero / n:5.1%})")
    print(f"    0 < r < range_min       {n_under:7d}  ({n_under / n:5.1%})")

    obs = downsample_scan(np.asarray(m.ranges, np.float32))

    print(f"\n  downsample_scan -> {obs.shape[0]} sectors, "
          f"min {obs.min():.3f} max {obs.max():.3f}")

    bad = (obs < -1e-6).any() or (obs > 1.0 + 1e-6).any()

    failed = [c for c in checks if not c[3]]

    print()

    if n_zero or n_under:
        print("  FAIL  this driver reports sub-range_min values. Those clip UP")
        print("        to LIDAR_RANGE_MIN and normalise to 0.0, which the")
        print("        policy reads as an obstacle against the bumper. If they")
        print("        mean 'no return', downsample_scan must map them to")
        print("        LIDAR_RANGE_MAX alongside the non-finite case.")
        return 1

    if bad:
        print("  FAIL  normalised sectors left [0, 1].")
        return 1

    if failed:
        print("  FAIL  geometry differs from the trained observation:")
        for label, actual, want, _ in failed:
            print(f"        {label}: got {actual}, trained on {want}")
        return 1

    print("  PASS  live /scan geometry matches the trained observation.")

    # Do NOT claim the encoding was verified when no beam was out of
    # range. In a 5 m maze with walls inside 6 m in every direction,
    # that is the normal case, and an earlier version of this check
    # printed PASS on a question it had not tested.
    if n_inf:
        print("        'no return' is encoded as inf -> normalises to 1.0, "
              "as training assumed.")
    elif n_atmax:
        print("        'no return' is encoded as range_max -> normalises to "
              "1.0, as training assumed.")
    else:
        print()
        print("  INCONCLUSIVE on the out-of-range encoding: not one beam of")
        print(f"        {n} exceeded range_max, so this run never exercised "
              f"it.")
        print(f"        Nearest/farthest sector this pose: "
              f"{obs.min():.3f} / {obs.max():.3f} normalised "
              f"(1.0 = ceiling).")
        print("        In a 5 m maze every direction has a wall inside the "
              "6 m")
        print("        sensor range, so the encoding cannot arise here and "
              "cannot")
        print("        affect the policy in this world. To actually test it, "
              "run")
        print("        this against an open world, or accept it as "
              "unexercised.")

    return 0


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
    p.add_argument("--check-scan", action="store_true",
                   help="compare a live /scan against the trained geometry "
                        "and exit (needs Gazebo up; no checkpoint required)")
    p.add_argument("--scan-topic", default="/scan")
    args = p.parse_args()

    if args.check_scan:
        raise SystemExit(check_scan(args.scan_topic))

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

    print()
    print("  The baseline's 'sec' is blank on purpose, not missing. SAC's")
    print("  seconds come from this simulator at a 10 Hz control rate; the")
    print("  baseline executes cell-to-cell moves through the Gazebo")
    print("  controller at roughly 7 s each. Printing both in one column")
    print("  would compare a headless integrator against a physics sim and")
    print("  flatter whichever ran in the cheaper one.")
    print()
    print("  path/opt is the honest cross-system comparison: both are")
    print("  measured against the same BFS shortest path, and it is")
    print("  dimensionless, so neither clock enters it.")
    print()
    print("  One asymmetry in it, and it favours the BASELINE. The DFS")
    print("  ratio counts cell moves, i.e. the cell-centre polyline, which")
    print("  is the shortest path a cell-level planner could possibly")
    print("  drive. The real robot cannot drive it - a 0.369 m minimum")
    print("  turning radius means every corner is an arc - so the baseline's")
    print("  true driven distance is longer than what is printed here.")
    print("  SAC's ratio is measured actual travel, arcs included. Read the")
    print("  baseline's path/opt as a lower bound and SAC's as exact.")

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
