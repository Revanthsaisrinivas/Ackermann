#!/usr/bin/env python3

"""
The maze layout, read from `worlds/maze.world`.

WHY THIS FILE EXISTS
--------------------
The layout used to live in two places: the 12 inner wall models in
`maze.world`, and a hand-written `blocked_edges` set in
`rl_environment.py`. Nothing checked that they agreed. Editing the world
and forgetting the Python set (or the reverse) trains a policy for one
maze and drives it in another, and the only symptom is a robot that
navigates badly.

So the world file is now the single source of truth. `blocked_edges()`
parses it; `rl_environment` calls that instead of carrying its own copy.

WHAT COUNTS AS AN INNER WALL
----------------------------
Every wall in the world is a box model with a `<pose>` and a
`<size>`. A wall is an INNER wall when:

  * its box is 0.50 m tall (the wall panels; the floor and ceiling
    planes are not), and
  * it is no longer than the grid is wide (the four boundary walls are
    5.1 m and are implied by the grid bounds anyway).

A wall thinner in x than in y is vertical: it sits at x = pose.x and
separates the two columns either side of it, over every row its length
covers. A horizontal wall is the same with the axes swapped. A wall
2.0 m long therefore blocks TWO edges, which is why `wall_B3_D3` and
`wall_E3_E5` each contribute a pair.

THE SIGNATURE
-------------
`signature()` hashes the canonical edge set. `train.py` stamps it into
the Q-table and `run_policy.py` reports it, so a table trained on one
layout is identifiable when it is loaded against another. It is derived
purely from the edges, so reformatting the world file, renaming a wall
model or reordering the models does not change it; moving, adding or
removing a wall does.
"""

import hashlib
import os
import re
import xml.etree.ElementTree as ET


# Grid geometry. Kept here rather than imported from rl_environment so
# that parsing the world does not depend on the environment module.
GRID_SIZE = 5
CELL_SIZE = 1.0

# A wall panel is this tall. The floor/ceiling planes are not.
WALL_HEIGHT = 0.50

# Floating point slop when deciding which cells a wall spans.
TOLERANCE = 1e-6

ENV_VAR = "RL_MAZE_WORLD"

WORLD_RELATIVE = os.path.join(
    "robot_description",
    "worlds",
    "maze.world",
)


# ================================================================
# LOCATING THE WORLD FILE
# ================================================================

def world_path():
    """Return the path to maze.world, or None if it cannot be found.

    Order: $RL_MAZE_WORLD, then the installed package share directory,
    then the source tree beside this package.
    """

    override = os.environ.get(ENV_VAR)

    if override:
        return override

    # Installed share directory, when running from a colcon install.
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("robot_description")

        candidate = os.path.join(share, "worlds", "maze.world")

        if os.path.isfile(candidate):
            return candidate

    except Exception:
        # ament is unavailable, or the package is not installed yet.
        # The source tree below is the normal path during development.
        pass

    # Source tree: .../src/robot_controller/robot_controller/ -> .../src/
    src_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    candidate = os.path.join(src_root, WORLD_RELATIVE)

    if os.path.isfile(candidate):
        return candidate

    return None


# ================================================================
# PARSING
# ================================================================

def _wall_models(path):
    """Yield (name, pose_x, pose_y, size_x, size_y, size_z) per box model."""

    raw = open(path).read()

    # Gazebo accepts "--" inside an XML comment; ElementTree does not,
    # and maze.world contains one. Strip comments before parsing rather
    # than requiring the world file to be edited.
    clean = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    root = ET.fromstring(clean)

    for model in root.iter("model"):

        pose = model.find("pose")

        box = model.find(".//box/size")

        if pose is None or box is None:
            continue

        pose_values = pose.text.split()

        size_values = box.text.split()

        if len(pose_values) < 2 or len(size_values) < 3:
            continue

        yield (
            model.get("name", ""),
            float(pose_values[0]),
            float(pose_values[1]),
            float(size_values[0]),
            float(size_values[1]),
            float(size_values[2]),
        )


def _edges_for_wall(pose_x, pose_y, size_x, size_y):
    """Return the set of blocked edges one wall panel creates."""

    edges = set()

    if size_x < size_y:

        # Vertical wall: sits on the boundary between two columns.

        column = int(round(pose_x / CELL_SIZE))

        if column <= 0 or column >= GRID_SIZE:
            return edges

        y_start = pose_y - size_y / 2.0
        y_end = pose_y + size_y / 2.0

        row = int(round(y_start / CELL_SIZE))

        while row * CELL_SIZE + TOLERANCE < y_end:

            if 0 <= row < GRID_SIZE:
                edges.add(
                    frozenset({(column - 1, row), (column, row)})
                )

            row += 1

    else:

        # Horizontal wall: sits on the boundary between two rows.

        row = int(round(pose_y / CELL_SIZE))

        if row <= 0 or row >= GRID_SIZE:
            return edges

        x_start = pose_x - size_x / 2.0
        x_end = pose_x + size_x / 2.0

        column = int(round(x_start / CELL_SIZE))

        while column * CELL_SIZE + TOLERANCE < x_end:

            if 0 <= column < GRID_SIZE:
                edges.add(
                    frozenset({(column, row - 1), (column, row)})
                )

            column += 1

    return edges


def blocked_edges(path=None):
    """Parse the world file and return its inner walls as blocked edges.

    Raises FileNotFoundError when the world file cannot be located, so a
    missing world is a loud failure rather than a silently empty maze.
    """

    if path is None:
        path = world_path()

    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            "Could not find maze.world.\n"
            f"Set ${ENV_VAR} to its full path, or build the workspace so\n"
            "that robot_description is installed."
        )

    edges = set()

    for name, px, py, sx, sy, sz in _wall_models(path):

        if abs(sz - WALL_HEIGHT) > TOLERANCE:
            # Not a wall panel (floor, ceiling, marker).
            continue

        if max(sx, sy) > GRID_SIZE * CELL_SIZE:
            # A boundary wall. The grid bounds already block those.
            continue

        edges |= _edges_for_wall(px, py, sx, sy)

    return edges


# ================================================================
# SIGNATURE
# ================================================================

def canonical(edges):
    """A stable, order-independent text form of an edge set."""

    pairs = sorted(
        tuple(sorted(edge)) for edge in edges
    )

    return ";".join(
        f"{a[0]},{a[1]}-{b[0]},{b[1]}" for a, b in pairs
    )


def signature(edges=None, path=None):
    """Short hash identifying a layout. Same walls -> same signature."""

    if edges is None:
        edges = blocked_edges(path)

    digest = hashlib.sha256(
        canonical(edges).encode("utf-8")
    ).hexdigest()

    return digest[:12]


def describe(path=None):
    """One line naming the world file and its signature, for logs."""

    if path is None:
        path = world_path()

    try:
        edges = blocked_edges(path)

    except FileNotFoundError as error:
        return f"maze layout UNAVAILABLE ({error})"

    return (
        f"{path}  ({len(edges)} inner walls, signature {signature(edges)})"
    )


if __name__ == "__main__":

    print(describe())

    for pair in sorted(
        tuple(sorted(edge)) for edge in blocked_edges()
    ):
        print("   ", pair)
