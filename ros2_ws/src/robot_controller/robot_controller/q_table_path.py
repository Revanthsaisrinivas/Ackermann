#!/usr/bin/env python3

"""
The one place that decides where q_table.pkl lives.

WHY THIS EXISTS
---------------
There used to be four copies of q_table.pkl:

    src/robot_controller/robot_controller/q_table.pkl
    build/robot_controller/build/lib/robot_controller/q_table.pkl
    install/robot_controller/lib/python3.10/site-packages/robot_controller/q_table.pkl
    install/robot_controller/share/robot_controller/q_table.pkl

train.py saved next to `__file__` of whichever copy of the module
got imported, and run_policy.resolve_q_table_path() searched a
different order, so the file that training wrote and the file that
the policy read were routinely not the same file. A run once loaded
a stale 2456-state table out of build/ while the freshly trained
3744-state one sat in src/. Nothing warned about it - the policy
just quietly drove on bad Q-values.

colcon copies whatever setup.py lists, so the fix has two halves:
the q_table.pkl entries came out of setup.py's package_data and
data_files (so the build stops making copies), and every module
now asks this function instead of guessing from `__file__`.

THE RULE
--------
One path, no searching:

    $RL_Q_TABLE                              if set
    ~/.ros/rl_ackermann/q_table.pkl          otherwise

It deliberately lives outside the workspace, so it survives
`rm -rf build install`, is never copied by a build, and is the same
file no matter which copy of the package you imported.
"""

import os


ENV_VAR = "RL_Q_TABLE"

DEFAULT_PATH = os.path.join(
    "~",
    ".ros",
    "rl_ackermann",
    "q_table.pkl",
)


def q_table_path():
    """Absolute path to the canonical Q-table. Never searches."""

    raw = os.environ.get(ENV_VAR)

    if raw:
        raw = raw.strip()

    if not raw:
        raw = DEFAULT_PATH

    return os.path.abspath(
        os.path.expanduser(
            os.path.expandvars(raw)
        )
    )


def ensure_q_table_dir(path=None):
    """Create the containing directory so a save cannot fail on it."""

    if path is None:
        path = q_table_path()

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    return path


def describe_q_table_source(path=None):
    """Short string saying WHICH rule produced the path, for logs."""

    if path is None:
        path = q_table_path()

    if os.environ.get(ENV_VAR, "").strip():
        origin = f"${ENV_VAR}"
    else:
        origin = "default"

    if os.path.isfile(path):
        status = f"{os.path.getsize(path)} bytes"
    else:
        status = "MISSING"

    return f"{path}  ({origin}, {status})"
