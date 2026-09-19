#!/usr/bin/env python3

"""
The one place that decides where SAC checkpoints live.

Follows the q_table_path.py precedent, and for the same reason. There
used to be four copies of q_table.pkl because train.py saved next to
whichever copy of the module got imported while run_policy.py searched a
different order; a run once loaded a stale 2456-state table while the
freshly trained one sat elsewhere, and nothing warned about it.

    $RL_SAC_MODEL                      if set
    ~/.ros/rl_ackermann/sac/           otherwise

Outside the workspace on purpose: it survives `rm -rf build install`,
colcon never copies it, and it is the same path whichever copy of the
package was imported.
"""

import os


ENV_VAR = "RL_SAC_MODEL"

DEFAULT_DIR = os.path.join("~", ".ros", "rl_ackermann", "sac")


def sac_dir():
    """Absolute checkpoint directory. Never searches."""

    raw = os.environ.get(ENV_VAR, "").strip() or DEFAULT_DIR

    return os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))


def checkpoint_path(name="policy"):
    """Full path to a named checkpoint (no extension)."""

    return os.path.join(sac_dir(), name)


def ensure_dir():

    d = sac_dir()
    os.makedirs(d, exist_ok=True)
    return d


def describe(name="policy"):
    """Which rule produced the path, and whether it exists."""

    p = checkpoint_path(name)

    origin = f"${ENV_VAR}" if os.environ.get(ENV_VAR, "").strip() else "default"

    status = "present" if os.path.exists(p + ".pt") else "MISSING"

    return f"{p}.pt  ({origin}, {status})"
