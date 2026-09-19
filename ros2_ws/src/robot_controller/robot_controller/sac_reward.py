#!/usr/bin/env python3

"""
Reward definition. Every coefficient is a named constant here; no
literals scattered through the environment.

Each term is returned separately so `sac_train.py` can log it to
TensorBoard. When an agent learns something degenerate, the term
responsible is visible immediately instead of being inferred from a
single scalar.

POTENTIAL-BASED SHAPING
-----------------------
The progress term is exact potential-based shaping:

    F(s, s') = gamma * Phi(s') - Phi(s)      with  Phi(s) = -K * d(s)
             = K * (d - gamma * d')

Note the gamma on d'. The naive form K*(d - d') is what most
implementations use, and it is NOT policy-invariant - it is the gamma
that carries Ng, Harada and Russell's guarantee that shaping cannot
change the optimal policy. Without it, a cycle that approaches and
retreats from the goal has slightly positive net shaping return, which
is precisely the reward-farming oscillation this design is trying to
exclude. With it, any closed loop nets exactly zero.

VERIFYING IT IN PRACTICE
------------------------
Theory guarantees the optimum is unchanged, not that learning finds it.
`sac_train.py` logs the correlation between episode return and success;
if return rises while success does not, something degenerate is being
farmed regardless of what the theory says.
"""

import numpy as np


# ============================================================
# COEFFICIENTS
# ============================================================

K_PROGRESS = 20.0          # per metre of potential
R_GOAL = 100.0             # terminal
R_COLLISION = -100.0       # terminal
R_TIME = -0.02             # per step
K_PROXIMITY = -2.0         # per metre inside PROXIMITY_BAND
PROXIMITY_BAND = 0.30      # m
K_SMOOTH = -0.1            # per unit L1 action change
R_REVERSE = -0.05          # per step while v < 0

GAMMA = 0.99               # must match the agent's discount


def compute(distance, prev_distance, min_lidar, action, prev_action,
            v, reached, collided, gamma=GAMMA):
    """Return (total, terms dict).

    distance/prev_distance - goal distance now and last step, metres
    min_lidar              - nearest return this step, metres
    action/prev_action     - normalised actions in [-1, 1]^2
    v                      - commanded speed, m/s (sign matters)
    """

    # Exact potential-based shaping; see the module docstring for why
    # the gamma on `distance` is not optional.
    r_progress = K_PROGRESS * (prev_distance - gamma * distance)

    r_time = R_TIME

    intrusion = max(0.0, PROXIMITY_BAND - float(min_lidar))
    r_proximity = K_PROXIMITY * intrusion

    delta_a = float(
        np.sum(np.abs(np.asarray(action) - np.asarray(prev_action)))
    )
    r_smooth = K_SMOOTH * delta_a

    r_reverse = R_REVERSE if v < 0.0 else 0.0

    r_terminal = 0.0
    if collided:
        r_terminal = R_COLLISION
    elif reached:
        r_terminal = R_GOAL

    total = (
        r_progress + r_time + r_proximity + r_smooth + r_reverse + r_terminal
    )

    return float(total), {
        "r_progress": r_progress,
        "r_time": r_time,
        "r_proximity": r_proximity,
        "r_smooth": r_smooth,
        "r_reverse": r_reverse,
        "r_terminal": r_terminal,
    }
