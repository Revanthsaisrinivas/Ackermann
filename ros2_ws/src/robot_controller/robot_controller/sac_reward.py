#!/usr/bin/env python3

"""
Reward definition. Every coefficient is a named constant here; no
literals scattered through the environment.

Each term is returned separately so `sac_train.py` can log it to
TensorBoard. When an agent learns something degenerate, the term
responsible is visible immediately instead of being inferred from a
single scalar.

PROGRESS SHAPING - AND WHY THE GAMMA CAME OUT
---------------------------------------------
The progress term is

    F(s, s') = Phi(s') - Phi(s)      with  Phi(s) = -K * d(s)
             = K * (d - d')

An earlier version of this file used the discounted potential form
`K * (d - gamma * d')`, on the grounds that the gamma carries Ng, Harada
and Russell's policy-invariance guarantee. The theorem is real, but the
justification written here for it was inverted, and the measured
consequence was the opposite of what was intended.

Set d' = d - the robot does not move at all - and compare:

    K*(d - gamma*d')  ->  K*d*(1 - gamma)  =  +0.2*d per step
    K*(d - d')        ->  0                =   0.0   per step

The discounted form pays a STATIONARY robot to be far from the goal. At
1.5 m that is +0.28/step against an R_TIME of -0.02 meant to discourage
exactly that - 14x too small to bite. Over an 800-step episode it comes
to +224, against +100 for actually reaching the goal. A timeout scored
better than a success.

Equivalently, for the undiscounted sum over a closed loop that returns
to where it started:

    gamma = 1     sum telescopes to K*(d_0 - d_T) = 0   exactly zero
    gamma = 0.99  sum = 0.01*K*sum(d_t)               > 0, farmable

so it is the gamma FORM that lets an approach-retreat cycle farm return,
not the naive one. The old docstring asserted the reverse.

Ng's theorem is not contradicted by this: it guarantees invariance of
the optimal policy under the DISCOUNTED objective, where the term does
telescope to gamma^T*Phi_T - Phi_0. But the guarantee is about the
optimum, not about what a function approximator chases on the way there,
and it is in any case already broken here because Phi != 0 at collision
terminals, which the theorem requires.

Measured on this project: with the gamma form, episode return climbed
60 -> 129 while success fell 40% -> 17%, and the return/success
correlation collapsed from 0.85 to 0.04.

VERIFYING IT IN PRACTICE
------------------------
Theory guarantees the optimum is unchanged, not that learning finds it.
`sac_train.py` logs the correlation between episode return and success;
if return rises while success does not, something degenerate is being
farmed regardless of what the theory says. That check is what caught
this, and it is the reason every term is logged separately.
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

GAMMA = 0.99               # the agent's discount, for reference

# Discount used INSIDE the shaping term. 1.0 on purpose - see the module
# docstring. Setting this to GAMMA restores the loiter income that made
# an 800-step timeout score +224 against +100 for reaching the goal.
SHAPING_GAMMA = 1.0


def compute(distance, prev_distance, min_lidar, action, prev_action,
            v, reached, collided, gamma=SHAPING_GAMMA):
    """Return (total, terms dict).

    distance/prev_distance - goal distance now and last step, metres
    min_lidar              - nearest return this step, metres
    action/prev_action     - normalised actions in [-1, 1]^2
    v                      - commanded speed, m/s (sign matters)
    """

    # Telescoping progress shaping. With gamma = SHAPING_GAMMA = 1.0 a
    # stationary robot earns exactly zero here, so R_TIME is the only
    # per-step pressure and it actually decides the episode.
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
