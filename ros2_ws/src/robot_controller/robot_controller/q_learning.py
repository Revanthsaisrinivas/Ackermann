#!/usr/bin/env python3

import pickle
import random


# The state encoding this agent's Q-tables are keyed on, stamped into
# every saved file and checked on load.
#
# It is NOT decoration. The table shipped with this workspace was keyed
# on an older encoding,
#
#     robot_x_robot_y_goal_x_goal_y_heading_east_north_west_south
#
# a 9-tuple of absolute positions. The controller has since moved to the
# 7-tuple relative form below. A 7-tuple can never be found in a table of
# 9-tuples, so `state in agent.q_table` was False for every state of
# every run - the policy silently degraded to its fallback and nothing
# said a word. Compare this tag before trusting a loaded table.
STATE_FORMAT = "dx_dy_heading_east_north_west_south"


class QLearningAgent:

    def __init__(
        self,
        actions,
        states=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):

        # Action order:
        #
        # 0 = EAST
        # 1 = NORTH
        # 2 = WEST
        # 3 = SOUTH

        self.actions = list(actions)

        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.epsilon_decay = epsilon_decay

        self.epsilon_min = epsilon_min

        self.q_table = {}

        if states is not None:

            self.initialize_q_table(
                states
            )

    # ============================================================
    # INITIALIZE STATES
    # ============================================================

    def initialize_q_table(
        self,
        states
    ):

        for state in states:

            self._ensure_state(
                state
            )

    # ============================================================
    # ENSURE STATE
    # ============================================================

    def _ensure_state(
        self,
        state
    ):

        if state not in self.q_table:

            self.q_table[state] = [

                0.0,

                0.0,

                0.0,

                0.0,
            ]

    # ============================================================
    # CHOOSE ACTION
    #
    # Training:
    # epsilon-greedy
    # ============================================================

    def choose_action(
        self,
        state,
        valid_actions
    ):

        self._ensure_state(
            state
        )

        if not valid_actions:

            return None

        # Exploration
        if random.random() < self.epsilon:

            return random.choice(
                valid_actions
            )

        # Exploitation
        return self.best_action(
            state,
            valid_actions
        )

    # ============================================================
    # BEST ACTION
    #
    # Priority when Q-values are equal:
    #
    # EAST -> NORTH -> WEST -> SOUTH
    # ============================================================

    def best_action(
        self,
        state,
        valid_actions
    ):

        self._ensure_state(
            state
        )

        if not valid_actions:

            return None

        # Keep priority ordering.
        # EAST -> NORTH -> WEST -> SOUTH
        ordered_actions = [
            action for action in [0, 1, 2, 3] if action in valid_actions
        ]

        q_values = self.q_table[
            state
        ]

        best_action = ordered_actions[0]
        best_value = q_values[best_action]
        tolerance = 1e-9

        for action in ordered_actions[1:]:

            q_value = q_values[action]

            # Equal values retain the earlier priority order.
            if abs(q_value - best_value) <= tolerance:
                continue

            if q_value > best_value + tolerance:
                best_value = q_value
                best_action = action

        return best_action

    # ============================================================
    # Q UPDATE
    # ============================================================

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        next_valid_actions,
        done=False,
    ):

        self._ensure_state(
            state
        )

        self._ensure_state(
            next_state
        )

        current_q = self.q_table[
            state
        ][action]

        # Terminal state
        if done:

            target = reward

        elif not next_valid_actions:

            target = reward

        else:

            max_next_q = max(

                self.q_table[
                    next_state
                ][next_action]

                for next_action
                in next_valid_actions
            )

            target = (

                reward

                +

                self.discount_factor

                *

                max_next_q
            )

        # Q-learning update
        self.q_table[
            state
        ][action] = (

            current_q

            +

            self.learning_rate

            *

            (
                target
                -
                current_q
            )
        )

    # ============================================================
    # EPSILON DECAY
    # ============================================================

    def decay_epsilon(self):

        self.epsilon *= (
            self.epsilon_decay
        )

        self.epsilon = max(

            self.epsilon,

            self.epsilon_min
        )

    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        filepath="q_table.pkl"
    ):

        data = {

            "q_table":
                self.q_table,

            "epsilon":
                self.epsilon,

            "learning_rate":
                self.learning_rate,

            "discount_factor":
                self.discount_factor,

            # dx = goal_x - robot_x, dy = goal_y - robot_y
            "state_format": STATE_FORMAT,

            "action_order": [

                "EAST",

                "NORTH",

                "WEST",

                "SOUTH",
            ],
        }

        with open(
            filepath,
            "wb"
        ) as file:

            pickle.dump(

                data,

                file
            )

    # ============================================================
    # LOAD
    # ============================================================

    def load(
        self,
        filepath="q_table.pkl"
    ):

        with open(
            filepath,
            "rb"
        ) as file:

            data = pickle.load(
                file
            )

        if (

            isinstance(
                data,
                dict
            )

            and

            "q_table" in data
        ):

            self.q_table = data[
                "q_table"
            ]

            self.epsilon = data.get(

                "epsilon",

                self.epsilon
            )

            self.learning_rate = data.get(

                "learning_rate",

                self.learning_rate
            )

            self.discount_factor = data.get(

                "discount_factor",

                self.discount_factor
            )

            # Keep the format tag so callers can verify the table was
            # written by the same state encoding they are about to look
            # up. A table keyed on an older encoding is not "empty" or
            # "undertrained" - every lookup simply misses, silently.
            self.state_format = data.get(
                "state_format"
            )

        else:

            # Compatibility with old raw Q-table
            self.q_table = data

            self.state_format = None