#!/usr/bin/env python3
"""
Q-learning agent for the 5x5 unknown-map maze.

Works with MazeEnvironment:
  - state is (robot_x, robot_y, goal_x, goal_y, heading,
    front_wall, left_wall, right_wall, back_wall) -- position/goal,
    the robot's current heading, and its local discovered wall
    observations
  - env.get_valid_actions() provides actions allowed by currently
    discovered information (never the full ground-truth wall map)
  - env.ACTIONS -> the 4 discrete actions (EAST/NORTH/WEST/SOUTH),
    passed in by the caller
  - Q-table entries are created lazily, the first time a state is
    actually encountered, rather than pre-populated up front: with
    heading + discovered walls folded into the state, the space is
    far larger than the old (robot, goal) state and mostly consists
    of configurations a given run will never visit
"""
import os
import pickle
import random
from collections import defaultdict


class QLearningAgent:

    def __init__(
        self,
        actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):
        self.actions = actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # state -> [Q(s, a0), Q(s, a1), Q(s, a2), Q(s, a3)], created
        # lazily on first access instead of pre-populated from
        # env.get_all_states(). NOTE: the factory is a bound method
        # rather than a lambda specifically so save() can still be
        # made to work -- lambdas (and therefore a raw
        # `defaultdict(lambda: ...)`) can't be pickled, and this
        # table gets pickled in save(). save() converts to a plain
        # dict before pickling either way, but keeping the factory
        # itself picklable-in-principle avoids a second landmine if
        # this table is ever pickled directly elsewhere.
        self.q_table = defaultdict(self._zero_row)

    def _zero_row(self):
        return [0.0 for _ in self.actions]

    # =====================================================
    # 1. ENSURE STATE EXISTS
    # =====================================================

    def _ensure_state(self, state):
        """The defaultdict already creates a zeroed row on first
        access, so this is technically redundant -- kept as an
        explicit, self-documenting guard at the top of the public
        methods below rather than relying on the side effect of a
        dict lookup."""

        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]

    # =====================================================
    # 2. CHOOSE ACTION (epsilon-greedy, restricted to valid actions)
    # =====================================================

    def choose_action(self, state, valid_actions):
        """
        With probability epsilon: pick a random VALID action
        (exploration). Otherwise: pick the valid action with the
        highest Q-value (exploitation). valid_actions should come
        from env.get_valid_actions() -- this is what guarantees the
        agent never intentionally selects an illegal move.
        """

        self._ensure_state(state)

        if not valid_actions:
            # No legal moves at all from this state (shouldn't happen
            # in this maze, since every cell has at least one open
            # side, but guard against it rather than crashing).
            return random.choice(self.actions)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_table[state]

        best_value = max(q_values[a] for a in valid_actions)

        best_actions = [
            a for a in valid_actions
            if q_values[a] == best_value
        ]

        return random.choice(best_actions)

    def best_action(self, state, valid_actions):
        """
        Deterministic greedy action selection.
        """

        self._ensure_state(state)

        if not valid_actions:
            return None

        q_values = self.q_table[state]

        best_action = valid_actions[0]

        for action in valid_actions:
            if q_values[action] > q_values[best_action]:
                best_action = action

        return best_action

    # =====================================================
    # 3. BELLMAN UPDATE
    # =====================================================

    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        next_valid_actions,
        done=False,
    ):
        """
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        next_valid_actions restricts the bootstrap max to actions that
        are actually legal from next_state (i.e. env.get_valid_actions()
        called AFTER stepping into next_state) -- since choose_action()
        already restricts action *selection* to valid actions, the
        Bellman target should be consistent with that and not bootstrap
        off Q-values for moves the agent would never actually take.

        done still short-circuits the bootstrap entirely: when an
        episode ends (goal reached OR timeout), there is no s' to
        continue from, so the target is just the reward itself.
        """

        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]

        if done or not next_valid_actions:
            target = reward
        else:
            target = reward + self.discount_factor * max(
                self.q_table[next_state][a] for a in next_valid_actions
            )

        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)

    # =====================================================
    # 4. DECAY EPSILON
    # =====================================================

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    # =====================================================
    # 5. SAVE Q-TABLE
    # =====================================================

    def save(self, filepath="q_table.pkl"):
        """Saves the Q-table AND the current hyperparameters (epsilon,
        learning_rate, discount_factor), so training can be resumed
        later from exactly where it left off, not just re-run
        inference against a frozen table.

        Converts the table to a plain dict first: a defaultdict's
        default_factory has to be picklable too, and even a bound
        method factory adds a needless dependency on this class's
        internals for something that's just data. A plain dict avoids
        that entirely and is what load() below expects back."""

        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor
            }, f)

    # =====================================================
    # 6. LOAD Q-TABLE
    # =====================================================

    def load(self, filepath="q_table.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "q_table" in data:
            loaded_table = data["q_table"]
            self.epsilon = data["epsilon"]
            self.learning_rate = data["learning_rate"]
            self.discount_factor = data["discount_factor"]
        else:
            # Older save format containing only the Q-table itself
            # (no "q_table" key) -- keep constructor values for
            # epsilon, learning_rate and discount_factor.
            loaded_table = data

        # Rebuild as a defaultdict so states not present in the saved
        # file (e.g. ones a resumed training run discovers that the
        # original run never reached) still get a zeroed row lazily,
        # instead of every lookup needing a manual _ensure_state()
        # call or a try/except.
        self.q_table = defaultdict(self._zero_row, loaded_table)


if __name__ == "__main__":
    # Minimal smoke test -- confirms the agent can be built against
    # MazeEnvironment's real interface and that a single update
    # actually moves a Q-value, without running a full training loop.
    from rl_environment import MazeEnvironment

    env = MazeEnvironment()

    agent = QLearningAgent(
        actions=env.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    print(f"Q-table starts with {len(agent.q_table)} states (lazy)")

    state = env.reset(start=(0, 1), goal=(4, 4))
    valid_actions = env.get_valid_actions()
    print(f"Start state: {state} | valid actions: {valid_actions}")

    action = agent.choose_action(state, valid_actions)
    next_state, reward, done = env.step(action)
    next_valid_actions = env.get_valid_actions()

    print(f"Took action {action}, got reward {reward}, done={done}")

    before = agent.q_table[state][action]
    agent.update_q_table(state, action, reward, next_state, next_valid_actions, done)
    after = agent.q_table[state][action]

    print(f"Q({state}, {action}) went from {before:.4f} -> {after:.4f}")
    print(f"Q-table now has {len(agent.q_table)} states")

    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon:.4f}")

    agent.save("test_q_table.pkl")

    reloaded = QLearningAgent(actions=env.ACTIONS)
    reloaded.load("test_q_table.pkl")

    print(
        f"Reloaded {len(reloaded.q_table)} states | "
        f"epsilon={reloaded.epsilon:.4f} | "
        f"lr={reloaded.learning_rate} | "
        f"gamma={reloaded.discount_factor}"
    )
    print(
        "Q-value round-trip matches:",
        reloaded.q_table[state][action] == after
    )

    # Confirm a never-seen state still gets a zeroed row lazily after
    # reload, rather than a KeyError.
    unseen_state = (4, 4, 0, 0, env.EAST, 0, 0, 0, 0)
    print(
        "Unseen state after reload auto-creates zero row:",
        reloaded.q_table[unseen_state]
    )

    if os.path.exists("test_q_table.pkl"):
        os.remove("test_q_table.pkl")