#!/usr/bin/env python3
"""
Q-learning agent for the 5x5 maze defined in maze.world.

Works directly against MazeEnvironment's interface (rl_environment.py):
  - env.get_all_states()    -> used to pre-populate the Q-table
  - env.get_valid_actions() -> used so the agent never intentionally
                             picks an illegal move, during both
                             exploration and exploitation
  - env.ACTIONS             -> the 4 discrete actions (EAST/NORTH/
                             WEST/SOUTH), passed in by the caller
"""
import os
import pickle
import random


class QLearningAgent:

    def __init__(
        self,
        actions,
        states=None,
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

        # state -> [Q(s, a0), Q(s, a1), Q(s, a2), Q(s, a3)]
        self.q_table = {}

        if states is not None:
            self.initialize_q_table(states)

    # =====================================================
    # 1. INITIALIZE Q-TABLE
    # =====================================================

    def initialize_q_table(self, states):
        """Pre-populate the Q-table with a zeroed row for every state
        returned by env.get_all_states(), so lookups never need to
        guess whether a state has been seen before."""

        for state in states:
            if state not in self.q_table:
                self.q_table[state] = [0.0 for _ in self.actions]

    def _ensure_state(self, state):
        """Defensive fallback: if a state was never pre-registered
        (e.g. get_all_states() wasn't called, or the environment's
        state space changes), create a zeroed row for it on the fly
        instead of raising a KeyError."""

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
        """Pure exploitation, no epsilon -- used by run_policy.py
        after training, where no exploration should happen."""

        self._ensure_state(state)

        if not valid_actions:
            return None

        q_values = self.q_table[state]

        best_value = max(q_values[a] for a in valid_actions)

        best_actions = [
            a for a in valid_actions
            if q_values[a] == best_value
        ]

        return random.choice(best_actions)

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
        inference against a frozen table."""

        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
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

        if isinstance(data, dict):
            self.q_table = data["q_table"]
            self.epsilon = data["epsilon"]
            self.learning_rate = data["learning_rate"]
            self.discount_factor = data["discount_factor"]
        else:
            # Older save format containing only the Q-table
            self.q_table = data
            # Keep constructor values for epsilon, learning_rate and discount_factor


if __name__ == "__main__":
    # Minimal smoke test -- confirms the agent can be built against
    # MazeEnvironment's real interface and that a single update
    # actually moves a Q-value, without running a full training loop.
    from rl_environment import MazeEnvironment

    env = MazeEnvironment()

    agent = QLearningAgent(
        actions=env.ACTIONS,
        states=env.get_all_states(),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    print(f"Q-table initialized with {len(agent.q_table)} states")

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
    if os.path.exists("test_q_table.pkl"):
        os.remove("test_q_table.pkl")