#!/usr/bin/env python3
"""
Train a Q-learning agent on the 5x5 maze.

This script:
1. Creates the maze environment.
2. Creates the Q-learning agent.
3. Trains the agent for multiple episodes.
4. Saves the learned Q-table.
"""

from .rl_environment import MazeEnvironment
from .q_learning import QLearningAgent
import os
import random

# =====================================================
# TRAINING PARAMETERS
# =====================================================

EPISODES = 1000
PRINT_EVERY = 5
SAVE_FILE = os.path.join(
    os.path.dirname(__file__),
    "q_table.pkl"
)


# =====================================================
# TRAINING FUNCTION
# =====================================================

def train():

    env = MazeEnvironment()

    agent = QLearningAgent(
        actions=env.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    success_count = 0

    # Build every (start, goal) pair once, up front, instead of
    # rebuilding the same list on every episode.
    pairs = []

    for gx in range(env.GRID_SIZE):
        for gy in range(env.GRID_SIZE):

            for rx in range(env.GRID_SIZE):
                for ry in range(env.GRID_SIZE):

                    if (rx, ry) != (gx, gy):
                        pairs.append(((rx, ry), (gx, gy)))

    cases_per_episode = len(pairs)

    for episode in range(1, EPISODES + 1):

        success_this_episode = 0

        # Shuffle so the Q-table isn't updated in the same fixed
        # goal-then-start order every single episode.
        random.shuffle(pairs)

        for start, goal in pairs:

            state = env.reset(
                start=start,
                goal=goal
            )

            done = False

            while not done:

                valid_actions = env.get_valid_actions()

                action = agent.choose_action(
                    state,
                    valid_actions
                )

                next_state, reward, done = env.step(action)

                next_valid_actions = env.get_valid_actions()

                agent.update_q_table(
                    state,
                    action,
                    reward,
                    next_state,
                    next_valid_actions,
                    done
                )

                state = next_state

            if reward == env.GOAL_REWARD:
                success_this_episode += 1

        success_count += success_this_episode

        agent.decay_epsilon()

        if episode % PRINT_EVERY == 0:

            total_cases = episode * cases_per_episode

            success_rate = (
                success_count / total_cases
            ) * 100

            print("-" * 50)
            print(f"Episode      : {episode}")
            print(f"Success Rate : {success_rate:.2f}%")
            print(f"Epsilon      : {agent.epsilon:.4f}")

    agent.save(SAVE_FILE)

    print("\nTraining Complete!")
    print(f"Episodes      : {EPISODES}")
    print(f"Successes     : {success_count}")
    print(f"Final Epsilon : {agent.epsilon:.4f}")
    print(f"Q-table saved : {SAVE_FILE}")


# =====================================================
# MAIN
# =====================================================

def main():
    train()


if __name__ == "__main__":
    main()