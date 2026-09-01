from .rl_environment import MazeEnvironment

env = MazeEnvironment()

state = env.reset()

print("Initial State:", state)

env.render()

state, reward, done = env.step(env.EAST)

print()

print("After moving EAST")

print("State :", state)
print("Reward:", reward)
print("Done :", done)

env.render()