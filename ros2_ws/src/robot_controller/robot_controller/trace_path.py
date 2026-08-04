from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

env.reset(start=(2,0), goal=(4,4))

actions = [
    ("EAST", env.EAST),
    ("EAST", env.EAST),
    ("NORTH", env.NORTH),
    ("NORTH", env.NORTH),
    ("NORTH", env.NORTH),
    ("NORTH", env.NORTH),
]

print("Start:", env.robot)

for name, action in actions:

    state, reward, done = env.step(action)

    print("----------------")
    print("Move :", name)
    print("Robot:", env.robot)
    print("Reward:", reward)
    print("Done:", done)
    