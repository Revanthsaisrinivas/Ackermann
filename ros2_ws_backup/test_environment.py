from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

env.reset(start=(2,0), goal=(4,4))

print("Robot:", env.robot)

names = {
    env.EAST: "EAST",
    env.NORTH: "NORTH",
    env.WEST: "WEST",
    env.SOUTH: "SOUTH"
}

print("\nChecking every direction:\n")

for action in env.ACTIONS:
    print(f"{names[action]} : {env.is_valid_move(action)}")

print("\nValid actions:")
print([names[a] for a in env.get_valid_actions()])
