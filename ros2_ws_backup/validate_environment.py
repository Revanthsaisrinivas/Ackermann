from robot_controller.rl_environment import MazeEnvironment

env = MazeEnvironment()

names = {
    env.EAST: "EAST",
    env.NORTH: "NORTH",
    env.WEST: "WEST",
    env.SOUTH: "SOUTH"
}

print("=" * 70)
print("VALIDATING ENTIRE MAZE")
print("=" * 70)

for y in reversed(range(env.GRID_SIZE)):
    for x in range(env.GRID_SIZE):

        env.reset(start=(x, y), goal=(4, 4))

        print(f"\nCELL ({x},{y})")

        valid = env.get_valid_actions()

        if not valid:
            print("  No valid actions!")
            continue

        for action in valid:

            start = env.robot

            next_state, reward, done = env.step(action)

            print(
                f"  {names[action]:6} : "
                f"{start} -> {env.robot} "
                f"Reward={reward}"
            )

            # Restore state
            env.robot = start
            env.visited = {start}
            env.current_steps = 0
