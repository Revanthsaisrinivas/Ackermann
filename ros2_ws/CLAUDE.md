# RL_Ackermann — ROS 2 + Gazebo Q-Learning maze navigation

Workspace root is `ros2_ws/`. Open **that** folder in the editor, not its parent.

## Build and run

Every shell starts clean, so chain the sourcing into each command:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && colcon build --packages-select robot_description robot_controller
```

Launch the simulator (leave running in its own terminal):

```bash
source install/setup.bash && ros2 launch robot_description gazebo.launch.py \
    x_cell:=0 y_cell:=0 heading:=0 drive_mode:=diff
```

Run the policy against it:

```bash
source install/setup.bash && python3 -m robot_controller.run_policy \
    --start 0 0 --goal 4 4 --heading 0
```

Train (long; overwrites the Q-table):

```bash
source install/setup.bash && python3 -m robot_controller.train
```

## Packages

- `robot_description` — URDF/xacro, `worlds/maze.world`, `launch/gazebo.launch.py`
- `robot_controller` — `rl_environment.py`, `q_learning.py`, `train.py`,
  `run_policy.py`, `grid_rl_controller.py`, `maze_memory.py`

## Conventions

- Grid is 5×5, `CELL_SIZE = 1.0 m`. Cell `(col, row)` centres on
  `(col + 0.5, row + 0.5)`. Nothing is at a negative coordinate.
- Actions and headings: `0=EAST 1=NORTH 2=WEST 3=SOUTH`. Ties always break
  in that order.
- RL state is the 7-tuple `(dx, dy, heading, east, north, west, south)`,
  `1 = BLOCKED`. `dx, dy` are goal-minus-robot, never absolute position.
- The executor must never read `blocked_edges`. The maze is unknown at run
  time; only live LIDAR plus remembered observations are allowed.

## Traps that have already cost time

**Do not edit `worlds/maze.world`.** The layout is fixed and its 12 inner
walls match the default `blocked_edges` in `rl_environment.py` exactly.

**There is exactly one `q_table.pkl`, and it is outside the workspace.**
`robot_controller/q_table_path.py` is the only thing that decides where:
`$RL_Q_TABLE` if set, otherwise `~/.ros/rl_ackermann/q_table.pkl`. `train.py`,
`run_policy.py` and `view_q_table.py` all call it, and `setup.py` no longer
packages the table, so colcon makes no copies. Do not reintroduce a
`os.path.dirname(__file__)` path or a candidate search — that is exactly
what let training write one file while the policy read another.

```bash
# should print the canonical path and nothing under build/ or install/
python3 -c "from robot_controller.q_table_path import describe_q_table_source as d; print(d())"
find build install -name q_table.pkl
```

A correctly trained table has ~3744 states out of 5184. The copy migrated
in from the old layout has 2456 — retrain to get back above that.
`src/robot_controller/robot_controller/q_table.pkl` is a leftover of the
old scheme, still tracked in git and no longer read by anything.

**`drive_mode` must be passed explicitly** to the launch file, or xacro
silently uses its own default and the logs won't tell you which drive
plugin loaded. `diff` pivots in place; `ackermann` cannot, and stalls every
in-place turn until the controller gains arc/K-turn primitives.

**Geometry is measured hardware, do not "tidy" it.** Wheelbase 0.213 m,
track 0.222 m, wheel radius 0.0505 m. `base_link`, `base_footprint` and the
LIDAR all sit on the **rear axle** deliberately: that is the pivot centre
with rear-wheel drive, and odometry, scan origin and rotation centre have
to coincide. Front wheels are near-frictionless (`mu = 0.02`) on purpose —
giving them grip turns this into a skid-steer whose driving moment arm
(0.1110 m) barely exceeds its resisting arm (0.1065 m), and the robot
stops being able to rotate at all.

**The LIDAR beam plane is at 0.1505 m** and the tyre tops are at 0.1010 m.
Lowering the mast makes the scanner see its own wheels.

**Wall detection thresholds are tight.** At a cell centre a wall face reads
0.45 m and an open direction reads 1.45 m. `wall_distance` sits between
them. Anything that makes the robot stop off-centre (low
`max_wheel_acceleration`, loose `position_tolerance`) pushes a real wall
past the threshold and it gets reported OPEN.

## Known-good facts to check against

- `maze.world` is fully connected; `(4,4)` is reachable from every cell.
- Optimal route `(0,0) → (4,4)` is 8 moves: E,E,E,E then N,N,N,N up the
  x=4 column.
- Dead ends are `(0,0) (2,2) (2,4) (3,4) (4,4)`.
- The memoryless greedy fallback livelocks at `(1,2)↔(2,2)` and
  `(1,4)↔(2,4)`; 6 of 24 start cells never finish. `maze_memory.py` fixes
  this — 0 failures across all 600 start/goal pairs, worst case 43 steps.

## Testing without the GUI

`gzclient` is the slow part and needs a display. The `gui` launch argument
now exists — pass `gui:=false` to run `gzserver` alone:

```bash
ros2 launch robot_description gazebo.launch.py \
    x_cell:=0 y_cell:=0 heading:=0 drive_mode:=diff gui:=false
```

Always bound automated runs, e.g. `timeout 180 ...`, since both the launch
and the policy block indefinitely on failure.

**`timeout` on `ros2 launch` orphans `gzserver`.** The launch process dies
but its child keeps the port, so the next launch fails with
`bind: Address already in use` and the robot never spawns — `/scan` simply
has no publisher. Clean up by exact name after every bounded run:

```bash
pkill -x gzserver; pkill -x gzclient
```

Use `-x`. A bare `pkill -f gazebo` also matches the shell running it, which
has already killed a wrapper mid-command in this project.

## Checking the LIDAR against the trained observation

```bash
# with Gazebo up (gui:=false is fine)
.venv-drl/bin/python -m robot_controller.sac_evaluate --check-scan
```

Measured on `maze.world`: beam count, `angle_min`, `range_min` and
`range_max` all match `robot_constants.py`. Two findings worth knowing:

- `angle_increment` is `2*pi/359`, not `2*pi/360` — Gazebo treats
  `angle_min` and `angle_max` as both inclusive, so the last beam
  duplicates the first. It accumulates to 1.0° by beam 359, which is
  0.10 of a 10° sector, below the min-pooling resolution.
- The out-of-range encoding is **unexercised here, not verified**. In a
  5 m maze every direction has a wall inside the 6 m sensor range, so no
  beam ever exceeds `range_max`. It cannot affect the policy in this
  world. Testing it needs an open world.

## Do not commit

`build/`, `install/`, `log/`.
