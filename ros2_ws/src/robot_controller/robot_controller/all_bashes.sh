#!/usr/bin/env bash
#
# Personal cheat-sheet of the commands used to run this project.
# Not meant to be executed top to bottom - copy/paste the block you
# need into the terminal it names. Every shell starts clean, so every
# block re-sources ROS/the workspace itself.
#
# ONE knob controls which maze every section below runs against: MAZE.
# maze.world is not a special case - it goes through the exact same
# commands as maze2.world, maze3.world, or any maze added later, just
# with a different value here. Set it once per shell, at the top.
#
#   export MAZE=maze.world      # or maze2.world, maze3.world, ...
#
# Workspace root for all of this:
#   cd ~/Robotics/RL_Ackermann/ros2_ws


# ============================================================
# 0. ONE-TIME / AFTER PULLING CHANGES: BUILD
# ============================================================

cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select robot_description robot_controller


# ============================================================
# 1. RUN A MAZE (simulator + policy, two terminals)
#
# RL_MAZE_WORLD/RL_Q_TABLE are derived from MAZE the same way in every
# terminal, for every maze - never left to the defaults, so Gazebo,
# run_policy and the saved table can never disagree about the layout.
# ============================================================

# ---- terminal 1: simulator ----
cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze.world        # or maze2.world, maze3.world, ...

ros2 launch robot_description gazebo.launch.py \
    x_cell:=0 y_cell:=0 heading:=0 drive_mode:=diff world:=$MAZE

# ---- terminal 2: policy ----
cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze.world        # must match terminal 1
export RL_MAZE_WORLD="$PWD/src/robot_description/worlds/$MAZE"
export RL_Q_TABLE="$HOME/.ros/rl_ackermann/q_table_${MAZE%.world}.pkl"

python3 -m robot_controller.run_policy --start 0 0 --goal 4 4 --heading 0


# ============================================================
# 2. TRAIN A MAZE'S Q-TABLE
#
# Same MAZE knob, same derived paths. ~25s; refuses to save a table
# that fails the BFS shortest-path check on any of the 2400
# (start, goal, heading) triples. Run this once for every maze,
# including maze.world, before section 1 will have a table to load.
# ============================================================

cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze.world        # or maze2.world, maze3.world, ...
export RL_MAZE_WORLD="$PWD/src/robot_description/worlds/$MAZE"
export RL_Q_TABLE="$HOME/.ros/rl_ackermann/q_table_${MAZE%.world}.pkl"

python3 -m robot_controller.train


# ============================================================
# 3. ADDING A BRAND NEW MAZE (e.g. maze3.world)
#
# Nothing new here - it's exactly sections 2 then 1, with MAZE set to
# the new file. Listed separately only as a reminder of the order:
# train before you try to run it.
# ============================================================

# 1. Build once (only needed the first time robot_controller's source changes)
cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select robot_controller

# 2. Train (section 2 above)
export MAZE=maze3.world
export RL_MAZE_WORLD="$PWD/src/robot_description/worlds/$MAZE"
export RL_Q_TABLE="$HOME/.ros/rl_ackermann/q_table_${MAZE%.world}.pkl"
python3 -m robot_controller.train

# 3. Run it (section 1 above, same MAZE value, same two terminals)


# ============================================================
# 4. VIEWING / EXPORTING A Q-TABLE
#
# Same MAZE knob again. The export script auto-detects which world
# file to verify against from the table's own stamped layout_source,
# so only RL_Q_TABLE strictly needs to be set here - but keep deriving
# it from MAZE anyway so every section in this file stays consistent.
# ============================================================

cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze.world        
export RL_Q_TABLE="$HOME/.ros/rl_ackermann/q_table_${MAZE%.world}.pkl"

# CSV, regenerated live from the table run_policy.py actually drives on:
python3 -m robot_controller.view_q_table

# XLSX (Summary/QTable/Labels sheets, live Best_Action formulas), same
# path convention - one file per RL_Q_TABLE, never overwrites another
# maze's export:
python3 -m robot_controller.unknown_qtable_xlsx

# Open the result in Windows Excel from WSL:
explorer.exe "$(wslpath -w "${RL_Q_TABLE%.pkl}.xlsx")"

# ...or in LibreOffice Calc, staying in Linux:
libreoffice --calc "${RL_Q_TABLE%.pkl}.xlsx" &


# ============================================================
# 5. SANITY CHECKS
# ============================================================

# Confirm which q_table.pkl a command will actually read/write, and
# that colcon hasn't made stray copies of it:
python3 -c "from robot_controller.q_table_path import describe_q_table_source as d; print(d())"
find build install -name "q_table.pkl"


# ============================================================
# 6. HEADLESS TESTING (no GUI, e.g. for scripted/CI runs)
# ============================================================

cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze.world        # or maze2.world, maze3.world, ...

timeout 180 ros2 launch robot_description gazebo.launch.py \
    x_cell:=0 y_cell:=0 heading:=0 drive_mode:=diff world:=$MAZE gui:=false

# `timeout` on `ros2 launch` orphans gzserver - always clean up by exact
# name afterwards (a bare `pkill -f gazebo` also kills the shell running it):
pkill -x gzserver; pkill -x gzclient

# Check the LIDAR against the trained observation (needs Gazebo up, gui:=false is fine):
.venv-drl/bin/python -m robot_controller.sac_evaluate --check-scan


# ============================================================
# 7. KILLING A STUCK RUN
# ============================================================

pkill -x gzserver; pkill -x gzclient


# to view any q table:
cd ~/Robotics/RL_Ackermann/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

export MAZE=maze2.world
export RL_Q_TABLE="$HOME/.ros/rl_ackermann/q_table_${MAZE%.world}.pkl"

python3 -m robot_controller.unknown_qtable_xlsx

explorer.exe "$(wslpath -w "${RL_Q_TABLE%.pkl}.xlsx")"

