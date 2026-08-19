from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import math

# Global configuration constants
CELL_SIZE = 1.0
GRID_WIDTH = 5
GRID_HEIGHT = 5
ROBOT_Z = 0.05

HEADING_TO_YAW = {
    0: 0.0,             # EAST
    1: math.pi / 2.0,   # NORTH
    2: math.pi,         # WEST
    3: -math.pi / 2.0   # SOUTH
}

HEADING_NAMES = {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH"}


# =========================================================
# KEYBOARD INPUT
# =========================================================
# Replaces the old `x_cell:=`/`y_cell:=`/`heading:=` ros2-launch
# command-line arguments entirely -- launch_setup() (called via
# OpaqueFunction) is just a normal Python function under the hood, so
# a blocking input() here works fine as long as this launch file is
# run from an interactive terminal, which it always has been so far.

def prompt_int(label, valid_values):
    while True:
        raw = input(label)
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if value not in valid_values:
            print(f"  Must be one of: {sorted(valid_values)}")
            continue
        return value


def launch_setup(context, *args, **kwargs):
    pkg_path = get_package_share_directory("robot_description")

    xacro_file = os.path.join(
        pkg_path,
        "urdf",
        "robot.urdf.xacro"
    )

    world_file = os.path.join(
        pkg_path,
        "worlds",
        "maze.world"
    )

    robot_description = ParameterValue(
        Command([
            "xacro",
            " ",
            xacro_file
        ]),
        value_type=str
    )

    print("=" * 50)
    print("Enter the robot's starting grid cell and heading")
    print("=" * 50)
    x_cell = prompt_int(
        f"  x_cell (0-{GRID_WIDTH - 1}): ", set(range(GRID_WIDTH))
    )
    y_cell = prompt_int(
        f"  y_cell (0-{GRID_HEIGHT - 1}): ", set(range(GRID_HEIGHT))
    )
    heading = prompt_int(
        "  heading (0=EAST, 1=NORTH, 2=WEST, 3=SOUTH): ", {0, 1, 2, 3}
    )

    # Convert grid cell indices to exact cell center world coordinates
    world_x = (x_cell + 0.5) * CELL_SIZE
    world_y = (y_cell + 0.5) * CELL_SIZE
    yaw = HEADING_TO_YAW[heading]

    # Print a clear, formatted startup message showing the full simulation configuration
    print("=" * 50)
    print("Launching Maze Simulation & Controller")
    print(f"Grid Cell : ({x_cell}, {y_cell})")
    print(f"Heading   : {heading}")
    print(f"World Pos : ({world_x:.2f}, {world_y:.2f})")
    print(f"Yaw       : {yaw:.2f} rad")
    print("=" * 50)

    gzserver = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            world_file,
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so"
        ],
        output="screen"
    )

    # Re-enabled: the Ogre AxisAlignedBox crash that made us disable
    # this was the lidar's <visualize>true</visualize> + exact-360deg
    # bug, fixed several iterations ago (visualize=false, angle
    # nudged to +/-3.13). Bringing the GUI back now because our two
    # non-visual verification methods (GetEntityState, gz model -p)
    # have both given inconsistent/unreliable answers -- direct visual
    # confirmation via gzclient is the one check that's actually been
    # trustworthy so far in this debugging process.
    gzclient = ExecuteProcess(
        cmd=["gzclient"],
        output="screen"
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }],
        output="screen"
    )

    spawn_args = [
        "-topic", "robot_description",
        "-entity", "rl_robot",
        "-x", str(world_x),
        "-y", str(world_y),
        "-z", str(ROBOT_Z),
        "-Y", str(yaw)
    ]

    # Debug: print the LITERAL argument list handed to spawn_entity.py,
    # not just the world_x/world_y variables above -- if there's ever a
    # mismatch between what we computed and what actually gets passed
    # to the Node, this is where it would show up.
    print(f"spawn_entity.py arguments: {spawn_args}")

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=spawn_args,
        output="screen"
    )

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'minimum_laser_range': 0.12,
            'maximum_laser_range': 10.0,
        }]
    )

    # PREVIOUS APPROACH (fixed timing delay before spawn_entity.py's
    # own -x/-y/-z/-Y args) reduced how often the pose got dropped but
    # didn't eliminate it -- it was still sometimes landing at 0,0,
    # and sometimes not spawning at all under load. A fixed delay is
    # fundamentally a guess about how long gzserver needs; it can
    # never be made reliable, only "usually long enough".
    #
    # NEW APPROACH: stop depending on spawn-time pose working at all.
    #   1. spawn_robot still runs after a delay (kept, harmless -- if
    #      the requested pose happens to apply at spawn time, fine).
    #   2. Once spawn_robot's PROCESS HAS ACTUALLY EXITED (meaning the
    #      entity is guaranteed to already exist -- this is an event,
    #      not a guess), a separate `ros2 service call
    #      /set_entity_state` explicitly moves the entity to
    #      the requested pose. By this point gzserver has been running
    #      and stepping for a while regardless of system load, so the
    #      race that was dropping the pose during spawn can't happen
    #      here -- the entity already exists and physics is already
    #      running before this call is even made.
    #   3. A `get_entity_state` call right after prints the entity's
    #      actual final pose to the terminal, so you can see directly
    #      whether the correction worked without needing a separate
    #      `gz model -p` check.
    SPAWN_DELAY_SEC = 8.0
    GZCLIENT_DELAY_SEC = 2.0

    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    set_pose_yaml = (
        "{state: {name: 'rl_robot', "
        f"pose: {{position: {{x: {world_x}, y: {world_y}, z: {ROBOT_Z}}}, "
        f"orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}}}, "
        "reference_frame: 'world'}}"
    )

    print(f"Will correct pose after spawn via: {set_pose_yaml}")

    fix_pose = ExecuteProcess(
        cmd=[
            "timeout", "15",
            "ros2", "service", "call",
            "/set_entity_state",
            "gazebo_msgs/srv/SetEntityState",
            set_pose_yaml
        ],
        output="screen"
    )

    # NOTE: GetEntityState's ROS response has proven unreliable for
    # verification -- in testing it returned state.name='', pose
    # (0,0,0), and twist full of nan while still reporting
    # success=True (looks like an "entity not found" fallback that
    # doesn't set success=False the way you'd expect). SetEntityState
    # itself worked fine (clean success=True with the exact requested
    # pose echoed back) -- it's specifically the read-back check that
    # can't be trusted. `gz model -p` talks to Gazebo directly rather
    # than through this ROS service wrapper, and is what we used
    # earlier in this conversation to get a trustworthy answer, so use
    # that for verification instead.
    verify_pose = ExecuteProcess(
        cmd=["gz", "model", "-m", "rl_robot", "-p"],
        output="screen"
    )

    delayed_gzclient = RegisterEventHandler(
        OnProcessStart(
            target_action=gzserver,
            on_start=[
                TimerAction(
                    period=GZCLIENT_DELAY_SEC,
                    actions=[gzclient]
                )
            ]
        )
    )

    delayed_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=gzserver,
            on_start=[
                TimerAction(
                    period=SPAWN_DELAY_SEC,
                    actions=[spawn_robot]
                )
            ]
        )
    )

    # Chained off spawn_robot's actual exit event (not a timer guess):
    # spawn_robot -> [1s settle] -> fix_pose -> verify_pose
    correct_pose_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                TimerAction(
                    period=1.0,
                    actions=[fix_pose]
                )
            ]
        )
    )

    print_final_pose = RegisterEventHandler(
        OnProcessExit(
            target_action=fix_pose,
            on_exit=[
                TimerAction(
                    period=0.5,
                    actions=[verify_pose]
                )
            ]
        )
    )

    return [
        gzserver,
        robot_state_publisher,
        slam_toolbox,
        delayed_gzclient,
        delayed_spawn,
        correct_pose_after_spawn,
        print_final_pose,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
    