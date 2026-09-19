from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import math

# Global configuration constants
CELL_SIZE = 1.0
GRID_WIDTH = 5
GRID_HEIGHT = 5

# Spawn height for base_footprint.
#
# This was 0.30 m. base_footprint sits on the ground plane, so 0.30
# dropped the robot 0.30 m onto the floor every launch. The wheel
# encoders do not turn during a free fall, so odometry still read
# (0, 0) while the chassis bounced and skidded somewhere else - and
# wait_until_ready() latches start_world_x/y from that odometry, so
# every cell centre the controller computed afterwards was offset by
# however far the landing threw it.
#
# The wheel radius is 0.0505 m, so 0.06 m is a 10 mm settle, not a drop.
ROBOT_Z = 0.06

HEADING_TO_YAW = {
    0: 0.0,             # EAST
    1: math.pi / 2.0,   # NORTH
    2: math.pi,         # WEST
    3: -math.pi / 2.0   # SOUTH
}


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

    drive_mode = LaunchConfiguration("drive_mode").perform(context)

    if drive_mode not in ("diff", "ackermann"):
        raise ValueError(
            f"Invalid drive_mode '{drive_mode}'. Use 'diff' or 'ackermann'."
        )

    # The xacro argument has to be passed explicitly. Without it the
    # file silently falls back to its own default and you have no way
    # to tell from the logs which drive plugin actually loaded.
    robot_description = ParameterValue(
        Command([
            "xacro", " ",
            xacro_file, " ",
            "drive_mode:=", drive_mode
        ]),
        value_type=str
    )

    x_cell = int(LaunchConfiguration("x_cell").perform(context))
    y_cell = int(LaunchConfiguration("y_cell").perform(context))
    heading = int(LaunchConfiguration("heading").perform(context))
    use_slam = LaunchConfiguration("use_slam").perform(context).lower() in (
        "1", "true", "yes"
    )

    if not (0 <= x_cell < GRID_WIDTH and 0 <= y_cell < GRID_HEIGHT):
        raise ValueError(
            f"Invalid cell ({x_cell}, {y_cell}). "
            f"Valid range: x=0-{GRID_WIDTH-1}, y=0-{GRID_HEIGHT-1}"
        )

    if heading not in HEADING_TO_YAW:
        raise ValueError(
            "Heading must be one of: "
            "0=EAST, 1=NORTH, 2=WEST, 3=SOUTH"
        )

    # Grid cell indices -> exact cell centre world coordinates
    world_x = (x_cell + 0.5) * CELL_SIZE
    world_y = (y_cell + 0.5) * CELL_SIZE
    yaw = HEADING_TO_YAW[heading]

    print("=" * 50)
    print("Launching Maze Simulation & Controller")
    print(f"Drive Mode: {drive_mode}")
    print(f"Grid Cell : ({x_cell}, {y_cell})")
    print(f"Heading   : {heading}")
    print(f"World Pos : ({world_x:.2f}, {world_y:.2f})")
    print(f"Yaw       : {yaw:.2f} rad")
    print(f"Spawn Z   : {ROBOT_Z:.3f} m")
    print(f"SLAM      : {use_slam}")
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

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "rl_robot",
            "-x", str(world_x),
            "-y", str(world_y),
            "-z", str(ROBOT_Z),
            "-Y", str(yaw)
        ],
        output="screen"
    )

    # gzclient is the slow part and needs a display. Automated checks
    # (policy rollouts, sac_evaluate --check-scan) only need gzserver,
    # and skipping the GUI leaves the cores for whatever is being
    # measured.
    show_gui = context.perform_substitution(
        LaunchConfiguration("gui")
    ).lower() not in ("false", "0", "no")

    delayed = [spawn_robot]
    if show_gui:
        delayed.insert(0, gzclient)

    actions = [
        gzserver,
        robot_state_publisher,
        TimerAction(
            period=2.0,
            actions=delayed
        ),
    ]

    # SLAM is off by default. The RL controller navigates purely from
    # raw /scan plus wheel odometry and never consults a map, so
    # slam_toolbox only burns CPU here. Turn it on with use_slam:=true
    # if you want the map for visualisation.
    if use_slam:
        actions.insert(2, Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[{"use_sim_time": True}]
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "x_cell",
            default_value="0",
            description="Grid X Cell"
        ),
        DeclareLaunchArgument(
            "y_cell",
            default_value="0",
            description="Grid Y Cell"
        ),
        DeclareLaunchArgument(
            "heading",
            default_value="0",
            description="0=EAST, 1=NORTH, 2=WEST, 3=SOUTH"
        ),
        DeclareLaunchArgument(
            "drive_mode",
            default_value="diff",
            description="diff = pivot-capable differential; ackermann = car-like"
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="false runs gzserver only - headless, no display "
                        "needed, and leaves the cores free for the run "
                        "being measured"
        ),
        DeclareLaunchArgument(
            "use_slam",
            default_value="false",
            description="Run slam_toolbox alongside (not needed by the RL controller)"
        ),
        OpaqueFunction(function=launch_setup)
    ])
