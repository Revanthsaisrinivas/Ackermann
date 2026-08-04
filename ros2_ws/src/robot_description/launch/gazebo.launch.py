from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import math

# Global configuration constants
CELL_SIZE = 1.0
GRID_WIDTH = 5
GRID_HEIGHT = 5
ROBOT_Z = 0.30

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

    robot_description = ParameterValue(
        Command([
            "xacro",
            " ",
            xacro_file
        ]),
        value_type=str
    )

    # Read and validate grid cell arguments and heading dynamically at launch time using OpaqueFunction
    x_cell = int(LaunchConfiguration("x_cell").perform(context))
    y_cell = int(LaunchConfiguration("y_cell").perform(context))
    heading = int(LaunchConfiguration("heading").perform(context))

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

    slam_toolbox = Node(
    package='slam_toolbox',
    executable='async_slam_toolbox_node',
    name='slam_toolbox',
    output='screen',
    parameters=[{
        'use_sim_time': True
    }]
    )
    
    return [
        gzserver,
        robot_state_publisher,
        slam_toolbox,
        TimerAction(
            period=2.0,
            actions=[
                gzclient,
                spawn_robot
            ]
        ),
    ]


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
        OpaqueFunction(function=launch_setup)
    ])