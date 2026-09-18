from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # q_table.pkl is deliberately NOT packaged. It used to be listed in
    # both package_data and data_files, so every colcon build made two
    # more copies of it; train.py then saved next to whichever copy got
    # imported while run_policy.py searched a different order, and the
    # two drifted apart. The table now lives at a single path outside
    # the workspace - see robot_controller/q_table_path.py ($RL_Q_TABLE,
    # falling back to ~/.ros/rl_ackermann/q_table.pkl).
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, [
        'package.xml',
    ]),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='revanth',
    maintainer_email='revanth@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'obstacle_avoidance = robot_controller.obstacle_avoidance:main',
        'train = robot_controller.train:main',
        'run_policy = robot_controller.run_policy:main',
    ],
},
)
