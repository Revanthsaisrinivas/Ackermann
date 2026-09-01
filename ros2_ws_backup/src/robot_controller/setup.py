from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, [
        'package.xml',
        'robot_controller/q_table.pkl',
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
