from glob import glob
from setuptools import find_packages, setup

package_name = 'rsf_waypoint_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/waypoints', glob('waypoints/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyo0221',
    maintainer_email='teiruyamashita@icloud.com',
    description='Waypoint navigation manager for the RSF robot.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'waypoint_core = rsf_waypoint_manager.waypoint_core:main',
            'waypoint_recorder = rsf_waypoint_manager.utils.recording_system:main',
        ],
    },
)
