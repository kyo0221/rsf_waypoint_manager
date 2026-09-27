import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    waypoint_manager_dir = get_package_share_directory('rsf_waypoint_manager')

    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value=os.path.join(
            waypoint_manager_dir, 'waypoints', 'tsudanuma2-3_wp.yaml'),
    )
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    waypoint_navigator_node = Node(
        package='rsf_waypoint_manager',
        executable='waypoint_core',
        name='waypoint_navigator',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    return LaunchDescription([
        waypoints_file_arg,
        use_sim_time_arg,
        waypoint_navigator_node,
    ])
