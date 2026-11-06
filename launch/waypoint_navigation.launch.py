import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    waypoint_manager_dir = get_package_share_directory('rsf_waypoint_manager')

    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value=os.path.join(
            waypoint_manager_dir, 'waypoints', 'tsudanuma2-3_wp.yaml'),
    )
    profiles_file_arg = DeclareLaunchArgument('profiles_file', default_value='')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    waypoint_navigator_node = Node(
        package='rsf_waypoint_manager',
        executable='waypoint_core',
        name='waypoint_navigator',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'profiles_file': LaunchConfiguration('profiles_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    queue_guard_node = Node(
        package='rsf_waypoint_manager',
        executable='queue_guard',
        name='queue_guard',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    yield_guard_node = Node(
        package='rsf_waypoint_manager',
        executable='yield_guard',
        name='yield_guard',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    zone_editor_node = Node(
        package='rsf_waypoint_manager',
        executable='no_refuge_zone_editor',
        name='no_refuge_zone_editor',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    def roadside_nodes(context):
        wp = LaunchConfiguration('waypoints_file').perform(context)
        zones = wp[:-len('_wp.yaml')] + '_roadside_zones.yaml' if wp.endswith('_wp.yaml') else ''
        sim = LaunchConfiguration('use_sim_time')
        return [
            Node(
                package='rsf_waypoint_manager',
                executable='keepout_zone_mask',
                name='keepout_zone_mask',
                parameters=[{
                    'zones_file': zones,
                    'mask_topic': '/roadside_filter_mask',
                    'soft_mask_topic': '/roadside_soft_filter_mask',
                    'use_sim_time': sim,
                }],
                output='screen',
            ),
            Node(
                package='rsf_waypoint_manager',
                executable='no_refuge_zone_editor',
                name='roadside_zone_editor',
                parameters=[{
                    'zones_file': zones,
                    'server_name': 'roadside_zone_editor',
                    'fill_color': [1.0, 0.3, 0.0, 0.35],
                    'use_sim_time': sim,
                }],
                output='screen',
            ),
        ]

    return LaunchDescription([
        waypoints_file_arg,
        profiles_file_arg,
        use_sim_time_arg,
        waypoint_navigator_node,
        queue_guard_node,
        yield_guard_node,
        zone_editor_node,
        OpaqueFunction(function=roadside_nodes),
    ])
