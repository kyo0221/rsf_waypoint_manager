from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value='recorded_waypoints.yaml',
    )
    record_interval_arg = DeclareLaunchArgument('record_interval', default_value='5.0')
    min_distance_arg = DeclareLaunchArgument('min_distance', default_value='0.5')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    waypoint_recorder_node = Node(
        package='rsf_waypoint_manager',
        executable='waypoint_recorder',
        name='waypoint_recorder',
        parameters=[{
            'output_file': LaunchConfiguration('output_file'),
            'record_interval': ParameterValue(
                LaunchConfiguration('record_interval'), value_type=float),
            'min_distance': ParameterValue(
                LaunchConfiguration('min_distance'), value_type=float),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
    )

    return LaunchDescription([
        output_file_arg,
        record_interval_arg,
        min_distance_arg,
        use_sim_time_arg,
        waypoint_recorder_node,
    ])
