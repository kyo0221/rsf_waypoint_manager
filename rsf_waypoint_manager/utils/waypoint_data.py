import math

import yaml
from geometry_msgs.msg import PoseStamped


def load_waypoints(path):
    with open(path) as waypoint_file:
        data = yaml.safe_load(waypoint_file) or {}

    raw_waypoints = data.get('waypoints')
    if not raw_waypoints:
        raise ValueError(f'no waypoints found in {path}')

    waypoints = []
    for waypoint_id, waypoint in enumerate(raw_waypoints):
        if not isinstance(waypoint, dict):
            raise ValueError(f'waypoint {waypoint_id} must be a mapping with x/y')
        missing = [key for key in ('x', 'y') if key not in waypoint]
        if missing:
            raise ValueError(f'waypoint {waypoint_id} is missing required keys: {missing}')
        loaded = {
            'x': float(waypoint['x']),
            'y': float(waypoint['y']),
            'yaw': float(waypoint.get('yaw', 0.0)),
            'radius': float(waypoint.get('radius', 1.0)),
            'stop': bool(waypoint.get('stop', False)),
        }
        removed = [key for key in ('speed_limit', 'parameters') if key in waypoint]
        if removed:
            raise ValueError(
                f'waypoint {waypoint_id} uses removed keys {removed}; '
                'define a profile in the profiles file and set "profile"')
        if 'profile' in waypoint:
            loaded['profile'] = str(waypoint['profile'])
        waypoints.append(loaded)
    return waypoints


def save_waypoints(path, waypoints):
    with open(path, 'w') as waypoint_file:
        yaml.safe_dump({'waypoints': waypoints}, waypoint_file, sort_keys=False)


def yaw_to_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def to_pose_stamped(waypoint, stamp):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = stamp
    pose.pose.position.x = waypoint['x']
    pose.pose.position.y = waypoint['y']
    pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quaternion(waypoint['yaw'])
    return pose
