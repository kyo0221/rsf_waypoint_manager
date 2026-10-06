import rclpy
import yaml
from nav2_msgs.msg import SpeedLimit
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

DEFAULT = 'default'


def load_profiles(path, waypoints):
    data = {}
    if path:
        with open(path) as profile_file:
            data = yaml.safe_load(profile_file) or {}
    profiles = {DEFAULT: {'speed_limit': 0.0, 'parameters': {}}}
    for name, profile in (data.get('profiles') or {}).items():
        profile = profile or {}
        unknown = sorted(set(profile) - {'speed_limit', 'parameters'})
        if unknown:
            raise ValueError(f'profile {name} has unknown keys: {unknown}')
        profiles[name] = {
            'speed_limit': float(profile.get('speed_limit', 0.0)),
            'parameters': {
                remote: dict(params)
                for remote, params in (profile.get('parameters') or {}).items()},
        }
    missing = sorted({w['profile'] for w in waypoints if 'profile' in w} - set(profiles))
    if missing:
        raise ValueError(f'profiles not defined in {path or "profiles_file"}: {missing}')
    return profiles


class ProfileManager:

    def __init__(self, node, waypoints, path):
        self.node = node
        self.waypoints = waypoints
        self.profiles = load_profiles(path, waypoints)
        self.speed_pub = node.create_publisher(SpeedLimit, 'speed_limit', 1)
        self.profile_pub = node.create_publisher(
            String, '~/profile', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.clients = {}
        self.defaults = {}
        self.current = None

    def profile_for(self, index):
        for waypoint in reversed(self.waypoints[:index + 1]):
            if 'profile' in waypoint:
                return waypoint['profile']
        return DEFAULT

    def client(self, remote):
        if remote not in self.clients:
            self.clients[remote] = AsyncParameterClient(self.node, remote)
        return self.clients[remote]

    def fetch_defaults(self):
        wanted = {}
        for profile in self.profiles.values():
            for remote, params in profile['parameters'].items():
                wanted.setdefault(remote, set()).update(params)
        for remote, names in wanted.items():
            names = sorted(name for name in names if (remote, name) not in self.defaults)
            if not names:
                continue
            client = self.client(remote)
            if not client.wait_for_services(timeout_sec=2.0):
                self.node.get_logger().error(f'parameter services of {remote} not available')
                continue
            future = client.get_parameters(names)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            if future.result() is None:
                self.node.get_logger().error(f'failed to get parameters of {remote}')
                continue
            for name, value in zip(names, future.result().values):
                self.defaults[(remote, name)] = value

    def apply(self, name):
        if name == self.current:
            return
        profile = self.profiles[name]
        previous = self.profiles[self.current]['parameters'] if self.current else {}
        self.current = name

        msg = SpeedLimit()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = profile['speed_limit']
        self.speed_pub.publish(msg)
        self.profile_pub.publish(String(data=name))

        target = {
            (remote, param): value
            for remote, params in profile['parameters'].items()
            for param, value in params.items()}
        updates = {}
        for remote, params in previous.items():
            for param in params:
                key = (remote, param)
                if key not in target and key in self.defaults:
                    updates.setdefault(remote, []).append(self.default_msg(key))
        for (remote, param), value in target.items():
            updates.setdefault(remote, []).append(Parameter(param, value=value).to_parameter_msg())
        for remote, params in updates.items():
            self.client(remote).set_parameters(params, callback=self.on_set(remote))
        self.node.get_logger().info(
            f'profile {name}: speed_limit {profile["speed_limit"]}, '
            f'parameters {profile["parameters"] or "default"}')

    def reset(self):
        self.apply(DEFAULT)

    def default_msg(self, key):
        msg = Parameter(key[1]).to_parameter_msg()
        msg.value = self.defaults[key]
        return msg

    def on_set(self, remote):
        def callback(future):
            result = future.result()
            if result is None or not all(r.successful for r in result.results):
                self.node.get_logger().error(f'failed to set parameters on {remote}: {result}')
        return callback
