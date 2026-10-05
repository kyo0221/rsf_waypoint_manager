import rclpy
from nav2_msgs.msg import SpeedLimit
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient


class SectionSettings:

    def __init__(self, node, waypoints):
        self.node = node
        self.waypoints = waypoints
        self.speed_pub = node.create_publisher(SpeedLimit, 'speed_limit', 1)
        self.clients = {}
        self.defaults = {}
        self.current = None

    def client(self, remote):
        if remote not in self.clients:
            self.clients[remote] = AsyncParameterClient(self.node, remote)
        return self.clients[remote]

    def fetch_defaults(self):
        wanted = {}
        for waypoint in self.waypoints:
            for remote, params in waypoint.get('parameters', {}).items():
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

    def apply(self, index):
        waypoint = self.waypoints[index] if index is not None else {}
        target = {
            (remote, name): value
            for remote, params in waypoint.get('parameters', {}).items()
            for name, value in params.items()}
        speed = float(waypoint.get('speed_limit', 0.0))
        state = (speed, tuple(sorted(target.items())))
        if state == self.current:
            return
        previous = dict(self.current[1]) if self.current else {}
        self.current = state

        msg = SpeedLimit()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = speed
        self.speed_pub.publish(msg)

        updates = {}
        for key in previous:
            if key not in target and key in self.defaults:
                updates.setdefault(key[0], []).append(self.default_msg(key))
        for (remote, name), value in target.items():
            updates.setdefault(remote, []).append(Parameter(name, value=value).to_parameter_msg())
        for remote, params in updates.items():
            self.client(remote).set_parameters(params, callback=self.on_set(remote))
        self.node.get_logger().info(
            f'section settings for waypoint {index}: speed_limit {speed}, '
            f'parameters {dict(target) if target else "default"}')

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
