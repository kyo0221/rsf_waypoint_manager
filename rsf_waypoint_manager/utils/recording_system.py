import math

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_srvs.srv import Trigger


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WaypointRecorder(Node):

    def __init__(self):
        super().__init__('waypoint_recorder')
        self.declare_parameter('output_file', 'recorded_waypoints.yaml')
        self.declare_parameter('record_interval', 5.0)
        self.declare_parameter('min_distance', 0.5)
        self.latest_pose = None
        self.waypoints = []

        self.create_subscription(PoseWithCovarianceStamped, 'mcl_pose', self.on_pose, 10)
        self.create_timer(self.get_parameter('record_interval').value, self.record)
        self.create_service(Trigger, '~/save', self.save)

    def on_pose(self, message):
        self.latest_pose = message.pose.pose

    def record(self):
        if self.latest_pose is None:
            return

        x = self.latest_pose.position.x
        y = self.latest_pose.position.y
        if self.waypoints:
            last = self.waypoints[-1]
            if math.hypot(x - last['x'], y - last['y']) < self.get_parameter('min_distance').value:
                return

        self.waypoints.append({
            'x': round(x, 3),
            'y': round(y, 3),
            'yaw': round(quaternion_to_yaw(self.latest_pose.orientation), 3),
        })

    def save(self, request, response):
        if not self.waypoints:
            response.success = False
            response.message = 'no waypoints recorded'
            return response

        output_file = self.get_parameter('output_file').value
        with open(output_file, 'w') as waypoint_file:
            yaml.safe_dump({'waypoints': self.waypoints}, waypoint_file, sort_keys=False)

        response.success = True
        response.message = f'saved {len(self.waypoints)} waypoints to {output_file}'
        return response


def main():
    rclpy.init()
    node = WaypointRecorder()
    rclpy.spin(node)
    rclpy.try_shutdown()
