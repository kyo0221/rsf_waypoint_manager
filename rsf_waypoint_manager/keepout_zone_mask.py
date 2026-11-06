import os

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rsf_waypoint_manager.utils.zone_mask import rasterize_zones
from rsf_waypoint_manager.yield_logic import load_zones


class KeepoutZoneMask(Node):
    """走行禁止の多角形 yaml を OccupancyGrid にして costmap filter 用に出す (yaml が変われば出し直す)"""

    def __init__(self):
        super().__init__('keepout_zone_mask')
        self.declare_parameter('zones_file', '')
        self.declare_parameter('mask_topic', '/roadside_filter_mask')
        self.declare_parameter('soft_mask_topic', '')  # 空なら出さない (local costmap 用に衝突未満の値で出す)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('margin', 1.0)
        self.declare_parameter('value', 100)
        self.declare_parameter('soft_value', 97)  # nav2 の変換で cost 249 (254 の衝突にならない)
        self.declare_parameter('check_period', 1.0)
        self.path = self.get_parameter('zones_file').value
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(OccupancyGrid, self.get_parameter('mask_topic').value, qos)
        soft_topic = self.get_parameter('soft_mask_topic').value
        self.soft_pub = self.create_publisher(OccupancyGrid, soft_topic, qos) if soft_topic else None
        self.mtime = None
        self.reload(force=True)
        self.create_timer(self.get_parameter('check_period').value, self.reload)

    def file_mtime(self):
        try:
            return os.stat(self.path).st_mtime
        except OSError:
            return None

    def reload(self, force=False):
        mtime = self.file_mtime()
        if not force and mtime == self.mtime:
            return
        self.mtime = mtime
        zones = load_zones(self.path) if self.path else []
        self.publish(self.pub, zones, self.get_parameter('value').value)
        if self.soft_pub:
            self.publish(self.soft_pub, zones, self.get_parameter('soft_value').value)
        self.get_logger().info(f'{len(zones)} keepout zones from {self.path}')

    def publish(self, pub, zones, value):
        res = float(self.get_parameter('resolution').value)
        ox, oy, w, h, data = rasterize_zones(zones, res, float(self.get_parameter('margin').value), value)
        msg = OccupancyGrid()
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = res
        msg.info.width, msg.info.height = w, h
        msg.info.origin.position.x, msg.info.origin.position.y = ox, oy
        msg.info.origin.orientation.w = 1.0
        msg.data = data
        pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(KeepoutZoneMask())


if __name__ == '__main__':
    main()
