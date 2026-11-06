import math
import os

import rclpy
from geometry_msgs.msg import Point
from interactive_markers import InteractiveMarkerServer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker, MarkerArray

from rsf_waypoint_manager.utils.zone_edit import fill_triangles, move_corner, outline_points, save_after_drag
from rsf_waypoint_manager.yield_logic import load_zones, zones_file_for

COLORS = {'crosswalk': (1.0, 0.85, 0.0), 'roadway': (1.0, 0.2, 0.2)}


class NoRefugeZoneEditor(Node):
    """退避禁止の範囲を rviz に出し, 4 隅のドラッグで編集して yaml に保存する"""

    def __init__(self):
        super().__init__('no_refuge_zone_editor')  # ノード名は launch の name で変える
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('no_refuge_zones_file', '')
        self.declare_parameter('zones_file', '')  # 指定があれば no_refuge_zones_file より優先
        self.declare_parameter('server_name', 'no_refuge_zone_editor')
        self.declare_parameter('fill_color', [-1.0])  # RGBA. 3 要素未満なら type ごとの色
        self.path = self.get_parameter('zones_file').value or zones_file_for(
            self.get_parameter('waypoints_file').value, self.get_parameter('no_refuge_zones_file').value)
        server_name = self.get_parameter('server_name').value
        self.fill_color = [float(v) for v in self.get_parameter('fill_color').value]
        self.zones, self.mtime = [], None
        self.pub = self.create_publisher(
            MarkerArray, f'/{server_name}/zones', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.server = InteractiveMarkerServer(self, server_name)
        self.reload(force=True)
        self.create_timer(1.0, self.reload)

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
        self.zones = load_zones(self.path) if self.path else []
        self.get_logger().info(f'{len(self.zones)} no-refuge zones loaded from {self.path}')
        self.build_handles()
        self.publish()

    def build_handles(self):
        self.server.clear()
        for zi, zone in enumerate(self.zones):
            for ci, (x, y) in enumerate(zone['points']):
                self.server.insert(self.make_handle(zi, ci, x, y, zone['type'], self.fill_color), feedback_callback=self.on_feedback)
        self.server.applyChanges()

    @staticmethod
    def make_handle(zi, ci, x, y, kind, fill_color=()):
        marker = InteractiveMarker()
        marker.header.frame_id = 'map'
        marker.name = f'{zi}_{ci}'
        marker.description = ''
        marker.scale = 1.0
        marker.pose.position.x, marker.pose.position.y = x, y
        control = InteractiveMarkerControl()
        control.name = 'move_xy'
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.orientation.w, control.orientation.y = math.sqrt(0.5), math.sqrt(0.5)  # 平面の法線を z に
        control.always_visible = True
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.5
        r, g, b = fill_color[:3] if len(fill_color) >= 3 else COLORS.get(kind, (1.0, 1.0, 1.0))
        sphere.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        control.markers.append(sphere)
        marker.controls.append(control)
        return marker

    def publish(self):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        for zi, zone in enumerate(self.zones):
            r, g, b = self.fill_color[:3] if len(self.fill_color) >= 3 else COLORS.get(zone['type'], (1.0, 1.0, 1.0))
            fill_a = self.fill_color[3] if len(self.fill_color) >= 4 else 0.25
            for ns, kind, a in (('fill', Marker.TRIANGLE_LIST, fill_a), ('outline', Marker.LINE_STRIP, 1.0)):
                m = Marker()
                m.header.frame_id = 'map'
                m.ns, m.id, m.type, m.action = ns, zi, kind, Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 1.0 if kind == Marker.TRIANGLE_LIST else 0.12
                m.color = ColorRGBA(r=r, g=g, b=b, a=a)
                pts = fill_triangles(zone['points']) if kind == Marker.TRIANGLE_LIST else outline_points(zone['points'])
                m.points = [Point(x=x, y=y, z=0.05) for x, y in pts]
                array.markers.append(m)
            t = Marker()
            t.header.frame_id = 'map'
            t.ns, t.id, t.type, t.action = 'name', zi, Marker.TEXT_VIEW_FACING, Marker.ADD
            cx = sum(p[0] for p in zone['points']) / len(zone['points'])
            cy = sum(p[1] for p in zone['points']) / len(zone['points'])
            t.pose.position.x, t.pose.position.y, t.pose.position.z = cx, cy, 1.0
            t.pose.orientation.w = 1.0
            t.scale.z = 1.0
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f'{zone["name"]} ({zone["type"]})'
            array.markers.append(t)
        self.pub.publish(array)

    def on_feedback(self, feedback):
        zi, ci = (int(v) for v in feedback.marker_name.split('_'))
        if zi >= len(self.zones):
            return
        pos = feedback.pose.position
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.zones = move_corner(self.zones, zi, ci, pos.x, pos.y)
            self.publish()
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            try:
                self.zones = save_after_drag(self.path, self.zones, zi, ci, pos.x, pos.y)
                self.mtime = self.file_mtime()
                self.get_logger().info(f'saved {self.zones[zi]["name"]} corner {ci} to {self.path}')
            except Exception as e:
                self.get_logger().error(f'failed to save zones to {self.path}: {e}')
            self.publish()


def main():
    rclpy.init()
    rclpy.spin(NoRefugeZoneEditor())


if __name__ == '__main__':
    main()
