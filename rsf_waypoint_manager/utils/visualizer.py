import math

from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
    MarkerArray,
)
from rclpy.qos import DurabilityPolicy, QoSProfile

from rsf_waypoint_manager.utils.waypoint_data import yaw_to_quaternion


class WaypointVisualizer:

    def __init__(self, node):
        self.node = node
        self.publisher = node.create_publisher(
            MarkerArray, '~/waypoints',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def create_marker(self, waypoint, waypoint_id):
        marker = InteractiveMarker()
        marker.header.frame_id = 'map'
        marker.name = f'waypoint_{waypoint_id}'
        marker.description = f'Waypoint {waypoint_id}'
        marker.scale = max(2.0 * waypoint['radius'], 1.0)
        marker.pose.position.x = waypoint['x']
        marker.pose.position.y = waypoint['y']
        marker.pose.orientation.z, marker.pose.orientation.w = yaw_to_quaternion(
            waypoint['yaw'])

        plane_control = self.make_vertical_axis_control(
            'move', InteractiveMarkerControl.MOVE_PLANE, 'Drag to move waypoint')
        plane_control.orientation_mode = InteractiveMarkerControl.INHERIT
        plane_control.always_visible = True
        plane_control.markers = self.create_markers(waypoint, waypoint_id)
        marker.controls.append(plane_control)

        rotate_control = self.make_vertical_axis_control(
            'rotate', InteractiveMarkerControl.ROTATE_AXIS, 'Drag to rotate waypoint')
        marker.controls.append(rotate_control)

        resize_control = InteractiveMarkerControl()
        resize_control.name = 'resize'
        resize_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        resize_control.orientation_mode = InteractiveMarkerControl.FIXED
        resize_control.orientation.z = math.sqrt(0.5)
        resize_control.orientation.w = math.sqrt(0.5)
        resize_control.description = 'Drag axis to resize waypoint'
        marker.controls.append(resize_control)
        return marker

    def create_markers(self, waypoint, waypoint_id):
        markers = []
        cylinder = Marker()
        cylinder.type = Marker.CYLINDER
        cylinder.pose.orientation.w = 1.0
        cylinder.scale.x = cylinder.scale.y = 2.0 * waypoint['radius']
        cylinder.scale.z = 0.01
        cylinder.color.r = 1.0 if waypoint['stop'] else 0.0
        cylinder.color.g = 0.0 if waypoint['stop'] else 1.0
        cylinder.color.a = 0.6
        markers.append(cylinder)

        arrow = Marker()
        arrow.type = Marker.ARROW
        arrow.pose.orientation.w = 1.0
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 1.0, 0.1, 0.1
        arrow.color.r = 1.0
        arrow.color.a = 0.7
        markers.append(arrow)

        label = Marker()
        label.type = Marker.TEXT_VIEW_FACING
        label.pose.orientation.w = 1.0
        label.scale.x = label.scale.y = label.scale.z = 1.0
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.pose.position.z = 1.5
        label.text = str(waypoint_id)
        markers.append(label)
        return markers

    def publish(self, waypoints):
        marker_array = MarkerArray()
        stamp = self.node.get_clock().now().to_msg()
        for waypoint_id, waypoint in enumerate(waypoints):
            qz, qw = yaw_to_quaternion(waypoint['yaw'])
            for marker in self.create_markers(waypoint, waypoint_id):
                marker.header.frame_id = 'map'
                marker.header.stamp = stamp
                marker.ns = 'waypoints'
                marker.id = len(marker_array.markers)
                marker.action = Marker.ADD
                marker.pose.position.x = waypoint['x']
                marker.pose.position.y = waypoint['y']
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw
                marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

    @staticmethod
    def make_vertical_axis_control(name, interaction_mode, description):
        control = InteractiveMarkerControl()
        control.name = name
        control.interaction_mode = interaction_mode
        control.orientation_mode = InteractiveMarkerControl.FIXED
        control.orientation.y = math.sqrt(0.5)
        control.orientation.w = math.sqrt(0.5)
        control.description = description
        return control
