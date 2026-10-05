import math

import rclpy
from interactive_markers import InteractiveMarkerServer, MenuHandler
from std_srvs.srv import Trigger
from visualization_msgs.msg import InteractiveMarkerFeedback

from rsf_waypoint_manager.utils.waypoint_data import save_waypoints


class WaypointEditor:

    def __init__(self, node, waypoints, waypoints_file, visualizer):
        self.node = node
        self.waypoints = waypoints
        self.waypoints_file = waypoints_file
        self.visualizer = visualizer
        self.server_node = rclpy.create_node('waypoint_editor', use_global_arguments=False)
        rclpy.get_global_executor().add_node(self.server_node)
        self.server = InteractiveMarkerServer(self.server_node, 'waypoint_editor')
        self.drag_state = {}
        node.create_service(Trigger, '~/save_waypoint', self.save)
        self.refresh()

    def refresh(self):
        self.server.clear()
        self.visualizer.publish(self.waypoints)
        for waypoint_id, waypoint in enumerate(self.waypoints):
            marker = self.visualizer.create_marker(waypoint, waypoint_id)
            self.server.insert(marker, feedback_callback=self.on_feedback)
            menu = MenuHandler()
            menu.insert('Add waypoint', callback=self.add_waypoint)
            menu.insert('Remove waypoint', callback=self.remove_waypoint)
            stop = menu.insert('Stop waypoint', callback=self.toggle_stop)
            menu.setCheckState(
                stop, MenuHandler.CHECKED if waypoint['stop'] else MenuHandler.UNCHECKED)
            menu.apply(self.server, marker.name)
        self.server.applyChanges()

    def on_feedback(self, feedback):
        waypoint_id = int(feedback.marker_name.split('_')[-1])
        waypoint = self.waypoints[waypoint_id]

        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.drag_state[feedback.marker_name] = {
                'y': feedback.pose.position.y,
                'radius': waypoint['radius'],
            }
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.update_waypoint(waypoint, feedback)
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.update_waypoint(waypoint, feedback)
            self.drag_state.pop(feedback.marker_name, None)
            self.refresh()

    def update_waypoint(self, waypoint, feedback):
        if feedback.control_name == 'resize':
            drag = self.drag_state.get(feedback.marker_name)
            if drag is None:
                drag = {'y': feedback.pose.position.y, 'radius': waypoint['radius']}
                self.drag_state[feedback.marker_name] = drag
            waypoint['radius'] = min(
                5.0, max(0.1, drag['radius'] + feedback.pose.position.y - drag['y']))
            return

        waypoint['x'] = feedback.pose.position.x
        waypoint['y'] = feedback.pose.position.y
        orientation = feedback.pose.orientation
        waypoint['yaw'] = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))

    def add_waypoint(self, feedback):
        waypoint_id = int(feedback.marker_name.split('_')[-1])
        waypoint = self.waypoints[waypoint_id]
        self.waypoints.insert(waypoint_id + 1, {
            'x': waypoint['x'] + math.cos(waypoint['yaw']),
            'y': waypoint['y'] + math.sin(waypoint['yaw']),
            'yaw': waypoint['yaw'],
            'radius': waypoint['radius'],
            'stop': False,
        })
        self.refresh()

    def remove_waypoint(self, feedback):
        if len(self.waypoints) > 1:
            del self.waypoints[int(feedback.marker_name.split('_')[-1])]
            self.refresh()

    def toggle_stop(self, feedback):
        waypoint = self.waypoints[int(feedback.marker_name.split('_')[-1])]
        waypoint['stop'] = not waypoint['stop']
        self.refresh()

    def save(self, request, response):
        save_waypoints(self.waypoints_file, self.waypoints)
        response.success = True
        response.message = f'saved {len(self.waypoints)} waypoints to {self.waypoints_file}'
        self.node.get_logger().info(response.message)
        return response
