#!/usr/bin/env python3
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator

from rsf_waypoint_manager.utils.navigation_system import WaypointSystem
from rsf_waypoint_manager.utils.waypoint_editor import WaypointEditor
from rsf_waypoint_manager.utils.visualizer import WaypointVisualizer
from rsf_waypoint_manager.utils.waypoint_data import load_waypoints


class WaypointCore(BasicNavigator):

    def __init__(self):
        super().__init__(node_name='waypoint_navigator')
        self.declare_parameter('waypoints_file', '')
        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.waypoints = load_waypoints(self.waypoints_file)
        self.visualizer = WaypointVisualizer(self)
        self.system = WaypointSystem(self, self.waypoints)
        self.editor = WaypointEditor(
            self, self.waypoints, self.waypoints_file, self.visualizer, self.system)

    def run(self):
        self.system.run()


def main():
    rclpy.init()
    node = WaypointCore()
    node.run()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
