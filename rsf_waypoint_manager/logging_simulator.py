import rclpy
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from rsf_waypoint_manager.utils.visualizer import WaypointVisualizer
from rsf_waypoint_manager.utils.waypoint_data import load_waypoints


class LoggingSimulator(Node):

    def __init__(self):
        super().__init__('logging_simulator')
        self.planning = self.declare_parameter('planning', False).value
        self.control = self.declare_parameter('control', False).value
        waypoints_file = self.declare_parameter('waypoints_file', '').value
        if waypoints_file:
            WaypointVisualizer(self).publish(load_waypoints(waypoints_file))
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.controller = ActionClient(self, FollowPath, 'follow_path')
        self.lifecycle = self.create_client(
            ManageLifecycleNodes, 'lifecycle_manager_logging_simulator/manage_nodes')
        self.started = False
        self.computing = False
        if self.planning or self.control:
            self.create_subscription(Path, '/plan', self.on_plan, 1)

    def on_plan(self, plan):
        if not self.started and self.lifecycle.service_is_ready():
            self.started = True
            self.lifecycle.call_async(
                ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP))
        if not plan.poses:
            return
        if not self.planning:
            self.follow(plan)
        elif not self.computing and self.planner.server_is_ready():
            self.computing = True
            goal = ComputePathToPose.Goal(goal=plan.poses[-1], planner_id='GridBased')
            self.planner.send_goal_async(goal).add_done_callback(self.on_accepted)

    def on_accepted(self, future):
        handle = future.result()
        if handle.accepted:
            handle.get_result_async().add_done_callback(self.on_computed)
        else:
            self.computing = False

    def on_computed(self, future):
        self.computing = False
        path = future.result().result.path
        if path.poses:
            self.follow(path)

    def follow(self, path):
        if self.control and self.controller.server_is_ready():
            self.controller.send_goal_async(FollowPath.Goal(path=path, controller_id='FollowPath'))


def main():
    rclpy.init()
    try:
        rclpy.spin(LoggingSimulator())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
