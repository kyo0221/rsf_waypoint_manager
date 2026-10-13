from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowWaypoints
from nav2_simple_commander.robot_navigator import TaskResult
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


class WaypointFollowerClient(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.follow_waypoints_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

    def followWaypoints(self, poses):
        while not self.follow_waypoints_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'FollowWaypoints' action server not available, waiting...")
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses
        self.get_logger().info(f'Following {len(goal_msg.poses)} goals....')
        send_goal_future = self.follow_waypoints_client.send_goal_async(goal_msg, self._feedback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error(f'Following {len(poses)} waypoints request was rejected!')
            return False
        self.result_future = self.goal_handle.get_result_async()
        return True

    def cancelTask(self):
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)

    def isTaskComplete(self):
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if not self.result_future.result():
            return False
        self.status = self.result_future.result().status
        return True

    def getFeedback(self):
        return self.feedback

    def getResult(self):
        return {
            GoalStatus.STATUS_SUCCEEDED: TaskResult.SUCCEEDED,
            GoalStatus.STATUS_ABORTED: TaskResult.FAILED,
            GoalStatus.STATUS_CANCELED: TaskResult.CANCELED,
        }.get(self.status, TaskResult.UNKNOWN)

    def _feedback(self, msg):
        self.feedback = msg.feedback
