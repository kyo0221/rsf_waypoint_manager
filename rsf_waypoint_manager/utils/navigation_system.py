from enum import Enum

import rclpy
from nav2_simple_commander.robot_navigator import TaskResult
from std_srvs.srv import Trigger

from rsf_waypoint_manager.utils.section_settings import SectionSettings
from rsf_waypoint_manager.utils.waypoint_data import to_pose_stamped


class State(Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'


class WaypointSystem:

    def __init__(self, node, waypoints):
        self.node = node
        self.waypoints = waypoints
        self.state = State.IDLE
        self.start_index = 0
        self.goal_end_index = 0
        self.pause_requested = False
        self.pending_start = False
        self.pending_pause = False
        self.pending_resume = False
        self.pending_next_waypoint = False
        self.pending_previous_waypoint = False
        self.sections = SectionSettings(node, waypoints)
        node.create_service(Trigger, '~/start', self.on_start)
        node.create_service(Trigger, '~/start_waypoint_nav', self.on_start)
        node.create_service(Trigger, '~/pause', self.on_pause)
        node.create_service(Trigger, '~/resume', self.on_resume)
        node.create_service(Trigger, '~/next_waypoint', self.on_next_waypoint)
        node.create_service(Trigger, '~/prev_waypoint', self.on_previous_waypoint)

    def on_start(self, request, response):
        if self.state != State.IDLE:
            response.success = False
            response.message = 'already active'
        elif not self.node.follow_waypoints_client.server_is_ready():
            response.success = False
            response.message = 'follow_waypoints action server not available'
        else:
            self.pending_start = True
            response.success = True
        return response

    def on_pause(self, request, response):
        response.success = self.state == State.RUNNING
        if response.success:
            self.pending_pause = True
        else:
            response.message = 'already paused' if self.state == State.PAUSED else 'not running'
        return response

    def on_resume(self, request, response):
        response.success = self.state == State.PAUSED
        if response.success:
            self.request_resume()
        else:
            response.message = 'not paused' if self.state == State.RUNNING else 'not running'
        return response

    def request_resume(self):
        if self.state == State.PAUSED:
            self.pending_resume = True

    def on_next_waypoint(self, request, response):
        response.success = self.state == State.STOPPED
        if response.success:
            self.pending_next_waypoint = True
        else:
            response.message = 'not stopped at a stop waypoint'
        return response

    def request_next_waypoint(self):
        if self.state == State.STOPPED:
            self.pending_next_waypoint = True

    def on_previous_waypoint(self, request, response):
        response.success = self.state == State.STOPPED and self.start_index > 0
        if response.success:
            self.pending_previous_waypoint = True
        else:
            response.message = 'no previous waypoint is available'
        return response

    def run(self):
        while rclpy.ok():
            if self.state == State.RUNNING:
                if self.node.isTaskComplete():
                    self.handle_result()
                else:
                    self.sections.apply(self.start_index + self.current_waypoint_index())
            else:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.process_requests()

    def process_requests(self):
        if self.pending_start:
            self.pending_start = False
            self.send_from(0)
        if self.pending_pause:
            self.pending_pause = False
            self.pause_requested = True
            self.node.cancelTask()
        if self.pending_resume:
            self.pending_resume = False
            self.send_from(self.start_index)
        if self.pending_next_waypoint:
            self.pending_next_waypoint = False
            self.send_from(self.start_index)
        if self.pending_previous_waypoint:
            self.pending_previous_waypoint = False
            self.send_from(max(0, self.start_index - 1))

    def handle_result(self):
        pause_requested = self.pause_requested
        self.pause_requested = False
        self.sections.apply(None)

        if self.node.getResult() == TaskResult.SUCCEEDED:
            if (self.goal_end_index < len(self.waypoints) - 1 and
                    self.waypoints[self.goal_end_index]['stop']):
                self.start_index = self.goal_end_index + 1
                self.node.get_logger().info(
                    f'stopped at waypoint {self.goal_end_index}; call next_waypoint to continue')
                self.state = State.STOPPED
            else:
                self.node.get_logger().info('finished')
                self.state = State.IDLE
        elif pause_requested:
            self.start_index += self.current_waypoint_index()
            self.node.get_logger().info(f'paused at waypoint {self.start_index}')
            self.state = State.PAUSED
        else:
            self.node.get_logger().warn('waypoint following failed, stopping')
            self.state = State.IDLE

    def current_waypoint_index(self):
        feedback = self.node.getFeedback()
        if feedback is None:
            return 0
        remaining = len(self.waypoints) - self.start_index
        index = feedback.current_waypoint
        return index if 0 <= index < remaining else 0

    def send_from(self, index):
        if not self.node.follow_waypoints_client.server_is_ready():
            self.node.get_logger().error('follow_waypoints action server not available, stopping')
            self.state = State.IDLE
            return

        self.start_index = index
        self.goal_end_index = next(
            (waypoint_index for waypoint_index in range(index, len(self.waypoints))
             if self.waypoints[waypoint_index]['stop']),
            len(self.waypoints) - 1)
        self.node.feedback = None
        self.sections.fetch_defaults()
        self.sections.apply(index)
        stamp = self.node.get_clock().now().to_msg()
        poses = [
            to_pose_stamped(waypoint, stamp)
            for waypoint in self.waypoints[index:self.goal_end_index + 1]
        ]
        if not self.node.followWaypoints(poses):
            self.node.get_logger().error('waypoint goal was rejected')
            self.state = State.IDLE
            return
        self.state = State.RUNNING
