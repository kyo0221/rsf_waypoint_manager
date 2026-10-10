from enum import Enum
import math

import rclpy
from nav2_simple_commander.robot_navigator import TaskResult
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from rsf_waypoint_manager.utils.profile_manager import ProfileManager
from rsf_waypoint_manager.utils.waypoint_data import to_pose_stamped

PLANNER = '/planner_server'
GOAL_TOLERANCE_PARAMETER = 'GridBased.tolerance'
GOAL_TOLERANCE_MARGIN = 0.5
MIN_GOAL_TOLERANCE = 0.125


class State(Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'


class WaypointSystem:

    def __init__(self, node, waypoints, profiles_file):
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
        self.entered_goal = False
        self.result = 'idle'
        self.status = None
        self.status_pub = node.create_publisher(
            String, '~/status', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.profiles = ProfileManager(node, waypoints, profiles_file)
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
            self.pending_resume = True
        else:
            response.message = 'not paused' if self.state == State.RUNNING else 'not running'
        return response

    def on_next_waypoint(self, request, response):
        response.success = self.state == State.STOPPED
        if response.success:
            self.pending_next_waypoint = True
        else:
            response.message = 'not stopped at a stop waypoint'
        return response

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
                    index = self.start_index + self.current_waypoint_index()
                    self.profiles.apply(self.profiles.profile_for(index))
                    self.check_entered(index)
            else:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            self.process_requests()
            self.publish_status()

    def publish_status(self):
        total = len(self.waypoints)
        if self.state == State.RUNNING:
            status = f'running {self.start_index + self.current_waypoint_index()} {total}'
        elif self.state == State.STOPPED:
            status = f'stopped {self.goal_end_index} {total}'
        elif self.state == State.PAUSED:
            status = f'paused {self.start_index} {total}'
        else:
            status = f'{self.result} {self.start_index + self.current_waypoint_index()} {total}'
        if status != self.status:
            self.status = status
            self.status_pub.publish(String(data=status))

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

    def robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except Exception:
            return None
        return transform.transform.translation.x, transform.transform.translation.y

    def check_entered(self, index):
        if self.entered_goal or self.pause_requested:
            return
        position = self.robot_position()
        if position is None:
            return
        waypoint = self.waypoints[index]
        if math.hypot(waypoint['x'] - position[0], waypoint['y'] - position[1]) > waypoint['radius']:
            return
        self.node.get_logger().info(f'entered waypoint {index}')
        if index < self.goal_end_index:
            self.send_from(index + 1)
        else:
            self.entered_goal = True
            self.node.cancelTask()

    def handle_result(self):
        pause_requested = self.pause_requested
        self.pause_requested = False
        entered_goal = self.entered_goal
        self.entered_goal = False
        self.profiles.reset()

        if entered_goal or self.node.getResult() == TaskResult.SUCCEEDED:
            if (self.goal_end_index < len(self.waypoints) - 1 and
                    self.waypoints[self.goal_end_index]['stop']):
                self.start_index = self.goal_end_index + 1
                self.node.get_logger().info(
                    f'stopped at waypoint {self.goal_end_index}; call next_waypoint to continue')
                self.state = State.STOPPED
            else:
                self.node.get_logger().info('finished')
                self.result = 'finished'
                self.state = State.IDLE
        elif pause_requested:
            self.start_index += self.current_waypoint_index()
            self.node.get_logger().info(f'paused at waypoint {self.start_index}')
            self.state = State.PAUSED
        else:
            self.node.get_logger().warn('waypoint following failed, stopping')
            self.result = 'failed'
            self.state = State.IDLE

    def set_goal_tolerance(self, radius):
        tolerance = max(radius - GOAL_TOLERANCE_MARGIN, MIN_GOAL_TOLERANCE)
        self.profiles.client(PLANNER).set_parameters(
            [Parameter(GOAL_TOLERANCE_PARAMETER, value=tolerance).to_parameter_msg()],
            callback=self.profiles.on_set(PLANNER))

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
            self.result = 'failed'
            self.state = State.IDLE
            return

        self.start_index = index
        self.entered_goal = False
        self.goal_end_index = next(
            (waypoint_index for waypoint_index in range(index, len(self.waypoints))
             if self.waypoints[waypoint_index]['stop']),
            len(self.waypoints) - 1)
        self.node.feedback = None
        self.profiles.fetch_defaults()
        self.profiles.apply(self.profiles.profile_for(index))
        self.set_goal_tolerance(self.waypoints[index]['radius'])
        stamp = self.node.get_clock().now().to_msg()
        poses = [
            to_pose_stamped(waypoint, stamp)
            for waypoint in self.waypoints[index:self.goal_end_index + 1]
        ]
        if not self.node.followWaypoints(poses):
            self.node.get_logger().error('waypoint goal was rejected')
            self.result = 'failed'
            self.state = State.IDLE
            return
        self.state = State.RUNNING
