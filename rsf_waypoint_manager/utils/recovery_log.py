from action_msgs.msg import GoalStatus, GoalStatusArray
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

BEHAVIORS = ['spin', 'wait', 'drive_on_heading', 'backup']
NAMES = {
    GoalStatus.STATUS_EXECUTING: 'start',
    GoalStatus.STATUS_SUCCEEDED: 'succeeded',
    GoalStatus.STATUS_ABORTED: 'aborted',
    GoalStatus.STATUS_CANCELED: 'canceled',
}


class RecoveryLog:

    def __init__(self, node, system, path):
        self.node = node
        self.system = system
        self.path = path
        self.seen = {}
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for name in BEHAVIORS:
            node.create_subscription(
                GoalStatusArray, f'/{name}/_action/status',
                lambda msg, name=name: self.on_status(name, msg), qos)
        if self.path:
            with open(self.path, 'a') as f:
                f.write('time,event,waypoint,x,y,detail\n')

    def on_status(self, name, msg):
        for status in msg.status_list:
            key = (name, bytes(status.goal_info.goal_id.uuid))
            if self.seen.get(key) == status.status:
                continue
            self.seen[key] = status.status
            label = NAMES.get(status.status)
            if label is None:
                continue
            self.write(f'{name}_{label}', self.system.current_index())
        if len(self.seen) > 2000:
            self.seen.clear()

    def write(self, event, index, detail=''):
        position = self.system.robot_position()
        x, y = position if position is not None else (float('nan'), float('nan'))
        self.node.get_logger().info(f'event {event} waypoint {index} at ({x:.2f}, {y:.2f}) {detail}'.rstrip())
        if self.path:
            now = self.node.get_clock().now().nanoseconds * 1e-9
            with open(self.path, 'a') as f:
                f.write(f'{now:.2f},{event},{index},{x:.2f},{y:.2f},{detail}\n')
