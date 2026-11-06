import math
import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from nav2_msgs.action import BackUp, NavigateToPose, Spin
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy import ndimage
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from rsf_waypoint_manager.yield_logic import (
    align_angle, backup_distance, blocked_ahead, choose_fixed_refuge, cluster_points, find_leader, find_retreat, find_oncoming, find_refuge, follow_speed,
    LeaderMonitor, near_stop_ahead, next_waypoint_index, obstacle_layer_action, points_near, predict_track, route_lane,
    should_resend_refuge, should_skip_waypoint, smooth_limit, stalled, Tracker, tracks_in_frame, YieldStateMachine)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class YieldGuard(Node):

    def __init__(self):
        super().__init__('yield_guard')
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('odom_frame', 'rsf_odom')
        self.declare_parameter('target_hold', 10.0)
        self.declare_parameter('follow_half_width', 0.6)
        self.declare_parameter('follow_min_speed', 0.15)
        self.declare_parameter('follow_min_travel', 0.6)
        self.declare_parameter('follow_standoff', 3.5)  # 2.0 だと前走のコストを避けて左右に振れる
        self.declare_parameter('follow_time_gap', 1.5)
        self.declare_parameter('follow_max_speed', 1.0)
        self.declare_parameter('leader_hold', 30.0)
        self.declare_parameter('leader_release', 10.0)
        self.declare_parameter('enabled', False)
        self.declare_parameter('follow_enabled', True)
        # 追従中に global の障害物層を切る (前走を避けて歩行者側を回る経路を防ぐ). 地図に無いコーンも経路計画から見えなくなる点に注意
        self.declare_parameter('follow_layer_off', True)
        self.declare_parameter('half_width', 1.2)
        self.declare_parameter('ahead', 18.0)
        self.declare_parameter('min_ahead', 2.5)
        self.declare_parameter('min_speed', 0.3)
        self.declare_parameter('min_travel', 0.6)
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('clear_time', 3.0)
        self.declare_parameter('confirm_time', 0.5)
        self.declare_parameter('max_wait', 20.0)
        self.declare_parameter('cooldown', 5.0)
        # その場の探索: 前方斜め前 (前方 refuge_ahead[m] まで, 左右 refuge_lateral[m]) を refuge_step 刻み
        self.declare_parameter('refuge_ahead', 3.0)
        self.declare_parameter('refuge_lateral', [0.5, 2.0])  # 0.5 からにしてレーン内の小さな寄りも使う (ポールの間など)
        self.declare_parameter('refuge_step', 0.5)
        self.declare_parameter('refuge_min_sep', 1.0)  # 相手の横位置からこれ未満の点は相手の進路から外れきれない
        self.declare_parameter('skip_blocked_waypoint', True)
        self.declare_parameter('unstick_enabled', True)  # 正面を塞がれて 8 秒動けなければ下がって走り直す
        # 固定の退避点 (空なら waypoints_file の <name>_wp.yaml に対する <name>_refuge_points.yaml)
        self.declare_parameter('refuge_points_file', '')
        self.declare_parameter('refuge_behind', 3.0)  # 前方と真横に無いときだけ後方を探す範囲
        self.declare_parameter('refuge_fixed_ahead', 5.0)
        self._refuge_points_path, self._refuge_points = None, []
        self.declare_parameter('refuge_max_cost', 98)  # local costmap の値 (0-100). 99 以上はロボット中心が置けない
        self.declare_parameter('static_margin', 0.4)
        self.declare_parameter('min_height', 0.3)
        self.declare_parameter('max_height', 2.0)
        self.route, self.stops, self.profiles = [], [], []
        path = self.get_parameter('waypoints_file').value
        if path:
            with open(path) as f:
                waypoints = yaml.safe_load(f)['waypoints']
            self.route = [(float(w['x']), float(w['y'])) for w in waypoints]
            self.stops = [bool(w.get('stop', False)) for w in waypoints]
            self.profiles = [w.get('profile') for w in waypoints]
        self.tracker = Tracker(window=2.0, min_span=1.0)
        self.machine = YieldStateMachine(
            self.get_parameter('clear_time').value, self.get_parameter('max_wait').value,
            self.get_parameter('cooldown').value, confirm_time=self.get_parameter('confirm_time').value)
        self.oncoming = None
        self.decision_lane = None
        self.target_id = None
        self.target_last = None
        self.leader_last = None
        self.limiting = False
        self.leader_monitor = LeaderMonitor(self.get_parameter('leader_release').value, 0.15)
        self.last_odom_pose = None
        self.yielded_ids = set()
        self.waiting_for = False
        self.lane = None
        self.robot_s = 0.0
        self.robot_xy = None
        self.nav_status = ''
        self.paused_index = None
        self.static = None
        self.goal_handle = None
        self.goal_done = False
        self.refuge, self.refuge_retries, self.refuge_sent = None, 0, 0.0
        self.buf = Buffer()
        TransformListener(self.buf, self)
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            PointCloud2, '/ground_segmentation/obstacle_points', self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(String, '/waypoint_navigator/status', self.on_status, transient)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, transient)
        self.local_costmap = None
        self.keepout = None  # 歩行者側の帯 (keepout マスク). 待機点は歩行者側を優先する
        self.create_subscription(OccupancyGrid, '/keepout_filter_mask', lambda m: setattr(self, 'keepout', m), transient)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.on_local_costmap, 1)
        self.pause_cli = self.create_client(Trigger, '/waypoint_navigator/pause')
        self.start_cli = self.create_client(Trigger, '/waypoint_navigator/start')
        self.param_cli = self.create_client(SetParameters, '/waypoint_navigator/set_parameters')
        self.global_param_cli = self.create_client(SetParameters, '/global_costmap/global_costmap/set_parameters')
        self.layer_off = False
        self.decision_xy = None
        self.limit_value, self.limit_time = None, 0.0
        self.target_index = None
        self.skipped_index, self.skip_to = None, None
        self.blocked_wp = (None, 0.0, 0.0)
        self.obstacle_xy = np.zeros((0, 2))
        self.last_following = -math.inf
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin = ActionClient(self, Spin, 'spin')
        self.backup = ActionClient(self, BackUp, 'backup')
        self.reverse_pub = self.create_publisher(Twist, '/cmd_vel_smoothed', 10)
        self.motion, self.unstick_index, self.held = [], None, (0.0, False)
        self.last_backup, self.backup_dist = None, 0.4
        self.create_subscription(Float32, '/waypoint_navigator/speed_override', self.on_override, 10)
        self.state_pub = self.create_publisher(String, '~/state', transient)
        self.oncoming_pub = self.create_publisher(Bool, '~/oncoming_ahead', 10)
        self.blocker = None
        self.speed_pub = self.create_publisher(Float32, '/waypoint_navigator/speed_override', 10)
        self.create_timer(0.1, self.tick)

    def on_status(self, msg):
        words = msg.data.split()
        self.nav_status = words[0] if words else ''
        if self.nav_status not in ('running', 'paused'):
            # 走行が終わった / 止められた: 次の走行では同じ waypoint をまた飛ばせるようにする
            self.skipped_index = None
        if len(words) > 1 and words[1].isdigit():
            self.target_index = int(words[1])
            # 飛ばした waypoint より手前から走り直したら (やり直し等) 同じ waypoint をまた飛ばせるようにする
            if self.skipped_index is not None and not (
                    self.skipped_index <= self.target_index <= self.skipped_index + 1):
                self.skipped_index = None
        if self.nav_status == 'paused' and len(words) > 1:
            self.paused_index = int(words[1])
            if self.unstick_index is not None and self.unstick_index >= 0:
                self.back_up()
            if self.skip_to is not None:
                # 一時停止の取り消し処理が終わる前に再開を送ると新しい追従の要求が失われるので 0.5 秒おく
                index, self.skip_to = self.skip_to, None
                self.restart_timer = self.create_timer(0.5, lambda i=index: self.restart_once(i))

    def on_map(self, msg):
        grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width) >= 65
        cells = max(1, math.ceil(self.get_parameter('static_margin').value / msg.info.resolution))
        self.static = (msg.info, ndimage.binary_dilation(grid, iterations=cells))

    def lookup(self, target, source, stamp=None):
        tf = self.buf.lookup_transform(target, source, stamp or rclpy.time.Time())
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_of(q)

    def on_cloud(self, msg):
        if not self.route:
            return
        odom = self.get_parameter('odom_frame').value
        try:
            rx, ry, _ = self.lookup('map', 'base_footprint')
            ox, oy, oyaw = self.lookup('map', msg.header.frame_id)
            map_odom = self.lookup('map', odom)
            odom_map = self.lookup(odom, 'map')
        except Exception:
            return
        ahead = self.get_parameter('ahead').value
        half = self.get_parameter('half_width').value
        lane = route_lane(self.route, rx, ry, ahead + 2.0, hint=self.target_index)
        robot_s, _ = lane.project(rx, ry)
        pts = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if msg.header.frame_id == 'base_footprint':
            low, high = self.get_parameter('min_height').value, self.get_parameter('max_height').value
            pts = pts[(pts[:, 2] >= low) & (pts[:, 2] <= high)]
        if len(pts):
            c, s = math.cos(oyaw), math.sin(oyaw)
            xy = np.column_stack([ox + c * pts[:, 0] - s * pts[:, 1], oy + s * pts[:, 0] + c * pts[:, 1]])
            near = np.hypot(xy[:, 0] - rx, xy[:, 1] - ry) < ahead + 2.0
            xy = xy[near]
            if len(xy):
                ps, pl = lane.project_many(xy)
                xy = xy[(np.abs(pl) < half + 0.5) & (ps > robot_s - 2.0) & (ps < robot_s + ahead + 1.0)]
            if len(xy) and self.static is not None:
                info, mask = self.static
                mi = np.floor((xy[:, 0] - info.origin.position.x) / info.resolution).astype(int)
                mj = np.floor((xy[:, 1] - info.origin.position.y) / info.resolution).astype(int)
                ok = (mi >= 0) & (mi < info.width) & (mj >= 0) & (mj < info.height)
                hit = np.zeros(len(xy), dtype=bool)
                hit[ok] = mask[mj[ok], mi[ok]]
                xy = xy[~hit]
        else:
            xy = np.zeros((0, 2))
        now = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.reset_on_odom_jump(now, odom)
        # 追跡は odom 系で行う (自己位置の補正で map 系の位置が滑っても静止物が動いて見えない)
        c, s = math.cos(odom_map[2]), math.sin(odom_map[2])
        xy_odom = np.column_stack([
            odom_map[0] + c * xy[:, 0] - s * xy[:, 1], odom_map[1] + s * xy[:, 0] + c * xy[:, 1]])
        odom_tracks = self.tracker.update(now, cluster_points(xy_odom))
        tracks = tracks_in_frame(odom_tracks, *map_odom)
        self.lane, self.robot_s, self.robot_xy = lane, robot_s, (rx, ry)
        self.obstacle_xy = xy
        oncoming = find_oncoming(
            lane, robot_s, [tr for tr in tracks if tr.id not in self.yielded_ids], half, ahead,
            self.get_parameter('min_speed').value, self.get_parameter('min_hits').value,
            self.get_parameter('min_ahead').value, now, min_travel=self.get_parameter('min_travel').value)
        self.oncoming = oncoming
        for tr in odom_tracks:
            if tr.id == self.target_id:
                self.target_last = tr
        self.update_leader(lane, robot_s, tracks, odom_tracks, map_odom, now)
        self.skip_blocked_waypoint(xy, rx, ry)
        self.check_stall(xy, rx, ry, now)
        self.waiting_for = False
        if self.target_last is not None:
            last = tracks_in_frame([self.target_last], *map_odom)[0]
            self.blocker = ((last.x, last.y), now)
            s, lateral = lane.project(last.x, last.y)
            ahead_of_robot = s > robot_s and abs(lateral) <= 2.0 * half
            if now - self.target_last.t <= self.get_parameter('target_hold').value:
                self.waiting_for = ahead_of_robot
            elif ahead_of_robot and self.machine.state == 'waiting' and points_near(xy, (last.x, last.y), 0.8):
                # 追跡が途切れて (ID が変わる等) 見失っても, 最後に見えた位置に物体が残っていれば相手はまだいる
                self.waiting_for = True
        if self.machine.state == 'waiting' and not self.waiting_for:
            # 待機中は前方 10m 以内を 0.1m/s 以上で近づく物体を相手として追い直す (旋回での ID 変化や相手の減速で再開しないため)
            slow = find_oncoming(lane, robot_s, tracks, 2.0 * half, 10.0, 0.1,
                                 self.get_parameter('min_hits').value, now=now)
            if slow is not None:
                # target_last は odom 座標で持つ (上で map 座標へ直して使う)
                self.target_id = slow.id
                self.target_last = next(tr for tr in odom_tracks if tr.id == slow.id)
                self.waiting_for = True

    def reset_on_odom_jump(self, now, odom):
        """odom 上のロボット位置が飛んだら (LIO のリセット等) 追跡履歴が静止物を動いて見せるので全て捨てる"""
        try:
            x, y, _ = self.lookup(odom, 'base_footprint')
        except Exception:
            return
        last = self.last_odom_pose
        self.last_odom_pose = (now, x, y)
        if last is None or now - last[0] > 1.0 or math.hypot(x - last[1], y - last[2]) < 1.0 + 2.0 * (now - last[0]):
            return
        self.get_logger().warn('odometry jumped, resetting tracks')
        self.tracker.clear()
        self.leader_last = None
        self.target_last = None

    def update_leader(self, lane, robot_s, tracks, odom_tracks, map_odom, now):
        """同じ向きに進む前方の相手を追い越さずに追従する. 一度追従した相手は止まっても最後の位置の後ろで待つ"""
        half = self.get_parameter('follow_half_width').value
        candidates = [tr for tr in tracks if tr.id not in self.leader_monitor.released]
        leader = find_leader(lane, robot_s, candidates, half, self.get_parameter('ahead').value,
                             self.get_parameter('follow_min_speed').value, self.get_parameter('min_hits').value, now,
                             min_travel=self.get_parameter('follow_min_travel').value)
        if leader is not None:
            if self.leader_last is None or self.leader_last.id != leader.id:
                self.get_logger().info(f'following object at {leader.s - robot_s:.1f} m')
            self.leader_last = next(tr for tr in odom_tracks if tr.id == leader.id)
        elif self.leader_last is not None:
            seen = next((tr for tr in odom_tracks if tr.id == self.leader_last.id), None)
            if seen is not None:
                self.leader_last = seen
        gap = None
        if self.leader_last is not None and now - self.leader_last.t <= self.get_parameter('leader_hold').value:
            # 検知が一瞬途切れても最後の速度で位置を進める (途切れるたびに上限と経路が切り替わると蛇行する)
            last = tracks_in_frame([predict_track(self.leader_last, now, 3.0)], *map_odom)[0]
            s, lateral = lane.project(last.x, last.y)
            h = lane.heading(s)
            moving = last.vx * math.cos(h) + last.vy * math.sin(h)
            if s > robot_s and abs(lateral) <= 2.0 * half and self.leader_monitor.update(now, last.id, moving):
                gap = s - robot_s
            elif s > robot_s and last.id in self.leader_monitor.released:
                self.get_logger().info(f'leader at {s - robot_s:.1f} m has not moved, stop following')
        if gap is None:
            self.leader_last = None
        enabled = self.get_parameter('follow_enabled').value and self.machine.state == 'idle'
        if gap is not None and enabled:
            target = follow_speed(
                gap, self.get_parameter('follow_standoff').value, self.get_parameter('follow_time_gap').value,
                self.get_parameter('follow_max_speed').value)
            dt = now - self.limit_time if self.limit_value is not None else 0.0
            # 上限を急に上げ下げすると低速の MPPI が旋回で調整して蛇行するので, 変化をなめらかにする
            self.limit_value = smooth_limit(self.limit_value, target, max(0.0, min(dt, 0.5)), 0.3, 1.0)
            self.limit_time = now
            self.speed_pub.publish(Float32(data=max(0.001, self.limit_value)))
            self.limiting = True
        elif self.limiting:
            self.speed_pub.publish(Float32(data=0.0))
            self.limiting = False
            self.limit_value = None
        self.update_obstacle_layer()

    @staticmethod
    def set_parameter(client, name, ptype, **value):
        request = SetParameters.Request()
        request.parameters = [ParameterMsg(name=name, value=ParameterValue(type=ptype, **value))]
        return client.call_async(request)

    def set_bool(self, client, name, value):
        return self.set_parameter(client, name, ParameterType.PARAMETER_BOOL, bool_value=bool(value))

    def set_start_index(self, index):
        future = self.set_parameter(self.param_cli, 'start_index', ParameterType.PARAMETER_INTEGER, integer_value=int(index))
        future.add_done_callback(lambda _: self.start_cli.call_async(Trigger.Request()))

    def update_obstacle_layer(self):
        """動いている前走ロボットを global の障害物層に載せると追い越す経路を引くので, 追従中は層を切る"""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.limiting:
            self.last_following = now
        # 追跡が一瞬切れただけで層を戻すと, その間に前走を避ける経路が引かれて左右に振れるので 6 秒待つ
        following = now - self.last_following < 6.0
        if not self.get_parameter('follow_layer_off').value and not self.layer_off:
            return
        action = obstacle_layer_action(following and self.get_parameter('follow_layer_off').value, self.layer_off,
                                       self.get_parameter('follow_enabled').value)
        if action == 'forget':
            self.layer_off = False
        elif action is not None:
            self.layer_off = action == 'disable'
            self.get_logger().info(f'global obstacle layer {"off" if self.layer_off else "on"} (following)')
            self.set_bool(self.global_param_cli, 'obstacle_layer.enabled', not self.layer_off)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.machine.state == 'moving' and not self.goal_done and now - self.machine.since > 15.0:
            # 待機点に向かえない (物体に塞がれた等) ときはその場で待つ
            self.get_logger().warn('could not reach refuge, waiting in place')
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            self.goal_done = True
        action = self.machine.step(
            now, enabled=self.get_parameter('enabled').value, running=self.nav_status == 'running',
            paused=self.nav_status == 'paused', oncoming=self.oncoming is not None, goal_done=self.goal_done,
            waiting_for=self.waiting_for)
        if action == 'pause':
            self.decision_lane = self.lane
            self.decision_xy = (self.oncoming.x, self.oncoming.y)
            self.target_id = self.oncoming.id
            self.target_last = None
            self.yielded_ids.add(self.oncoming.id)
            self.get_logger().info(f'oncoming object at {self.oncoming.s - self.robot_s:.1f} m, yielding')
            self.pause_cli.call_async(Trigger.Request())
        elif action == 'goto':
            self.goto_refuge()
        elif action == 'resume':
            self.target_id = None
            self.target_last = None
            if self.goal_handle is None:
                self.leave_refuge()
            else:
                # 待機点へのゴールの取り消し (MPPI の reset) と再開時のパラメータ変更 (同じ reset) が重ならないよう, 取り消しの完了から 0.5 秒おく
                self.goal_handle.cancel_goal_async().add_done_callback(lambda _: self.after(0.5, self.leave_refuge))
                self.goal_handle = None
        self.state_pub.publish(String(data=self.machine.state))
        self.oncoming_pub.publish(Bool(data=self.oncoming_ahead(now)))

    def oncoming_ahead(self, now):
        """待避の原因になった対向ロボットが, 最後に見えた位置 (前方 8m 以内のレーン上) にまだ残っていれば True"""
        if self.blocker is None or self.lane is None or self.obstacle_xy is None or now - self.blocker[1] > 60.0:
            return False
        s, lateral = self.lane.project(*self.blocker[0])
        return (0.0 < s - self.robot_s <= 8.0 and abs(lateral) <= 1.5
                and points_near(self.obstacle_xy, self.blocker[0], 1.0))

    def after(self, delay, callback):
        timer = None

        def fire():
            timer.cancel()
            self.destroy_timer(timer)
            callback()
        timer = self.create_timer(delay, fire)

    def leave_refuge(self):
        if not self.turn_to_exit():
            self.continue_route()

    def turn_to_exit(self):
        """待機点から再開する前に, レーンへ戻る向き (2m 先のレーン中央) へその場で旋回する. 旋回したら True"""
        if self.refuge is None or self.lane is None or self.robot_xy is None:
            return False
        try:
            _, _, yaw = self.lookup('map', 'base_footprint')
        except Exception:
            return False
        s, _ = self.lane.project(*self.robot_xy)
        tx, ty = self.lane.point(s + 2.0, 0.0)
        turn = align_angle(yaw, math.atan2(ty - self.robot_xy[1], tx - self.robot_xy[0]), 0.3)
        self.refuge = None
        if turn is None or not self.spin.wait_for_server(timeout_sec=0.5):
            return False
        self.get_logger().info(f'turning to exit the refuge ({turn:+.2f} rad)')
        goal = Spin.Goal()
        goal.target_yaw = float(turn)
        goal.time_allowance = Duration(sec=6)
        self.spin.send_goal_async(goal).add_done_callback(self.on_exit_spin_accepted)
        return True

    def on_exit_spin_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.continue_route()
            return
        handle.get_result_async().add_done_callback(lambda _: self.continue_route())

    def skip_blocked_waypoint(self, xy, rx, ry):
        """向かっている waypoint (停止点以外) の上に物体があれば, 一時停止して次の waypoint から再開する"""
        k = self.target_index
        if (not self.get_parameter('skip_blocked_waypoint').value or k is None or k >= len(self.route) - 1
                or k == self.skipped_index or self.nav_status != 'running' or self.machine.state != 'idle'):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        # 動いている相手を追従中と, 停止点の手前 20m 以内は飛ばさない. 0.5 秒続けて塞がれていたら飛ばす
        moving_leader = (self.leader_last is not None and math.hypot(self.leader_last.vx, self.leader_last.vy)
                         >= self.get_parameter('follow_min_speed').value)
        # 飛ばした先でプロファイルが切り替わるなら飛ばさない
        next_profile = self.profiles[k + 1] if k + 1 < len(self.profiles) else None
        blocked = (not moving_leader and not next_profile and not near_stop_ahead(self.route, self.stops, k, 20.0)
                   and should_skip_waypoint(self.route[k], self.stops[k], (rx, ry), xy, 0.6, 7.0))
        # (waypoint, 塞がれ始めた時刻, 最後に塞がれていた時刻). 点群の一瞬の途切れ (1 秒以内) では数え直さない
        wp, first, last = self.blocked_wp
        if wp != k or now - last > 1.0:
            wp, first = (k, now) if blocked else (None, now)
        if blocked:
            last = now
        self.blocked_wp = (wp, first, last)
        if blocked and wp == k and now - first >= 0.5:
            self.blocked_wp = (None, now, now)
            self.skipped_index, self.skip_to = k, k + 1
            self.get_logger().info(f'waypoint {k} is occupied by an object, skipping to waypoint {k + 1}')
            self.pause_cli.call_async(Trigger.Request())

    def on_override(self, msg):
        # queue_guard が前の物体の後ろで意図して止めている (待機列) 間は立て直さない. 0 は上限なし
        self.held = (self.get_clock().now().nanoseconds * 1e-9, 0.0 < msg.data < 0.5)

    def check_stall(self, xy, rx, ry, now):
        """正面 (0.9m 以内, 横 0.65m 以内) を塞がれて 8 秒ほとんど動けなければ, 一時停止して下がり (同じ場所で繰り返すほど長く), 同じ waypoint から走り直す"""
        self.motion = [m for m in self.motion if m[0] >= now - 10.0] + [(now, (rx, ry))]
        held = self.held[1] and now - self.held[0] < 1.0
        if (not self.get_parameter('unstick_enabled').value or self.unstick_index is not None or held
                or self.nav_status != 'running' or self.machine.state != 'idle' or self.target_index is None
                or not stalled(self.motion, now, 8.0, 0.1)):
            return
        try:
            pose = self.lookup('map', 'base_footprint')
        except Exception:
            return
        if not blocked_ahead(xy, pose, 0.9, 0.65):
            return
        self.backup_dist, self.last_backup = backup_distance(self.last_backup, now, (rx, ry))
        self.get_logger().info(f'stuck in front of an object, backing up {self.backup_dist:.1f} m '
                               f'and retrying waypoint {self.target_index}')
        self.unstick_index, self.motion = self.target_index, []
        self.pause_cli.call_async(Trigger.Request())

    def back_up(self):
        index, self.unstick_index = self.unstick_index, -1
        goal = BackUp.Goal()
        goal.target.x, goal.speed = self.backup_dist, 0.1
        goal.time_allowance = Duration(sec=int(self.backup_dist / 0.1) + 5)

        def finish(_=None):
            self.unstick_index, self.motion = None, []
            self.set_start_index(index)
        if not self.backup.wait_for_server(timeout_sec=0.5):
            self.after(0.5, finish)
            return
        def done(result):
            # 外形が既に物体に触れていると BackUp は衝突判定で中止する: collision_monitor の手前へ直接後退の速度を出す
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.after(0.5, finish)
            else:
                self.reverse(min(self.backup_dist / 0.1, 12.0), finish)
        self.backup.send_goal_async(goal).add_done_callback(
            lambda f: f.result().get_result_async().add_done_callback(lambda r: done(r.result()))
            if f.result().accepted else self.reverse(min(self.backup_dist / 0.1, 12.0), finish))

    def reverse(self, duration, then):
        self.get_logger().info(f'backup aborted, reversing directly for {duration:.0f} s')
        end = self.get_clock().now().nanoseconds * 1e-9 + duration
        timer = None

        def step():
            if self.get_clock().now().nanoseconds * 1e-9 < end:
                cmd = Twist()
                cmd.linear.x = -0.1
                self.reverse_pub.publish(cmd)
                return
            self.reverse_pub.publish(Twist())
            timer.cancel()
            self.destroy_timer(timer)
            self.after(0.5, then)
        timer = self.create_timer(0.1, step)

    def restart_once(self, index):
        self.restart_timer.cancel()
        self.destroy_timer(self.restart_timer)
        self.set_start_index(index)

    def continue_route(self):
        """待機した位置より先の waypoint から走行を再開する (元の waypoint へは戻らない)"""
        index = self.paused_index if self.paused_index is not None else 0
        if self.robot_xy is not None and self.route:
            index = next_waypoint_index(self.route, self.stops, self.robot_xy, index)
        self.get_logger().info(f'lane clear, continuing from waypoint {index}')
        self.set_start_index(index)

    def goto_refuge(self):
        """待機点を決めて向かう: 固定の退避点をその場で確かめ, 駄目ならその場で探索, それも無ければレーン内で後退"""
        self.goal_done = False
        self.refuge = None
        refuge, source, why, side = None, '', None, None
        cost = self.costmap_lookup()
        if self.decision_lane is not None and self.robot_xy is not None and self.decision_xy is not None and cost is not None:
            refuge, source, why, side = self.choose_refuge(self.decision_lane, cost)
        if refuge is None or not self.nav.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f'no refuge available, waiting in place (rejected: {why}, pedestrian side: {side})')
            self.goal_done = True
            return
        self.get_logger().info(f'moving to refuge ({refuge[0]:.1f}, {refuge[1]:.1f}) [{source}]')
        self.refuge, self.refuge_retries = refuge, 0
        self.send_refuge()

    def choose_refuge(self, lane, cost):
        """(待機点 (x, y, yaw) か None, 選び方, 外れた理由の件数, 歩行者側) を返す"""
        limit = self.get_parameter('refuge_max_cost').value
        min_sep = self.get_parameter('refuge_min_sep').value
        _, opponent_l = lane.project(*self.decision_xy)
        robot_l = lane.project(*self.robot_xy)[1]
        why = {'cost': 0, 'path': 0, 'exit': 0, 'sep': 0}

        def path_clear(a, b):
            n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1))
            return all(cost(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) < 99 for k in range(1, n + 1))

        def exit_clear(p):
            # 待機点で向きをそろえた後, レーンへ斜め前に戻れること (正面のコーン等で出られない点を除く)
            ps, _ = lane.project(*p)
            return path_clear(p, lane.point(ps + 2.0, robot_l))

        def usable(p):
            # 外れた理由を数える (見つからなかったときのログ用)
            for key, ok in (('sep', lambda: abs(lane.project(*p)[1] - opponent_l) >= min_sep),
                            ('cost', lambda: cost(*p) <= limit),
                            ('path', lambda: path_clear(self.robot_xy, p)), ('exit', lambda: exit_clear(p))):
                if not ok():
                    why[key] += 1
                    return False
            return True

        def usable_free(p):
            if cost(*p) > limit:
                why['cost'] += 1
                return False
            if not exit_clear(p):
                why['exit'] += 1
                return False
            return True

        refuge = choose_fixed_refuge(lane, self.robot_xy, self.decision_xy, self.refuge_points(),
                                     self.get_parameter('refuge_fixed_ahead').value, check=usable)
        source = 'fixed'
        # 車道側へははみ出さない: ロボットから見て歩行者側 (keepout の帯がある側) だけを探す
        side = self.pedestrian_side(lane)
        if refuge is None and side is not None:
            refuge = find_refuge(
                lane, self.robot_xy, self.decision_xy, free=usable_free, path_clear=path_clear,
                ahead=self.get_parameter('refuge_ahead').value, lateral=tuple(self.get_parameter('refuge_lateral').value),
                step=self.get_parameter('refuge_step').value, sides=(side,), min_sep=min_sep,
                behind=self.get_parameter('refuge_behind').value)
            source = 'search'
        if refuge is None:
            # ポールの間などを抜けて相手が横を通れる所まで 2-5m 後ろへ下がる (歩行者側を優先, 出口の確認はしない)
            lats = (side * 1.0, side * 1.5, side * 0.5, 0.0) if side is not None else (0.0,)
            refuge = find_retreat(lane, self.robot_xy, free=lambda p: cost(*p) <= limit and path_clear(self.robot_xy, p),
                                  dists=tuple(np.arange(2.0, 5.01, 0.5)), laterals=lats)
            source = 'retreat'
        return refuge, source, why, side

    def refuge_points(self):
        """固定の退避点 [(x, y)] (refuge_points_file. 読み込みはパスが変わったときだけ)"""
        path = self.get_parameter('refuge_points_file').value
        if not path:
            wp = self.get_parameter('waypoints_file').value
            path = wp[:-len('_wp.yaml')] + '_refuge_points.yaml' if wp.endswith('_wp.yaml') else ''
        if path != self._refuge_points_path:
            self._refuge_points_path, self._refuge_points = path, []
            if path and os.path.exists(path):
                with open(path) as f:
                    self._refuge_points = [(float(p['x']), float(p['y']))
                                           for p in (yaml.safe_load(f) or {}).get('refuge_points') or []]
            self.get_logger().info(f'{len(self._refuge_points)} fixed refuge points loaded from {path}')
        return self._refuge_points

    def pedestrian_side(self, lane):
        """ロボットから見てレーンの左 (+1) / 右 (-1) のどちらに歩行者側の帯があるか. 左右 3.5m 以内に無ければ None"""
        robot_s, robot_l = lane.project(*self.robot_xy)
        for d in np.arange(0.5, 3.51, 0.25):
            for side in (1.0, -1.0):
                if self.on_pedestrian_side(lane.point(robot_s, robot_l + side * d)):
                    return side
        return None

    def on_pedestrian_side(self, p):
        m = self.keepout
        if m is None:
            return False
        i = int((p[0] - m.info.origin.position.x) / m.info.resolution)
        j = int((p[1] - m.info.origin.position.y) / m.info.resolution)
        return 0 <= i < m.info.width and 0 <= j < m.info.height and m.data[j * m.info.width + i] > 0

    def on_local_costmap(self, msg):
        self.local_costmap = msg

    def costmap_lookup(self):
        """map 座標 (x, y) -> local costmap の値 (範囲外と不明は 100) を返す関数. costmap か TF が無ければ None"""
        m = self.local_costmap
        if m is None:
            return None
        try:
            ox, oy, oyaw = self.lookup(m.header.frame_id, 'map')
        except Exception:
            return None
        info = m.info
        grid = np.array(m.data, dtype=np.int16).reshape(info.height, info.width)
        c, s = math.cos(oyaw), math.sin(oyaw)

        def cost(x, y):
            u, v = ox + c * x - s * y, oy + s * x + c * y
            i = int((u - info.origin.position.x) / info.resolution)
            j = int((v - info.origin.position.y) / info.resolution)
            if not (0 <= i < info.width and 0 <= j < info.height) or grid[j, i] < 0:
                return 100
            return int(grid[j, i])
        return cost

    def send_refuge(self):
        refuge = self.refuge
        self.refuge_sent = self.get_clock().now().nanoseconds * 1e-9
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = refuge[0], refuge[1]
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(refuge[2] / 2), math.cos(refuge[2] / 2)
        self.nav.send_goal_async(goal).add_done_callback(self.on_goal_accepted)

    def on_goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goal_done = True
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_goal_result)

    def on_goal_result(self, future):
        self.goal_handle = None
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.refuge_sent
        if self.machine.state == 'moving' and should_resend_refuge(
                future.result().status, elapsed, self.refuge_retries, canceled_by_us=False):
            self.refuge_retries += 1
            self.get_logger().warn(f'refuge goal ended after {elapsed:.2f} s, resending')
            self.resend_timer = self.create_timer(0.3, self.resend_once)
            return
        if self.machine.state == 'moving' and future.result().status == 4 and self.align_heading():
            return
        self.goal_done = True

    def align_heading(self):
        """待機点に着いたらレーンの向き (相手が来る方) へその場で向きをそろえる. LiDAR の見る範囲を相手側に固定するため"""
        try:
            _, _, yaw = self.lookup('map', 'base_footprint')
        except Exception:
            return False
        turn = align_angle(yaw, self.refuge[2], 0.2)
        if turn is None or not self.spin.wait_for_server(timeout_sec=0.5):
            return False
        self.get_logger().info(f'aligning heading at refuge ({turn:+.2f} rad)')
        goal = Spin.Goal()
        goal.target_yaw = float(turn)
        goal.time_allowance = Duration(sec=8)
        self.spin.send_goal_async(goal).add_done_callback(self.on_spin_accepted)
        return True

    def on_spin_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goal_done = True
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.on_spin_result)

    def on_spin_result(self, future):
        self.goal_handle = None
        self.goal_done = True

    def resend_once(self):
        self.resend_timer.cancel()
        self.destroy_timer(self.resend_timer)
        if self.machine.state == 'moving':
            self.send_refuge()


def main():
    rclpy.init()
    rclpy.spin(YieldGuard())


if __name__ == '__main__':
    main()
