import math

import numpy as np
import rclpy
import yaml
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy import ndimage
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener


def dilate(mask, resolution, radius):
    """bool マスクを半径 radius [m] の円で膨らませる"""
    cells = math.ceil(radius / resolution)
    if cells <= 0:
        return mask
    y, x = np.ogrid[-cells:cells + 1, -cells:cells + 1]
    return ndimage.binary_dilation(mask, structure=(x * x + y * y) <= cells * cells)


def static_mask(grid, resolution, margin):
    """占有 (>=65) セルを margin [m] だけ膨らませた bool マスク"""
    return dilate(grid >= 65, resolution, max(margin, resolution))


def tall_mask(points, info, radius):
    """格子の座標系の点 [[x, y], ...] (背の高い物の点) のあるセルを radius [m] 膨らませた bool マスク"""
    mask = np.zeros((info.height, info.width), dtype=bool)
    if len(points) == 0:
        return mask
    i = np.floor((points[:, 0] - info.origin.position.x) / info.resolution).astype(int)
    j = np.floor((points[:, 1] - info.origin.position.y) / info.resolution).astype(int)
    ok = (i >= 0) & (i < info.width) & (j >= 0) & (j < info.height)
    mask[j[ok], i[ok]] = True
    return dilate(mask, info.resolution, radius)


def queue_objects(lethal, resolution, min_size, max_size, tall=None):
    """前に並ぶ物体として数えるセルの bool マスク.
    max_size [m] を超える塊 (壁・植え込み) は数えない. min_size [m] 未満の塊 (コーン等) は,
    背の高い点 (tall) と重なるときだけ数える: 人は LiDAR で脚 2 本の小さな塊に分かれても, 胴体の点が上にある"""
    labels, count = ndimage.label(lethal, structure=np.ones((3, 3)))
    has_tall = np.zeros(count + 1, dtype=bool)
    if tall is not None:
        has_tall[np.unique(labels[tall & lethal])] = True
        has_tall[0] = False
    keep = np.zeros(count + 1, dtype=bool)
    lower, limit = min_size / resolution, max_size / resolution
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        size = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        keep[k] = size <= limit and (size >= lower or has_tall[k])
    return keep[labels]


def static_hit(statics, px, py):
    """map 座標の点 (px, py) が, いずれかの静的マスク [(info, mask), ...] に入っていれば True"""
    hit = np.zeros(np.shape(px), dtype=bool)
    for info, mask in statics:
        mi = np.floor((px - info.origin.position.x) / info.resolution).astype(int)
        mj = np.floor((py - info.origin.position.y) / info.resolution).astype(int)
        ok = (mi >= 0) & (mi < info.width) & (mj >= 0) & (mj < info.height)
        hit[ok] |= mask[mj[ok], mi[ok]]
    return hit


class QueueGuard(Node):

    def __init__(self):
        super().__init__('queue_guard')
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('enabled', False)
        self.declare_parameter('half_width', 1.5)
        self.declare_parameter('lookahead', 4.5)
        self.declare_parameter('standoff', 1.5)
        self.declare_parameter('time_gap', 1.5)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('static_margin', 0.5)
        self.declare_parameter('lethal', 100)
        self.declare_parameter('max_object_size', 1.5)
        self.declare_parameter('use_route', False)  # 経路 (/plan) でなく waypoint の線に沿って前の物体を探す (経路が列を回り込んでも止まる)
        self.declare_parameter('min_object_size', 0.0)  # これより小さい塊 (コーン等) は, 背の高い点が上に無ければ前の物体として数えない
        # 背の高い物 (人) の判定: 点群でこの高さ [m] (base_footprint 基準) 以上の点がある塊は小さくても数える. 0 で使わない
        self.declare_parameter('tall_height', 0.0)
        self.declare_parameter('tall_radius', 0.3)  # 胴体の点と脚の塊を結びつける距離 [m]
        self.declare_parameter('tall_points_topic', '/ground_segmentation/obstacle_points')
        # 前走物体として数えない静的マスク. 芝生マスク (keepout) を local costmap に入れると,
        # その縁が小さな塊として「前方の物体」に見えて停止するため, /map と同様に除外する
        self.declare_parameter('static_mask_topics', ['/map', '/grass_filter_mask'])
        self.route = []
        path = self.get_parameter('waypoints_file').value
        if path:
            with open(path) as f:
                self.route = [(float(w['x']), float(w['y'])) for w in yaml.safe_load(f)['waypoints']]
        self.plan = []
        self.local = None
        self.grid = None
        self.objects = None
        self.statics = {}
        self.cloud = None
        self.active = False
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/plan', self.on_plan, 1)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.on_local, 1)
        self.create_subscription(OccupancyGridUpdate, '/local_costmap/costmap_updates', self.on_local_update, 10)
        for topic in self.get_parameter('static_mask_topics').value:
            self.create_subscription(
                OccupancyGrid, topic, lambda msg, topic=topic: self.on_map(topic, msg), transient)
        self.create_subscription(PointCloud2, self.get_parameter('tall_points_topic').value, self.on_cloud, 1)
        self.pub = self.create_publisher(Float32, '/waypoint_navigator/speed_override', 10)
        self.distance_pub = self.create_publisher(Float32, '~/obstacle_distance', 10)
        self.buf = Buffer()
        TransformListener(self.buf, self)
        self.create_timer(0.1, self.tick)

    def on_plan(self, msg):
        self.plan = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_local(self, msg):
        self.local = msg
        self.grid = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width).copy()
        self.objects = None

    def on_local_update(self, msg):
        if self.grid is None:
            return
        patch = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.height, msg.width)
        self.grid[msg.y:msg.y + msg.height, msg.x:msg.x + msg.width] = patch
        self.objects = None

    def on_cloud(self, msg):
        self.cloud = msg
        self.objects = None

    def tall_cells(self):
        """点群のうち tall_height 以上の点を local costmap の格子に落としたマスク. 使わない設定なら None"""
        height = self.get_parameter('tall_height').value
        if height <= 0.0 or self.cloud is None:
            return None
        points = point_cloud2.read_points_numpy(self.cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        points = points[points[:, 2] >= height]
        if len(points):
            ox, oy, oyaw = self.pose(self.local.header.frame_id, self.cloud.header.frame_id)
            c, s = math.cos(oyaw), math.sin(oyaw)
            points = np.stack([ox + points[:, 0] * c - points[:, 1] * s, oy + points[:, 0] * s + points[:, 1] * c], axis=1)
        return tall_mask(points[:, :2], self.local.info, self.get_parameter('tall_radius').value)

    def object_mask(self):
        if self.objects is None:
            self.objects = queue_objects(
                self.grid >= self.get_parameter('lethal').value, self.local.info.resolution,
                self.get_parameter('min_object_size').value, self.get_parameter('max_object_size').value,
                self.tall_cells())
        return self.objects

    def on_map(self, topic, msg):
        grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.statics[topic] = (msg.info, static_mask(grid, msg.info.resolution, self.get_parameter('static_margin').value))

    def pose(self, target, source):
        tf = self.buf.lookup_transform(target, source, rclpy.time.Time())
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def lane(self, rx, ry, lookahead):
        pts = list(self.plan)
        if self.get_parameter('use_route').value and self.route and pts:
            # 経路の終点 (今向かっている waypoint) から先の waypoint の線 (往復で重なる区間の取り違えを防ぐ)
            ex, ey = pts[-1]
            j = min(range(len(self.route)), key=lambda i: math.hypot(self.route[i][0] - ex, self.route[i][1] - ey))
            pts = [(rx, ry)] + self.route[j:j + 3]
        elif pts:
            i0 = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - rx, pts[i][1] - ry))
            pts = pts[i0:]
        else:
            pts = [(rx, ry)]
        if self.route and not self.get_parameter('use_route').value:
            ex, ey = pts[-1]
            j = min(range(len(self.route)), key=lambda i: math.hypot(self.route[i][0] - ex, self.route[i][1] - ey))
            pts += self.route[j + 1:j + 4]
        xs, ys, ds = [pts[0][0]], [pts[0][1]], [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-6:
                continue
            n = max(1, int(seg / 0.1))
            for k in range(1, n + 1):
                t = k / n
                xs.append(x0 + t * (x1 - x0))
                ys.append(y0 + t * (y1 - y0))
                ds.append(ds[-1] + seg / n)
                if ds[-1] >= lookahead:
                    return np.array(xs), np.array(ys), np.array(ds)
        return np.array(xs), np.array(ys), np.array(ds)

    def obstacle_distance(self):
        rx, ry, _ = self.pose('map', 'base_footprint')
        xs, ys, ds = self.lane(rx, ry, self.get_parameter('lookahead').value)
        if len(xs) < 2:
            return None
        hx = np.gradient(xs)
        hy = np.gradient(ys)
        norm = np.hypot(hx, hy) + 1e-9
        nx, ny = -hy / norm, hx / norm
        half = self.get_parameter('half_width').value
        offsets = np.arange(-half, half + 1e-6, 0.05)
        px = xs[:, None] + nx[:, None] * offsets[None, :]
        py = ys[:, None] + ny[:, None] * offsets[None, :]
        g = self.local
        ox, oy, oyaw = self.pose(g.header.frame_id, 'map')
        c, s = math.cos(oyaw), math.sin(oyaw)
        lx = ox + px * c - py * s
        ly = oy + px * s + py * c
        ci = np.floor((lx - g.info.origin.position.x) / g.info.resolution).astype(int)
        cj = np.floor((ly - g.info.origin.position.y) / g.info.resolution).astype(int)
        inside = (ci >= 0) & (ci < g.info.width) & (cj >= 0) & (cj < g.info.height)
        objects = self.object_mask()
        hit = np.zeros(ci.shape, dtype=bool)
        hit[inside] = objects[cj[inside], ci[inside]]
        hit &= ~static_hit(list(self.statics.values()), px, py)
        rows = np.where(hit.any(axis=1))[0]
        if len(rows) == 0:
            return None
        return float(ds[rows[0]])

    def tick(self):
        enabled = self.get_parameter('enabled').value
        if not enabled or self.local is None:
            if self.active:
                self.pub.publish(Float32(data=0.0))
                self.active = False
            return
        try:
            distance = self.obstacle_distance()
        except Exception as e:
            self.get_logger().warn(f'queue guard skipped: {e}', throttle_duration_sec=5.0)
            return
        if distance is None:
            speed = 0.0
            self.distance_pub.publish(Float32(data=-1.0))
        else:
            standoff = self.get_parameter('standoff').value
            speed = (distance - standoff) / self.get_parameter('time_gap').value
            speed = min(self.get_parameter('max_speed').value, max(0.005, speed))
            self.distance_pub.publish(Float32(data=distance))
        self.pub.publish(Float32(data=speed))
        self.active = True


def main():
    rclpy.init()
    rclpy.spin(QueueGuard())


if __name__ == '__main__':
    main()
