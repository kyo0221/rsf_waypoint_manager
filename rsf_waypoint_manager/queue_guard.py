import math

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy import ndimage
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener


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
        self.route = []
        path = self.get_parameter('waypoints_file').value
        if path:
            with open(path) as f:
                self.route = [(float(w['x']), float(w['y'])) for w in yaml.safe_load(f)['waypoints']]
        self.plan = []
        self.local = None
        self.grid = None
        self.objects = None
        self.static = None
        self.active = False
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/plan', self.on_plan, 1)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.on_local, 1)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, transient)
        self.pub = self.create_publisher(Float32, '/waypoint_navigator/speed_override', 10)
        self.distance_pub = self.create_publisher(Float32, '~/obstacle_distance', 10)
        self.buf = Buffer()
        TransformListener(self.buf, self)
        self.create_timer(0.1, self.tick)

    def on_plan(self, msg):
        self.plan = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_local(self, msg):
        self.local = msg
        self.grid = np.frombuffer(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.objects = None

    def object_mask(self):
        if self.objects is None:
            lethal = self.grid >= self.get_parameter('lethal').value
            labels, count = ndimage.label(lethal, structure=np.ones((3, 3)))
            small = np.zeros(count + 1, dtype=bool)
            limit = self.get_parameter('max_object_size').value / self.local.info.resolution
            for k, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) <= limit:
                    small[k] = True
            self.objects = small[labels]
        return self.objects

    def on_map(self, msg):
        grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        cells = max(1, math.ceil(self.get_parameter('static_margin').value / msg.info.resolution))
        y, x = np.ogrid[-cells:cells + 1, -cells:cells + 1]
        self.static = (msg.info, ndimage.binary_dilation(grid >= 65, structure=(x * x + y * y) <= cells * cells))

    def pose(self, target, source):
        tf = self.buf.lookup_transform(target, source, rclpy.time.Time())
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def lane(self, rx, ry, lookahead):
        pts = list(self.plan)
        if pts:
            i0 = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - rx, pts[i][1] - ry))
            pts = pts[i0:]
        else:
            pts = [(rx, ry)]
        if self.route:
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
        if self.static is not None:
            info, mask = self.static
            mi = np.floor((px - info.origin.position.x) / info.resolution).astype(int)
            mj = np.floor((py - info.origin.position.y) / info.resolution).astype(int)
            ok = (mi >= 0) & (mi < info.width) & (mj >= 0) & (mj < info.height)
            static_hit = np.zeros(ci.shape, dtype=bool)
            static_hit[ok] = mask[mj[ok], mi[ok]]
            hit &= ~static_hit
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
