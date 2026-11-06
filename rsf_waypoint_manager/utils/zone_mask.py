"""走行禁止の多角形を OccupancyGrid 用の配列にする / コーン列から多角形を作る (ROS 非依存)"""
import math

from rsf_waypoint_manager.yield_logic import point_in_polygon


def rasterize_zones(zones, resolution, margin, value):
    """(origin_x, origin_y, width, height, data): セル中心が多角形内なら value, 外は 0 (行優先)"""
    pts = [p for z in zones for p in z['points']]
    if not pts:
        return 0.0, 0.0, 1, 1, [0]
    ox = min(p[0] for p in pts) - margin
    oy = min(p[1] for p in pts) - margin
    width = int(math.ceil((max(p[0] for p in pts) + margin - ox) / resolution - 1e-9))
    height = int(math.ceil((max(p[1] for p in pts) + margin - oy) / resolution - 1e-9))
    data = [0] * (width * height)
    for z in zones:
        poly = z['points']
        c0 = max(0, int(math.floor((min(p[0] for p in poly) - ox) / resolution)))
        c1 = min(width - 1, int(math.ceil((max(p[0] for p in poly) - ox) / resolution)))
        r0 = max(0, int(math.floor((min(p[1] for p in poly) - oy) / resolution)))
        r1 = min(height - 1, int(math.ceil((max(p[1] for p in poly) - oy) / resolution)))
        for r in range(r0, r1 + 1):
            cy = oy + (r + 0.5) * resolution
            for c in range(c0, c1 + 1):
                if point_in_polygon((ox + (c + 0.5) * resolution, cy), poly):
                    data[r * width + c] = int(value)
    return ox, oy, width, height, data


def nearest_on_route(route, x, y):
    """経路 (点列をつないだ線) の上で (x, y) に最も近い点"""
    if len(route) == 1:
        return route[0]
    best, bd = route[0], float('inf')
    for (ax, ay), (bx, by) in zip(route, route[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / max(dx * dx + dy * dy, 1e-12)))
        px, py = ax + dx * t, ay + dy * t
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            best, bd = (px, py), d
    return best


def polyline_to_zones(points, route, width, name_prefix):
    """折れ線の各区間を, 経路から遠い側へ width[m] 広げた 4 角形にして返す"""
    zones = []
    for i in range(len(points) - 1):
        (ax, ay), (bx, by) = points[i], points[i + 1]
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-9 or not route:
            continue
        nx, ny = -(by - ay) / length, (bx - ax) / length
        mx, my = (ax + bx) / 2, (ay + by) / 2
        rx, ry = nearest_on_route(route, mx, my)
        if (rx - mx) * nx + (ry - my) * ny > 0:  # 法線が経路側を向いていたら反転
            nx, ny = -nx, -ny
        zones.append({'name': f'{name_prefix}_{i}', 'type': 'roadway',
                      'points': [[ax, ay], [bx, by], [bx + nx * width, by + ny * width],
                                 [ax + nx * width, ay + ny * width]]})
    return zones
