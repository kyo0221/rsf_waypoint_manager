"""退避禁止の範囲の編集ロジック (ROS 非依存)"""
import copy

from rsf_waypoint_manager.yield_logic import save_zones


def move_corner(zones, zone_index, corner, x, y):
    """zone_index 番の範囲の corner 番の隅を (x, y) に動かした新しいリスト (元は変えない)"""
    if not 0 <= zone_index < len(zones) or not 0 <= corner < len(zones[zone_index]['points']):
        raise IndexError((zone_index, corner))
    out = copy.deepcopy(zones)
    out[zone_index]['points'][corner] = [float(x), float(y)]
    return out


def fill_triangles(points):
    """TRIANGLE_LIST 用の頂点列 (扇形に三角形分割)"""
    pts = [tuple(p) for p in points]
    out = []
    for i in range(1, len(pts) - 1):
        out += [pts[0], pts[i], pts[i + 1]]
    return out


def outline_points(points):
    """LINE_STRIP 用の閉じた輪郭"""
    pts = [tuple(p) for p in points]
    return pts + pts[:1]


def save_after_drag(path, zones, zone_index, corner, x, y):
    """隅を動かして保存し, 新しいリストを返す"""
    new = move_corner(zones, zone_index, corner, x, y)
    save_zones(path, new)
    return new
