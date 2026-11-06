import numpy as np
from nav_msgs.msg import MapMetaData

from rsf_waypoint_manager.queue_guard import queue_objects

RES = 0.05
INFO = MapMetaData()
INFO.width, INFO.height, INFO.resolution = 80, 80, RES


def blob(grid, cx, cy, size):
    """中心 (cx, cy) [m], 一辺 size [m] の正方形の塊を lethal にする"""
    n = round(size / RES)
    i0, j0 = round(cx / RES) - n // 2, round(cy / RES) - n // 2
    grid[j0:j0 + n, i0:i0 + n] = True


def column(cx, cy, z0, z1):
    """(cx, cy) に z0 から z1 まで縦に並ぶ点 (ロボットの足元の高さが基準)"""
    return np.array([[cx, cy, z] for z in np.arange(z0, z1 + 1e-6, 0.05)])


def objects(lethal, points=None):
    return queue_objects(lethal, RES, min_size=0.48, max_size=1.5, points=points, info=INFO, radius=0.3, tall_span=0.78)


def test_large_object_counts_without_points():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.6)  # 待機列のロボット
    assert objects(lethal)[40, 40]


def test_cone_is_ignored():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)
    assert not objects(lethal, column(2.0, 2.0, 0.05, 0.7)).any()


def test_cone_on_an_uphill_is_still_ignored():
    """上り坂の先のコーンは, ロボットの足元から見て頂点が 0.85m を超えるが, 縦の幅は 0.7m のまま"""
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)
    assert not objects(lethal, column(2.0, 2.0, 0.25, 0.9)).any()


def test_leg_blobs_under_a_torso_count_as_a_person():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 1.9, 2.0, 0.15)  # 左脚
    blob(lethal, 2.15, 2.0, 0.15)  # 右脚
    pts = np.vstack([column(1.9, 2.0, 0.1, 0.9), column(2.15, 2.0, 0.1, 0.9), column(2.02, 2.0, 0.9, 1.1)])
    o = objects(lethal, pts)
    assert o[40, 38] and o[40, 43]


def test_person_on_an_uphill_still_counts():
    """点群は足元から 1.1m までなので, 0.2m 上り坂の人は地面から 0.9m までしか見えない"""
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.3)
    assert objects(lethal, column(2.0, 2.0, 0.3, 1.1)).any()


def test_points_far_from_a_cone_do_not_extend_it():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)
    pts = np.vstack([column(2.0, 2.0, 0.05, 0.7), column(3.0, 2.0, 0.05, 1.1)])
    assert not objects(lethal, pts).any()


def test_object_larger_than_max_is_ignored_even_if_tall():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 2.0)  # 壁・植え込み
    assert not objects(lethal, column(2.0, 2.0, 0.0, 1.1)).any()


def test_points_outside_the_grid_are_dropped():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)
    assert not objects(lethal, column(99.0, 99.0, 0.0, 1.1)).any()
