import numpy as np
from nav_msgs.msg import MapMetaData

from rsf_waypoint_manager.queue_guard import queue_objects, tall_mask

RES = 0.05


def info(width, height, resolution=RES, ox=0.0, oy=0.0):
    i = MapMetaData()
    i.width, i.height, i.resolution = width, height, resolution
    i.origin.position.x, i.origin.position.y = ox, oy
    return i


def blob(grid, cx, cy, size):
    """中心 (cx, cy) [m], 一辺 size [m] の正方形の塊を lethal にする"""
    n = round(size / RES)
    i0, j0 = round(cx / RES) - n // 2, round(cy / RES) - n // 2
    grid[j0:j0 + n, i0:i0 + n] = True


def test_large_object_counts_without_tall_points():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.6)  # 待機列のロボット
    objects = queue_objects(lethal, RES, min_size=0.48, max_size=1.5)
    assert objects[40, 40]


def test_small_object_without_tall_points_is_ignored():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)  # コーン
    objects = queue_objects(lethal, RES, min_size=0.48, max_size=1.5, tall=np.zeros_like(lethal))
    assert not objects.any()


def test_leg_blobs_under_tall_points_count_as_a_person():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 1.9, 2.0, 0.15)  # 左脚
    blob(lethal, 2.15, 2.0, 0.15)  # 右脚
    tall = tall_mask(np.array([[2.02, 2.0]]), info(80, 80), radius=0.3)  # 胴体の点
    objects = queue_objects(lethal, RES, min_size=0.48, max_size=1.5, tall=tall)
    assert objects[40, 38] and objects[40, 43]


def test_tall_points_far_from_a_cone_do_not_keep_it():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 0.4)
    tall = tall_mask(np.array([[3.0, 2.0]]), info(80, 80), radius=0.3)
    objects = queue_objects(lethal, RES, min_size=0.48, max_size=1.5, tall=tall)
    assert not objects.any()


def test_object_larger_than_max_is_ignored_even_if_tall():
    lethal = np.zeros((80, 80), dtype=bool)
    blob(lethal, 2.0, 2.0, 2.0)  # 壁・植え込み
    tall = tall_mask(np.array([[2.0, 2.0]]), info(80, 80), radius=0.3)
    assert not queue_objects(lethal, RES, min_size=0.48, max_size=1.5, tall=tall).any()


def test_tall_mask_uses_grid_origin_and_drops_points_outside():
    m = tall_mask(np.array([[10.25, 5.25], [0.0, 0.0], [99.0, 99.0]]), info(20, 20, ox=10.0, oy=5.0), radius=0.0)
    assert m[5, 5]
    assert m.sum() == 1


def test_tall_mask_without_points_is_empty():
    assert not tall_mask(np.zeros((0, 2)), info(10, 10), radius=0.3).any()
