import numpy as np
from nav_msgs.msg import MapMetaData

from rsf_waypoint_manager.queue_guard import static_hit, static_mask


def info(width, height, resolution=0.1, ox=0.0, oy=0.0):
    i = MapMetaData()
    i.width, i.height, i.resolution = width, height, resolution
    i.origin.position.x, i.origin.position.y = ox, oy
    return i


def test_static_mask_dilates_occupied_cells_by_margin():
    grid = np.zeros((11, 11), dtype=np.int16)
    grid[5, 5] = 100
    mask = static_mask(grid, 0.1, 0.2)
    assert mask[5, 5] and mask[5, 7] and mask[3, 5]
    assert not mask[5, 8]
    assert not mask[0, 0]


def test_static_mask_ignores_free_and_unknown():
    grid = np.array([[0, -1, 64]], dtype=np.int16)
    assert not static_mask(grid, 0.1, 0.0).any()


def test_static_hit_combines_all_masks():
    a = np.zeros((10, 10), dtype=bool)
    a[0, 0] = True  # (0.05, 0.05)
    b = np.zeros((10, 10), dtype=bool)
    b[0, 0] = True  # (2.05, 0.05): 原点がずれた別マスク (芝生マスク)
    statics = [(info(10, 10), a), (info(10, 10, ox=2.0), b)]
    px = np.array([[0.05, 2.05, 1.05]])
    py = np.array([[0.05, 0.05, 0.05]])
    assert static_hit(statics, px, py).tolist() == [[True, True, False]]


def test_static_hit_outside_masks_is_false():
    statics = [(info(10, 10), np.ones((10, 10), dtype=bool))]
    px = np.array([[-0.5, 5.0]])
    py = np.array([[0.5, 0.5]])
    assert static_hit(statics, px, py).tolist() == [[False, False]]


def test_static_hit_without_masks_is_all_false():
    px = np.zeros((2, 3))
    assert not static_hit([], px, px).any()
