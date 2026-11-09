"""global 用芝生マスク (tsukuba_grass_center.pgm) の確認: WP10-11 の通路は中央が縁より低く, 他の区間と芝生は元のまま."""
import os

import numpy as np
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', 'rsf_navigation_executor', 'config')
WAYPOINTS = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'tsukuba_wp.yaml')
RES, OX, OY = 0.2, -80.0, -10.0


def read_pgm(name):
    data = open(os.path.join(CONFIG, name), 'rb').read()
    fields = data.split(maxsplit=4)
    w, h = int(fields[1]), int(fields[2])
    return np.flipud(np.frombuffer(fields[4], dtype=np.uint8)[:w * h].reshape(h, w)).astype(int)


def test_global_mask_points_to_center_pgm():
    assert yaml.safe_load(open(os.path.join(CONFIG, 'tsukuba_grass.yaml')))['image'] == 'tsukuba_grass_center.pgm'
    assert yaml.safe_load(open(os.path.join(CONFIG, 'tsukuba_grass_soft.yaml')))['image'] == 'tsukuba_grass.pgm'


def test_wp10_to_wp11_corridor_is_lowest_in_the_middle():
    base, center = read_pgm('tsukuba_grass.pgm'), read_pgm('tsukuba_grass_center.pgm')
    assert (center >= base).all()
    assert (center[base >= 100] == 100).all()
    assert (center < 100).sum() == (base < 100).sum()  # 芝生は増やさない
    w = yaml.safe_load(open(WAYPOINTS))['waypoints']
    x = (w[10]['x'] + w[11]['x']) / 2.0
    col = int((x - OX) / RES)
    y = (w[10]['y'] + w[11]['y']) / 2.0
    row = int((y - OY) / RES)
    up = next(r for r in range(row, row + 40) if center[r, col] >= 100)
    down = next(r for r in range(row, row - 40, -1) if center[r, col] >= 100)
    profile = center[down + 1:up, col]
    mid = len(profile) // 2
    assert profile[mid] < profile[1] and profile[mid] < profile[-2]
    assert profile.argmin() in range(mid - 2, mid + 3)  # 最も低いのは通路の中央付近
