"""tsudanuma_wp.yaml の検証."""
import os

import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(__file__)
WP_FILE = os.path.join(HERE, '..', 'waypoints', 'tsudanuma_wp.yaml')
CFG = os.path.join(HERE, '..', '..', 'rsf_navigation_executor', 'config')


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)['waypoints']


def test_count_and_keys():
    w = load(WP_FILE)
    assert len(w) == 342
    for p in w:
        for k in ('x', 'y', 'yaw', 'radius', 'stop'):
            assert k in p


def test_profiles_exist():
    with open(os.path.join(CFG, 'nav_profiles.yaml')) as f:
        profiles = yaml.safe_load(f)['profiles']
    for p in load(WP_FILE):
        if 'profile' in p:
            assert p['profile'] in profiles


def test_waypoints_in_free_space_with_clearance():
    with open(os.path.join(CFG, 'tsudanuma.yaml')) as f:
        m = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(CFG, m['image'])))
    res = m['resolution']
    ox, oy = m['origin'][0], m['origin'][1]
    h = img.shape[0]
    occ = np.argwhere(img < 100)
    occ_xy = np.stack([ox + (occ[:, 1] + 0.5) * res, oy + (h - 1 - occ[:, 0] + 0.5) * res], axis=1)
    for i, p in enumerate(load(WP_FILE)):
        col = int((p['x'] - ox) / res)
        row = h - 1 - int((p['y'] - oy) / res)
        assert 0 <= row < h and 0 <= col < img.shape[1], i
        assert img[row, col] >= 100, f'WP{i} が占有セル上'
        d = np.hypot(occ_xy[:, 0] - p['x'], occ_xy[:, 1] - p['y']).min()
        assert d >= 0.6, f'WP{i} は占有まで {d:.2f} m'


def test_route_starts_with_no_following_profile():
    """既定だと yield_guard が脇の静止物を前走とみなして止まるので, 始点から追従を切る wide_road にする"""
    w = load(WP_FILE)
    assert w[0].get('profile') == 'wide_road'
    assert all('profile' not in p for p in w[1:])
