"""待機列の後ろで止まるプロファイルが, 歩道のコーン列 (経路の横 0.7-1.0m, 直径 0.36m) で止まり続けない設定になっているか"""
import os

import yaml

PROFILES = os.path.join(os.path.dirname(__file__), '..', '..', 'rsf_navigation_executor', 'config', 'nav_profiles.yaml')
CONE_BLOB = 0.45  # costmap 上のコーンの塊の大きさ (0.05m 格子で 9 セル)


def queue_guard(name):
    return yaml.safe_load(open(PROFILES))['profiles'][name]['parameters']['/queue_guard']


def test_queue_profiles_ignore_cones_but_count_people():
    for name in ('following', 'signal_queue', 'speed_down_following'):
        qg = queue_guard(name)
        assert qg['enabled']
        assert qg['min_object_size'] > CONE_BLOB, name
        assert 0.7 < qg['tall_span'] <= 0.8, name  # 点の縦の幅: コーン (0.7m) は越え, 0.2m 上り坂の人 (0.8m) は入る


def test_signal_queue_looks_only_at_the_waypoint_line():
    """信号前の列は waypoint の線上に並ぶ. 横のコーン列 (経路の横 0.7m 以上, 表面は 0.52m 以上) は検出幅の外"""
    qg = queue_guard('signal_queue')
    assert qg['use_route']
    assert qg['half_width'] < 0.7 - 0.18
    assert qg['half_width'] >= 0.4  # 線上に並ぶ人 (半径 0.25) が少しずれても拾う
