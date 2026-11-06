"""tsukuba_wp.yaml の profile の付け方の確認."""
import os

import yaml

WP_FILE = os.path.join(os.path.dirname(__file__), '..', 'waypoints', 'tsukuba_wp.yaml')


def load():
    with open(WP_FILE) as f:
        return yaml.safe_load(f)['waypoints']


def test_following_starts_at_wp10_not_wp9():
    """WP9 で止まり続けたので, 細い通路の追従は WP10 から. WP9 は既定 (確認走行区間の wide_road を引き継ぐ)"""
    w = load()
    assert 'profile' not in w[9]
    assert w[10].get('profile') == 'following'
