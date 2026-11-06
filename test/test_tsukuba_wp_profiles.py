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


def profile_at(w, k):
    """WP k へ向かう区間に効くプロファイル (k から逆にたどって最初に付いているもの)"""
    return next((w[i]['profile'] for i in range(k, -1, -1) if 'profile' in w[i]), 'default')


def test_pole_gates_on_wp112_to_115_use_poles_profile():
    """WP112->113 と WP114->115 のポールの門 (x 465, 484) も, 歩行者側へ回り込まず門の真ん中を通す"""
    w = load()
    for k in (113, 114, 115):
        assert profile_at(w, k) == 'road_edge_keep_poles'
    assert profile_at(w, 116) == 'road_edge_keep'


def test_hotel_section_wp146_to_169_keeps_road_edge():
    """ホテル周辺 (WP146-169) は走行領域が狭く, はみ出しが失格に直結するので road_edge_keep で走る"""
    w = load()
    for k in range(146, 170):
        assert profile_at(w, k) == 'road_edge_keep'


def test_hotel_section_wp146_to_169_uses_small_radius():
    """ホテル周辺 (WP146-169) は走行領域が狭いので, waypoint の半径を元 (1.3, 2.0m) の 2/3 にして経路の角を内側へ切らない"""
    w = load()
    assert {w[k]['radius'] for k in range(146, 170)} <= {0.87, 1.33}
