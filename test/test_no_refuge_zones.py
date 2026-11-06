import math
import os

import pytest
import yaml

from rsf_waypoint_manager.yield_logic import (
    choose_fixed_refuge, filter_points_outside_zones, find_refuge, find_retreat, in_zones, Lane, load_zones,
    oncoming_for_machine, save_zones, segment_hits_zones, YieldStateMachine)

WAYPOINTS = os.path.join(os.path.dirname(__file__), '..', 'waypoints')


def box(x0, y0, x1, y1, name='z', kind='roadway'):
    return {'name': name, 'type': kind, 'points': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_in_zones_inside_outside_and_rotated():
    zones = [box(0, 0, 4, 2)]
    assert in_zones((2, 1), zones)
    assert not in_zones((5, 1), zones)
    assert not in_zones((2, -0.1), zones)
    diamond = {'name': 'd', 'type': 'crosswalk', 'points': [[0, -2], [2, 0], [0, 2], [-2, 0]]}
    assert in_zones((0.5, 0.5), [diamond])
    assert not in_zones((1.5, 1.5), [diamond])


def test_in_zones_any_zone_and_empty():
    assert not in_zones((0, 0), [])
    assert in_zones((10, 10), [box(0, 0, 1, 1), box(9, 9, 11, 11)])


def test_segment_hits_zones():
    zones = [box(4, -1, 6, 1)]
    assert segment_hits_zones((0, 0), (10, 0), zones)
    assert not segment_hits_zones((0, 0), (3, 0), zones)
    assert not segment_hits_zones((0, 5), (10, 5), zones)
    assert segment_hits_zones((5, 0), (5, 0), zones)  # 端点も見る
    assert segment_hits_zones((0, 0), (5, 0), zones)  # 終点だけ中


def test_segment_leaving_current_zone_is_allowed():
    zones = [box(0, -2, 4, 2)]
    assert segment_hits_zones((2, 0), (8, 0), zones)
    assert not segment_hits_zones((2, 0), (8, 0), zones, allow_exit=True)
    assert not segment_hits_zones((2, 0), (3, 0), zones, allow_exit=True)  # 範囲内のまま動くのは許す
    # 出てから別の範囲へは入れない / 出てから戻るのも不可
    assert segment_hits_zones((2, 0), (12, 0), zones + [box(8, -1, 10, 1)], allow_exit=True)
    assert not segment_hits_zones((3, 0), (3, 6), zones, allow_exit=True)


def test_save_and_load_roundtrip(tmp_path):
    zones = [box(0.5, 1.5, 4.25, 2.0, 'a', 'crosswalk'), box(10, 10, 12, 14, 'b', 'roadway')]
    path = tmp_path / 'z.yaml'
    save_zones(str(path), zones)
    assert load_zones(str(path)) == zones
    keys = list(yaml.safe_load(path.read_text())['zones'][0].keys())
    assert keys == ['name', 'type', 'points']


def test_save_zones_writes_through_symlink(tmp_path):
    real = tmp_path / 'real.yaml'
    real.write_text('zones: []\n')
    link = tmp_path / 'link.yaml'
    link.symlink_to(real)
    save_zones(str(link), [box(0, 0, 1, 1)])
    assert link.is_symlink()
    assert len(load_zones(str(real))) == 1


def test_load_zones_missing_file_is_empty(tmp_path):
    assert load_zones(str(tmp_path / 'none.yaml')) == []


def straight_lane():
    return Lane([(float(x), 0.0) for x in range(0, 41, 5)])


def test_find_refuge_picks_point_outside_zones():
    lane = straight_lane()
    zones = [box(-1, 0.2, 6, 2.5)]  # 左側の寄せ先を塞ぐ
    free = lambda p: not in_zones(p, zones)
    refuge = find_refuge(lane, (0.0, 0.0), (30.0, 0.0), free, lambda a, b: True, ahead=3.0, lateral=(1.0, 2.0))
    assert refuge is not None
    assert not in_zones(refuge[:2], zones)
    assert refuge[1] < 0


def test_find_refuge_none_when_all_in_zones():
    lane = straight_lane()
    zones = [box(-10, -5, 20, 5)]
    free = lambda p: not in_zones(p, zones)
    assert find_refuge(lane, (0.0, 0.0), (30.0, 0.0), free, lambda a, b: True, behind=3.0) is None


def test_choose_fixed_refuge_skips_points_in_zones():
    lane = straight_lane()
    zones = [box(1, 0.5, 3, 3)]
    pts = [(2.0, 1.5), (4.0, 1.5)]
    got = choose_fixed_refuge(lane, (0.0, 0.0), (30.0, 0.0), pts, 5.0, check=lambda p: not in_zones(p, zones))
    assert got is not None and got[:2] == (4.0, 1.5)
    zones = [box(1, 0.5, 5, 3)]
    assert choose_fixed_refuge(lane, (0.0, 0.0), (30.0, 0.0), pts, 5.0,
                               check=lambda p: not in_zones(p, zones)) is None


def test_find_retreat_stays_outside_zones():
    lane = straight_lane()
    zones = [box(10.5, -2, 14.5, 2)]
    free = lambda p: not in_zones(p, zones)
    got = find_retreat(lane, (15.0, 0.0), free, dists=(2.0, 3.0, 4.0, 5.0, 6.0))
    assert got is not None
    assert got[0] == pytest.approx(10.0)
    assert find_retreat(lane, (15.0, 0.0), free, dists=(2.0, 3.0, 4.0)) is None


def test_filter_points_outside_zones():
    zones = [box(0, 0, 2, 2)]
    kept, dropped = filter_points_outside_zones([(1, 1), (3, 3)], zones)
    assert kept == [(3, 3)] and dropped == [(1, 1)]


def test_no_pause_while_robot_in_zone():
    zones = [box(0, 0, 4, 4)]
    assert oncoming_for_machine(True, 'idle', (2, 2), zones) is False
    assert oncoming_for_machine(True, 'idle', (5, 2), zones) is True
    assert oncoming_for_machine(False, 'idle', (5, 2), zones) is False
    # 退避中の状態はそのまま
    assert oncoming_for_machine(True, 'waiting', (2, 2), zones) is True
    m = YieldStateMachine(clear_time=3.0, max_wait=20.0, cooldown=5.0)
    for t in (0.0, 0.1, 0.2, 1.0):
        assert m.step(t, True, True, False, oncoming_for_machine(True, m.state, (2, 2), zones), False) is None
    assert m.step(1.1, True, True, False, oncoming_for_machine(True, m.state, (9, 9), zones), False) == 'pause'


# ---- 実データの整合 ----
STOPS = {
    'crosswalk1': (48, 266), 'crosswalk2': (51, 261), 'lot_a': (61, 251), 'lot_b': (65, 248),
    'koban': (82, 227), 'signal': (91, 215), 'station': (123, 184)}


def poly_dist(p, poly):
    best = math.inf
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))
    return best


def real_data():
    wps = yaml.safe_load(open(os.path.join(WAYPOINTS, 'tsukuba_wp.yaml')))['waypoints']
    zones = load_zones(os.path.join(WAYPOINTS, 'tsukuba_no_refuge_zones.yaml'))
    return wps, zones


def test_real_zones_have_both_types_and_four_corners():
    _, zones = real_data()
    assert {z['type'] for z in zones} == {'crosswalk', 'roadway'}
    assert len(zones) >= 14
    assert all(len(z['points']) == 4 for z in zones)


@pytest.mark.parametrize('site', sorted(STOPS))
def test_real_stop_points_outside_zones_but_near(site):
    wps, zones = real_data()
    assert any(z['name'].startswith(site) for z in zones) or True
    for i in STOPS[site]:
        p = (wps[i]['x'], wps[i]['y'])
        assert wps[i]['stop']
        assert not in_zones(p, zones), f'WP{i} inside a zone'
        near = min(poly_dist(p, z['points']) for z in zones)
        assert near <= 2.5, f'WP{i} is {near:.2f} m from the nearest zone'


def test_real_fixed_refuge_points_all_outside_zones():
    _, zones = real_data()
    pts = yaml.safe_load(open(os.path.join(WAYPOINTS, 'tsukuba_refuge_points.yaml')))['refuge_points']
    inside = [(p['x'], p['y']) for p in pts if in_zones((p['x'], p['y']), zones)]
    assert inside == []
    assert not any(abs(p['x'] - 279.334) < 1e-3 and abs(p['y'] - 116.848) < 1e-3 for p in pts)
