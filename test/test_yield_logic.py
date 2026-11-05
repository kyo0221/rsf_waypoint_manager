import math

import numpy as np
import pytest

from rsf_waypoint_manager.yield_logic import (
    cluster_points, find_oncoming, Lane, route_lane, Track, Tracker, YieldStateMachine)


def straight_lane():
    return Lane([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)])


def test_lane_projects_arclength_and_left_positive_lateral():
    lane = Lane([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    s, lat = lane.project(5.0, 1.0)
    assert s == pytest.approx(5.0)
    assert lat == pytest.approx(1.0)
    s, lat = lane.project(11.0, 4.0)
    assert s == pytest.approx(14.0)
    assert lat == pytest.approx(-1.0)


def test_lane_point_inverts_projection():
    lane = Lane([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    x, y = lane.point(14.0, -1.0)
    assert (x, y) == pytest.approx((11.0, 4.0))


def test_route_lane_starts_behind_robot_and_reaches_ahead():
    route = [(float(10 * i), 0.0) for i in range(10)]
    lane = route_lane(route, 25.0, 0.3, ahead=20.0)
    s, lat = lane.project(25.0, 0.3)
    assert s > 0.0
    assert lat == pytest.approx(0.3)
    assert lane.length - s >= 20.0


def test_cluster_points_groups_nearby_points():
    pts = np.array([[0.0, 0.0], [0.1, 0.05], [0.15, -0.05], [5.0, 5.0], [5.1, 5.1], [5.05, 4.95], [9.0, 9.0]])
    clusters = cluster_points(pts, cell=0.2, min_points=3)
    centers = sorted((c[0], c[1]) for c in clusters)
    assert len(centers) == 2
    assert centers[0] == pytest.approx((0.083, 0.0), abs=0.01)
    assert centers[1] == pytest.approx((5.05, 5.017), abs=0.01)


def test_tracker_estimates_velocity_of_moving_object():
    tracker = Tracker()
    for k in range(6):
        t = 0.2 * k
        tracks = tracker.update(t, [(10.0 - 1.0 * t, 0.0), (3.0, 2.0)])
    moving = [tr for tr in tracks if tr.y < 1.0][0]
    static = [tr for tr in tracks if tr.y > 1.0][0]
    assert moving.vx == pytest.approx(-1.0, abs=0.15)
    assert abs(static.vx) < 0.05 and abs(static.vy) < 0.05
    assert moving.hits == 6


def test_tracker_drops_stale_tracks():
    tracker = Tracker(max_age=0.5)
    tracker.update(0.0, [(1.0, 1.0)])
    tracks = tracker.update(1.0, [])
    assert tracks == []


def make_tracks(objects):
    tracker = Tracker()
    tracks = []
    for k in range(6):
        t = 0.2 * k
        tracks = tracker.update(t, [(x + vx * t, y + vy * t) for x, y, vx, vy in objects])
    return tracks


def test_find_oncoming_detects_object_approaching_in_lane():
    tracks = make_tracks([(15.0, 0.3, -1.0, 0.0)])
    hit = find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3)
    assert hit is not None
    assert hit.s == pytest.approx(13.8, abs=0.3)


# 同方向, 静止 (コーン), 帯の外 (歩行者側), 後方, 遠すぎる
@pytest.mark.parametrize('obj', [
    (15.0, 0.3, 1.0, 0.0),
    (15.0, 0.3, 0.0, 0.0),
    (15.0, 2.5, -1.0, 0.0),
    (0.5, 0.0, -1.0, 0.0),
    (40.0, 0.0, -1.0, 0.0),
])
def test_find_oncoming_ignores_other_objects(obj):
    tracks = make_tracks([obj])
    assert find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3) is None


def run_machine(m, seq):
    actions = []
    for now, kw in seq:
        a = m.step(now, **kw)
        if a:
            actions.append(a)
    return actions


BASE = dict(enabled=True, running=True, paused=False, oncoming=False, goal_done=False)


def test_state_machine_pause_goto_wait_resume():
    m = YieldStateMachine(clear_time=3.0, max_wait=60.0, cooldown=5.0)
    actions = run_machine(m, [
        (0.0, BASE),
        (1.0, dict(BASE, oncoming=True)),
        (1.5, dict(BASE, running=False, paused=True, oncoming=True)),
        (5.0, dict(BASE, running=False, paused=True, oncoming=True, goal_done=True)),
        (8.0, dict(BASE, running=False, paused=True, oncoming=False, goal_done=True)),
        (10.0, dict(BASE, running=False, paused=True, oncoming=False, goal_done=True)),
        (11.5, dict(BASE, running=False, paused=True, oncoming=False, goal_done=True)),
    ])
    assert actions == ['pause', 'goto', 'resume']


def test_state_machine_ignores_oncoming_when_disabled_or_not_running():
    m = YieldStateMachine(clear_time=3.0, max_wait=60.0, cooldown=5.0)
    actions = run_machine(m, [
        (0.0, dict(BASE, enabled=False, oncoming=True)),
        (1.0, dict(BASE, running=False, oncoming=True)),
    ])
    assert actions == []


def test_state_machine_resumes_after_max_wait():
    m = YieldStateMachine(clear_time=3.0, max_wait=20.0, cooldown=5.0)
    seq = [(0.0, dict(BASE, oncoming=True)), (0.5, dict(BASE, running=False, paused=True, oncoming=True))]
    seq += [(1.0 + k, dict(BASE, running=False, paused=True, oncoming=True, goal_done=True)) for k in range(25)]
    assert run_machine(m, seq) == ['pause', 'goto', 'resume']


def test_state_machine_cooldown_blocks_immediate_retrigger():
    m = YieldStateMachine(clear_time=1.0, max_wait=60.0, cooldown=5.0)
    actions = run_machine(m, [
        (0.0, dict(BASE, oncoming=True)),
        (0.5, dict(BASE, running=False, paused=True, oncoming=True)),
        (1.0, dict(BASE, running=False, paused=True, goal_done=True)),
        (2.5, dict(BASE, running=False, paused=True, goal_done=True)),
        (3.0, dict(BASE, oncoming=True)),
        (9.0, dict(BASE, oncoming=True)),
    ])
    assert actions == ['pause', 'goto', 'resume', 'pause']


def test_state_machine_gives_up_when_pause_never_takes_effect():
    m = YieldStateMachine(clear_time=3.0, max_wait=60.0, cooldown=5.0, pause_timeout=5.0)
    seq = [(0.0, dict(BASE, oncoming=True))] + [(1.0 + k, dict(BASE, oncoming=True)) for k in range(6)]
    actions = run_machine(m, seq)
    assert actions == ['pause']
    assert m.state == 'idle'


def test_lane_point_heading():
    lane = Lane([(0.0, 0.0), (0.0, 10.0)])
    assert lane.heading(5.0) == pytest.approx(math.pi / 2)


def test_project_many_matches_project():
    lane = Lane([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    pts = np.array([[5.0, 1.0], [11.0, 4.0], [-3.0, 0.5], [12.0, 12.0]])
    s, lat = lane.project_many(pts)
    for k, (x, y) in enumerate(pts):
        assert (s[k], lat[k]) == pytest.approx(lane.project(x, y))


def test_find_oncoming_ignores_objects_closer_than_min_ahead():
    tracks = make_tracks([(5.0, 0.0, -1.0, 0.0)])
    lane = straight_lane()
    assert find_oncoming(lane, 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3,
                         min_ahead=2.5) is None
    assert find_oncoming(lane, 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3,
                         min_ahead=0.5) is not None


def test_state_machine_stays_at_refuge_while_yielded_object_is_ahead():
    m = YieldStateMachine(clear_time=3.0, max_wait=20.0, cooldown=5.0)
    seq = [(0.0, dict(BASE, oncoming=True)),
           (0.5, dict(BASE, running=False, paused=True, oncoming=True)),
           (2.0, dict(BASE, running=False, paused=True, goal_done=True, waiting_for=True))]
    seq += [(3.0 + k, dict(BASE, running=False, paused=True, goal_done=True, waiting_for=True)) for k in range(10)]
    assert run_machine(m, seq) == ['pause', 'goto']
    assert m.state == 'waiting'
    later = [(13.0 + k, dict(BASE, running=False, paused=True, goal_done=True)) for k in range(4)]
    assert run_machine(m, later) == ['resume']


def test_state_machine_resumes_after_patience_when_object_never_leaves():
    m = YieldStateMachine(clear_time=3.0, max_wait=20.0, cooldown=5.0)
    seq = [(0.0, dict(BASE, oncoming=True)), (0.5, dict(BASE, running=False, paused=True, oncoming=True))]
    seq += [(1.0 + k, dict(BASE, running=False, paused=True, goal_done=True, waiting_for=True)) for k in range(25)]
    assert run_machine(m, seq) == ['pause', 'goto', 'resume']


def noisy_tracks(x0, vx, seconds=1.5, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    tracker = Tracker()
    tracks = []
    for k in range(int(seconds / 0.1) + 1):
        t = 0.1 * k
        tracks = tracker.update(t, [(x0 + vx * t + rng.uniform(-noise, noise), 0.2 + rng.uniform(-noise, noise))])
    return tracks


def test_jittering_static_object_is_not_oncoming():
    for seed in range(5):
        tracks = noisy_tracks(8.0, 0.0, seed=seed)
        assert find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.25,
                             min_hits=3, min_ahead=2.5) is None


def test_noisy_moving_object_is_oncoming():
    for seed in range(5):
        tracks = noisy_tracks(14.0, -0.8, seed=seed)
        assert find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.25,
                             min_hits=3, min_ahead=2.5) is not None


def test_implausibly_fast_object_is_ignored():
    tracks = make_tracks([(15.0, 0.0, -6.0, 0.0)])
    assert find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.25,
                         min_hits=3, min_ahead=2.5) is None


def test_static_object_with_jumping_centroid_is_not_oncoming():
    pattern = [0.0, -0.5, -0.5, 0.0, -0.5, 0.0, 0.0, -0.5, -0.5, -0.5, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0]
    for shift in range(len(pattern)):
        tracker = Tracker()
        lane = straight_lane()
        for k in range(len(pattern)):
            tracks = tracker.update(0.1 * k, [(8.0 + pattern[(k + shift) % len(pattern)], 0.2)])
            assert find_oncoming(lane, 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.25, min_hits=3,
                                 min_ahead=2.5) is None, (shift, k)


def test_find_oncoming_ignores_tracks_not_seen_in_latest_frame():
    tracks = make_tracks([(15.0, 0.3, -1.0, 0.0)])
    lane = straight_lane()
    assert find_oncoming(lane, 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3, now=1.0) is not None
    assert find_oncoming(lane, 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.2, min_hits=3, now=1.5) is None


def test_state_machine_requires_oncoming_to_persist_for_confirm_time():
    m = YieldStateMachine(clear_time=3.0, max_wait=20.0, cooldown=5.0, confirm_time=0.5)
    blips = [(0.0, dict(BASE, oncoming=True)), (0.1, dict(BASE, oncoming=True)), (0.2, BASE),
             (1.0, dict(BASE, oncoming=True)), (1.2, BASE)]
    assert run_machine(m, blips) == []
    steady = [(2.0 + 0.1 * k, dict(BASE, oncoming=True)) for k in range(7)]
    assert run_machine(m, steady) == ['pause']


def line_route(n=6, step=10.0):
    return [(step * i, 0.0) for i in range(n)]


def test_next_waypoint_index_skips_waypoint_already_passed():
    from rsf_waypoint_manager.yield_logic import next_waypoint_index
    stops = [False] * 6
    assert next_waypoint_index(line_route(), stops, (24.0, 1.5), current=2) == 3


def test_next_waypoint_index_keeps_current_when_not_yet_reached():
    from rsf_waypoint_manager.yield_logic import next_waypoint_index
    stops = [False] * 6
    assert next_waypoint_index(line_route(), stops, (15.0, 1.5), current=2) == 2


def test_next_waypoint_index_skips_waypoint_right_beside_robot():
    from rsf_waypoint_manager.yield_logic import next_waypoint_index
    stops = [False] * 6
    assert next_waypoint_index(line_route(), stops, (29.5, 1.5), current=2, margin=1.0) == 4


def test_next_waypoint_index_never_skips_stop_waypoint():
    from rsf_waypoint_manager.yield_logic import next_waypoint_index
    stops = [False, False, False, True, False, False]
    assert next_waypoint_index(line_route(), stops, (35.0, 1.5), current=2) == 3


def test_next_waypoint_index_ignores_return_route_on_same_sidewalk():
    from rsf_waypoint_manager.yield_logic import next_waypoint_index
    route = line_route() + [(45.0, 0.3), (35.0, 0.3), (25.0, 0.3), (15.0, 0.3)]
    stops = [False] * len(route)
    assert next_waypoint_index(route, stops, (24.0, 1.5), current=2) == 3


def track_series(positions, dt=0.1, **kw):
    tracker = Tracker(**kw)
    tracks = []
    for k, x in enumerate(positions):
        tracks = tracker.update(dt * k, [(x, 0.2)])
    return tracks


def test_slow_oncoming_robot_is_detected_by_travel():
    tracks = track_series([14.0 - 0.6 * 0.1 * k for k in range(21)], window=2.0, min_span=1.0)
    hit = find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.3, min_hits=3,
                        min_ahead=2.5, min_travel=0.6)
    assert hit is not None


def test_bounded_centroid_drift_is_not_oncoming():
    drift = [8.0 - min(0.4, 0.04 * k) for k in range(21)]
    tracks = track_series(drift, window=2.0, min_span=1.0)
    assert find_oncoming(straight_lane(), 2.0, tracks, half_width=1.2, ahead=20.0, min_speed=0.3, min_hits=3,
                         min_ahead=2.5, min_travel=0.6) is None


def test_tracks_in_frame_moves_position_and_rotates_velocity():
    from rsf_waypoint_manager.yield_logic import tracks_in_frame
    tr = Track(7, 1.0, 0.0, vx=1.0, vy=0.0, hits=5, reliable=True, span=1.5)
    out = tracks_in_frame([tr], 10.0, 5.0, math.pi / 2)
    assert (out[0].x, out[0].y) == pytest.approx((10.0, 6.0))
    assert (out[0].vx, out[0].vy) == pytest.approx((0.0, 1.0))
    assert out[0].id == 7 and out[0].reliable and out[0].span == 1.5
    assert (tr.x, tr.y) == (1.0, 0.0)


def test_find_leader_picks_same_direction_mover_in_lane():
    from rsf_waypoint_manager.yield_logic import find_leader
    tracks = make_tracks([(8.0, 0.2, 0.4, 0.0), (12.0, 0.1, 0.0, 0.0), (6.0, 1.5, 0.5, 0.0), (15.0, 0.0, -1.0, 0.0)])
    leader = find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.2, min_hits=3)
    assert leader is not None
    assert leader.y == pytest.approx(0.2)


def test_find_leader_none_without_same_direction_mover():
    from rsf_waypoint_manager.yield_logic import find_leader
    tracks = make_tracks([(12.0, 0.1, 0.0, 0.0), (15.0, 0.0, -1.0, 0.0)])
    assert find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.2, min_hits=3) is None


def test_follow_speed_keeps_standoff_and_never_returns_zero():
    from rsf_waypoint_manager.yield_logic import follow_speed
    assert follow_speed(6.0, standoff=1.5, time_gap=1.5, max_speed=1.0) == pytest.approx(1.0)
    assert follow_speed(2.25, standoff=1.5, time_gap=1.5, max_speed=1.0) == pytest.approx(0.5)
    v = follow_speed(1.0, standoff=1.5, time_gap=1.5, max_speed=1.0)
    assert 0.0 < v < 0.01


def test_find_leader_ignores_bounded_forward_drift_of_static_object():
    from rsf_waypoint_manager.yield_logic import find_leader
    drift = [8.0 + min(0.4, 0.04 * k) for k in range(21)]
    tracks = track_series(drift, window=2.0, min_span=1.0)
    assert find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.15, min_hits=3,
                       min_travel=0.6) is None


def test_find_leader_detects_slow_robot_by_travel():
    from rsf_waypoint_manager.yield_logic import find_leader
    tracks = track_series([5.0 + 0.4 * 0.1 * k for k in range(21)], window=2.0, min_span=1.0)
    assert find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.15, min_hits=3,
                       min_travel=0.6) is not None


def test_leader_monitor_releases_leader_that_stays_still():
    from rsf_waypoint_manager.yield_logic import LeaderMonitor
    m = LeaderMonitor(release_after=10.0, moving_speed=0.15)
    assert m.update(0.0, 7, 0.4) is True
    assert m.update(5.0, 7, 0.0) is True
    assert m.update(9.9, 7, 0.05) is True
    assert m.update(10.1, 7, 0.0) is False
    assert m.update(30.0, 7, 0.5) is False


def test_leader_monitor_keeps_leader_that_moves_again():
    from rsf_waypoint_manager.yield_logic import LeaderMonitor
    m = LeaderMonitor(release_after=10.0, moving_speed=0.15)
    m.update(0.0, 3, 0.4)
    m.update(8.0, 3, 0.0)
    assert m.update(9.0, 3, 0.3) is True
    assert m.update(18.0, 3, 0.0) is True
    assert m.update(19.5, 3, 0.0) is False


def test_find_leader_rejects_object_size_drift_over_three_seconds():
    from rsf_waypoint_manager.yield_logic import find_leader
    drift = [8.0 + min(0.8, 0.04 * k) for k in range(31)]
    tracks = track_series(drift, window=3.0, min_span=1.0)
    assert find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.15, min_hits=3,
                       min_travel=1.0) is None


def test_find_leader_accepts_slow_robot_over_three_seconds():
    from rsf_waypoint_manager.yield_logic import find_leader
    tracks = track_series([5.0 + 0.4 * 0.1 * k for k in range(31)], window=3.0, min_span=1.0)
    assert find_leader(straight_lane(), 2.0, tracks, half_width=0.6, ahead=15.0, min_speed=0.15, min_hits=3,
                       min_travel=1.0) is not None


def test_refuge_goal_canceled_right_after_send_is_retried():
    # 一時停止の取り消しが後から届いて待機点ゴールが消されたら, 送り直す
    from rsf_waypoint_manager.yield_logic import should_resend_refuge
    canceled, aborted = 5, 6
    assert should_resend_refuge(canceled, elapsed=0.05, retries=0, canceled_by_us=False) is True
    assert should_resend_refuge(aborted, elapsed=0.3, retries=1, canceled_by_us=False) is True


def test_refuge_goal_not_retried_when_finished_or_too_many_times():
    from rsf_waypoint_manager.yield_logic import should_resend_refuge
    canceled, aborted, succeeded = 5, 6, 4
    assert should_resend_refuge(succeeded, elapsed=0.05, retries=0, canceled_by_us=False) is False
    assert should_resend_refuge(canceled, elapsed=0.05, retries=0, canceled_by_us=True) is False
    assert should_resend_refuge(canceled, elapsed=0.05, retries=3, canceled_by_us=False) is False
    # 走り出してから失敗したもの (経路が無い等) はその場で待つ
    assert should_resend_refuge(aborted, elapsed=8.0, retries=0, canceled_by_us=False) is False


def test_global_obstacle_layer_is_disabled_while_following_a_leader():
    # 動いている前走ロボットを global の障害物層に載せると, 追い越す経路 (歩行者側) が引かれる
    from rsf_waypoint_manager.yield_logic import obstacle_layer_action
    assert obstacle_layer_action(following=True, layer_off=False, follow_enabled=True) == 'disable'
    assert obstacle_layer_action(following=True, layer_off=True, follow_enabled=True) is None
    assert obstacle_layer_action(following=False, layer_off=True, follow_enabled=True) == 'enable'
    assert obstacle_layer_action(following=False, layer_off=False, follow_enabled=True) is None


def test_global_obstacle_layer_left_to_profile_when_follow_disabled():
    # queue_guard を使うプロファイル (follow_enabled=false) は自分で障害物層を切るので戻さない
    from rsf_waypoint_manager.yield_logic import obstacle_layer_action
    assert obstacle_layer_action(following=False, layer_off=True, follow_enabled=False) == 'forget'
    assert obstacle_layer_action(following=False, layer_off=False, follow_enabled=False) is None


def test_route_lane_uses_current_waypoint_on_overlapping_out_and_back():
    # 往路 (x: 0->40) と復路 (x: 40->0) が同じ道を通るとき, 向かっている waypoint の近くの区間でレーンを作る
    route = [(float(x), 0.0) for x in range(0, 41, 10)] + [(float(x), 0.3) for x in range(40, -1, -10)]
    out = route_lane(route, 15.0, 0.1, 12.0)
    assert out.heading(0.0) == pytest.approx(0.0, abs=0.1)
    back = route_lane(route, 15.0, 0.1, 12.0, hint=8)
    assert abs(back.heading(0.0)) == pytest.approx(math.pi, abs=0.1)


def test_smooth_limit_rises_slowly_and_falls_faster():
    # 追従の速度上限を急に上げ下げすると, 低速の MPPI が旋回で調整して蛇行する
    from rsf_waypoint_manager.yield_logic import smooth_limit
    assert smooth_limit(0.2, 1.0, dt=0.1, up=0.3, down=1.0) == pytest.approx(0.23)
    assert smooth_limit(0.8, 0.2, dt=0.1, up=0.3, down=1.0) == pytest.approx(0.7)
    assert smooth_limit(0.5, 0.52, dt=0.1, up=0.3, down=1.0) == pytest.approx(0.52)
    assert smooth_limit(None, 0.4, dt=0.1, up=0.3, down=1.0) == pytest.approx(0.4)


def test_predict_track_extrapolates_briefly_lost_leader():
    # 前走ロボットの検知が一瞬途切れても, 速度で位置を進めて追従を続ける (経路の左右の切り替わりによる蛇行の防止)
    from rsf_waypoint_manager.yield_logic import predict_track
    tr = Track(id=1, x=10.0, y=0.0, vx=0.4, vy=0.0, t=5.0)
    p = predict_track(tr, now=6.0, max_dt=3.0)
    assert (p.x, p.y, p.t) == (pytest.approx(10.4), 0.0, 5.0)
    far = predict_track(tr, now=20.0, max_dt=3.0)
    assert far.x == pytest.approx(11.2)
    assert tr.x == 10.0



def test_find_refuge_picks_forward_diagonal_away_from_the_oncoming_side():
    # 相手がレーンの右寄り (lateral<0) なら左の点. 前方斜め前 (前方 3m まで, 出会う地点より手前) を優先する
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, -0.5), free=lambda p: True, path_clear=lambda a, b: True)
    assert r[0] == pytest.approx(8.0) and r[1] == pytest.approx(2.0) and r[2] == pytest.approx(0.0)


def test_find_refuge_stays_short_of_the_meeting_point():
    # 相手まで 4m なら前方は 4/2-0.5 = 1.5m まで
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (9.0, -0.5), free=lambda p: True, path_clear=lambda a, b: True)
    assert r[0] == pytest.approx(6.5) and r[1] == pytest.approx(2.0)


def test_find_refuge_falls_back_to_the_side_when_opponent_is_close():
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (6.5, -0.5), free=lambda p: True, path_clear=lambda a, b: True)
    assert r[0] == pytest.approx(5.0) and r[1] == pytest.approx(2.0)


def test_find_refuge_uses_other_side_when_far_side_is_blocked():
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, -0.5), free=lambda p: p[1] < 0.0, path_clear=lambda a, b: True)
    assert r[0] == pytest.approx(8.0) and r[1] == pytest.approx(-2.0)


def test_find_refuge_skips_points_whose_way_is_blocked():
    # 3m 先の点への直線が塞がれていれば, 手前の点を選ぶ
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, -0.5), free=lambda p: p[1] > 0.0,
                    path_clear=lambda a, b: b[0] <= 6.6)
    assert r[0] == pytest.approx(6.5) and r[1] == pytest.approx(2.0)


def test_find_refuge_returns_none_when_nothing_is_free():
    from rsf_waypoint_manager.yield_logic import find_refuge
    assert find_refuge(straight_lane(), (5.0, 0.0), (12.0, 0.0), free=lambda p: False,
                       path_clear=lambda a, b: True) is None


def test_align_angle_returns_signed_turn_or_none():
    # 待機点に着いたら, レーンの向き (相手が来る方) へ向きをそろえる. 回転量は -pi〜pi の短い方
    from rsf_waypoint_manager.yield_logic import align_angle
    assert align_angle(0.5, 0.0, tol=0.2) == pytest.approx(-0.5)
    assert align_angle(3.0, -3.0, tol=0.2) == pytest.approx(2 * math.pi - 6.0)
    assert align_angle(0.1, 0.0, tol=0.2) is None


def test_tracker_clear_keeps_ids_unique():
    # odom が飛んで追跡をやり直しても, 前の相手と同じ ID を振らない (待避済みの相手と取り違えて検知を逃さない)
    tracker = Tracker()
    tracker.update(0.0, [(1.0, 0.0), (5.0, 0.0)])
    used = {tr.id for tr in tracker.tracks}
    tracker.clear()
    assert tracker.tracks == []
    tracker.update(1.0, [(9.0, 0.0)])
    assert tracker.tracks[0].id not in used


def test_target_still_present_near_its_last_position():
    # 相手の追跡が途切れても, 最後に見えた位置の近くに点が残っていれば相手はまだそこにいる (コーンなど離れた点は見ない)
    from rsf_waypoint_manager.yield_logic import points_near
    pts = np.array([[7.0, 0.2], [9.0, 1.2]])
    assert points_near(pts, (7.3, 0.0), 0.8) is True
    assert points_near(pts, (5.0, 0.0), 0.8) is False
    assert points_near(np.zeros((0, 2)), (7.3, 0.0), 0.8) is False


def test_find_refuge_limited_to_given_side():
    # 歩行者側 (keepout の帯がある側) だけを探す. 帯の手前 (レーン内の歩行者側寄り) の点も使える
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, -0.5), free=lambda p: p[1] > -1.6, path_clear=lambda a, b: True,
                    sides=(-1.0,))
    assert r[1] == pytest.approx(-1.5)
    assert find_refuge(straight_lane(), (5.0, 0.0), (12.0, -0.5), free=lambda p: p[1] > 0.0,
                       path_clear=lambda a, b: True, sides=(-1.0,)) is None


def test_find_refuge_rejects_points_still_in_opponents_path():
    # 相手の横位置から min_sep 未満の点は待避にならない (相手が止まったままになる)
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, 0.0), free=lambda p: abs(p[1]) <= 0.6,
                    path_clear=lambda a, b: True, lateral=(0.5, 2.0), min_sep=1.0)
    assert r is None


def test_find_refuge_falls_back_to_behind_only_when_forward_and_side_fail():
    # 前方と真横に使える点が無いときだけ, 後方 (behind m まで) を探す
    from rsf_waypoint_manager.yield_logic import find_refuge
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, 0.0), free=lambda p: p[0] < 4.6,
                    path_clear=lambda a, b: True, lateral=(0.5, 2.0), min_sep=1.0, behind=3.0, sides=(-1.0,))
    assert r[0] == pytest.approx(4.5) and r[1] == pytest.approx(-2.0)
    r = find_refuge(straight_lane(), (5.0, 0.0), (12.0, 0.0), free=lambda p: True,
                    path_clear=lambda a, b: True, lateral=(0.5, 2.0), min_sep=1.0, behind=3.0, sides=(-1.0,))
    assert r[0] > 5.0


def test_choose_fixed_refuge_nearest_ahead_before_meeting_point():
    # 固定の退避点のうち, 前方で出会う地点 (相手との間隔の半分 - 0.5m) より手前の最も近い点
    from rsf_waypoint_manager.yield_logic import choose_fixed_refuge
    points = [(3.0, -1.5), (6.0, -1.5), (9.0, -1.5), (12.0, -1.5)]
    r = choose_fixed_refuge(straight_lane(), (5.0, 0.0), (16.0, 0.0), points, ahead=5.0)
    assert r[:2] == (6.0, -1.5) and r[2] == pytest.approx(0.0)
    # 相手が近いと出会う地点より先の点は使えない
    assert choose_fixed_refuge(straight_lane(), (5.0, 0.0), (7.0, 0.0), points, ahead=5.0) is None


def test_choose_fixed_refuge_skips_points_failing_check():
    from rsf_waypoint_manager.yield_logic import choose_fixed_refuge
    points = [(6.0, -1.5), (9.0, -1.5)]
    r = choose_fixed_refuge(straight_lane(), (5.0, 0.0), (20.0, 0.0), points, ahead=5.0, check=lambda p: p[0] > 7.0)
    assert r[:2] == (9.0, -1.5)
    assert choose_fixed_refuge(straight_lane(), (5.0, 0.0), (20.0, 0.0), points, ahead=5.0,
                               check=lambda p: False) is None


def test_choose_fixed_refuge_ignores_points_of_other_lanes():
    # 横に離れた点 (別の道の点) は使わない
    from rsf_waypoint_manager.yield_logic import choose_fixed_refuge
    r = choose_fixed_refuge(straight_lane(), (5.0, 0.0), (20.0, 0.0), [(6.0, 8.0), (7.5, -1.2)], ahead=5.0)
    assert r[:2] == (7.5, -1.2)


def test_skip_blocked_waypoint_only_when_near_and_not_stop():
    # 向かっている waypoint の上に物体があり, ロボットが近づいていて, 停止点でなければ次へ飛ばす
    from rsf_waypoint_manager.yield_logic import should_skip_waypoint
    wp = (10.0, 0.0)
    box = np.array([[10.1, 0.1], [10.2, -0.1]])
    assert should_skip_waypoint(wp, False, (5.0, 0.0), box, radius=0.6, within=7.0) is True
    assert should_skip_waypoint(wp, True, (5.0, 0.0), box, radius=0.6, within=7.0) is False
    assert should_skip_waypoint(wp, False, (1.0, 0.0), box, radius=0.6, within=7.0) is False
    assert should_skip_waypoint(wp, False, (5.0, 0.0), np.array([[11.0, 1.0]]), radius=0.6, within=7.0) is False


def test_find_retreat_returns_nearest_free_lane_point_behind():
    # 待避点が無いとき, レーンの中央を後ろへ下がる点 (近い順). 向きはレーンの向きのまま (後退で下がる)
    from rsf_waypoint_manager.yield_logic import find_retreat
    r = find_retreat(straight_lane(), (10.0, 0.0), free=lambda p: True)
    assert r[0] == pytest.approx(8.5) and r[1] == pytest.approx(0.0) and r[2] == pytest.approx(0.0)
    r = find_retreat(straight_lane(), (10.0, 0.0), free=lambda p: p[0] < 8.1)
    assert r[0] == pytest.approx(8.0)
    assert find_retreat(straight_lane(), (10.0, 0.0), free=lambda p: False) is None


def test_find_retreat_tries_lateral_offsets_at_each_distance():
    # 下がった先で歩行者側 (右 = -1) へ寄せた点を先に試す. 近い距離を優先
    from rsf_waypoint_manager.yield_logic import find_retreat
    r = find_retreat(straight_lane(), (10.0, 0.0), free=lambda p: p[0] < 7.6,
                     dists=(2.0, 2.5, 3.0), laterals=(-1.0, 0.0))
    assert r[0] == pytest.approx(7.5) and r[1] == pytest.approx(-1.0)
    r = find_retreat(straight_lane(), (10.0, 0.0), free=lambda p: p[1] > -0.5,
                     dists=(2.0,), laterals=(-1.0, 0.0))
    assert r[0] == pytest.approx(8.0) and r[1] == pytest.approx(0.0)


def test_near_stop_ahead_detects_stop_within_route_distance():
    # 信号の停止点の手前 (経路に沿って within[m] 以内) の waypoint は飛ばさない (待機列を追い越さない)
    from rsf_waypoint_manager.yield_logic import near_stop_ahead
    route = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
    stops = [False, False, False, True, False]
    assert near_stop_ahead(route, stops, 2, 20.0) is True
    assert near_stop_ahead(route, stops, 1, 20.0) is True
    assert near_stop_ahead(route, stops, 0, 20.0) is False
    assert near_stop_ahead(route, stops, 4, 20.0) is False


def test_stop_ahead_within_returns_stop_index():
    # 経路に沿って within[m] 以内の先の停止点の番号 (待機列の区間). 無ければ None
    from rsf_waypoint_manager.yield_logic import stop_ahead_within
    route = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (25.0, 0.0), (40.0, 0.0)]
    stops = [False, False, False, True, False]
    assert stop_ahead_within(route, stops, 2, 12.0) == 3
    assert stop_ahead_within(route, stops, 3, 12.0) == 3
    assert stop_ahead_within(route, stops, 1, 12.0) is None
    assert stop_ahead_within(route, stops, 4, 12.0) is None


def test_stalled_needs_full_window_without_motion():
    from rsf_waypoint_manager.yield_logic import stalled
    h = [(0.0, (0.0, 0.0)), (4.0, (0.05, 0.0)), (8.5, (0.08, 0.0))]
    assert stalled(h, 8.5, 8.0, 0.1)
    assert not stalled(h, 8.5, 8.0, 0.05)
    assert not stalled(h[1:], 8.5, 8.0, 0.1)


def test_blocked_ahead_checks_points_in_front_of_robot():
    from rsf_waypoint_manager.yield_logic import blocked_ahead
    assert blocked_ahead([(0.4, 0.1)], (0.0, 0.0, 0.0), 0.5, 0.45)
    assert not blocked_ahead([(-0.4, 0.0), (0.4, 0.6), (0.8, 0.0)], (0.0, 0.0, 0.0), 0.5, 0.45)
    assert blocked_ahead([(0.0, 0.4)], (0.0, 0.0, math.pi / 2), 0.5, 0.45)


def test_backup_distance_grows_when_stuck_again_at_same_place():
    from rsf_waypoint_manager.yield_logic import backup_distance
    assert backup_distance(None, 10.0, (0.0, 0.0)) == (0.4, (10.0, (0.0, 0.0), 1))
    d, last = backup_distance((10.0, (0.0, 0.0), 1), 30.0, (0.3, 0.0))
    assert d == pytest.approx(0.8) and last[2] == 2
    d, last = backup_distance((30.0, (0.3, 0.0), 3), 50.0, (0.3, 0.0))
    assert d == pytest.approx(1.2)
    d, last = backup_distance((10.0, (0.0, 0.0), 2), 100.0, (0.0, 0.0))
    assert d == pytest.approx(0.4)
    d, last = backup_distance((10.0, (0.0, 0.0), 2), 20.0, (3.0, 0.0))
    assert d == pytest.approx(0.4)
