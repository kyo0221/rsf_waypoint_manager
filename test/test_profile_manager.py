from rsf_waypoint_manager.utils.profile_manager import merge_profile, near_stop


def wp(x, stop=False):
    return {'x': x, 'y': 0.0, 'stop': stop}


def test_near_stop_true_only_for_final_approach_to_stop():
    waypoints = [wp(0.0), wp(10.0), wp(14.0), wp(16.0, stop=True), wp(30.0)]
    assert near_stop(waypoints, 3, 5.0)
    assert near_stop(waypoints, 2, 5.0)
    assert not near_stop(waypoints, 1, 5.0)
    assert not near_stop(waypoints, 4, 5.0)


def test_merge_profile_overlays_parameters_and_keeps_base_ones():
    base = {'speed_limit': 0.5, 'parameters': {'/a': {'x': 1, 'y': 2}}}
    overlay = {'speed_limit': 0.0, 'parameters': {'/a': {'y': 3}, '/b': {'z': True}}}
    merged = merge_profile(base, overlay)
    assert merged == {'speed_limit': 0.5, 'parameters': {'/a': {'x': 1, 'y': 3}, '/b': {'z': True}}}
    assert base['parameters']['/a'] == {'x': 1, 'y': 2}


def test_split_deferred_holds_controller_params_while_pausing_and_flushes_later():
    # 一時停止 (経路追従の取り消しと同時) の間は controller_server 宛てを送らずに持っておき, 次の走行開始で新しい値を優先して送る
    from rsf_waypoint_manager.utils.profile_manager import split_deferred
    send, held = split_deferred({'/controller_server': {'a': 1, 'b': 2}, '/costmap': {'c': 3}}, {}, defer=True)
    assert send == {'/costmap': {'c': 3}} and held == {'/controller_server': {'a': 1, 'b': 2}}
    send, held = split_deferred({'/controller_server': {'b': 5}}, held, defer=False)
    assert send == {'/controller_server': {'a': 1, 'b': 5}} and held == {}
    send, held = split_deferred({}, {'/controller_server': {'a': 1}}, defer=False)
    assert send == {'/controller_server': {'a': 1}} and held == {}


def test_profile_for_skips_signal_queue_while_oncoming_robot_blocks():
    # 待避の原因になった対向ロボットが前に残っている間は, 待機列とみなさず追い越さない設定を重ねない
    from rsf_waypoint_manager.utils.profile_manager import queue_profile_name
    assert queue_profile_name('wide_road', near=True, oncoming=False, available=True) == 'wide_road+signal_queue'
    assert queue_profile_name('wide_road', near=True, oncoming=True, available=True) == 'wide_road'
    assert queue_profile_name('wide_road', near=False, oncoming=False, available=True) == 'wide_road'
    assert queue_profile_name('wide_road', near=True, oncoming=False, available=False) == 'wide_road'
