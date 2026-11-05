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
