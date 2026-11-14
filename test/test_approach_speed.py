"""停止前の減速上限は, 向かっている waypoint が終点そのものの時だけ掛かる"""
import math

from rsf_waypoint_manager.utils.navigation_system import WaypointSystem


def make(waypoints, goal_end_index):
    system = WaypointSystem.__new__(WaypointSystem)
    system.waypoints = waypoints
    system.goal_end_index = goal_end_index
    return system


def loop_course():
    return make([
        {'x': 10.0, 'y': 0.0, 'radius': 2.0},
        {'x': 0.2, 'y': -0.9, 'radius': 2.0},
    ], 1)


def test_no_limit_while_heading_to_other_waypoint_near_goal():
    assert loop_course().approach_speed(0, (0.0, 0.0)) == 0.0


def test_min_speed_when_heading_to_goal_inside_radius():
    assert loop_course().approach_speed(1, (0.0, 0.0)) == 0.2


def test_decelerates_near_goal():
    system = make([{'x': 3.0, 'y': 0.0, 'radius': 2.0}], 0)
    assert math.isclose(system.approach_speed(0, (0.0, 0.0)), math.sqrt(0.32), rel_tol=1e-6)


def test_no_limit_far_from_goal():
    system = make([{'x': 5.0, 'y': 0.0, 'radius': 2.0}], 0)
    assert system.approach_speed(0, (0.0, 0.0)) == 0.0


def test_no_limit_without_position():
    assert loop_course().approach_speed(1, None) == 0.0
