import pytest

from rsf_waypoint_manager.utils.zone_edit import fill_triangles, move_corner, outline_points, save_after_drag
from rsf_waypoint_manager.yield_logic import load_zones


def zones():
    return [
        {'name': 'a', 'type': 'crosswalk', 'points': [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]},
        {'name': 'b', 'type': 'roadway', 'points': [[10.0, 0.0], [14.0, 0.0], [14.0, 2.0], [10.0, 2.0]]}]


def test_move_corner_changes_only_that_corner():
    before = zones()
    after = move_corner(before, 1, 2, 15.5, 3.25)
    assert after[1]['points'][2] == [15.5, 3.25]
    assert after[1]['points'][:2] == before[1]['points'][:2]
    assert after[0] == before[0]
    assert before == zones()  # 元は変えない


def test_move_corner_validates_index():
    with pytest.raises(IndexError):
        move_corner(zones(), 5, 0, 0.0, 0.0)
    with pytest.raises(IndexError):
        move_corner(zones(), 0, 4, 0.0, 0.0)


def test_fill_triangles_covers_quad_with_two_triangles():
    tris = fill_triangles(zones()[0]['points'])
    assert len(tris) == 6  # 2 triangles x 3 vertices
    assert tris[0] == (0.0, 0.0) and tris[1] == (4.0, 0.0) and tris[2] == (4.0, 2.0)
    assert tris[3:] == [(0.0, 0.0), (4.0, 2.0), (0.0, 2.0)]


def test_outline_is_closed():
    pts = outline_points(zones()[0]['points'])
    assert len(pts) == 5 and pts[0] == pts[-1]


def test_save_after_drag_roundtrip(tmp_path):
    path = tmp_path / 'z.yaml'
    new = save_after_drag(str(path), zones(), 0, 3, -1.0, 5.0)
    assert load_zones(str(path)) == new
    assert load_zones(str(path))[0]['points'][3] == [-1.0, 5.0]
    assert load_zones(str(path))[1] == zones()[1]
