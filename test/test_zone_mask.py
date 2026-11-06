import math

from rsf_waypoint_manager.utils.zone_mask import polyline_to_zones, rasterize_zones


def square(x0, y0, size, name='s'):
    return {'name': name, 'type': 'roadway',
            'points': [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]]}


def test_rasterize_single_square():
    ox, oy, w, h, data = rasterize_zones([square(10.0, 20.0, 2.0)], 0.1, 1.0, 100)
    assert (ox, oy) == (9.0, 19.0)
    assert (w, h) == (40, 40)
    assert len(data) == w * h
    assert sum(1 for v in data if v == 100) == 400
    assert set(data) == {0, 100}
    # 行優先: セル (col, row) の中心が多角形の内側
    assert data[15 * w + 15] == 100  # (10.55, 20.55)
    assert data[5 * w + 5] == 0


def test_rasterize_two_polygons_bounding_box():
    ox, oy, w, h, data = rasterize_zones([square(0.0, 0.0, 1.0), square(5.0, 3.0, 1.0)], 0.5, 0.5, 80)
    assert (ox, oy) == (-0.5, -0.5)
    assert (w, h) == (14, 10)
    assert sum(1 for v in data if v == 80) == 8


def test_rasterize_empty():
    assert rasterize_zones([], 0.1, 1.0, 100) == (0.0, 0.0, 1, 1, [0])


def seg_cross(a, b, c, d):
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def test_polyline_to_zones_goes_opposite_to_route():
    route = [(float(x), 0.0) for x in range(0, 11)]
    zones = polyline_to_zones([(0.0, 3.0), (5.0, 3.0), (10.0, 3.0)], route, 2.0, 'rs')
    assert [z['name'] for z in zones] == ['rs_0', 'rs_1']
    for z in zones:
        assert z['type'] == 'roadway' and len(z['points']) == 4
        ys = [p[1] for p in z['points']]
        assert min(ys) >= 3.0 - 1e-9 and math.isclose(max(ys), 5.0)  # 経路 (y=0) と反対の +y 側
    # 反対側の経路でも逆向きに出る
    zones = polyline_to_zones([(0.0, -3.0), (10.0, -3.0)], route, 2.0, 'rs')
    assert math.isclose(min(p[1] for p in zones[0]['points']), -5.0)
    assert max(p[1] for p in zones[0]['points']) <= -3.0 + 1e-9


def test_polyline_to_zones_does_not_cross_route():
    route = [(float(x), 0.0) for x in range(0, 11)]
    zones = polyline_to_zones([(0.0, 3.0), (5.0, 3.0)], route, 2.0, 'rs')
    pts = zones[0]['points']
    for i in range(4):
        for j in range(len(route) - 1):
            assert not seg_cross(pts[i], pts[(i + 1) % 4], route[j], route[j + 1])
