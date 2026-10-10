from ppe.rules.geometry import point_in_polygon
from ppe.rules.roi import FULL_FRAME, ROI, Zone

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_point_in_polygon_inside_outside_border():
    assert point_in_polygon((0.5, 0.5), SQUARE)
    assert not point_in_polygon((0.1, 0.5), SQUARE)
    assert point_in_polygon((0.2, 0.5), SQUARE)       # borde cuenta como dentro
    assert point_in_polygon((0.8, 0.8), SQUARE)       # vértice


def test_point_in_concave_polygon():
    l_shape = [(0, 0), (1, 0), (1, 0.4), (0.4, 0.4), (0.4, 1), (0, 1)]
    assert point_in_polygon((0.2, 0.8), l_shape)
    assert not point_in_polygon((0.8, 0.8), l_shape)


def test_roi_uses_feet_and_is_resolution_independent():
    roi = ROI([Zone("A", SQUARE)])
    # los pies están en (500, 700) sobre 1000x1000 -> (0.5, 0.7): dentro
    assert roi.locate((400, 100, 600, 700), (1000, 1000)) == "A"
    # misma persona en proporción, imagen 2000x500
    assert roi.locate((800, 25, 1200, 350), (2000, 500)) == "A"
    # pies fuera (y = 0.95)
    assert roi.locate((400, 100, 600, 950), (1000, 1000)) is None


def test_roi_person_outside_returns_none():
    roi = ROI([Zone("A", SQUARE)])
    assert roi.locate((0, 0, 50, 100), (1000, 1000)) is None


def test_roi_disabled_or_empty_is_full_frame():
    assert ROI([Zone("A", SQUARE)], enabled=False).locate((0, 0, 10, 10), (100, 100)) == FULL_FRAME
    assert ROI([]).locate((0, 0, 10, 10), (100, 100)) == FULL_FRAME


def test_roi_from_config_per_camera():
    cfg = {"default_zones": [{"id": "DEF", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
           "cameras": {"CAM_02": [{"id": "ESPECIAL", "polygon": SQUARE}]}}
    assert ROI.from_config(cfg, "CAM_01").locate((0, 0, 10, 10), (100, 100)) == "DEF"
    assert ROI.from_config(cfg, "CAM_02").locate((0, 0, 10, 10), (100, 100)) is None
