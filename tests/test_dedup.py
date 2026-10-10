from ppe.inference.postprocess import class_nms, filter_by_confidence
from ppe.rules.tracker import IoUTracker
from ppe.types import Detection


def d(cls, box, conf):
    return Detection(cls, conf, box)


def test_class_nms_removes_same_class_duplicates_only():
    dets = [d("person", (0, 0, 100, 200), 0.9), d("person", (2, 2, 101, 201), 0.7), d("helmet", (2, 2, 101, 201), 0.8)]
    kept, removed = class_nms(dets, 0.8)
    assert removed == 1
    assert {(k.cls, k.conf) for k in kept} == {("person", 0.9), ("helmet", 0.8)}


def test_class_nms_keeps_distant_boxes():
    dets = [d("person", (0, 0, 50, 100), 0.9), d("person", (200, 0, 250, 100), 0.8)]
    assert class_nms(dets, 0.5)[1] == 0


def test_filter_by_confidence_per_class():
    dets = [d("person", (0, 0, 1, 1), 0.30), d("person", (0, 0, 1, 1), 0.40), d("helmet", (0, 0, 1, 1), 0.30)]
    kept, dropped = filter_by_confidence(dets, {"person": 0.35}, 0.25)
    assert dropped == 1 and len(kept) == 2


def test_tracker_keeps_ids_across_frames_and_creates_new_ones():
    t = IoUTracker(0.3, max_age=2)
    assert t.update([(0, 0, 100, 200), (300, 0, 400, 200)]) == [1, 2]
    assert t.update([(5, 0, 105, 200), (305, 0, 405, 200)]) == [1, 2]
    assert t.update([(5, 0, 105, 200), (305, 0, 405, 200), (600, 0, 700, 200)]) == [1, 2, 3]


def test_tracker_drops_old_tracks():
    t = IoUTracker(0.3, max_age=1)
    t.update([(0, 0, 100, 200)])
    t.update([]); t.update([])
    assert t.update([(0, 0, 100, 200)]) == [2]
