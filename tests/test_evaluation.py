from ppe.evaluation.detection import average_precision, evaluate_detections, iou_matrix, match_greedy
from ppe.evaluation.events import evaluate_events
import numpy as np


def box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def gt_img(src="a"):
    return {"record_type": "image", "source_id": src, "frame_id": 0, "image_width": 1000, "image_height": 1000}


def gt(cls, b, src="a"):
    return {"record_type": "annotation", "source_id": src, "frame_id": 0, "model_class": cls, "bbox": box(*b)}


def pred(cls, b, conf, src="a"):
    return {"source_id": src, "frame_id": 0, "model_class": cls, "class_name": cls, "confidence": conf, "bbox": box(*b)}


def test_iou_matrix_basic():
    m = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert np.allclose(m, [[1.0, 1 / 3, 0.0]])


def test_perfect_predictions_give_perfect_metrics():
    g = [gt_img(), gt("person", (0, 0, 100, 200)), gt("helmet", (10, 0, 50, 30))]
    p = [pred("person", (0, 0, 100, 200), 0.9), pred("helmet", (10, 0, 50, 30), 0.8)]
    r = evaluate_detections(p, g)
    assert r["global"]["micro"]["f1"] == 1.0 and r["global"]["map50"] == 1.0 and r["global"]["map50_95"] == 1.0


def test_tp_fp_fn_counts_and_causes():
    g = [gt_img(), gt("person", (0, 0, 100, 200)), gt("person", (300, 0, 400, 200)), gt("helmet", (10, 0, 50, 30))]
    p = [pred("person", (0, 0, 100, 200), 0.9),               # TP
         pred("person", (700, 700, 800, 900), 0.8),            # FP: fondo
         pred("vest", (10, 0, 50, 30), 0.7)]                   # FP: confusión + FN del casco
    r = evaluate_detections(p, g)
    pc = r["per_class"]
    assert (pc["person"]["tp"], pc["person"]["fp"], pc["person"]["fn"]) == (1, 1, 1)
    assert (pc["helmet"]["tp"], pc["helmet"]["fn"]) == (0, 1)
    assert pc["vest"]["fp"] == 1
    assert r["false_positive_causes"]["false_alarm_background"] == 1
    assert any(k.startswith("class_confusion") for k in r["false_positive_causes"])
    assert any(k.startswith("class_confusion") for k in r["annotations_without_prediction"]["by_cause"])


def test_duplicate_detection_is_a_fp():
    g = [gt_img(), gt("person", (0, 0, 100, 200))]
    p = [pred("person", (0, 0, 100, 200), 0.9), pred("person", (1, 1, 100, 200), 0.8)]
    r = evaluate_detections(p, g)
    assert r["per_class"]["person"]["fp"] == 1 and r["false_positive_causes"]["duplicate_detection"] == 1


def test_confidence_operating_point_filters_but_map_uses_all():
    g = [gt_img(), gt("person", (0, 0, 100, 200))]
    p = [pred("person", (0, 0, 100, 200), 0.3)]
    r = evaluate_detections(p, g, conf_op=0.5)
    assert r["per_class"]["person"]["fn"] == 1 and r["per_class"]["person"]["ap50"] == 1.0
    assert r["annotations_without_prediction"]["by_cause"].get("below_confidence_threshold") == 1


def test_predictions_outside_labeled_scope_are_ignored():
    g = [gt_img("a"), gt("person", (0, 0, 100, 200), "a")]
    p = [pred("person", (0, 0, 100, 200), 0.9, "a"), pred("person", (0, 0, 100, 200), 0.9, "zzz")]
    assert evaluate_detections(p, g)["scope"]["predictions_in_scope"] == 1


def test_images_without_detections_reported():
    g = [gt_img("a"), gt("person", (0, 0, 100, 200), "a"), gt_img("b")]
    r = evaluate_detections([], g)
    assert r["images_without_detections"]["count"] == 1 and r["images_empty_in_both"] == 1


def test_average_precision_known_value():
    conf = np.array([0.9, 0.8, 0.7]); tp = np.array([True, False, True])
    ap, _, _ = average_precision(conf, tp, 2)
    assert 0.7 < ap < 0.85
    assert average_precision(np.array([]), np.array([], dtype=bool), 0)[0] is None


def ev(src, typ, frame=0, bbox=(0, 0, 100, 200), lat=0.0, ts=0.0, cam="CAM_01"):
    return {"source_id": src, "camera_id": cam, "event_type": typ, "frame_id": frame, "timestamp_seconds": ts,
            "latency_seconds": lat, "bbox": box(*bbox)}


def test_event_matching_duplicates_missed_and_fp():
    gt_e = [ev("a", "persona_sin_casco"), ev("b", "persona_sin_chaleco")]
    pr_e = [ev("a", "persona_sin_casco", lat=1.0),
            ev("a", "persona_sin_casco", frame=2),              # duplicado del mismo evento esperado
            ev("c", "persona_sin_casco")]                       # FP (no esperado)
    r = evaluate_events(pr_e, gt_e, tol_frames=5)
    assert r["counts"] == {"predicted_events": 3, "expected_events": 2, "matched": 1, "duplicates": 1,
                           "false_positives": 1, "missed": 1}
    assert r["duplicate_rate"] == round(1 / 3, 4)
    assert r["overall_excluding_duplicates_as_fp"]["precision"] == 0.5
    assert r["latency_seconds"]["pipeline_condition_to_event"]["mean"] == 1.0


def test_two_people_same_source_matched_by_bbox():
    gt_e = [ev("a", "persona_sin_casco", bbox=(0, 0, 100, 200)), ev("a", "persona_sin_casco", bbox=(500, 0, 600, 200))]
    pr_e = [ev("a", "persona_sin_casco", bbox=(502, 0, 601, 200)), ev("a", "persona_sin_casco", bbox=(1, 0, 100, 200))]
    r = evaluate_events(pr_e, gt_e)
    assert r["counts"]["matched"] == 2 and r["overall"]["f1"] == 1.0


def test_event_scope_restricts_to_reviewed_sources():
    r = evaluate_events([ev("zzz", "persona_sin_casco")], [], scope_sources={"a"})
    assert r["counts"]["predicted_events"] == 0


def test_wrong_event_type_is_reported_as_missed_cause():
    r = evaluate_events([ev("a", "persona_sin_chaleco")], [ev("a", "persona_sin_casco")])
    assert r["missed_events"]["by_cause"].get("wrong_event_type_or_severity") == 1 and r["counts"]["false_positives"] == 1
