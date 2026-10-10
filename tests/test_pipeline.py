"""Prueba de punta a punta con un detector simulado (no requiere GPU ni ultralytics)."""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ppe.pipeline import run_pipeline
from ppe.types import Detection

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "inference.yaml").read_text(encoding="utf-8"))
RULES = yaml.safe_load((ROOT / "configs" / "rules.yaml").read_text(encoding="utf-8"))


class FakeDetector:
    """Devuelve, en el espacio de 640x640 del modelo, una persona con casco y SIN chaleco."""
    names = {0: "helmet", 1: "no_helmet", 2: "no_vest", 3: "person", 4: "vest"}

    def predict(self, img):
        assert img.shape[:2] == (640, 640)                 # letterbox deja siempre 640x640
        return [Detection("person", 0.9, (200.0, 140.0, 400.0, 600.0)),
                Detection("helmet", 0.8, (260.0, 150.0, 340.0, 220.0))]


def make_images(folder: Path, sizes):
    folder.mkdir(parents=True, exist_ok=True)
    for i, (w, h) in enumerate(sizes):
        img = np.full((h, w, 3), 120 + i, dtype=np.uint8)
        cv2.imencode(".jpg", img)[1].tofile(str(folder / f"img_{i:03d}.jpg"))


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(tmp_path, mode, sizes, **overrides):
    src = tmp_path / "imgs"
    make_images(src, sizes)
    (src / "zz_corrupta.jpg").write_bytes(b"basura")                       # no debe detener el lote
    cfg = json.loads(json.dumps(CFG))
    cfg["input"]["mode"] = mode
    for k, v in overrides.items():
        cfg["input"][k] = v
    out = tmp_path / "out"
    metrics = run_pipeline(cfg, RULES, FakeDetector(), src, out, "run_test", {"seed": 42})
    return out, metrics


def test_independent_mode_mixed_resolutions_events_and_persistence(tmp_path):
    out, metrics = run(tmp_path, "independent", [(1280, 720), (320, 240), (640, 640), (1920, 1080)])
    preds, events = read_jsonl(out / "predictions.jsonl"), read_jsonl(out / "events.jsonl")
    assert metrics["volume"]["files_failed"] == 1 and metrics["volume"]["frames"] == 4
    assert len(events) == 4 and all(e["event_type"] == "persona_sin_chaleco" and e["severity"] == "media" for e in events)
    for p in preds:                                                        # cajas dentro de la imagen original
        b, W, H = p["bbox"], p["image_width"], p["image_height"]
        assert 0 <= b["x1"] < b["x2"] <= W and 0 <= b["y1"] < b["y2"] <= H
        assert p["schema_version"] and p["prediction_id"] and p["model_version"] == "1.0.0"
    person = next(p for p in preds if p["model_class"] == "person" and p["image_width"] == 1280)
    assert person["class_name"] == "persona" and person["event_generated"] and person["track_id"] is None
    # 1280x720 -> letterbox 640x360 con 140 px de borde arriba: (200,140,400,600) -> (400, 0, 800, 920)->recortado a H
    assert person["bbox"]["x1"] == 400.0 and person["bbox"]["x2"] == 800.0 and person["bbox"]["y1"] == 0.0
    helmet = next(p for p in preds if p["model_class"] == "helmet" and p["image_width"] == 1280)
    assert helmet["inside_roi"] is True
    idx = read_jsonl(out / "images_index.jsonl")
    assert sum(r["status"] == "failed" for r in idx) == 1
    for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)                                                   # cada línea es JSON válido
    assert (out / "metrics.json").exists()


def test_sequence_mode_requires_persistence_and_emits_once(tmp_path):
    out, metrics = run(tmp_path, "sequence", [(640, 480)] * 8, fps=2.0, source_id="video_001")
    events = read_jsonl(out / "events.jsonl")
    assert len(events) == 1                                                # persiste, pero se emite una sola vez
    e = events[0]
    assert e["source_id"] == "video_001" and e["track_id"] == 1
    assert e["frame_id"] == 4 and e["start_frame_id"] == 0                 # 5 frames (min_frames=5)
    assert e["latency_seconds"] == 2.0                                     # frame 4 a 2 fps
    preds = read_jsonl(out / "predictions.jsonl")
    persons = [p for p in preds if p["model_class"] == "person"]
    assert [p["event_generated"] for p in persons].count(True) == 1
    assert {p["track_id"] for p in persons} == {1}


def test_person_outside_roi_generates_no_event(tmp_path):
    rules = json.loads(json.dumps(RULES))
    rules["roi"]["default_zones"] = [{"id": "ARRIBA", "polygon": [[0, 0], [1, 0], [1, 0.3], [0, 0.3]]}]
    src = tmp_path / "imgs"
    make_images(src, [(640, 480)])
    out = tmp_path / "out"
    run_pipeline(json.loads(json.dumps(CFG)), rules, FakeDetector(), src, out, "r", {})
    assert read_jsonl(out / "events.jsonl") == []
    person = next(p for p in read_jsonl(out / "predictions.jsonl") if p["model_class"] == "person")
    assert person["inside_roi"] is False and person["event_generated"] is False


def test_missing_classes_in_model_fail_fast(tmp_path):
    class Wrong(FakeDetector):
        names = {0: "cat"}
    src = tmp_path / "imgs"
    make_images(src, [(640, 480)])
    try:
        run_pipeline(json.loads(json.dumps(CFG)), RULES, Wrong(), src, tmp_path / "o", "r", {})
    except ValueError as exc:
        assert "clases" in str(exc)
    else:
        raise AssertionError("debía fallar")


def test_empty_folder_raises(tmp_path):
    (tmp_path / "vacia").mkdir()
    try:
        run_pipeline(json.loads(json.dumps(CFG)), RULES, FakeDetector(), tmp_path / "vacia", tmp_path / "o", "r", {})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("debía fallar")
