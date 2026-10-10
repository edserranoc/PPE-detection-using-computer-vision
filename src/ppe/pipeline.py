"""Pipeline de inferencia de EPP sobre una carpeta de imágenes.

Flujo por imagen:
    leer/validar → adaptar resolución → detectar → volver a coordenadas originales →
    umbrales por clase + NMS por clase → asociar persona↔EPP → ROI → condición de
    incumplimiento → tracking (modo secuencia) → persistencia temporal / duplicados →
    predictions.jsonl + events.jsonl + images_index.jsonl + metrics.json

Un error en una imagen se registra y NO detiene el lote.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from ppe.inference.postprocess import class_nms, filter_by_confidence
from ppe.inference.preprocess import ImageReadError, preprocess, read_image
from ppe.monitoring.metrics import RunMonitor
from ppe.persistence.writers import JsonlWriter
from ppe.rules.association import associate
from ppe.rules.events import EventManager, PersonObs, condition_key, missing_ppe
from ppe.rules.roi import ROI
from ppe.rules.tracker import IoUTracker
from ppe.types import SCHEMA_VERSION, Detection

LOG = logging.getLogger("ppe.pipeline")
MODES = ("independent", "sequence")


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def collect_images(source: Path, extensions: list[str], recursive: bool) -> list[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if source.is_file():
        return [source]
    it = source.rglob("*") if recursive else source.glob("*")
    return sorted((p for p in it if p.is_file() and p.suffix.lower() in exts), key=_natural_key)


def group_sources(paths: list[Path], root: Path, mode: str, forced_source_id: str | None):
    """Devuelve [(source_id, [rutas])]. En modo secuencia, cada subcarpeta es una fuente (como un video)."""
    if mode == "independent":
        out = []
        for p in paths:
            rel = p.relative_to(root) if root.is_dir() else Path(p.name)
            out.append((rel.with_suffix("").as_posix(), [p]))
        return out
    groups: dict[Path, list[Path]] = {}
    for p in paths:
        groups.setdefault(p.parent, []).append(p)
    if forced_source_id and len(groups) == 1:
        return [(forced_source_id, next(iter(groups.values())))]
    return [(d.name or "source", sorted(ps, key=_natural_key)) for d, ps in sorted(groups.items())]


# --------------------------------------------------------------------------- #
# Anotación visual (opcional)
# --------------------------------------------------------------------------- #
def _annotate(img: np.ndarray, dets: list[Detection], roi: ROI, size: tuple[int, int],
              out_names: dict, events: list[dict]) -> np.ndarray:
    out = img.copy()
    for zid, pts in roi.polygons_px(size):
        cv2.polylines(out, [np.array(pts, dtype=np.int32)], True, (255, 200, 0), 2)
        cv2.putText(out, zid, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    thick = max(2, size[0] // 400)
    for d in dets:
        bad = d.condition is not None and d.inside_roi
        color = (0, 0, 255) if bad else (0, 200, 0)
        if d.cls.startswith("no_"):
            color = (0, 128, 255)
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{out_names.get(d.cls, d.cls)} {d.conf:.2f}"
        if d.event_generated:
            label += f" | {d.event_type} [{d.severity}]"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(path))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: dict, rules: dict, detector, source: Path, output_dir: Path,
                 run_id: str, traceability: dict) -> dict:
    inp, pre, thr, out_cfg = cfg["input"], cfg["preprocessing"], cfg["thresholds"], cfg["output"]
    mode = inp.get("mode", "independent")
    if mode not in MODES:
        raise ValueError(f"input.mode inválido: {mode} (válidos: {MODES})")
    camera_id = inp.get("camera_id", "CAM_01")
    fps = float(inp.get("fps", 1.0))
    imgsz = int(cfg["model"]["imgsz"])
    model_version = str(cfg["model"].get("version", "1.0.0"))
    strategy = rules["decision"]["strategy"]

    roles = {v: k for k, v in rules["classes"].items() if k != "person"}  # clase del modelo → rol
    person_cls = rules["classes"]["person"]
    out_names = rules.get("output_names", {})

    missing_cls = {c for c in rules["classes"].values() if c not in set(detector.names.values())}
    if missing_cls:
        raise ValueError(f"El modelo no tiene las clases {sorted(missing_cls)} definidas en rules.yaml "
                         f"(clases del modelo: {sorted(detector.names.values())}).")

    roi = ROI.from_config(rules["roi"], camera_id)
    persistence = rules["persistence"]["sequence" if mode == "sequence" else "independent"]
    manager = EventManager(persistence, rules["events"], model_version, run_id)
    monitor = RunMonitor()

    paths = collect_images(source, inp.get("extensions", [".jpg", ".jpeg", ".png"]), bool(inp.get("recursive", True)))
    monitor.files_found = len(paths)
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {source}")
    groups = group_sources(paths, source, mode, inp.get("source_id"))
    LOG.info("Imágenes: %d | fuentes: %d | modo: %s | resize: %s | imgsz: %d", len(paths), len(groups), mode,
             pre["resize_mode"], imgsz)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotate = out_cfg.get("save_annotated", False)
    done = 0
    root = source if source.is_dir() else source.parent

    with JsonlWriter(output_dir / "predictions.jsonl") as w_pred, JsonlWriter(output_dir / "events.jsonl") as w_evt, \
            JsonlWriter(output_dir / "images_index.jsonl") as w_idx:
        try:
            for source_id, files in groups:
                tracker = IoUTracker(**rules["tracking"]) if mode == "sequence" else None
                for frame_id, path in enumerate(files):
                    done += 1
                    if done % 50 == 0:
                        LOG.info("Procesadas %d/%d", done, len(paths))
                    rel_file = path.relative_to(root).as_posix() if path != root else path.name
                    index_row = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "source_id": source_id,
                                 "camera_id": camera_id, "frame_id": frame_id if mode == "sequence" else 0,
                                 "image_file": rel_file, "image_path": str(path)}
                    try:
                        t0 = time.perf_counter()
                        img = read_image(path, int(inp.get("min_side", 16)))
                        h, w = img.shape[:2]
                        net_in, info = preprocess(img, imgsz, pre["resize_mode"], int(pre.get("pad_value", 114)))
                        t1 = time.perf_counter()
                        raw = detector.predict(net_in)
                        t2 = time.perf_counter()

                        dets: list[Detection] = []
                        for d in raw:
                            d.bbox = info.to_original(d.bbox)
                            if (d.bbox[2] - d.bbox[0]) >= 1 and (d.bbox[3] - d.bbox[1]) >= 1:
                                dets.append(d)
                        dets, n_conf = filter_by_confidence(dets, thr.get("per_class"), float(thr["conf_default"]))
                        dets, n_dup = class_nms(dets, float(thr["dedup_iou"]))

                        persons = [d for d in dets if d.cls == person_cls]
                        ppe = [d for d in dets if d.cls in roles]
                        if tracker is not None:
                            ids = tracker.update([p.bbox for p in persons])
                            for p, tid in zip(persons, ids):
                                p.track_id = tid
                            keys = ids
                        else:
                            keys = list(range(1, len(persons) + 1))

                        per_person, assigned = associate(persons, ppe, roles, rules["association"])
                        obs: list[PersonObs] = []
                        for i, p in enumerate(persons):
                            p.zone_id = roi.locate(p.bbox, (w, h))
                            p.inside_roi = p.zone_id is not None
                            mh, mv = missing_ppe(per_person[i], strategy)
                            p.condition = condition_key(mh, mv)
                            obs.append(PersonObs(keys[i], p.track_id, p.bbox, p.conf, p.prediction_id,
                                                 p.condition, p.inside_roi, p.zone_id))
                        for k, item in enumerate(ppe):
                            if k in assigned:
                                owner = persons[assigned[k]]
                                item.track_id, item.inside_roi, item.zone_id = owner.track_id, owner.inside_roi, owner.zone_id

                        ts = frame_id / fps if mode == "sequence" else 0.0
                        fid = frame_id if mode == "sequence" else 0
                        events = manager.update(source_id, camera_id, fid, ts, obs)
                        by_pred = {p.prediction_id: p for p in persons}
                        for e in events:
                            p = by_pred[e["prediction_id"]]
                            p.event_generated, p.event_type, p.severity = True, e["event_type"], e["severity"]
                            e["image_file"] = rel_file
                            w_evt.write(e)

                        for d in sorted(dets, key=lambda x: -x.conf):
                            w_pred.write({
                                "schema_version": SCHEMA_VERSION, "run_id": run_id,
                                "prediction_id": d.prediction_id, "source_id": source_id, "camera_id": camera_id,
                                "frame_id": fid, "timestamp_seconds": round(ts, 3), "image_file": rel_file,
                                "image_width": w, "image_height": h, "model_version": model_version,
                                "model_class": d.cls, "class_name": out_names.get(d.cls, d.cls),
                                "confidence": round(d.conf, 4),
                                "bbox": {"x1": round(d.bbox[0], 1), "y1": round(d.bbox[1], 1),
                                         "x2": round(d.bbox[2], 1), "y2": round(d.bbox[3], 1)},
                                "track_id": d.track_id, "inside_roi": d.inside_roi, "zone_id": d.zone_id,
                                "violation_condition": d.condition,
                                "event_generated": d.event_generated, "event_type": d.event_type,
                                "severity": d.severity,
                            })

                        t3 = time.perf_counter()
                        monitor.image_ok(source_id, camera_id, w, h, len(dets), (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t0) * 1e3)
                        monitor.add_detections(dets, source_id, camera_id)
                        monitor.add_discarded(n_conf, n_dup)
                        monitor.add_events(events, source_id, camera_id)
                        w_idx.write({**index_row, "status": "ok", "image_width": w, "image_height": h,
                                     "resize_mode": pre["resize_mode"], "n_detections": len(dets),
                                     "n_persons": len(persons), "n_events": len(events),
                                     "inference_ms": round((t2 - t1) * 1e3, 2)})
                        if annotate and (out_cfg.get("annotate", "events") == "all" or events):
                            _save_image(output_dir / "annotated" / source_id / f"{path.stem}.jpg",
                                        _annotate(img, dets, roi, (w, h), out_names, events))
                    except Exception as exc:  # noqa: BLE001 - un archivo malo no detiene el lote
                        kind = "imagen inválida" if isinstance(exc, ImageReadError) else "error inesperado"
                        LOG.error("Fallo en %s (%s): %s", path, kind, exc)
                        if not isinstance(exc, ImageReadError):
                            LOG.debug(traceback.format_exc())
                        monitor.image_failed(str(path), f"{type(exc).__name__}: {exc}")
                        w_idx.write({**index_row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
                manager.end_source(source_id)
        except KeyboardInterrupt:
            LOG.warning("Interrumpido por el usuario: se guardan los resultados parciales.")
            monitor.interrupted = True

    metrics = monitor.finalize(
        {**traceability, "run_id": run_id, "model_version": model_version, "decision_strategy": strategy,
         "input_mode": mode, "resize_mode": pre["resize_mode"], "imgsz": imgsz},
        dict(manager.stats),
    )
    import json  # noqa: PLC0415

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Listo: %d frames, %d detecciones, %d eventos, %d fallidos.", metrics["volume"]["frames"],
             metrics["volume"]["detections"], metrics["volume"]["events"], metrics["volume"]["files_failed"])
    return metrics


def mirror_latest(output_dir: Path, latest_dir: Path) -> None:
    """Copia los entregables de la corrida a `outputs/latest/` (ruta fija para el reporte)."""
    latest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.jsonl", "events.jsonl", "images_index.jsonl", "metrics.json",
                 "run_config.yaml", "inference.log"):
        if (output_dir / name).exists():
            shutil.copy2(output_dir / name, latest_dir / name)
    if (output_dir / "annotated").is_dir():
        shutil.copytree(output_dir / "annotated", latest_dir / "annotated", dirs_exist_ok=True)
