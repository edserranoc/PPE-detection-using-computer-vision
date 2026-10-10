"""Monitoreo operacional de una corrida: volumen, rendimiento, calidad, segmentación y trazabilidad."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from ppe.types import SCHEMA_VERSION, Detection


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size), "mean": round(float(a.mean()), 4), "min": round(float(a.min()), 4),
        "p10": round(float(np.percentile(a, 10)), 4), "p50": round(float(np.percentile(a, 50)), 4),
        "p90": round(float(np.percentile(a, 90)), 4), "max": round(float(a.max()), 4),
    }


class RunMonitor:
    MAX_SOURCES_LISTED = 50

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self.files_found = 0
        self.failed: list[dict] = []
        self.frames = 0
        self.no_detection_images = 0
        self.t_read: list[float] = []
        self.t_infer: list[float] = []
        self.t_total: list[float] = []
        self.conf_by_class: dict[str, list[float]] = defaultdict(list)
        self.det_by_class: Counter = Counter()
        self.discarded: Counter = Counter()
        self.by_camera: dict[str, Counter] = defaultdict(Counter)
        self.by_source: dict[str, Counter] = defaultdict(Counter)
        self.events_by_type: Counter = Counter()
        self.events_by_severity: Counter = Counter()
        self.latencies: list[float] = []
        self.resolutions: Counter = Counter()
        self.interrupted = False

    def image_ok(self, source: str, camera: str, w: int, h: int, n_det: int,
                 read_ms: float, infer_ms: float, total_ms: float) -> None:
        self.frames += 1
        self.t_read.append(read_ms)
        self.t_infer.append(infer_ms)
        self.t_total.append(total_ms)
        self.resolutions[f"{w}x{h}"] += 1
        self.by_camera[camera]["frames"] += 1
        self.by_source[source]["frames"] += 1
        if n_det == 0:
            self.no_detection_images += 1

    def image_failed(self, path: str, error: str) -> None:
        self.failed.append({"file": path, "error": error[:300]})

    def add_detections(self, dets: list[Detection], source: str, camera: str) -> None:
        for d in dets:
            self.conf_by_class[d.cls].append(d.conf)
            self.det_by_class[d.cls] += 1
        self.by_camera[camera]["detections"] += len(dets)
        self.by_source[source]["detections"] += len(dets)

    def add_discarded(self, below_threshold: int, duplicates: int) -> None:
        self.discarded["below_threshold"] += below_threshold
        self.discarded["duplicate_detections_nms"] += duplicates

    def add_events(self, events: list[dict], source: str, camera: str) -> None:
        for e in events:
            self.events_by_type[e["event_type"]] += 1
            self.events_by_severity[e["severity"]] += 1
            self.latencies.append(e["latency_seconds"])
        self.by_camera[camera]["events"] += len(events)
        self.by_source[source]["events"] += len(events)

    def finalize(self, traceability: dict, event_stats: dict) -> dict:
        total = time.perf_counter() - self.t0
        n_det = sum(self.det_by_class.values())
        n_events = sum(self.events_by_type.values())
        total_det = max(n_det, 1)
        top_sources = dict(sorted(self.by_source.items(), key=lambda kv: -kv[1]["detections"])[: self.MAX_SOURCES_LISTED])
        return {
            "schema_version": SCHEMA_VERSION,
            "volume": {
                "files_found": self.files_found, "files_processed": self.frames,
                "files_failed": len(self.failed), "videos": 0, "frames": self.frames,
                "detections": n_det, "events": n_events,
            },
            "performance": {
                "total_seconds": round(total, 3),
                "mean_ms_per_frame": round(float(np.mean(self.t_total)), 2) if self.t_total else None,
                "mean_inference_ms": round(float(np.mean(self.t_infer)), 2) if self.t_infer else None,
                "p95_inference_ms": round(float(np.percentile(self.t_infer, 95)), 2) if self.t_infer else None,
                "mean_read_preprocess_ms": round(float(np.mean(self.t_read)), 2) if self.t_read else None,
                "fps_end_to_end": round(self.frames / total, 2) if total > 0 else None,
                "fps_inference_only": round(1000 / float(np.mean(self.t_infer)), 2) if self.t_infer else None,
                "failed_files": self.failed[:50],
                "interrupted": self.interrupted,
            },
            "quality": {
                "confidence_by_class": {c: _stats(v) for c, v in sorted(self.conf_by_class.items())},
                "class_distribution": {c: {"count": n, "pct": round(100 * n / total_det, 2)}
                                       for c, n in sorted(self.det_by_class.items())},
                "discarded": dict(self.discarded),
                "images_without_detections": self.no_detection_images,
                "events_suppressed_as_duplicates": {k: v for k, v in event_stats.items() if k.startswith("suppressed")},
                "event_latency_seconds": _stats(self.latencies),
                "input_resolutions": dict(self.resolutions.most_common(20)),
            },
            "segmentation": {
                "by_class": dict(self.det_by_class),
                "by_camera": {k: dict(v) for k, v in self.by_camera.items()},
                "by_source": {k: dict(v) for k, v in top_sources.items()},
                "n_sources": len(self.by_source),
                "by_event_type": dict(self.events_by_type),
                "by_severity": dict(self.events_by_severity),
                "by_period": {"run_date": self.started_at.date().isoformat(), "run_hour_utc": self.started_at.hour},
            },
            "traceability": {**traceability, "run_started_at": self.started_at.isoformat()},
        }
