"""Reglas de negocio: condición de incumplimiento, severidad, persistencia y duplicados."""
from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass

from ppe.rules.association import PersonPPE
from ppe.rules.geometry import iou
from ppe.types import SCHEMA_VERSION, BBox

STRATEGIES = ("association", "explicit", "hybrid")


# --------------------------------------------------------------------------- #
# Condición y severidad
# --------------------------------------------------------------------------- #
def missing_ppe(ppe: PersonPPE, strategy: str) -> tuple[bool, bool]:
    """(falta_casco, falta_chaleco) según la estrategia de decisión.

    - association: falta si no hay EPP positivo asociado a la persona.
    - explicit:    falta si hay una clase `no_*` asociada con más confianza que el positivo.
    - hybrid:      falta salvo que haya un positivo asociado con confianza >= a la del `no_*`.
    """

    def missing(pos: float, neg: float) -> bool:
        if strategy == "association":
            return pos <= 0
        if strategy == "explicit":
            return neg > 0 and neg > pos
        if strategy == "hybrid":
            return not (pos > 0 and pos >= neg)
        raise ValueError(f"Estrategia desconocida: {strategy} (válidas: {STRATEGIES})")

    return missing(ppe.helmet, ppe.no_helmet), missing(ppe.vest, ppe.no_vest)


def condition_key(missing_helmet: bool, missing_vest: bool) -> str | None:
    if missing_helmet and missing_vest:
        return "sin_casco_y_chaleco"
    if missing_helmet:
        return "sin_casco"
    if missing_vest:
        return "sin_chaleco"
    return None


def resolve_event(condition: str, events_cfg: dict) -> tuple[str, str]:
    """(event_type, severity) configurables para una condición."""
    entry = events_cfg["types"][condition]
    return entry["event_type"], entry["severity"]


# --------------------------------------------------------------------------- #
# Persistencia temporal + duplicados
# --------------------------------------------------------------------------- #
@dataclass
class PersonObs:
    key: int                   # id de seguimiento (o índice local en modo imágenes independientes)
    track_id: int | None       # id a reportar en la salida
    bbox: BBox
    conf: float
    prediction_id: str
    condition: str | None
    inside_roi: bool
    zone_id: str | None


class EventManager:
    """Convierte condiciones frame a frame en eventos consolidados.

    - Persistencia: exige `min_frames` frames y `min_seconds` de condición continua
      (tolerando huecos de hasta `max_gap_frames`).
    - Duplicados: (1) un evento por condición continua; (2) `cooldown_seconds` para el mismo
      (fuente, track, tipo); (3) fusión de eventos del mismo tipo con caja muy solapada
      dentro de `merge_window_seconds` aunque cambie el track_id.
    """

    def __init__(self, persistence: dict, events_cfg: dict, model_version: str, run_id: str) -> None:
        self.min_frames = int(persistence.get("min_frames", 1))
        self.min_seconds = float(persistence.get("min_seconds", 0.0))
        self.max_gap = int(persistence.get("max_gap_frames", 0))
        self.cooldown = float(events_cfg.get("cooldown_seconds", 30.0))
        self.merge_window = float(events_cfg.get("merge_window_seconds", 2.0))
        self.merge_iou = float(events_cfg.get("merge_iou", 0.5))
        self.events_cfg = events_cfg
        self.model_version = model_version
        self.run_id = run_id
        self._states: dict[tuple, dict] = {}
        self._last_emit: dict[tuple, float] = {}
        self._recent: dict[tuple, list[dict]] = {}
        self.stats: Counter = Counter()

    def update(
        self, source_id: str, camera_id: str, frame_id: int, timestamp: float, persons: list[PersonObs]
    ) -> list[dict]:
        events: list[dict] = []
        for p in persons:
            if not p.inside_roi or p.condition is None:
                continue
            k = (source_id, p.key, p.condition)
            st = self._states.get(k)
            if st is None or frame_id - st["last_frame"] > self.max_gap + 1:
                st = {"start_frame": frame_id, "start_ts": timestamp, "count": 0,
                      "emitted": False, "suppressed": False, "recent": None}
                self._states[k] = st
            st["count"] += 1
            st["last_frame"] = frame_id
            if st["emitted"] and st["recent"] is not None:
                st["recent"].update(bbox=p.bbox, ts=timestamp)

            if st["emitted"] or st["suppressed"]:
                continue
            if st["count"] < self.min_frames or timestamp - st["start_ts"] < self.min_seconds:
                continue

            reason = self._duplicate_reason(k, source_id, p, timestamp)
            if reason:
                st["suppressed"] = True
                self.stats[reason] += 1
                continue
            event_type, severity = resolve_event(p.condition, self.events_cfg)
            st["emitted"] = True
            rec = {"bbox": p.bbox, "ts": timestamp, "track": p.key}
            st["recent"] = rec
            self._recent.setdefault((source_id, event_type), []).append(rec)
            self._last_emit[k] = timestamp
            self.stats["emitted"] += 1
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": str(uuid.uuid4()),
                    "run_id": self.run_id,
                    "model_version": self.model_version,
                    "source_id": source_id,
                    "camera_id": camera_id,
                    "event_type": event_type,
                    "severity": severity,
                    "track_id": p.track_id,
                    "zone_id": p.zone_id,
                    "start_frame_id": st["start_frame"],
                    "frame_id": frame_id,
                    "start_timestamp_seconds": round(st["start_ts"], 3),
                    "timestamp_seconds": round(timestamp, 3),
                    "latency_seconds": round(timestamp - st["start_ts"], 3),
                    "persistence_frames": st["count"],
                    "confidence": round(p.conf, 4),
                    "bbox": {"x1": round(p.bbox[0], 1), "y1": round(p.bbox[1], 1),
                             "x2": round(p.bbox[2], 1), "y2": round(p.bbox[3], 1)},
                    "prediction_id": p.prediction_id,
                }
            )
        return events

    def _duplicate_reason(self, k: tuple, source_id: str, p: PersonObs, ts: float) -> str | None:
        last = self._last_emit.get(k)
        if last is not None and ts - last < self.cooldown:
            return "suppressed_cooldown"
        event_type, _ = resolve_event(p.condition, self.events_cfg)
        for r in self._recent.get((source_id, event_type), []):
            if r["track"] != p.key and ts - r["ts"] <= self.merge_window and iou(r["bbox"], p.bbox) >= self.merge_iou:
                return "suppressed_track_switch"
        return None

    def end_source(self, source_id: str) -> None:
        """Libera el estado de una fuente ya procesada."""
        self._states = {k: v for k, v in self._states.items() if k[0] != source_id}
        self._last_emit = {k: v for k, v in self._last_emit.items() if k[0] != source_id}
        self._recent = {k: v for k, v in self._recent.items() if k[0] != source_id}
