"""Tracker IoU voraz y determinista para secuencias de frames (sin dependencias)."""
from __future__ import annotations

from ppe.rules.geometry import iou
from ppe.types import BBox


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, boxes: list[BBox]) -> list[int]:
        """Asigna un id persistente a cada caja del frame actual (mismo orden de entrada)."""
        pairs = sorted(
            ((iou(t["bbox"], b), tid, j) for tid, t in self._tracks.items() for j, b in enumerate(boxes)),
            reverse=True,
        )
        used_tracks: set[int] = set()
        ids: dict[int, int] = {}
        for score, tid, j in pairs:
            if score < self.iou_threshold:
                break
            if tid in used_tracks or j in ids:
                continue
            used_tracks.add(tid)
            ids[j] = tid
            self._tracks[tid] = {"bbox": boxes[j], "age": 0}

        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        for j, box in enumerate(boxes):
            if j not in ids:
                ids[j] = self._next_id
                self._tracks[self._next_id] = {"bbox": box, "age": 0}
                self._next_id += 1
        return [ids[j] for j in range(len(boxes))]
