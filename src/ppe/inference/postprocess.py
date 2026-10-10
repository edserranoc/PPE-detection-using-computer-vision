"""Umbrales configurables por clase y control de detecciones duplicadas."""
from __future__ import annotations

from ppe.rules.geometry import iou
from ppe.types import Detection


def filter_by_confidence(
    dets: list[Detection], thresholds: dict[str, float] | None, default: float
) -> tuple[list[Detection], int]:
    thresholds = thresholds or {}
    kept = [d for d in dets if d.conf >= thresholds.get(d.cls, default)]
    return kept, len(dets) - len(kept)


def class_nms(dets: list[Detection], iou_threshold: float) -> tuple[list[Detection], int]:
    """NMS por clase: descarta cajas de la misma clase que solapan con una de mayor confianza."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.conf, reverse=True):
        if all(k.cls != d.cls or iou(k.bbox, d.bbox) <= iou_threshold for k in kept):
            kept.append(d)
    return kept, len(dets) - len(kept)
