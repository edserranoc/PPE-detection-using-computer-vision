"""Zonas de interés (ROI) configurables como polígonos."""
from __future__ import annotations

from dataclasses import dataclass

from ppe.rules.geometry import bottom_center, center, point_in_polygon
from ppe.types import BBox

FULL_FRAME = "FULL_FRAME"


@dataclass
class Zone:
    zone_id: str
    polygon: list[tuple[float, float]]


class ROI:
    """Evalúa si una persona está dentro de alguna zona.

    Los polígonos se definen en coordenadas normalizadas (0-1) por defecto, de modo que
    funcionan con cualquier resolución de entrada. El punto de referencia de la persona
    es el centro inferior de su caja (los pies).
    """

    def __init__(
        self,
        zones: list[Zone],
        enabled: bool = True,
        normalized: bool = True,
        reference: str = "bottom_center",
    ) -> None:
        self.zones = zones
        self.enabled = enabled
        self.normalized = normalized
        self.reference = reference

    @classmethod
    def from_config(cls, roi_cfg: dict, camera_id: str) -> "ROI":
        raw = (roi_cfg.get("cameras") or {}).get(camera_id, roi_cfg.get("default_zones") or [])
        zones = [Zone(str(z["id"]), [(float(x), float(y)) for x, y in z["polygon"]]) for z in raw]
        return cls(
            zones,
            enabled=bool(roi_cfg.get("enabled", True)),
            normalized=roi_cfg.get("coordinates", "normalized") == "normalized",
            reference=roi_cfg.get("reference_point", "bottom_center"),
        )

    def _point(self, bbox: BBox, size: tuple[int, int]) -> tuple[float, float]:
        px, py = (bottom_center(bbox) if self.reference == "bottom_center" else center(bbox))
        if self.normalized:
            w, h = size
            return px / w, py / h
        return px, py

    def locate(self, bbox: BBox, size: tuple[int, int]) -> str | None:
        """Devuelve el id de la zona que contiene a la persona, o None si está fuera."""
        if not self.enabled or not self.zones:
            return FULL_FRAME
        point = self._point(bbox, size)
        for zone in self.zones:
            if point_in_polygon(point, zone.polygon):
                return zone.zone_id
        return None

    def polygons_px(self, size: tuple[int, int]) -> list[tuple[str, list[tuple[int, int]]]]:
        w, h = size
        out = []
        for z in self.zones:
            pts = [(int(x * w), int(y * h)) if self.normalized else (int(x), int(y)) for x, y in z.polygon]
            out.append((z.zone_id, pts))
        return out
