"""Tipos compartidos por el pipeline."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles de la imagen original
SCHEMA_VERSION = "1.0"


@dataclass
class Detection:
    cls: str                      # nombre de clase del modelo (p. ej. "person")
    conf: float
    bbox: BBox
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    track_id: int | None = None
    inside_roi: bool | None = None
    zone_id: str | None = None
    condition: str | None = None  # condición de incumplimiento detectada en este frame
    event_generated: bool = False
    event_type: str | None = None
    severity: str | None = None
