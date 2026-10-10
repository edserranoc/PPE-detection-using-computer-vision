"""Geometría de cajas y polígonos (sin dependencias externas)."""
from __future__ import annotations

from ppe.types import BBox


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(inner: BBox, outer: BBox) -> float:
    """Fracción del área de `inner` que cae dentro de `outer` (intersección / área de inner)."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def bottom_center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, b[3]


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting; los puntos sobre el borde cuentan como dentro."""
    x, y = point
    eps = 1e-9
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if (
            abs(cross) < eps
            and min(x1, x2) - eps <= x <= max(x1, x2) + eps
            and min(y1, y2) - eps <= y <= max(y1, y2) + eps
        ):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside
