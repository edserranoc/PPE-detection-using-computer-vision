"""Lectura robusta de imágenes y adaptación de resolución (letterbox / resize).

El modelo se entrenó a un tamaño fijo (p. ej. 640x640). Imágenes con otra resolución se
adaptan antes de inferir y las cajas se devuelven al sistema de coordenadas ORIGINAL.

Modos:
- letterbox: reescala conservando la proporción y rellena con bordes (gris 114) hasta
  imgsz x imgsz. Es el mismo preprocesado del entrenamiento: no distorsiona a las personas.
- resize:    estira la imagen a imgsz x imgsz (distorsiona la proporción).
- none:      no toca la imagen; Ultralytics aplica su propio letterbox interno.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ppe.types import BBox

MODES = ("letterbox", "resize", "none")


class ImageReadError(Exception):
    """Imagen inexistente, vacía, ilegible o demasiado pequeña."""


def read_image(path: Path, min_side: int = 16) -> np.ndarray:
    if not path.is_file():
        raise ImageReadError(f"No existe el archivo: {path}")
    if path.stat().st_size == 0:
        raise ImageReadError(f"Archivo vacío: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)  # soporta rutas con espacios/tildes
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageReadError(f"No se pudo decodificar la imagen: {path}")
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        raise ImageReadError(f"Imagen demasiado pequeña ({w}x{h}): {path}")
    return img


@dataclass
class PreprocessInfo:
    mode: str
    orig_w: int
    orig_h: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

    def to_original(self, bbox: BBox) -> BBox:
        """Mapea una caja del espacio preprocesado al de la imagen original (recortada a sus límites)."""
        x1 = (bbox[0] - self.pad_x) / self.scale_x
        y1 = (bbox[1] - self.pad_y) / self.scale_y
        x2 = (bbox[2] - self.pad_x) / self.scale_x
        y2 = (bbox[3] - self.pad_y) / self.scale_y
        clip = lambda v, hi: float(min(max(v, 0.0), hi))  # noqa: E731
        return (clip(x1, self.orig_w), clip(y1, self.orig_h), clip(x2, self.orig_w), clip(y2, self.orig_h))


def preprocess(img: np.ndarray, imgsz: int, mode: str = "letterbox", pad_value: int = 114):
    if mode not in MODES:
        raise ValueError(f"resize_mode inválido: {mode} (válidos: {MODES})")
    h, w = img.shape[:2]
    info = PreprocessInfo(mode=mode, orig_w=w, orig_h=h)
    if mode == "none":
        return img, info

    if mode == "resize":
        interp = cv2.INTER_AREA if (w > imgsz or h > imgsz) else cv2.INTER_LINEAR
        out = cv2.resize(img, (imgsz, imgsz), interpolation=interp)
        info.scale_x, info.scale_y = imgsz / w, imgsz / h
        return out, info

    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(1, round(w * r)), max(1, round(h * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    left, top = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    info.scale_x = info.scale_y = r
    info.pad_x, info.pad_y = float(left), float(top)
    return canvas, info
