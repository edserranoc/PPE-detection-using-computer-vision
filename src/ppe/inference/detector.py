"""Envoltorio del modelo YOLO (Ultralytics). Se importa de forma tardía."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ppe.types import Detection


class Detector:
    def __init__(self, weights: str | Path, imgsz: int, conf: float, iou: float,
                 device: str | int | None = None, max_det: int = 300) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        if not Path(weights).is_file():
            raise FileNotFoundError(f"No se encontró el modelo: {weights}")
        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.device, self.max_det = imgsz, conf, iou, device, max_det
        names = self.model.names
        self.names: dict[int, str] = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    def predict(self, img_bgr: np.ndarray) -> list[Detection]:
        kwargs = {"device": self.device} if self.device is not None else {}
        res = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 max_det=self.max_det, verbose=False, **kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [Detection(cls=self.names[int(c)], conf=float(s), bbox=tuple(float(v) for v in b))
                for b, s, c in zip(xyxy, conf, cls)]
