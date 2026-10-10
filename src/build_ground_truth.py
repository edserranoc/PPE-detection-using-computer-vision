#!/usr/bin/env python
"""Convierte las etiquetas YOLO de una partición en ground_truth.jsonl (y eventos esperados).

Uso:
    python scripts/build_ground_truth.py --split-dir src/data/kaggle_dataset/test --out data/ground_truth
    python scripts/build_ground_truth.py --split-dir src/data/kaggle_dataset/test --out data/ground_truth --derive-events

Notas:
- Las cajas se escriben en píxeles de la imagen ORIGINAL; source_id = ruta relativa sin extensión (igual que
  el modo `independent` de inference.py), frame_id = 0.
- --derive-events aplica las reglas de negocio sobre las anotaciones para obtener eventos esperados. Eso es una
  verdad terreno DERIVADA (aísla los errores del detector de los de la lógica); revísala manualmente antes de
  presentarla como verdad terreno revisada.
- Si quieres evaluar sólo una muestra revisada, usa --sample N --seed S y documenta el tamaño y la selección.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

for _p in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
    sys.path.insert(0, str(_p))
from ppe.config import load_yaml  # noqa: E402
from ppe.data.dataset import class_names, find_data_yaml, label_path_for, labels_dir_for, list_images, parse_label_file  # noqa: E402
from ppe.rules.association import associate  # noqa: E402
from ppe.rules.events import condition_key, missing_ppe, resolve_event  # noqa: E402
from ppe.rules.roi import ROI  # noqa: E402
from ppe.types import SCHEMA_VERSION, Detection  # noqa: E402

LOG = logging.getLogger("ppe.gt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, help="Carpeta de la partición (con images/ y labels/) o su carpeta images.")
    ap.add_argument("--dataset-root", default=None, help="Raíz con data.yaml (por defecto, la carpeta padre de --split-dir).")
    ap.add_argument("--out", default="data/ground_truth")
    ap.add_argument("--camera-id", default="CAM_01")
    ap.add_argument("--sample", type=int, default=0, help="Usa sólo N imágenes al azar (muestra revisada).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--derive-events", action="store_true")
    ap.add_argument("--rules", default="configs/rules.yaml")
    ap.add_argument("--strategy", default="explicit", choices=["association", "explicit", "hybrid"],
                    help="Estrategia para derivar eventos de las anotaciones (por defecto: lo etiquetado explícitamente).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split_dir = Path(args.split_dir).resolve()
    img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    lbl_dir = labels_dir_for(img_dir)
    root = Path(args.dataset_root).resolve() if args.dataset_root else split_dir.parent
    yaml_path = find_data_yaml(root) or find_data_yaml(split_dir)
    if yaml_path is None:
        sys.exit("No se encontró data.yaml con los nombres de clase (usa --dataset-root).")
    names = class_names(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))

    images = list_images(img_dir)
    if args.sample and args.sample < len(images):
        images = sorted(random.Random(args.seed).sample(images, args.sample))
    rules = load_yaml(args.rules)
    roles = {v: k for k, v in rules["classes"].items() if k != "person"}
    out_names = rules.get("output_names", {})
    roi = ROI.from_config(rules["roi"], args.camera_id)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ann = n_evt = 0
    with (out / "ground_truth.jsonl").open("w", encoding="utf-8") as f_gt, \
            (out / "ground_truth_events.jsonl").open("w", encoding="utf-8") as f_ev:
        for path in images:
            with Image.open(path) as im:
                w, h = im.size
            source_id = path.relative_to(img_dir).with_suffix("").as_posix()
            base = {"source_id": source_id, "camera_id": args.camera_id, "frame_id": 0}
            f_gt.write(json.dumps({"schema_version": SCHEMA_VERSION, "record_type": "image", **base,
                                   "image_file": path.relative_to(img_dir).as_posix(), "image_path": str(path),
                                   "image_width": w, "image_height": h}, ensure_ascii=False) + "\n")
            lp = label_path_for(path, img_dir, lbl_dir)
            boxes = parse_label_file(lp, len(names))[0] if lp.exists() else []
            dets = []
            for c, xc, yc, bw, bh in boxes:
                if not 0 <= c < len(names):
                    continue
                x1, y1 = max(0.0, (xc - bw / 2) * w), max(0.0, (yc - bh / 2) * h)
                x2, y2 = min(float(w), (xc + bw / 2) * w), min(float(h), (yc + bh / 2) * h)
                cls = names[c]
                dets.append(Detection(cls, 1.0, (x1, y1, x2, y2)))
                f_gt.write(json.dumps({"schema_version": SCHEMA_VERSION, "record_type": "annotation", **base,
                                       "image_file": path.relative_to(img_dir).as_posix(),
                                       "model_class": cls, "class_name": out_names.get(cls, cls),
                                       "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)}},
                                      ensure_ascii=False) + "\n")
                n_ann += 1
            if args.derive_events:
                persons = [d for d in dets if d.cls == rules["classes"]["person"]]
                ppe = [d for d in dets if d.cls in roles]
                per_person, _ = associate(persons, ppe, roles, rules["association"])
                for p, info in zip(persons, per_person):
                    cond = condition_key(*missing_ppe(info, args.strategy))
                    if cond and roi.locate(p.bbox, (w, h)) is not None:
                        etype, sev = resolve_event(cond, rules["events"])
                        f_ev.write(json.dumps({**base, "event_type": etype, "severity": sev, "timestamp_seconds": 0.0,
                                               "bbox": {"x1": round(p.bbox[0], 1), "y1": round(p.bbox[1], 1),
                                                        "x2": round(p.bbox[2], 1), "y2": round(p.bbox[3], 1)},
                                               "derived_from_annotations": True}, ensure_ascii=False) + "\n")
                        n_evt += 1
    LOG.info("Imágenes: %d | anotaciones: %d | eventos esperados: %d -> %s", len(images), n_ann, n_evt, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
