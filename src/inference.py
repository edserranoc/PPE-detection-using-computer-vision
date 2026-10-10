#!/usr/bin/env python
"""Pipeline de inferencia de EPP sobre una carpeta de imágenes.

Uso:
    python scripts/inference.py --source src/data/kaggle_dataset/test/images
    python scripts/inference.py --source <carpeta> --mode sequence --fps 5 --camera-id CAM_02
    python scripts/inference.py --source <carpeta> --resize-mode letterbox --imgsz 640 --save-annotated

Salidas en outputs/<run_id>/ (y copia en outputs/latest/):
    predictions.jsonl  events.jsonl  images_index.jsonl  metrics.json  run_config.yaml  inference.log
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

for _p in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
    sys.path.insert(0, str(_p))
from ppe.config import apply_overrides, config_hash, load_yaml  # noqa: E402
from ppe.pipeline import mirror_latest, run_pipeline  # noqa: E402

LOG = logging.getLogger("ppe.inference")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Carpeta de imágenes (o una imagen).")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--rules", default="configs/rules.yaml")
    ap.add_argument("--weights", default=None, help="Sobrescribe model.weights.")
    ap.add_argument("--output", default=None, help="Carpeta de salida (por defecto outputs/<run_id>).")
    ap.add_argument("--mode", choices=["independent", "sequence"], default=None)
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--conf", type=float, default=None, help="Umbral de confianza por defecto.")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--resize-mode", choices=["letterbox", "resize", "none"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="clave=valor", help="Overrides de configs/inference.yaml.")
    ap.add_argument("--set-rules", nargs="*", default=[], metavar="clave=valor",
                    help="Overrides de configs/rules.yaml (p. ej. decision.strategy=explicit).")
    return ap.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    args = parse_args()
    cfg, rules = load_yaml(args.config), load_yaml(args.rules)

    overrides = list(args.set)
    for flag, key in [("weights", "model.weights"), ("mode", "input.mode"), ("camera_id", "input.camera_id"),
                      ("fps", "input.fps"), ("conf", "thresholds.conf_default"), ("imgsz", "model.imgsz"),
                      ("resize_mode", "preprocessing.resize_mode"), ("device", "model.device")]:
        value = getattr(args, flag)
        if value is not None:
            overrides.append(f"{key}={value}")
    apply_overrides(cfg, overrides)
    apply_overrides(rules, args.set_rules)
    if args.save_annotated:
        cfg["output"]["save_annotated"] = True

    chash = config_hash(cfg, rules)
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{chash[:8]}"
    output_dir = Path(args.output) if args.output else Path(cfg["output"]["dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "inference.log", encoding="utf-8")])
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump({"inference": cfg, "rules": rules, "config_hash": chash, "cli": vars(args)},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    source = Path(args.source)
    if not source.exists():
        LOG.error("No existe --source: %s", source)
        return 2

    try:
        from ppe.inference.detector import Detector  # noqa: PLC0415

        weights = Path(cfg["model"]["weights"])
        thr = cfg["thresholds"]
        low_conf = min([float(thr["conf_default"]), *map(float, (thr.get("per_class") or {}).values())])
        detector = Detector(weights, int(cfg["model"]["imgsz"]), low_conf, float(thr["model_nms_iou"]),
                            cfg["model"].get("device"), int(cfg["model"].get("max_det", 300)))
        try:
            import torch  # noqa: PLC0415
            import ultralytics  # noqa: PLC0415
            torch.manual_seed(seed)
            versions = {"torch": torch.__version__, "ultralytics": ultralytics.__version__}
        except ImportError:
            versions = {}
        trace = {"weights": str(weights), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                 "config_hash": chash, "seed": seed, "python": platform.python_version(),
                 "platform": platform.platform(), "git_commit": _git_commit(), **versions,
                 "source": str(source), "executed_at": datetime.now().isoformat()}
        run_pipeline(cfg, rules, detector, source, output_dir, run_id, trace)
    except (FileNotFoundError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001
        LOG.exception("Error fatal en el pipeline")
        return 1

    if cfg["output"].get("mirror_latest", True):
        mirror_latest(output_dir, Path(cfg["output"]["dir"]) / "latest")
    LOG.info("Resultados en %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
