#!/usr/bin/env python
"""Fine-tuning de YOLOv8n para detección de EPP.

Uso:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --set train.epochs=2 train.batch=8
    python scripts/train.py --config configs/train.yaml --dry-run   # sólo prepara y valida datos

Salidas:
    models/best_model.pt            mejor checkpoint
    models/model_metadata.json      trazabilidad completa del entrenamiento
    reports/offline/metrics_*.json  métricas de Ultralytics en val y test
    runs/train/<name>/              artefactos de Ultralytics (curvas, results.csv, etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.data.dataset import (  # noqa: E402
    class_names,
    find_data_yaml,
    label_path_for,
    labels_dir_for,
    list_images,
    resolve_splits,
    source_id,
)

LOG = logging.getLogger("ppe.train")
BEST_CRITERION = (
    "Ultralytics selecciona best.pt por 'fitness' en validación = 0.1·mAP@0.5 + 0.9·mAP@0.5:0.95 "
    "(evaluado al final de cada época)."
)


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        if "=" not in item:
            sys.exit(f"Override inválido (usa clave=valor): {item}")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOG.warning("torch no está instalado; se fija sólo random/numpy.")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def dataset_fingerprint(split_dirs: dict[str, Path]) -> dict:
    """Huella reproducible del dataset (nombres de imagen + contenido de etiquetas)."""
    h = hashlib.sha256()
    counts = {}
    for split, img_dir in sorted(split_dirs.items()):
        lbl_dir = labels_dir_for(img_dir)
        images = list_images(img_dir)
        counts[split] = len(images)
        for p in images:
            h.update(p.name.encode())
            lp = label_path_for(p, img_dir, lbl_dir)
            if lp.exists():
                h.update(lp.read_bytes())
    return {"sha256": h.hexdigest(), "images_per_split": counts}


def split_train_val(train_dir: Path, fraction: float, seed: int, out_dir: Path) -> tuple[Path, Path]:
    """Crea listas train/val desde `train` agrupando por imagen fuente (evita fuga)."""
    images = list_images(train_dir)
    groups: dict[str, list[Path]] = {}
    for p in images:
        groups.setdefault(source_id(p) or p.stem, []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = int(len(images) * fraction)
    val, train = [], []
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])
    out_dir.mkdir(parents=True, exist_ok=True)
    f_train, f_val = out_dir / "train_split.txt", out_dir / "val_split.txt"
    f_train.write_text("\n".join(str(p.resolve()) for p in sorted(train)) + "\n", encoding="utf-8")
    f_val.write_text("\n".join(str(p.resolve()) for p in sorted(val)) + "\n", encoding="utf-8")
    LOG.warning(
        "No había partición de validación: se creó desde train (%d train / %d val), agrupando por imagen fuente. "
        "Si las imágenes son copias aumentadas casi idénticas aún puede haber fuga; revisa el EDA.",
        len(train), len(val),
    )
    return f_train, f_val


def prepare_data_yaml(cfg: dict, run_dir: Path) -> tuple[Path, dict]:
    root = Path(cfg["data"]["root"]).resolve()
    if not root.is_dir():
        sys.exit(f"No existe data.root: {root}. Descarga el dataset (ver README).")
    yaml_path = Path(cfg["data"]["yaml"]).resolve() if cfg["data"].get("yaml") else find_data_yaml(root)
    ds_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path else {}
    names = cfg["data"].get("names") or class_names(ds_cfg)
    if not names:
        sys.exit("Sin nombres de clase: define data.yaml o data.names en la configuración.")

    split_dirs = resolve_splits(root, yaml_path, ds_cfg)
    if "train" not in split_dirs:
        sys.exit(f"No se encontró la partición de entrenamiento en {root}.")
    for split, d in split_dirs.items():
        if labels_dir_for(d) == d:
            sys.exit(
                f"Estructura no soportada en '{split}' ({d}): las etiquetas deben estar en una carpeta "
                "'labels' hermana de 'images' (p. ej. train/images + train/labels)."
            )

    fingerprint = dataset_fingerprint(split_dirs)
    runtime = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    created_val = False
    if "val" not in split_dirs:
        frac = float(cfg["data"].get("val_fraction_if_missing", 0.15))
        f_train, f_val = split_train_val(split_dirs["train"], frac, cfg["train"]["seed"], run_dir)
        runtime["train"], runtime["val"] = str(f_train), str(f_val)
        created_val = True
    else:
        runtime["train"], runtime["val"] = str(split_dirs["train"]), str(split_dirs["val"])
    if "test" in split_dirs:
        runtime["test"] = str(split_dirs["test"])

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "data_runtime.yaml"
    out.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False), encoding="utf-8")
    info = {
        "source_root": str(root),
        "source_yaml": str(yaml_path) if yaml_path else None,
        "class_names": names,
        "split_dirs": {k: str(v) for k, v in split_dirs.items()},
        "validation_created_from_train": created_val,
        "fingerprint": fingerprint,
    }
    return out, info


# --------------------------------------------------------------------------- #
# Entorno / hardware
# --------------------------------------------------------------------------- #
def hardware_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def enable_mlflow(cfg: dict) -> None:
    from ultralytics import settings

    settings.update({"mlflow": True})
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["tracking"].get("experiment", "ppe-detection")
    LOG.info("MLflow activado (%s).", os.environ["MLFLOW_TRACKING_URI"])


# --------------------------------------------------------------------------- #
# Evaluación rápida con Ultralytics (la evaluación completa va en evaluate.py)
# --------------------------------------------------------------------------- #
def evaluate_split(model, data_yaml: Path, split: str, cfg: dict, project: Path) -> dict:
    m = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=cfg["train"]["imgsz"],
        batch=cfg["train"]["batch"],
        conf=cfg["eval"]["conf"],
        iou=cfg["eval"]["iou"],
        plots=True,
        project=str(project),
        name=f"eval_{split}",
        exist_ok=True,
        verbose=False,
        **({"device": cfg["train"]["device"]} if cfg["train"].get("device") is not None else {}),
    )
    box = m.box
    f1 = lambda p, r: float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0  # noqa: E731
    per_class = {}
    for k, idx in enumerate(box.ap_class_index):
        p, r = float(box.p[k]), float(box.r[k])
        per_class[m.names[int(idx)]] = {
            "precision": p, "recall": r, "f1": f1(p, r),
            "ap50": float(box.ap50[k]), "ap50_95": float(box.ap[k]),
        }
    return {
        "split": split,
        "conf_threshold_for_pr_curves": cfg["eval"]["conf"],
        "global": {
            "precision": float(box.mp), "recall": float(box.mr), "f1": f1(float(box.mp), float(box.mr)),
            "map50": float(box.map50), "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
        "plots_dir": str(m.save_dir),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="clave=valor", help="Sobrescribe valores del YAML.")
    ap.add_argument("--dry-run", action="store_true", help="Valida datos y escribe data_runtime.yaml sin entrenar.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path(args.config), args.set)
    seed = cfg["train"]["seed"]
    set_seed(seed)

    project = Path(cfg["output"]["project"]).resolve()
    run_dir = project / f"{cfg['output']['name']}_setup"
    data_yaml, data_info = prepare_data_yaml(cfg, run_dir)
    LOG.info("Dataset: %s", json.dumps({k: data_info[k] for k in ("split_dirs", "class_names")}, ensure_ascii=False))
    LOG.info("data.yaml de entrenamiento: %s", data_yaml)
    if args.dry_run:
        LOG.info("--dry-run: datos validados; no se entrena.")
        return

    from ultralytics import YOLO  # import tardío: el dry-run no necesita ultralytics

    if cfg.get("tracking", {}).get("mlflow"):
        enable_mlflow(cfg)

    train_kwargs = {k: v for k, v in {**cfg["train"], **cfg["augment"]}.items() if v is not None}
    train_kwargs.update(
        data=str(data_yaml), project=str(project), name=cfg["output"]["name"], pretrained=True, plots=True
    )

    model = YOLO(cfg["model"]["weights"])
    LOG.info("Entrenando %s con %s", cfg["model"]["arch"], {k: train_kwargs[k] for k in ("imgsz", "epochs", "batch", "optimizer", "lr0")})
    started = datetime.now(timezone.utc)
    t0 = time.time()
    model.train(**train_kwargs)
    train_seconds = time.time() - t0

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    if not best.exists():
        best = save_dir / "weights" / "best.pt"

    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    best_copy = models_dir / "best_model.pt"
    shutil.copy2(best, best_copy)

    # Evaluación del mejor checkpoint
    best_model = YOLO(str(best_copy))
    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split in ("val", "test"):
        if split in data_info["split_dirs"] or (split == "val" and data_info["validation_created_from_train"]):
            metrics[split] = evaluate_split(best_model, data_yaml, split, cfg, save_dir)
            (reports_dir / f"metrics_{split}.json").write_text(
                json.dumps(metrics[split], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            g = metrics[split]["global"]
            LOG.info("[%s] P=%.3f R=%.3f F1=%.3f mAP50=%.3f mAP50-95=%.3f", split, g["precision"], g["recall"], g["f1"], g["map50"], g["map50_95"])

    metadata = {
        "model_version": cfg["output"]["model_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": cfg["model"]["arch"],
        "initial_weights": cfg["model"]["weights"],
        "checkpoint": str(best_copy),
        "checkpoint_sha256": hashlib.sha256(best_copy.read_bytes()).hexdigest(),
        "best_checkpoint_criterion": BEST_CRITERION,
        "epochs_requested": cfg["train"]["epochs"],
        "epochs_completed": int(getattr(trainer, "epoch", -1)) + 1,
        "best_fitness": float(getattr(trainer, "best_fitness", float("nan"))),
        "training": {
            "imgsz": cfg["train"]["imgsz"], "batch": cfg["train"]["batch"],
            "optimizer": cfg["train"]["optimizer"], "lr0": cfg["train"]["lr0"], "lrf": cfg["train"]["lrf"],
            "seed": seed, "deterministic": cfg["train"]["deterministic"],
            "started_at": started.isoformat(), "duration_seconds": round(train_seconds, 1),
        },
        "augmentations": cfg["augment"],
        "dataset": data_info,
        "hardware": hardware_info(),
        "metrics": {s: m["global"] for s, m in metrics.items()},
        "run_dir": str(save_dir),
        "config": cfg,
        "ultralytics_args": vars(trainer.args) if hasattr(trainer, "args") else None,
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    LOG.info("Listo. Modelo: %s | metadata: %s", best_copy, models_dir / "model_metadata.json")


if __name__ == "__main__":
    main()
