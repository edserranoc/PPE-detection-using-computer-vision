"""Utilidades compartidas para localizar y leer datasets en formato YOLO.

Se usan tanto en el EDA como en el entrenamiento, de modo que ambos scripts
interpreten la estructura del dataset exactamente igual.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

log = logging.getLogger("ppe.data")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}
IGNORED_LABEL_FILES = {"classes.txt", "readme.txt", "readme.dataset.txt"}


# --------------------------------------------------------------------------- #
# Descubrimiento de estructura
# --------------------------------------------------------------------------- #
def find_data_yaml(root: Path) -> Path | None:
    """Busca un YAML de dataset (con la clave `names`) dentro de `root`."""
    candidates = sorted(
        root.rglob("*.y*ml"),
        key=lambda p: (p.name not in ("data.yaml", "dataset.yaml"), len(p.parts)),
    )
    for path in candidates:
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - YAML ajenos al dataset
            continue
        if isinstance(cfg, dict) and "names" in cfg:
            return path
    return None


def class_names(cfg: dict) -> list[str]:
    """Devuelve los nombres de clase ordenados por id (lista o dict en el YAML)."""
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def _has_images(directory: Path) -> bool:
    return directory.is_dir() and any(
        f.suffix.lower() in IMG_EXTS for f in directory.iterdir() if f.is_file()
    )


def resolve_splits(root: Path, yaml_path: Path | None, cfg: dict) -> dict[str, Path]:
    """Ubica la carpeta de imágenes de cada partición (train/val/test).

    Primero intenta con las rutas del YAML (relativas al YAML, a `path:` o a
    `root`); si no existen -muy común en exportaciones de Roboflow- prueba las
    estructuras habituales: `<split>/images`, `images/<split>` y `<split>`.
    """
    bases: list[Path] = []
    if yaml_path is not None:
        bases.append(yaml_path.parent)
        if cfg.get("path"):
            bases.append((yaml_path.parent / str(cfg["path"])))
    bases.append(root)

    found: dict[str, Path] = {}
    for split, aliases in SPLIT_ALIASES.items():
        value = cfg.get(split)
        if value is None and split == "val":
            value = cfg.get("valid")
        match: Path | None = None

        if isinstance(value, str):
            for base in bases:
                cand = (base / value).resolve()
                if _has_images(cand):
                    match = cand
                    break

        if match is None:
            for alias in aliases:
                for cand in (root / alias / "images", root / "images" / alias, root / alias):
                    if _has_images(cand):
                        match = cand.resolve()
                        break
                if match:
                    break

        if match is None:
            for alias in aliases:
                for directory in sorted(root.rglob(alias)):
                    if "runs" in directory.parts:
                        continue
                    for cand in (directory / "images", directory):
                        if _has_images(cand):
                            match = cand.resolve()
                            break
                    if match:
                        break
                if match:
                    break

        if match is not None:
            found[split] = match
    return found


def labels_dir_for(images_dir: Path) -> Path:
    """Carpeta de etiquetas equivalente (misma regla que usa Ultralytics)."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    sibling = images_dir.parent / "labels"
    return sibling if sibling.is_dir() else images_dir


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path: Path, img_dir: Path, lbl_dir: Path) -> Path:
    return (lbl_dir / img_path.relative_to(img_dir)).with_suffix(".txt")


def source_id(path: str | Path) -> str | None:
    """Id de la imagen original en exportaciones Roboflow (`<nombre>.rf.<hash>`).

    Las copias aumentadas de una misma imagen comparten este id; si caen en
    particiones distintas hay fuga de información.
    """
    stem = Path(path).stem
    return stem.split(".rf.")[0] if ".rf." in stem else None


# --------------------------------------------------------------------------- #
# Lectura de etiquetas
# --------------------------------------------------------------------------- #
def parse_label_file(
    path: Path, n_classes: int, tol: float = 1e-3
) -> tuple[list[tuple[int, float, float, float, float]], list[tuple[str, int]]]:
    """Lee un .txt YOLO. Devuelve (cajas, problemas).

    Cajas: (clase, xc, yc, w, h) normalizadas. Problemas: (tipo, nº de línea).
    Los polígonos de segmentación se convierten a su caja envolvente.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[tuple[str, int]] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls_f = float(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(("malformed_line", lineno))
            continue
        if cls_f != int(cls_f):
            issues.append(("malformed_line", lineno))
            continue
        cls = int(cls_f)

        if len(vals) == 4:
            xc, yc, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            issues.append(("polygon_annotation", lineno))
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            issues.append(("malformed_line", lineno))
            continue

        if not 0 <= cls < n_classes:
            issues.append(("class_out_of_range", lineno))
        if w <= 0 or h <= 0:
            issues.append(("degenerate_box", lineno))
        if (
            xc - w / 2 < -tol
            or yc - h / 2 < -tol
            or xc + w / 2 > 1 + tol
            or yc + h / 2 > 1 + tol
        ):
            issues.append(("box_out_of_bounds", lineno))
        boxes.append((cls, xc, yc, w, h))

    rounded = {tuple(np.round(b, 4)) for b in boxes}
    if len(rounded) < len(boxes):
        issues.append(("duplicate_box", 0))
    return boxes, issues


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #
def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - sólo para detectar duplicados
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """Hash perceptual de diferencias (64 bits con size=8)."""
    gray = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(gray, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Número de bits en 1 de un arreglo uint64."""
    mask = np.uint64(0xFFFF)
    total = np.zeros(x.shape, dtype=np.uint8)
    for shift in (0, 16, 32, 48):
        total += _POP16[((x >> np.uint64(shift)) & mask).astype(np.intp)]
    return total
