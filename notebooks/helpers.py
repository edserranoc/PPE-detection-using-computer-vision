"""Utilidades compartidas por los notebooks (los notebooks no contienen lógica del pipeline: la reutilizan)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # en Jupyter display() es un builtin; fuera de él usamos IPython o print
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)


def setup() -> Path:
    """Cambia el directorio de trabajo a la raíz del repo y habilita `import ppe`."""
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    return repo


def env(name: str, default: str) -> str:
    """Lee PPE_<name> del entorno (sirve para apuntar los notebooks a otras rutas sin editarlos)."""
    return os.getenv(f"PPE_{name}", default)


def run(cmd: list[str], tail: int = 15) -> subprocess.CompletedProcess:
    """Ejecuta un comando del repo (scripts/...) y muestra el final de su salida."""
    print("$", " ".join(str(c) for c in cmd))
    res = subprocess.run([str(c) for c in cmd], text=True, capture_output=True)
    lines = (res.stdout + res.stderr).strip().splitlines()
    print("\n".join(lines[-tail:]))
    if res.returncode != 0:
        raise RuntimeError(f"El comando falló (código {res.returncode}). Revisa la salida de arriba.")
    return res


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl_df(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def exists_or_hint(path: str | Path, hint: str) -> bool:
    ok = Path(path).exists()
    if not ok:
        print(f"[falta] {path}\n        -> {hint}")
    return ok


def show_images(paths, titles=None, cols: int = 3, size: tuple[float, float] = (5.5, 4.2)) -> None:
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        print("No hay imágenes para mostrar todavía.")
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (ax, p) in enumerate(zip(axes, paths)):
        ax.imshow(mpimg.imread(str(p)))
        ax.set_title(titles[i] if titles else Path(p).name, fontsize=8)
    plt.tight_layout()
    plt.show()


def read_bgr(path: str | Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_resized_split(src_split: Path, dst_root: Path, long_side: int, limit: int = 0) -> Path:
    """Copia una partición (images/ + labels/) reescalando cada imagen a `long_side` px en su lado mayor.

    Las etiquetas YOLO están normalizadas, así que siguen siendo válidas. Sirve para probar el modelo con
    resoluciones distintas a las del entrenamiento. Devuelve la carpeta de la partición copiada.
    """
    from ppe.data.dataset import label_path_for, labels_dir_for, list_images

    img_dir = src_split / "images" if (src_split / "images").is_dir() else src_split
    lbl_dir = labels_dir_for(img_dir)
    out_img, out_lbl = dst_root / "test" / "images", dst_root / "test" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    images = list_images(img_dir)
    if limit:
        images = images[:limit]
    for p in images:
        img = read_bgr(p)
        h, w = img.shape[:2]
        r = long_side / max(h, w)
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        small = cv2.resize(img, (max(1, round(w * r)), max(1, round(h * r))), interpolation=interp)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        buf.tofile(str(out_img / f"{p.stem}.jpg"))
        lp = label_path_for(p, img_dir, lbl_dir)
        if lp.exists():
            (out_lbl / f"{p.stem}.txt").write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
    for yaml_file in list(src_split.parent.glob("*.y*ml")) + list(src_split.glob("*.y*ml")):
        (dst_root / yaml_file.name).write_text(yaml_file.read_text(encoding="utf-8"), encoding="utf-8")
    return dst_root / "test"


def run_experiment(name: str, images_dir: Path, split_dir: Path, dataset_root: Path, conf: float = 0.25,
                   inference_args: list[str] | None = None, rules_args: list[str] | None = None) -> dict:
    """Inferencia + verdad terreno + evaluación para una configuración. Devuelve una fila de métricas."""
    py = sys.executable
    out, gt, ev = Path(f"outputs/exp_{name}"), Path(f"reports/experiments/{name}/gt"), Path(f"reports/experiments/{name}")
    cmd = [py, "scripts/inference.py", "--source", images_dir, "--output", out, "--conf", conf, *(inference_args or [])]
    if rules_args:
        cmd += ["--set-rules", *rules_args]
    run(cmd, tail=3)
    run([py, "scripts/build_ground_truth.py", "--split-dir", split_dir, "--dataset-root", dataset_root,
         "--out", gt, "--derive-events"], tail=2)
    run([py, "scripts/evaluate_predictions.py", "--predictions", out / "predictions.jsonl",
         "--ground-truth", gt / "ground_truth.jsonl", "--events", out / "events.jsonl",
         "--ground-truth-events", gt / "ground_truth_events.jsonl", "--images-index", out / "images_index.jsonl",
         "--out", ev, "--conf", conf, "--error-images", 0], tail=3)
    pm, em, mm = read_json(ev / "prediction_metrics.json"), read_json(ev / "event_metrics.json"), read_json(out / "metrics.json")
    return {
        "experimento": name,
        "mAP50*": pm["global"]["map50"], "mAP50-95*": pm["global"]["map50_95"],
        "F1 pred. (micro)": pm["global"]["micro"]["f1"], "F1 pred. (macro)": pm["global"]["macro"]["f1"],
        "P eventos": em["overall"]["precision"], "R eventos": em["overall"]["recall"], "F1 eventos": em["overall"]["f1"],
        "duplicados %": round(100 * em["duplicate_rate"], 2), "omitidos": em["counts"]["missed"],
        "ms/inferencia": mm["performance"]["mean_inference_ms"], "imágenes": mm["volume"]["frames"],
    }
