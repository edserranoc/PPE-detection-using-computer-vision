#!/usr/bin/env python
"""EDA y validación del dataset de EPP (formato YOLO).

Genera `dataset_report.json` y las visualizaciones del análisis exploratorio.

Uso:
    python scripts/eda.py --root data/ppe --out reports/eda
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.data.dataset import (  # noqa: E402
    class_names,
    dhash,
    find_data_yaml,
    label_path_for,
    labels_dir_for,
    list_images,
    md5_file,
    parse_label_file,
    popcount64,
    resolve_splits,
    source_id,
    IGNORED_LABEL_FILES,
)

LOG = logging.getLogger("ppe.eda")
SMALL_REL_AREA = (32 * 32) / (640 * 640)  # objeto < 32x32 px a 640x640


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _limited(items: list, k: int = 50) -> dict:
    return {"count": len(items), "examples": items[:k]}


def _quantiles(values: np.ndarray, qs=(0, 5, 50, 95, 100)) -> dict:
    if len(values) == 0:
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


# --------------------------------------------------------------------------- #
# Análisis por partición
# --------------------------------------------------------------------------- #
def analyze_split(split: str, img_dir: Path, names: list[str], tol: float, max_images: int):
    lbl_dir = labels_dir_for(img_dir)
    images = list_images(img_dir)
    truncated = bool(max_images) and len(images) > max_images
    if truncated:
        images = images[:max_images]
    n_cls = len(names)

    def name_of(c: int) -> str:
        return names[c] if 0 <= c < n_cls else f"id_{c}"

    records, corrupt, missing, empty = [], [], [], []
    boxes_all, per_image_counts = [], []
    obj_per_class, img_per_class = Counter(), Counter()
    resolutions = Counter()
    issue_count, issue_examples = Counter(), defaultdict(list)

    for i, path in enumerate(images, start=1):
        if i % 1000 == 0:
            LOG.info("[%s] %d/%d imágenes", split, i, len(images))
        rec = {"split": split, "path": str(path), "md5": None, "dhash": None, "w": None, "h": None}
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                rec["w"], rec["h"] = im.size
                rec["dhash"] = dhash(im)
            rec["md5"] = md5_file(path)
            resolutions[(rec["w"], rec["h"])] += 1
        except Exception as exc:  # noqa: BLE001 - cualquier fallo = imagen ilegible
            corrupt.append({"path": str(path), "error": repr(exc)[:200]})
            records.append(rec)
            continue
        records.append(rec)

        lp = label_path_for(path, img_dir, lbl_dir)
        if not lp.exists():
            missing.append(str(path))
            continue

        boxes, issues = parse_label_file(lp, n_cls, tol)
        for kind, lineno in issues:
            issue_count[kind] += 1
            if len(issue_examples[kind]) < 10:
                issue_examples[kind].append({"label": str(lp), "line": lineno})

        per_image_counts.append(len(boxes))
        if not boxes:
            empty.append(str(lp))
            continue
        arr = np.array(boxes, dtype=float)
        boxes_all.append(arr)
        classes = arr[:, 0].astype(int)
        obj_per_class.update(classes.tolist())
        img_per_class.update(set(classes.tolist()))

    orphans: list[str] = []
    if not truncated and lbl_dir.is_dir() and lbl_dir != img_dir:
        expected = {label_path_for(p, img_dir, lbl_dir) for p in images}
        for txt in lbl_dir.rglob("*.txt"):
            if txt not in expected and txt.name.lower() not in IGNORED_LABEL_FILES:
                orphans.append(str(txt))

    boxes_arr = np.concatenate(boxes_all) if boxes_all else np.zeros((0, 5))
    counts = np.array(per_image_counts, dtype=float)
    rel_area = boxes_arr[:, 3] * boxes_arr[:, 4]
    known_ids = sorted(set(range(n_cls)) | set(obj_per_class))

    summary = {
        "images_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
        "truncated_by_max_images": truncated,
        "n_images": len(images),
        "n_readable": len(images) - len(corrupt),
        "n_objects": int(len(boxes_arr)),
        "objects_per_class": {name_of(c): int(obj_per_class.get(c, 0)) for c in known_ids},
        "images_per_class": {name_of(c): int(img_per_class.get(c, 0)) for c in known_ids},
        "corrupt_images": _limited(corrupt),
        "images_without_label_file": _limited(missing),
        "orphan_label_files": _limited(orphans),
        "images_with_empty_label": _limited(empty),
        "label_issues": {
            k: {"count": int(v), "examples": issue_examples[k]} for k, v in issue_count.items()
        },
        "resolutions": {
            "unique": len(resolutions),
            "top": [{"w": w, "h": h, "count": c} for (w, h), c in resolutions.most_common(10)],
            "min": min(resolutions) if resolutions else None,
            "max": max(resolutions) if resolutions else None,
        },
        "boxes_per_image": (
            {
                "mean": float(counts.mean()),
                "median": float(np.median(counts)),
                "p95": float(np.percentile(counts, 95)),
                "max": int(counts.max()),
            }
            if len(counts)
            else {}
        ),
        "bbox_rel_area": _quantiles(rel_area),
        "small_objects_pct": float((rel_area < SMALL_REL_AREA).mean() * 100) if len(rel_area) else None,
    }
    return summary, records, boxes_arr, per_image_counts


# --------------------------------------------------------------------------- #
# Duplicados y fuga de información
# --------------------------------------------------------------------------- #
def find_duplicates(records: list[dict], near_thr: int, skip_near: bool) -> dict:
    valid = [r for r in records if r["md5"] and r["dhash"] is not None]
    fmt = lambda g: [{"split": r["split"], "path": r["path"]} for r in g]  # noqa: E731

    by_md5 = defaultdict(list)
    for r in valid:
        by_md5[r["md5"]].append(r)
    exact = [g for g in by_md5.values() if len(g) > 1]
    cross_exact = [g for g in exact if len({r["split"] for r in g}) > 1]

    by_src = defaultdict(list)
    for r in valid:
        sid = source_id(r["path"])
        if sid:
            by_src[sid].append(r)
    cross_src = [g for g in by_src.values() if len({r["split"] for r in g}) > 1]

    out = {
        "exact_duplicate_groups": {
            "count": len(exact),
            "images_involved": sum(len(g) for g in exact),
            "examples": [fmt(g) for g in exact[:20]],
        },
        "exact_duplicates_across_splits": {
            "count": len(cross_exact),
            "examples": [fmt(g) for g in cross_exact[:20]],
        },
        "same_source_across_splits_roboflow": {
            "count": len(cross_src),
            "examples": [fmt(g)[:6] for g in cross_src[:20]],
        },
        "near_duplicates": None,
    }
    if skip_near or len(valid) < 2:
        return out

    n = len(valid)
    hashes = np.array([r["dhash"] for r in valid], dtype=np.uint64)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        dist = popcount64(hashes[i + 1 :] ^ hashes[i])
        for j in np.nonzero(dist <= near_thr)[0]:
            ra, rb = find(i), find(i + 1 + int(j))
            if ra != rb:
                parent[rb] = ra

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(valid[i])
    groups = [g for g in clusters.values() if len(g) > 1]
    cross = [g for g in groups if len({r["split"] for r in g}) > 1]
    leaked_eval = sum(
        1
        for g in cross
        if any(r["split"] == "train" for r in g)
        for r in g
        if r["split"] in ("val", "test")
    )
    out["near_duplicates"] = {
        "hamming_threshold": near_thr,
        "clusters": len(groups),
        "images_involved": sum(len(g) for g in groups),
        "clusters_across_splits": len(cross),
        "val_test_images_with_train_twin": leaked_eval,
        "examples_across_splits": [fmt(g)[:6] for g in cross[:20]],
    }
    return out


# --------------------------------------------------------------------------- #
# Recomendaciones automáticas
# --------------------------------------------------------------------------- #
def build_recommendations(report: dict) -> list[str]:
    recs: list[str] = []
    splits = report["splits"]
    total = lambda key: sum(s[key]["count"] for s in splits.values())  # noqa: E731

    if total("corrupt_images"):
        recs.append(f"Excluir {total('corrupt_images')} imagen(es) corruptas/ilegibles antes de entrenar.")
    if total("images_without_label_file") or total("orphan_label_files"):
        recs.append("Hay imágenes sin etiqueta o etiquetas huérfanas: revisar la correspondencia imagen↔etiqueta.")
    for kind, msg in {
        "class_out_of_range": "Corregir ids de clase fuera de rango.",
        "box_out_of_bounds": "Cajas fuera de límites: recortar a [0,1] o corregir la anotación.",
        "degenerate_box": "Eliminar cajas con ancho o alto <= 0.",
        "malformed_line": "Corregir líneas mal formadas en las etiquetas.",
        "duplicate_box": "Hay cajas duplicadas dentro de una misma imagen.",
        "polygon_annotation": "Hay polígonos de segmentación; se tratan como su caja envolvente.",
    }.items():
        n = sum(s["label_issues"].get(kind, {}).get("count", 0) for s in splits.values())
        if n:
            recs.append(f"{msg} ({n} casos)")

    missing = [s for s in ("train", "val", "test") if s not in splits]
    if missing:
        recs.append(f"Faltan particiones: {', '.join(missing)}. Crear una partición sin fuga (por fuente o video).")

    dup = report["duplicates"]
    leak = (
        dup["exact_duplicates_across_splits"]["count"]
        + dup["same_source_across_splits_roboflow"]["count"]
        + ((dup["near_duplicates"] or {}).get("clusters_across_splits", 0))
    )
    if leak:
        recs.append("Posible fuga de información entre particiones: volver a dividir agrupando por imagen fuente.")

    imb = report["imbalance"]
    if imb["max_min_ratio"] and imb["max_min_ratio"] > 10:
        recs.append(
            f"Desbalance fuerte (razón {imb['max_min_ratio']:.1f}:1): reportar métricas por clase y "
            "considerar sobremuestreo de clases raras."
        )
    if imb["classes_without_objects"]:
        recs.append(f"Clases sin objetos: {', '.join(imb['classes_without_objects'])}.")

    small = [s["small_objects_pct"] for s in splits.values() if s["small_objects_pct"] is not None]
    if small and max(small) > 30:
        recs.append("Más del 30 % de los objetos son pequeños (<32 px a 640): evaluar imgsz mayor (800–960).")
    if any(s["resolutions"]["unique"] > 20 for s in splits.values()):
        recs.append("Resoluciones muy heterogéneas: el letterbox de YOLO las normaliza, pero documentar el efecto.")
    return recs


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_images_per_split(splits: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(splits)
    values = [splits[s]["n_images"] for s in labels]
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.bar_label(bars)
    ax.set(title="Imágenes por partición", ylabel="Imágenes")
    fig.tight_layout()
    fig.savefig(out / "images_per_split.png", dpi=150)
    plt.close(fig)


def fig_class_distribution(splits: dict, names: list[str], out: Path) -> None:
    classes = list(next(iter(splits.values()))["objects_per_class"])
    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.9), 4.5))
    for k, (split, s) in enumerate(splits.items()):
        vals = [s["objects_per_class"].get(c, 0) for c in classes]
        ax.bar(x + k * width, vals, width, label=split)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set(title="Objetos por clase y partición", ylabel="Objetos (escala log)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "objects_per_class.png", dpi=150)
    plt.close(fig)


def fig_resolutions(records: list[dict], out: Path) -> None:
    counts = Counter((r["w"], r["h"]) for r in records if r["w"])
    if not counts:
        return
    pts = np.array(list(counts))
    sizes = np.array(list(counts.values()), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=20 + 400 * sizes / sizes.max(), alpha=0.6)
    for (w, h), c in counts.most_common(5):
        ax.annotate(f"{w}x{h} ({c})", (w, h), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set(title="Resoluciones (tamaño = nº de imágenes)", xlabel="Ancho (px)", ylabel="Alto (px)")
    fig.tight_layout()
    fig.savefig(out / "resolutions.png", dpi=150)
    plt.close(fig)


def fig_bbox_area(all_boxes: np.ndarray, names: list[str], out: Path) -> None:
    if len(all_boxes) == 0:
        return
    data, labels = [], []
    for c in sorted(set(all_boxes[:, 0].astype(int))):
        sel = all_boxes[all_boxes[:, 0] == c]
        data.append(sel[:, 3] * sel[:, 4])
        labels.append(names[c] if 0 <= c < len(names) else f"id_{c}")
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(SMALL_REL_AREA, ls="--", c="r", lw=1, label="32x32 px @640")
    ax.set_yscale("log")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set(title="Área relativa de las cajas por clase", ylabel="Área (fracción de la imagen)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bbox_area_per_class.png", dpi=150)
    plt.close(fig)


def fig_objects_per_image(counts: list[int], out: Path) -> None:
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=range(0, max(counts) + 2), color="#F58518")
    ax.set(title="Objetos por imagen", xlabel="Objetos", ylabel="Imágenes")
    fig.tight_layout()
    fig.savefig(out / "objects_per_image.png", dpi=150)
    plt.close(fig)


def fig_center_heatmap(all_boxes: np.ndarray, out: Path) -> None:
    if len(all_boxes) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    h = ax.hist2d(all_boxes[:, 1], all_boxes[:, 2], bins=40, range=[[0, 1], [0, 1]], cmap="magma")
    ax.invert_yaxis()
    fig.colorbar(h[3], ax=ax)
    ax.set(title="Centros de las cajas", xlabel="x (norm.)", ylabel="y (norm.)")
    fig.tight_layout()
    fig.savefig(out / "box_centers_heatmap.png", dpi=150)
    plt.close(fig)


def fig_samples(splits_dirs: dict, names: list[str], out: Path, n: int, seed: int, tol: float) -> None:
    pool = []
    for split, img_dir in splits_dirs.items():
        lbl_dir = labels_dir_for(img_dir)
        for p in list_images(img_dir):
            lp = label_path_for(p, img_dir, lbl_dir)
            if lp.exists() and lp.stat().st_size > 0:
                pool.append((split, p, lp))
    if not pool:
        return
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))
    cmap = plt.get_cmap("tab20")
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (split, p, lp) in zip(axes, chosen):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        W, H = im.size
        draw = ImageDraw.Draw(im)
        boxes, _ = parse_label_file(lp, len(names), tol)
        for c, xc, yc, w, h in boxes:
            col = tuple(int(255 * v) for v in cmap(c % 20)[:3])
            draw.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                           outline=col, width=max(2, W // 300))
            label = names[c] if 0 <= c < len(names) else f"id_{c}"
            draw.text(((xc - w / 2) * W + 2, (yc - h / 2) * H + 2), label, fill=col)
        ax.imshow(im)
        ax.set_title(f"{split}: {p.name[:28]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "samples_with_boxes.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Carpeta raíz del dataset (descomprimido).")
    ap.add_argument("--yaml", default=None, help="Ruta al data.yaml (por defecto se busca en --root).")
    ap.add_argument("--names", default=None, help="Nombres de clase separados por coma si no hay YAML.")
    ap.add_argument("--out", default="reports/eda", help="Carpeta de salida.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-3, help="Tolerancia para cajas fuera de [0,1].")
    ap.add_argument("--near-dup-threshold", type=int, default=2, help="Distancia Hamming (dHash 64 bits).")
    ap.add_argument("--skip-near-dup", action="store_true", help="Omite duplicados perceptuales (más rápido).")
    ap.add_argument("--max-images", type=int, default=0, help="Limita imágenes por partición (pruebas rápidas).")
    ap.add_argument("--samples", type=int, default=12, help="Imágenes en la cuadrícula de ejemplo.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"No existe la carpeta: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(args.yaml).resolve() if args.yaml else find_data_yaml(root)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path else {}
    names = [n.strip() for n in args.names.split(",")] if args.names else class_names(cfg)
    if not names:
        sys.exit("No se encontraron nombres de clase: pasa --yaml o --names 'a,b,c'.")

    split_dirs = resolve_splits(root, yaml_path, cfg)
    if not split_dirs:
        sys.exit(f"No se encontraron imágenes en {root}. Revisa la estructura del dataset.")
    LOG.info("YAML: %s | clases (%d): %s", yaml_path, len(names), names)
    LOG.info("Particiones: %s", {k: str(v) for k, v in split_dirs.items()})

    summaries, all_records, boxes_by_split, counts_by_split = {}, [], {}, {}
    for split, img_dir in split_dirs.items():
        s, recs, boxes, counts = analyze_split(split, img_dir, names, args.tol, args.max_images)
        summaries[split], boxes_by_split[split], counts_by_split[split] = s, boxes, counts
        all_records.extend(recs)

    duplicates = find_duplicates(all_records, args.near_dup_threshold, args.skip_near_dup)

    totals = Counter()
    for s in summaries.values():
        totals.update(s["objects_per_class"])
    positive = {k: v for k, v in totals.items() if v > 0 and k in names}
    n_images_total = sum(s["n_images"] for s in summaries.values())
    imbalance = {
        "objects_per_class_total": dict(totals),
        "class_share_pct": {k: round(100 * v / max(sum(totals.values()), 1), 2) for k, v in totals.items()},
        "max_min_ratio": (max(positive.values()) / min(positive.values())) if positive else None,
        "classes_without_objects": [n for n in names if totals.get(n, 0) == 0],
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset": {"root": str(root), "yaml": str(yaml_path) if yaml_path else None, "classes": names},
        "totals": {
            "images": n_images_total,
            "objects": int(sum(s["n_objects"] for s in summaries.values())),
            "split_share_pct": {k: round(100 * s["n_images"] / max(n_images_total, 1), 2) for k, s in summaries.items()},
        },
        "splits": summaries,
        "duplicates": duplicates,
        "imbalance": imbalance,
        "parameters": {k: v for k, v in vars(args).items()},
    }
    report["recommendations"] = build_recommendations(report)

    (out / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )

    all_boxes = np.concatenate([b for b in boxes_by_split.values() if len(b)]) if any(
        len(b) for b in boxes_by_split.values()
    ) else np.zeros((0, 5))
    fig_images_per_split(summaries, out)
    fig_class_distribution(summaries, names, out)
    fig_resolutions([r for r in all_records if r["w"]], out)
    fig_bbox_area(all_boxes, names, out)
    fig_objects_per_image([c for cs in counts_by_split.values() for c in cs], out)
    fig_center_heatmap(all_boxes, out)
    fig_samples(split_dirs, names, out, args.samples, args.seed, args.tol)

    LOG.info("Imágenes: %s", {k: s["n_images"] for k, s in summaries.items()})
    LOG.info("Objetos por clase: %s", dict(totals))
    for rec in report["recommendations"]:
        LOG.warning("→ %s", rec)
    LOG.info("Listo en %.1f s. Reporte: %s", time.time() - t0, out / "dataset_report.json")


if __name__ == "__main__":
    main()
