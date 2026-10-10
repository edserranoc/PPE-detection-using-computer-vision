#!/usr/bin/env python
"""Desempeño de la solución: predictions.jsonl vs ground_truth.jsonl (y events.jsonl vs eventos esperados).

Uso:
    python scripts/evaluate_predictions.py \
        --predictions outputs/latest/predictions.jsonl --ground-truth data/ground_truth/ground_truth.jsonl \
        --events outputs/latest/events.jsonl --ground-truth-events data/ground_truth/ground_truth_events.jsonl \
        --images-index outputs/latest/images_index.jsonl --out reports/evaluation

Salidas (en --out): prediction_metrics.json, event_metrics.json, errors.jsonl,
    confusion_matrix_predictions.png, pr_curve_predictions.png, error_examples/*.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _p in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
    sys.path.insert(0, str(_p))
from ppe.evaluation.detection import box_of, evaluate_detections, key_of, load_jsonl, split_gt  # noqa: E402
from ppe.evaluation.events import evaluate_events  # noqa: E402

LOG = logging.getLogger("ppe.evaluate")


def plot_confusion(cm: dict, path: Path) -> None:
    m, labels = np.array(cm["matrix"]), cm["labels"]
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicho", ylabel="Real", title="Matriz de confusión (predicciones vs verdad terreno)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(curves: dict, per_class: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, (rec, prec) in curves.items():
        if len(rec):
            ap = per_class[c]["ap50"]
            ax.plot(rec, prec, label=f"{c} (AP50={ap:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Curva Precision–Recall (IoU 0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_error_examples(errors: list[dict], gt_records: list[dict], preds: list[dict], image_paths: dict,
                        out_dir: Path, n: int) -> int:
    """Guarda las N imágenes con más errores: GT en verde, FP en rojo, FN en naranja."""
    by_img: dict[tuple, list[dict]] = {}
    for e in errors:
        by_img.setdefault((e["source_id"], e["frame_id"]), []).append(e)
    ranked = sorted(by_img.items(), key=lambda kv: -len(kv[1]))[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for key, errs in ranked:
        path = image_paths.get(key)
        if not path or not Path(path).is_file():
            continue
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for g in (g for g in gt_records if g.get("record_type") != "image" and key_of(g) == key):
            x1, y1, x2, y2 = (int(v) for v in box_of(g))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
        for e in errs:
            x1, y1, x2, y2 = (int(v) for v in box_of(e))
            color = (0, 0, 255) if e["type"] == "FP" else (0, 140, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{e['type']} {e['class']}: {e['cause']}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        name = f"{str(key[0]).replace('/', '_')}_{key[1]}.jpg"
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            buf.tofile(str(out_dir / name))
            saved += 1
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--ground-truth-events", default=None)
    ap.add_argument("--images-index", default=None, help="images_index.jsonl (para localizar imágenes en las evidencias).")
    ap.add_argument("--out", default="reports/evaluation")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU de emparejamiento para P/R/F1/TP/FP/FN.")
    ap.add_argument("--conf", type=float, default=0.0, help="Punto de operación: sólo predicciones con confianza >= valor.")
    ap.add_argument("--tol-frames", type=int, default=5, help="Tolerancia temporal (frames) para emparejar eventos.")
    ap.add_argument("--event-iou", type=float, default=0.3)
    ap.add_argument("--error-images", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds, bad_p = load_jsonl(args.predictions)
    gts, bad_g = load_jsonl(args.ground_truth)
    if bad_p or bad_g:
        LOG.warning("Líneas JSON inválidas ignoradas: predicciones=%d, ground_truth=%d", bad_p, bad_g)

    res = evaluate_detections(preds, gts, args.iou, args.conf)
    errors, curves = res.pop("_errors"), res.pop("_pr_curves")
    res["input_files"] = {"predictions": args.predictions, "ground_truth": args.ground_truth,
                          "invalid_lines": {"predictions": bad_p, "ground_truth": bad_g}}
    (out / "prediction_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "errors.jsonl").open("w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    plot_confusion(res["confusion_matrix"], out / "confusion_matrix_predictions.png")
    plot_pr(curves, res["per_class"], out / "pr_curve_predictions.png")

    image_paths = {}
    for g in gts:
        if g.get("record_type") == "image" and g.get("image_path"):
            image_paths[key_of(g)] = g["image_path"]
    if args.images_index and Path(args.images_index).exists():
        for r in load_jsonl(args.images_index)[0]:
            if r.get("status") == "ok":
                image_paths.setdefault(key_of(r), r["image_path"])
    saved = draw_error_examples(errors, gts, preds, image_paths, out / "error_examples", args.error_images)

    g = res["global"]
    LOG.info("Imágenes revisadas: %d | objetos GT: %d | predicciones en alcance: %d",
             res["scope"]["labeled_images"], res["scope"]["gt_objects"], res["scope"]["predictions_in_scope"])
    LOG.info("micro P=%.3f R=%.3f F1=%.3f | mAP50=%s mAP50-95=%s | evidencias de error: %d imágenes",
             g["micro"]["precision"], g["micro"]["recall"], g["micro"]["f1"], g["map50"], g["map50_95"], saved)

    if args.events and args.ground_truth_events:
        pev, _ = load_jsonl(args.events)
        gev, _ = load_jsonl(args.ground_truth_events)
        labeled, _, _ = split_gt(gts)
        ev = evaluate_events(pev, gev, args.tol_frames, args.event_iou, {s for s, _ in labeled})
        (out / "event_metrics.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
        o = ev["overall"]
        LOG.info("Eventos: P=%.3f R=%.3f F1=%.3f | duplicados=%d (tasa %.3f) | omitidos=%d",
                 o["precision"], o["recall"], o["f1"], ev["counts"]["duplicates"], ev["duplicate_rate"],
                 ev["counts"]["missed"])
    LOG.info("Resultados en %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
