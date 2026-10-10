#!/usr/bin/env python
"""Selección de muestras para el nuevo ciclo (reetiquetado / reentrenamiento).

Categorías: baja confianza, errores (FP/FN), imágenes sin detección, alta confianza (control de calidad).
La selección se estratifica por cámara y clase para reducir sesgos.

Uso:
    python scripts/select_samples.py --run outputs/latest --errors reports/evaluation/errors.jsonl --n 100
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1] / "src", Path.cwd() / "src"):
    sys.path.insert(0, str(_p))
from ppe.evaluation.detection import load_jsonl  # noqa: E402

LOG = logging.getLogger("ppe.select")


def round_robin(rows: list[dict], n: int, strata_keys: tuple[str, ...]) -> list[dict]:
    """Toma n filas alternando entre estratos (cámara, clase) para maximizar diversidad."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in strata_keys)].append(r)
    picked = []
    for batch in zip_longest(*groups.values()):
        picked.extend(x for x in batch if x is not None)
        if len(picked) >= n:
            break
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/latest", help="Carpeta con predictions.jsonl e images_index.jsonl.")
    ap.add_argument("--errors", default=None, help="errors.jsonl de evaluate_predictions.py.")
    ap.add_argument("--out", default="reports/next_cycle_samples.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--low-conf-band", type=float, nargs=2, default=(0.25, 0.5))
    ap.add_argument("--high-conf", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)

    run = Path(args.run)
    preds, _ = load_jsonl(run / "predictions.jsonl")
    index, _ = load_jsonl(run / "images_index.jsonl")
    index = [r for r in index if r.get("status") == "ok"]
    by_img: dict[tuple, list[dict]] = defaultdict(list)
    for p in preds:
        by_img[(p["source_id"], p["frame_id"])].append(p)
    meta = {(r["source_id"], r["frame_id"]): r for r in index}

    def row(key, category, reason, score, klass="-"):
        m = meta.get(key, {})
        return {"category": category, "reason": reason, "score": round(score, 4), "source_id": key[0],
                "frame_id": key[1], "camera_id": m.get("camera_id", "-"), "class": klass,
                "image_path": m.get("image_path", ""), "image_file": m.get("image_file", "")}

    low, high, none_ = [], [], []
    lo, hi = args.low_conf_band
    for key in meta:
        dets = by_img.get(key, [])
        if not dets:
            none_.append(row(key, "sin_deteccion", "cero detecciones", 0.0))
            continue
        weakest = min(dets, key=lambda d: d["confidence"])
        if lo <= weakest["confidence"] < hi:
            low.append(row(key, "baja_confianza", f"min conf {weakest['confidence']:.2f}",
                           weakest["confidence"], weakest["model_class"]))
        if all(d["confidence"] >= args.high_conf for d in dets):
            high.append(row(key, "alta_confianza_QC", "control de calidad", min(d["confidence"] for d in dets),
                            dets[0]["model_class"]))
    low.sort(key=lambda r: abs(r["score"] - (lo + hi) / 2))   # lo más incierto primero
    errs = []
    if args.errors and Path(args.errors).exists():
        seen = set()
        for e in load_jsonl(args.errors)[0]:
            key = (e["source_id"], e["frame_id"])
            if (key, e["type"], e["class"]) in seen:
                continue
            seen.add((key, e["type"], e["class"]))
            errs.append(row(key, f"error_{e['type']}", e["cause"], e.get("confidence", 0.0), e["class"]))
    rng.shuffle(none_); rng.shuffle(high)

    quota = {"errors": 0.35, "low": 0.30, "none": 0.15, "high": 0.20}
    picked = []
    seen_imgs: set[tuple] = set()
    for name, rows in (("errors", errs), ("low", low), ("none", none_), ("high", high)):
        for r in round_robin(rows, int(args.n * quota[name]), ("camera_id", "class")):
            k = (r["source_id"], r["frame_id"])
            if k not in seen_imgs:
                seen_imgs.add(k)
                picked.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "reason", "score", "source_id", "frame_id", "camera_id", "class", "image_file", "image_path"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)
    counts = defaultdict(int)
    for r in picked:
        counts[r["category"]] += 1
    LOG.info("Muestras seleccionadas: %d %s -> %s", len(picked), dict(counts), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
