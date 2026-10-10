"""Evaluación de predicciones (predictions.jsonl) contra una verdad terreno revisada (ground_truth.jsonl).

Formato de ground_truth.jsonl (una línea por registro):
    {"record_type": "image", "source_id": "...", "frame_id": 0, "image_file": "...", "image_width": W, "image_height": H}
    {"record_type": "annotation", "source_id": "...", "frame_id": 0, "model_class": "helmet",
     "bbox": {"x1":..,"y1":..,"x2":..,"y2":..}}
Los registros "image" definen el ALCANCE de la evaluación (imágenes revisadas, incluso sin objetos);
las predicciones sobre imágenes fuera de ese alcance se ignoran (no se pueden juzgar).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IOU_RANGE = np.round(np.arange(0.5, 0.951, 0.05), 2)
SMALL_REL_AREA = (32 * 32) / (640 * 640)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def load_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """Lee un JSONL ignorando (y contando) líneas inválidas."""
    records, invalid = [], 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                invalid += 1
    return records, invalid


def cls_of(rec: dict) -> str:
    return rec.get("model_class") or rec["class_name"]


def box_of(rec: dict) -> tuple[float, float, float, float]:
    b = rec["bbox"]
    return float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])


def key_of(rec: dict) -> tuple[str, int]:
    return str(rec["source_id"]), int(rec.get("frame_id", 0))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_greedy(pred_boxes: np.ndarray, pred_conf: np.ndarray, gt_boxes: np.ndarray, thr: float):
    """Emparejamiento voraz por confianza descendente. Devuelve (tp[pred], gt_emparejado[gt], gt_de_cada_pred)."""
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    tp = np.zeros(n_p, dtype=bool)
    gt_matched = np.zeros(n_g, dtype=bool)
    pred_to_gt = np.full(n_p, -1, dtype=int)
    if n_p and n_g:
        ious = iou_matrix(pred_boxes, gt_boxes)
        for p in np.argsort(-pred_conf, kind="stable"):
            cand = np.where(gt_matched, -1.0, ious[p])
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                tp[p], gt_matched[j], pred_to_gt[p] = True, True, j
    return tp, gt_matched, pred_to_gt


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int):
    """AP con interpolación de 101 puntos (COCO). Devuelve (ap, recall, precision)."""
    if n_gt == 0:
        return None, np.array([]), np.array([])
    if len(conf) == 0:
        return 0.0, np.array([]), np.array([])
    order = np.argsort(-conf, kind="stable")
    t = tp[order]
    tpc, fpc = np.cumsum(t), np.cumsum(~t)
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    q = np.zeros(101)
    valid = idx < len(envelope)
    q[valid] = envelope[idx[valid]]
    return float(q.mean()), recall, precision


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def split_gt(gt_records: list[dict]):
    labeled: dict[tuple, dict] = {}
    annotations = []
    for r in gt_records:
        if r.get("record_type") == "image":
            labeled[key_of(r)] = r
        else:
            annotations.append(r)
    scope_explicit = bool(labeled)
    for a in annotations:
        labeled.setdefault(key_of(a), {"source_id": a["source_id"], "frame_id": a.get("frame_id", 0)})
    return labeled, annotations, scope_explicit


def evaluate_detections(preds: list[dict], gt_records: list[dict], iou_thr: float = 0.5,
                        conf_op: float = 0.0, classes: list[str] | None = None, max_examples: int = 25) -> dict:
    labeled, gts, scope_explicit = split_gt(gt_records)
    preds = [p for p in preds if key_of(p) in labeled]
    classes = classes or sorted({cls_of(r) for r in gts} | {cls_of(r) for r in preds})

    P: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    G: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in preds:
        P[key_of(p)][cls_of(p)].append(p)
    for g in gts:
        G[key_of(g)][cls_of(g)].append(g)

    # ---- AP por clase en 0.5:0.95 (usa TODAS las predicciones guardadas) ----
    ap_by_cls: dict[str, dict] = {}
    pr_curves: dict[str, tuple] = {}
    op_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in classes}
    for c in classes:
        n_gt = sum(len(G[k].get(c, [])) for k in labeled)
        confs = {t: [] for t in IOU_RANGE}
        tps = {t: [] for t in IOU_RANGE}
        for k in labeled:
            pl, gl = P[k].get(c, []), G[k].get(c, [])
            pb = np.array([box_of(x) for x in pl]).reshape(-1, 4)
            pc = np.array([x["confidence"] for x in pl], dtype=float)
            gb = np.array([box_of(x) for x in gl]).reshape(-1, 4)
            for t in IOU_RANGE:
                tp, _, _ = match_greedy(pb, pc, gb, float(t))
                confs[t].extend(pc.tolist()); tps[t].extend(tp.tolist())
                if abs(t - iou_thr) < 1e-9:
                    keep = pc >= conf_op
                    n_tp = int(tp[keep].sum())
                    op_counts[c]["tp"] += n_tp
                    op_counts[c]["fp"] += int(keep.sum()) - n_tp
                    op_counts[c]["pred"] += int(keep.sum())
        for t in IOU_RANGE:
            ap, rec, prec = average_precision(np.array(confs[t]), np.array(tps[t], dtype=bool), n_gt)
            ap_by_cls.setdefault(c, {})[float(t)] = ap
            if abs(t - 0.5) < 1e-9:
                pr_curves[c] = (rec, prec)
        op_counts[c]["gt"] = n_gt
        op_counts[c]["fn"] = n_gt - op_counts[c]["tp"]

    per_class = {}
    for c in classes:
        o = op_counts[c]
        aps = [v for v in ap_by_cls[c].values() if v is not None]
        per_class[c] = {"gt_objects": o["gt"], "predictions": o["pred"], **_prf(o["tp"], o["fp"], o["fn"]),
                        "ap50": round(ap_by_cls[c][0.5], 4) if ap_by_cls[c][0.5] is not None else None,
                        "ap50_95": round(float(np.mean(aps)), 4) if aps else None}
    with_gt = [c for c in classes if per_class[c]["gt_objects"] > 0]
    micro = _prf(sum(op_counts[c]["tp"] for c in classes), sum(op_counts[c]["fp"] for c in classes),
                 sum(op_counts[c]["fn"] for c in classes))
    macro = {m: round(float(np.mean([per_class[c][m] for c in with_gt])), 4) if with_gt else None
             for m in ("precision", "recall", "f1")}

    # ---- errores con causa probable + matriz de confusión (IoU = iou_thr, conf >= conf_op) ----
    errors, fp_causes, fn_causes = [], Counter(), Counter()
    labels = classes + ["background"]
    conf_mat = np.zeros((len(labels), len(labels)), dtype=int)  # filas = real, columnas = predicho
    cidx = {c: i for i, c in enumerate(labels)}
    bg = cidx["background"]
    for k, meta in labeled.items():
        W, H = meta.get("image_width"), meta.get("image_height")
        pl_all = [p for c in classes for p in P[k].get(c, [])]
        pl = [p for p in pl_all if p["confidence"] >= conf_op]
        gl = [g for c in classes for g in G[k].get(c, [])]
        pb = np.array([box_of(x) for x in pl]).reshape(-1, 4)
        gb = np.array([box_of(x) for x in gl]).reshape(-1, 4)
        ious = iou_matrix(pb, gb)

        # matriz de confusión: emparejamiento ignorando clase
        pc = np.array([x["confidence"] for x in pl], dtype=float)
        tp_any, gt_any, p2g = match_greedy(pb, pc, gb, iou_thr)
        for i, p in enumerate(pl):
            if p2g[i] >= 0:
                conf_mat[cidx[cls_of(gl[p2g[i]])], cidx[cls_of(p)]] += 1
            else:
                conf_mat[bg, cidx[cls_of(p)]] += 1
        for j, g in enumerate(gl):
            if not gt_any[j]:
                conf_mat[cidx[cls_of(g)], bg] += 1

        # emparejamiento por clase (define TP/FP/FN) y causas
        for c in classes:
            pl_idx = [i for i, p in enumerate(pl) if cls_of(p) == c]
            ps = [pl[i] for i in pl_idx]
            gs = [g for g in gl if cls_of(g) == c]
            pbc = np.array([box_of(x) for x in ps]).reshape(-1, 4)
            gbc = np.array([box_of(x) for x in gs]).reshape(-1, 4)
            tp, gm, _ = match_greedy(pbc, np.array([x["confidence"] for x in ps], dtype=float), gbc, iou_thr)
            same = iou_matrix(pbc, gbc)
            for i, p in enumerate(ps):
                if tp[i]:
                    continue
                best_same = float(same[i].max()) if same.size else 0.0
                other = [(float(ious[pl_idx[i], j]), cls_of(g)) for j, g in enumerate(gl) if cls_of(g) != c]
                best_other = max(other, default=(0.0, ""))
                if best_same >= iou_thr:
                    cause = "duplicate_detection"
                elif best_same >= 0.1:
                    cause = "localization_error"
                elif best_other[0] >= iou_thr:
                    cause = f"class_confusion(gt={best_other[1]})"
                else:
                    cause = "false_alarm_background"
                fp_causes[cause] += 1
                errors.append({"type": "FP", "source_id": k[0], "frame_id": k[1], "class": c, "confidence": p["confidence"],
                               "bbox": p["bbox"], "best_iou_same_class": round(best_same, 3), "cause": cause,
                               "image_file": p.get("image_file")})
            allp = P[k].get(c, [])
            allb = np.array([box_of(x) for x in allp]).reshape(-1, 4)
            for j, g in enumerate(gs):
                if gm[j]:
                    continue
                gbox = np.array([box_of(g)])
                best_same = float(iou_matrix(pbc, gbox).max()) if len(pbc) else 0.0
                oth = [(float(iou_matrix(np.array([box_of(p)]), gbox)[0, 0]), cls_of(p)) for p in pl if cls_of(p) != c]
                best_other = max(oth, default=(0.0, ""))
                low = float(iou_matrix(allb, gbox).max()) if len(allb) else 0.0
                if best_same >= 0.1:
                    cause = "localization_error"
                elif best_other[0] >= iou_thr:
                    cause = f"class_confusion(pred={best_other[1]})"
                elif low >= iou_thr:
                    cause = "below_confidence_threshold"
                else:
                    cause = "missed_detection"
                if W and H and (g_area := (box_of(g)[2] - box_of(g)[0]) * (box_of(g)[3] - box_of(g)[1])):
                    if g_area / (W * H) < SMALL_REL_AREA:
                        cause += "+small_object"
                fn_causes[cause] += 1
                errors.append({"type": "FN", "source_id": k[0], "frame_id": k[1], "class": c, "bbox": g["bbox"],
                               "best_iou_same_class": round(best_same, 3), "cause": cause,
                               "image_file": g.get("image_file") or meta.get("image_file")})

    gt_imgs_no_pred = [k for k in labeled if G[k] and not any(P[k].values())]
    pred_imgs_no_gt = [k for k in labeled if not G[k] and any(P[k].values())]
    empty_both = [k for k in labeled if not G[k] and not any(P[k].values())]
    fn_rows = [e for e in errors if e["type"] == "FN"]

    return {
        "params": {"iou_threshold": iou_thr, "confidence_operating_point": conf_op, "ap_iou_range": [0.5, 0.95]},
        "scope": {
            "labeled_images": len(labeled), "scope_defined_by_image_records": scope_explicit,
            "gt_objects": len(gts), "predictions_in_scope": len(preds),
            "note": "Las métricas sólo cubren las imágenes revisadas; predicciones en otras imágenes se ignoran. "
                    "mAP usa todas las predicciones guardadas: si la inferencia usó un umbral alto, el mAP queda truncado.",
        },
        "global": {"micro": micro, "macro": macro,
                   "map50": round(float(np.mean([per_class[c]["ap50"] for c in with_gt])), 4) if with_gt else None,
                   "map50_95": round(float(np.mean([per_class[c]["ap50_95"] for c in with_gt])), 4) if with_gt else None},
        "per_class": per_class,
        "images_without_detections": {
            "count": len(gt_imgs_no_pred),
            "meaning": "imágenes con objetos reales y cero predicciones: cada objeto cuenta como FN",
            "examples": [f"{s}#{f}" for s, f in gt_imgs_no_pred[:max_examples]]},
        "images_with_predictions_but_no_objects": {
            "count": len(pred_imgs_no_gt), "meaning": "imágenes sin objetos reales con predicciones: todas son FP",
            "examples": [f"{s}#{f}" for s, f in pred_imgs_no_gt[:max_examples]]},
        "images_empty_in_both": len(empty_both),
        "annotations_without_prediction": {"count": len(fn_rows), "by_cause": dict(fn_causes),
                                           "examples": fn_rows[:max_examples]},
        "false_positive_causes": dict(fp_causes),
        "confusion_matrix": {"labels": labels, "rows": "real", "columns": "predicho", "matrix": conf_mat.tolist()},
        "_errors": errors,                       # el script lo escribe a errors.jsonl y lo quita del JSON
        "_pr_curves": pr_curves,                 # idem (para graficar)
    }
