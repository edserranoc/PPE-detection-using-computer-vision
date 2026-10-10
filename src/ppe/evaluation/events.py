"""Desempeño de la lógica de eventos: events.jsonl vs eventos esperados.

Formato de ground_truth_events.jsonl:
    {"source_id": "...", "camera_id": "CAM_01", "event_type": "persona_sin_casco", "frame_id": 0,
     "timestamp_seconds": 0.0, "bbox": {"x1":..,"y1":..,"x2":..,"y2":..}, "scenario": "opcional"}
`frame_id`/`timestamp_seconds` marcan el instante en que debía generarse el evento;
`bbox` (opcional) desambigua varias personas con el mismo tipo de evento en una misma fuente.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ppe.evaluation.detection import _prf, box_of
from ppe.rules.geometry import iou


def _lat_stats(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0}
    a = np.asarray(vals, dtype=float)
    return {"count": int(a.size), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}


def evaluate_events(pred_events: list[dict], gt_events: list[dict], tol_frames: int = 5, min_iou: float = 0.3,
                    scope_sources: set[str] | None = None, max_examples: int = 25) -> dict:
    if scope_sources is not None:
        pred_events = [e for e in pred_events if str(e["source_id"]) in scope_sources]
    P, G = pred_events, gt_events

    cands = []
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            if str(p["source_id"]) != str(g["source_id"]) or p["event_type"] != g["event_type"]:
                continue
            dt = abs(int(p.get("frame_id", 0)) - int(g.get("frame_id", 0)))
            if dt > tol_frames:
                continue
            ov = iou(box_of(p), box_of(g)) if p.get("bbox") and g.get("bbox") else 1.0
            if ov < min_iou:
                continue
            cands.append((-ov, dt, i, j))
    cands.sort()
    p_to_g: dict[int, int] = {}
    g_matched: set[int] = set()
    for _, _, i, j in cands:
        if i not in p_to_g and j not in g_matched:
            p_to_g[i] = j
            g_matched.add(j)

    cand_by_pred = defaultdict(set)
    for _, _, i, j in cands:
        cand_by_pred[i].add(j)
    duplicates = [i for i in range(len(P)) if i not in p_to_g and cand_by_pred[i] & g_matched]
    false_pos = [i for i in range(len(P)) if i not in p_to_g and i not in set(duplicates)]
    missed = [j for j in range(len(G)) if j not in g_matched]

    # causa de omisión
    miss_rows, miss_causes = [], Counter()
    for j in missed:
        g = G[j]
        near = [p for p in P if str(p["source_id"]) == str(g["source_id"])
                and abs(int(p.get("frame_id", 0)) - int(g.get("frame_id", 0))) <= tol_frames
                and (not (p.get("bbox") and g.get("bbox")) or iou(box_of(p), box_of(g)) >= min_iou)]
        cause = "wrong_event_type_or_severity" if near else "no_event_generated(detección_o_regla)"
        miss_causes[cause] += 1
        miss_rows.append({**g, "cause": cause})

    pipeline_latency = [float(P[i].get("latency_seconds", 0.0)) for i in p_to_g]
    vs_gt = [float(P[i]["timestamp_seconds"]) - float(G[j]["timestamp_seconds"]) for i, j in p_to_g.items()
             if "timestamp_seconds" in G[j] and "timestamp_seconds" in P[i]]

    def breakdown(field: str) -> dict:
        keys = {str(x.get(field)) for x in P + G if x.get(field) is not None}
        out = {}
        for k in sorted(keys):
            tp = sum(1 for i in p_to_g if str(P[i].get(field)) == k)
            fp = sum(1 for i in false_pos if str(P[i].get(field)) == k)
            fn = sum(1 for j in missed if str(G[j].get(field)) == k)
            out[k] = _prf(tp, fp, fn)
        return out

    by_source = breakdown("source_id")
    return {
        "params": {"tolerance_frames": tol_frames, "min_bbox_iou": min_iou},
        "counts": {"predicted_events": len(P), "expected_events": len(G), "matched": len(p_to_g),
                   "duplicates": len(duplicates), "false_positives": len(false_pos), "missed": len(missed)},
        "overall": _prf(len(p_to_g), len(false_pos) + len(duplicates), len(missed)),
        "overall_excluding_duplicates_as_fp": _prf(len(p_to_g), len(false_pos), len(missed)),
        "duplicate_rate": round(len(duplicates) / len(P), 4) if P else 0.0,
        "missed_events": {"count": len(missed), "by_cause": dict(miss_causes), "examples": miss_rows[:max_examples]},
        "latency_seconds": {"pipeline_condition_to_event": _lat_stats(pipeline_latency),
                            "event_vs_ground_truth_timestamp": _lat_stats(vs_gt)},
        "by_event_type": breakdown("event_type"),
        "by_camera": breakdown("camera_id"),
        "by_scenario": breakdown("scenario"),
        "by_source": dict(list(by_source.items())[:50]),
        "false_positive_examples": [P[i] for i in false_pos[:max_examples]],
        "duplicate_examples": [P[i] for i in duplicates[:max_examples]],
    }
