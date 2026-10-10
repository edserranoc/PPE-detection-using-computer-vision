"""Asociación persona ↔ EPP.

Cada EPP se asigna a lo sumo a UNA persona: la que contiene al EPP (intersección sobre el
área del EPP, no IoU, porque un casco es diminuto frente a una persona) y cuyo centro cae
en la zona esperada del cuerpo (cabeza para casco, torso para chaleco).
"""
from __future__ import annotations

from dataclasses import dataclass

from ppe.rules.geometry import area, center, ioa
from ppe.types import Detection

ZONE_OF_ROLE = {"helmet": "head", "no_helmet": "head", "vest": "torso", "no_vest": "torso"}


@dataclass
class PersonPPE:
    """Máxima confianza de cada tipo de EPP asociado a una persona (0 = ausente)."""

    helmet: float = 0.0
    no_helmet: float = 0.0
    vest: float = 0.0
    no_vest: float = 0.0


def associate(
    persons: list[Detection],
    ppe: list[Detection],
    roles: dict[str, str],
    cfg: dict,
) -> tuple[list[PersonPPE], dict[int, int]]:
    """Devuelve (EPP por persona, {índice_ppe: índice_persona}).

    `roles` mapea nombre de clase del modelo → rol ("helmet", "no_helmet", "vest", "no_vest").
    """
    min_ioa = float(cfg.get("min_ioa", 0.5))
    zones = cfg.get("zones", {"head": [0.0, 0.40], "torso": [0.15, 0.80]})
    result = [PersonPPE() for _ in persons]
    assigned: dict[int, int] = {}

    for k, item in enumerate(ppe):
        role = roles.get(item.cls)
        if role not in ZONE_OF_ROLE:
            continue
        lo, hi = zones[ZONE_OF_ROLE[role]]
        cx, cy = center(item.bbox)
        best: tuple[float, float, int] | None = None
        for i, person in enumerate(persons):
            px1, py1, px2, py2 = person.bbox
            height = py2 - py1
            if height <= 0 or not (px1 <= cx <= px2):
                continue
            if not (lo <= (cy - py1) / height <= hi):
                continue
            score = ioa(item.bbox, person.bbox)
            if score < min_ioa:
                continue
            candidate = (score, -area(person.bbox), i)  # empate: persona más pequeña (más cercana al EPP)
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            i = best[2]
            assigned[k] = i
            setattr(result[i], role, max(getattr(result[i], role), item.conf))
    return result, assigned
