"""Carga de configuración YAML, overrides por CLI y huella de configuración."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """Aplica overrides `a.b.c=valor` (el valor se interpreta como YAML)."""
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override inválido (usa clave=valor): {item}")
        key, raw = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def config_hash(*cfgs: dict) -> str:
    blob = json.dumps(cfgs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
