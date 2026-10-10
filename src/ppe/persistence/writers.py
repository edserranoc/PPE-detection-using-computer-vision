"""Escritura de JSONL: una línea = un JSON válido, con flush inmediato (resistente a caídas)."""
from __future__ import annotations

import json
from pathlib import Path


class JsonlWriter:
    def __init__(self, path: Path, mode: str = "w") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(mode, encoding="utf-8")
        self.count = 0

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
