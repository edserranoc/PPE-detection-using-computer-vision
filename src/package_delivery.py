#!/usr/bin/env python
"""Empaqueta la entrega en uno o varios .zip (el correo no admite enlaces a la nube ni .py sueltos).

Uso:
    python scripts/package_delivery.py --out entrega --max-mb 20            # sin dataset
    python scripts/package_delivery.py --out entrega --max-mb 20 --include-data

Reparte los archivos en partes `entrega_parte1de3.zip`, `entrega_parte2de3.zip`, ... de tamaño <= --max-mb
(un archivo individual mayor al límite queda en su propia parte). Cada parte incluye un MANIFEST.txt.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "runs", "mlruns", ".pytest_cache", ".venv", "venv"}
ALWAYS_SKIP_SUFFIX = {".pyc"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="entrega")
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--include-data", action="store_true", help="Incluye src/data (dataset).")
    ap.add_argument("--all-runs", action="store_true", help="Incluye todas las corridas de outputs/ (por defecto sólo latest).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & EXCLUDE_DIRS or p.suffix in ALWAYS_SKIP_SUFFIX:
            continue
        if rel.parts[:2] == ("src", "data") and not args.include_data:
            continue
        if rel.parts[0] == "outputs" and len(rel.parts) > 2 and rel.parts[1] != "latest" and not args.all_runs:
            continue
        if rel.suffix == ".zip" and rel.name.startswith(Path(args.out).name):
            continue
        files.append(rel)

    limit = int(args.max_mb * 1024 * 1024)
    parts: list[list[Path]] = [[]]
    size = 0
    for rel in sorted(files, key=lambda r: -(root / r).stat().st_size):
        s = (root / rel).stat().st_size
        if parts[-1] and size + s > limit:
            parts.append([])
            size = 0
        parts[-1].append(rel)
        size += s

    total = len(parts)
    for i, part in enumerate(parts, start=1):
        name = f"{args.out}_parte{i}de{total}.zip" if total > 1 else f"{args.out}.zip"
        with zipfile.ZipFile(root / name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("MANIFEST.txt", f"Parte {i} de {total}\n" + "\n".join(r.as_posix() for r in sorted(part)) + "\n")
            for rel in part:
                zf.write(root / rel, rel.as_posix())
        print(f"{name}: {len(part)} archivos, {(root / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
