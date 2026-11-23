"""Verify hashes of the public Activity 9 evidence package.

This public copy intentionally avoids local absolute paths, protected repository
inventories, model files, credentials, and other internal implementation data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    "evidencias/Tweets.csv",
    "evidencias/Tweets_limpios.csv",
    "evidencias/fairness_por_aerolinea.csv",
    "evidencias/fairness_metodologia.json",
    "evidencias/rendimiento_local.csv",
    "evidencias/rendimiento_metodologia.json",
    "evidencias/dataset_provenance.json",
    "evidencias/integridad_actividades.json",
    "scripts/measure_activity9.py",
    "scripts/verify_integrity.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    result = {"root_relative": ".", "files": [], "ok": True}
    for relative in PUBLIC_FILES:
        path = ROOT / relative
        exists = path.is_file()
        item = {"path": relative, "exists": exists}
        if exists:
            item["bytes"] = path.stat().st_size
            item["sha256"] = sha256(path)
        else:
            result["ok"] = False
        result["files"].append(item)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
