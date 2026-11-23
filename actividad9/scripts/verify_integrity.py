"""Validate the public Activity 9 package against its SHA-256 manifest.

The manifest deliberately excludes itself and this verifier to avoid circular
references. It contains no local absolute paths or protected-project inventory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evidencias" / "integridad_actividades.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(json.dumps({"status": "manifest_missing", "path": "evidencias/integridad_actividades.json"}, indent=2))
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("expected_sha256")
    if not isinstance(expected, dict) or not expected:
        print(json.dumps({"status": "manifest_invalid", "reason": "expected_sha256 is missing or empty"}, indent=2))
        return 2

    files = []
    missing = []
    mismatches = []
    valid = []
    for relative, expected_hash in expected.items():
        path = ROOT / relative
        if not path.is_file():
            item = {"path": relative, "status": "missing"}
            files.append(item)
            missing.append(item)
            continue

        actual_hash = sha256(path)
        item = {
            "path": relative,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "status": "valid" if actual_hash == expected_hash else "hash_mismatch",
        }
        files.append(item)
        if item["status"] == "valid":
            valid.append(item)
        else:
            mismatches.append(item)

    status = "valid" if not missing and not mismatches else "invalid"
    result = {
        "status": status,
        "valid_count": len(valid),
        "missing_count": len(missing),
        "hash_mismatch_count": len(mismatches),
        "files": files,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
