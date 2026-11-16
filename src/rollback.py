"""Registro de versiones y rollback controlado de modelos."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_registry(stable_version: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "active_version": stable_version["version"],
        "stable_version": stable_version["version"],
        "versions": {stable_version["version"]: {**stable_version, "status": "stable"}},
        "events": [],
    }


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_active_model(registry: dict[str, Any]) -> Any:
    """Carga el modelo activo y verifica su existencia e integridad."""
    version = registry["active_version"]
    entry = registry["versions"][version]
    path = Path(entry["artifact_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = artifact_sha256(path)
    expected = entry.get("artifact_sha256")
    if expected and actual != expected:
        raise ValueError(f"SHA-256 inválido para {version}")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    registry["last_loaded"] = {"version": version, "artifact_sha256": actual}
    return model


def register_version(registry: dict[str, Any], version: dict[str, Any], *, status: str = "candidate") -> None:
    if version["version"] in registry["versions"]:
        raise ValueError(f"La versión ya existe: {version['version']}")
    registry["versions"][version["version"]] = {**version, "status": status}


def promote_version(registry: dict[str, Any], version: str, *, explicit_simulation_authorization: bool = False) -> dict[str, Any]:
    if version not in registry["versions"]:
        raise KeyError(f"Versión desconocida: {version}")
    status = registry["versions"][version].get("status")
    if status == "rejected_candidate" and not explicit_simulation_authorization:
        raise ValueError("No se puede promover un rejected_candidate sin autorización explícita de simulación")
    registry["active_version"] = version
    registry["events"].append({"event": "promotion", "version": version, "timestamp": datetime.now(timezone.utc).isoformat(), "explicit_simulation_authorization": explicit_simulation_authorization})
    return registry


def rollback(registry: dict[str, Any], target_version: str, *, reason: str, simulation: bool = True) -> dict[str, Any]:
    if target_version not in registry["versions"]:
        raise KeyError(f"Versión objetivo desconocida: {target_version}")
    target = registry["versions"][target_version]
    target_path = Path(target["artifact_path"])
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    actual_hash = artifact_sha256(target_path)
    if target.get("artifact_sha256") and actual_hash != target["artifact_sha256"]:
        raise ValueError(f"SHA-256 inválido para rollback de {target_version}")
    with target_path.open("rb") as stream:
        pickle.load(stream)
    previous = registry["active_version"]
    registry["active_version"] = target_version
    event = {
        "event": "rollback",
        "from_version": previous,
        "to_version": target_version,
        "reason": reason,
        "simulation": simulation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_artifact_sha256": actual_hash,
        "model_loaded_successfully": True,
    }
    registry["events"].append(event)
    registry["versions"][target_version]["status"] = "stable"
    return event
