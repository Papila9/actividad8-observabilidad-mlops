"""Registro de versiones y rollback controlado de modelos."""

from __future__ import annotations

import json
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


def register_version(registry: dict[str, Any], version: dict[str, Any], *, status: str = "candidate") -> None:
    if version["version"] in registry["versions"]:
        raise ValueError(f"La versión ya existe: {version['version']}")
    registry["versions"][version["version"]] = {**version, "status": status}


def promote_version(registry: dict[str, Any], version: str) -> dict[str, Any]:
    if version not in registry["versions"]:
        raise KeyError(f"Versión desconocida: {version}")
    registry["active_version"] = version
    registry["events"].append({"event": "promotion", "version": version, "timestamp": datetime.now(timezone.utc).isoformat()})
    return registry


def rollback(registry: dict[str, Any], target_version: str, *, reason: str, simulation: bool = True) -> dict[str, Any]:
    if target_version not in registry["versions"]:
        raise KeyError(f"Versión objetivo desconocida: {target_version}")
    previous = registry["active_version"]
    registry["active_version"] = target_version
    event = {
        "event": "rollback",
        "from_version": previous,
        "to_version": target_version,
        "reason": reason,
        "simulation": simulation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    registry["events"].append(event)
    return event
