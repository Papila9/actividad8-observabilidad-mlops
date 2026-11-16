"""Carga centralizada de configuraciones versionadas del proyecto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str, *, config_dir: Path | None = None) -> dict[str, Any]:
    """Carga una configuración permitida sin duplicar reglas en los scripts."""
    allowed = {"drift_alerts.json", "model_lifecycle.json"}
    filename = name if name.endswith(".json") else f"{name}.json"
    if filename not in allowed:
        raise ValueError(f"Configuración no permitida: {name}")
    path = (config_dir or ROOT / "config") / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"La configuración debe ser un objeto JSON: {path}")
    return value
