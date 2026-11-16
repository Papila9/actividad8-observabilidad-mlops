"""Reglas de alertamiento, deduplicación y cooldown para drift."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AlertManager:
    """Emite alertas con fingerprint, ventana y cooldown configurables."""

    def __init__(
        self,
        log_path: Path,
        *,
        cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        state_path: Path | None = None,
    ) -> None:
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.state_path = state_path or self.log_path.with_suffix(".state.json")
        self._last_emitted: dict[str, float] = {}
        self._seen_windows: set[str] = set()
        self._seen_slo_keys: set[str] = set()
        self.detections_generated = 0
        self.alerts_emitted = 0
        self.alerts_suppressed = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._last_emitted = {str(k): float(v) for k, v in state.get("last_emitted", {}).items()}
        self._seen_windows = set(state.get("seen_windows", []))
        self._seen_slo_keys = set(state.get("seen_slo_keys", []))
        self.detections_generated = int(state.get("detections_generated", 0))
        self.alerts_emitted = int(state.get("alerts_emitted", 0))
        self.alerts_suppressed = int(state.get("alerts_suppressed", 0))

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps({"last_emitted": self._last_emitted, "seen_windows": sorted(self._seen_windows), "seen_slo_keys": sorted(self._seen_slo_keys), "detections_generated": self.detections_generated, "alerts_emitted": self.alerts_emitted, "alerts_suppressed": self.alerts_suppressed}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def stats(self) -> dict[str, int]:
        return {"detections_generated": self.detections_generated, "alerts_emitted": self.alerts_emitted, "alerts_suppressed": self.alerts_suppressed}

    def _write(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def record_detection(self, detection: dict[str, Any], *, execution_id: str, window_id: str) -> None:
        """Persiste la detección aunque no produzca una alerta."""
        self.detections_generated += 1
        self._write(
            {
                "timestamp": _now_iso(),
                "event": "drift_detection",
                "execution_id": execution_id,
                "window_id": window_id,
                "data_drift": detection["data_drift"],
                "class_distribution_shift": detection["class_distribution_shift"],
                "model_drift": detection["model_drift"],
            }
        )
        self._save_state()

    def evaluate(self, detection: dict[str, Any], *, execution_id: str, window_id: str) -> list[dict[str, Any]]:
        candidates: list[tuple[str, str, str, dict[str, Any]]] = []
        data = detection["data_drift"]
        if data["severity"] in {"medium", "high"}:
            candidates.append(("data_drift", data["severity"], "text_distribution_changed", data))
        class_shift = detection["class_distribution_shift"]
        if class_shift.get("severity") in {"medium", "high"}:
            candidates.append(("class_distribution_shift", class_shift["severity"], "class_distribution_changed", class_shift))
        model = detection["model_drift"]
        if model.get("severity") in {"high", "critical"}:
            candidates.append(("model_drift", model["severity"], "supervised_metric_degraded", model))
        alerts: list[dict[str, Any]] = []
        for alert_type, severity, reason, details in candidates:
            fingerprint = hashlib.sha256(f"{alert_type}|{severity}|{reason}".encode()).hexdigest()[:16]
            if window_id in self._seen_windows:
                self.alerts_suppressed += 1
                continue
            last = self._last_emitted.get(fingerprint)
            if last is not None and self.clock() - last < self.cooldown_seconds:
                self.alerts_suppressed += 1
                continue
            alert = {
                "timestamp": _now_iso(),
                "event": "drift_alert",
                "alert_id": f"alert-{fingerprint}-{window_id}",
                "fingerprint": fingerprint,
                "execution_id": execution_id,
                "window_id": window_id,
                "alert_type": alert_type,
                "severity": severity,
                "reason": reason,
                "details": details,
            }
            alerts.append(alert)
            self._last_emitted[fingerprint] = self.clock()
            self.alerts_emitted += 1
            self._write(alert)
        self._seen_windows.add(window_id)
        self._save_state()
        return alerts

    def evaluate_slo(
        self, slo_result: dict[str, Any], *, execution_id: str, window_id: str
    ) -> list[dict[str, Any]]:
        """Emite alertas SLO usando el mismo fingerprint y cooldown del manager."""
        alerts: list[dict[str, Any]] = []
        for indicator, details in slo_result["slos"].items():
            if not details.get("applicable") or details.get("passed") is not False:
                continue
            key = f"slo|{indicator}|{window_id}"
            if key in self._seen_slo_keys:
                self.alerts_suppressed += 1
                continue
            budget = details["error_budget"]
            consumed = budget.get("budget_consumed_fraction")
            severity = "critical" if consumed is not None and consumed >= 1.0 else "high"
            fingerprint = hashlib.sha256(f"slo|{indicator}|{severity}".encode()).hexdigest()[:16]
            last = self._last_emitted.get(fingerprint)
            if last is not None and self.clock() - last < self.cooldown_seconds:
                self.alerts_suppressed += 1
                self._seen_slo_keys.add(key)
                continue
            alert = {
                "timestamp": _now_iso(),
                "event": "slo_alert",
                "alert_id": f"alert-{fingerprint}-{window_id}",
                "fingerprint": fingerprint,
                "execution_id": execution_id,
                "window_id": window_id,
                "alert_type": "slo_breach",
                "indicator": indicator,
                "severity": severity,
                "reason": "slo_not_met",
                "details": details,
            }
            alerts.append(alert)
            self._last_emitted[fingerprint] = self.clock()
            self.alerts_emitted += 1
            self._seen_slo_keys.add(key)
            self._write(alert)
        self._save_state()
        return alerts
