"""Clasificación y registro estructurado de incidentes académicos."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNBOOKS: dict[str, dict[str, Any]] = {
    "data_drift": {
        "severity_default": "high",
        "detection": "Comparar la ventana contra el baseline mediante KS/JSD.",
        "diagnosis": "Confirmar window_id, volumen, fuente, tokenización y duplicados.",
        "impact": "No inferir pérdida de precisión sin etiquetas reales; revisar exposición y clases.",
        "response": "Mantener observación, solicitar etiquetas y preparar evaluación supervisada.",
        "recovery": "Dos ventanas consecutivas sin alerta o evaluación supervisada aceptable.",
        "closure": "Registrar causa, evidencia, acciones y responsable académico.",
    },
    "slo_latency": {
        "severity_default": "critical",
        "detection": "Comparar p95 de la ventana con el SLO de 500 ms y su error budget.",
        "diagnosis": "Separar inferencia local de red, cola, I/O y MLflow; revisar errores y volumen.",
        "impact": "Incumplimiento técnico de latencia; no implica model drift.",
        "response": "Reducir carga experimental, revisar pipeline y evaluar rollback si existe versión estable.",
        "recovery": "Ventana posterior dentro de 500 ms y recuperación del presupuesto en el escenario controlado.",
        "closure": "Conservar p95, alertas, acción correctiva y resultado posterior.",
    },
    "error_rate": {
        "severity_default": "high",
        "detection": "Comparar error_rate y disponibilidad contra los SLO configurados.",
        "diagnosis": "Clasificar errores por tipo y confirmar que no son fallos de etiquetado.",
        "impact": "Predicciones no procesadas o inválidas; cuantificar solicitudes afectadas.",
        "response": "Aislar causa técnica, aplicar rollback solo con evidencia y mantener trazabilidad.",
        "recovery": "Error rate dentro del objetivo en una ventana posterior.",
        "closure": "Documentar volumen, errores, causa raíz y validación.",
    },
    "model_quality": {
        "severity_default": "critical",
        "detection": "Comparar Accuracy/F1 contra SLO usando etiquetas reales.",
        "diagnosis": "Verificar representatividad, etiquetas, versión y separación de datos.",
        "impact": "Solo puede afirmarse degradación de calidad con evaluación supervisada.",
        "response": "Detener promoción, comparar candidato y considerar reentrenamiento.",
        "recovery": "Métricas supervisadas cumplen criterios de aceptación.",
        "closure": "Registrar conjunto etiquetado, métricas y decisión de modelo.",
    },
}


def classify_alert(alert: dict[str, Any]) -> str:
    """Mapea una alerta de drift/SLO a un tipo de incidente."""
    alert_type = alert.get("alert_type")
    indicator = alert.get("indicator")
    if alert_type == "data_drift":
        return "data_drift"
    if alert_type == "slo_breach" and indicator == "latency_p95":
        return "slo_latency"
    if alert_type == "slo_breach" and indicator == "availability":
        return "error_rate"
    if alert_type == "model_drift":
        return "model_quality"
    return "error_rate"


def build_incident(alert: dict[str, Any], *, simulation: bool = True) -> dict[str, Any]:
    incident_type = classify_alert(alert)
    runbook = RUNBOOKS[incident_type]
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return {
        "incident_id": f"inc-{uuid.uuid4().hex[:12]}",
        "created_at": created_at,
        "incident_type": incident_type,
        "severity": alert.get("severity", runbook["severity_default"]),
        "status": "open",
        "simulation": {"is_simulation": simulation},
        "source_alert": alert,
        "runbook": runbook,
        "lifecycle": [
            {"step": "detection", "timestamp": created_at, "status": "completed", "result": "source_alert_recorded"},
            {"step": "classification", "timestamp": created_at, "status": "completed", "result": incident_type},
        ],
        "data_drift_boundary": "Data drift no implica model drift ni pérdida de precisión sin etiquetas reales.",
        "closure_note": "Pendiente de verificación y cierre del runbook; escenario académico controlado, no representa producción real." if simulation else None,
    }


def append_incident(path: Path, incident: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(incident, ensure_ascii=False, sort_keys=True) + "\n")
