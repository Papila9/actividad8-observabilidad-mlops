"""Registra dos incidentes controlados usando evidencias de Fases 3 y 4."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.incidentes import append_incident, build_incident


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-result", type=Path, required=True)
    parser.add_argument("--slo-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()

    drift = json.loads(args.drift_result.read_text(encoding="utf-8"))
    slo = json.loads(args.slo_result.read_text(encoding="utf-8"))
    drift_alert = drift["alerts"][0]
    latency_alert = next(alert for alert in slo["alerts"] if alert["indicator"] == "latency_p95")
    incidents = [
        build_incident(drift_alert, simulation=True),
        build_incident(latency_alert, simulation=True),
    ]
    execution_id = f"incidents-simulation-{uuid.uuid4().hex[:12]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    for incident in incidents:
        incident["execution_id"] = execution_id
        append_incident(args.log_path, incident)
    result = {
        "execution_id": execution_id,
        "execution_type": "controlled_incident_and_runbook_simulation",
        "simulation": {"is_simulation": True, "description": "Incidentes derivados de evidencias controladas; no representan producción."},
        "source_evidence": [str(args.drift_result.resolve()), str(args.slo_result.resolve())],
        "incidents": incidents,
        "log_path": str(args.log_path.resolve()),
    }
    result_path = args.output_dir / f"{execution_id}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Actividad8_Observabilidad_Incidentes")
    with mlflow.start_run(run_name=execution_id) as run:
        mlflow.set_tags({"execution_id": execution_id, "execution_type": result["execution_type"], "simulation": "true", "phase": "fase_5"})
        mlflow.log_params({"incident_count": len(incidents), "runbook_count": len({item["incident_type"] for item in incidents})})
        mlflow.log_metrics({"data_drift_incidents": 1.0, "slo_latency_incidents": 1.0})
        mlflow.log_dict(result, "incidents_simulation_result.json")
        mlflow.log_artifact(str(args.log_path), artifact_path="incidents")
        result["mlflow_run_id"] = run.info.run_id
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
