"""Genera evidencia controlada de cumplimiento e incumplimiento de SLO."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alertas import AlertManager
from src.metricas import calculate_window_metrics
from src.slo import evaluate_slos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    execution_id = f"slo-simulation-{uuid.uuid4().hex[:12]}"
    manager = AlertManager(args.log_path, cooldown_seconds=300)
    compliant_metrics = calculate_window_metrics(
        [120, 180, 220, 260, 300, 340, 380, 420, 460, 480], error_count=0
    )
    violation_metrics = calculate_window_metrics([606] * 100, error_count=2)
    compliant = evaluate_slos(compliant_metrics, error_count=0, config=config)
    violation = evaluate_slos(violation_metrics, error_count=2, config=config)
    alerts = manager.evaluate_slo(
        violation, execution_id=execution_id, window_id="slo-window-violation"
    )
    result = {
        "execution_id": execution_id,
        "execution_type": "controlled_slo_compliance_and_violation_simulation",
        "simulation": {
            "is_simulation": True,
            "description": "Ventanas controladas para demostrar cumplimiento e incumplimiento de SLO; no representan tráfico de producción.",
        },
        "config": config,
        "scenarios": {
            "compliant": compliant,
            "violation": violation,
        },
        "alerts": alerts,
        "log_path": str(args.log_path.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{execution_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Actividad8_Observabilidad_SLO")
    with mlflow.start_run(run_name=execution_id) as run:
        mlflow.set_tags(
            {
                "execution_id": execution_id,
                "execution_type": result["execution_type"],
                "simulation": "true",
                "phase": "fase_4",
            }
        )
        mlflow.log_params(
            {
                "evaluation_window": config["evaluation_window"],
                "budget_window": config["budget_window"],
                "availability_target": config["availability_target"],
                "latency_p95_target_ms": config["latency_p95_target_ms"],
                "accuracy_target": config["accuracy_target"],
                "f1_macro_target": config["f1_macro_target"],
            }
        )
        mlflow.log_metrics(
            {
                "compliant_availability": float(compliant["slis"]["availability"]),
                "compliant_latency_p95_ms": float(compliant["slis"]["latency_p95_ms"]),
                "violation_availability": float(violation["slis"]["availability"]),
                "violation_latency_p95_ms": float(violation["slis"]["latency_p95_ms"]),
                "violation_availability_budget_consumed": float(violation["slos"]["availability"]["error_budget"]["budget_consumed_fraction"]),
                "violation_latency_budget_consumed": float(violation["slos"]["latency_p95"]["error_budget"]["budget_consumed_fraction"]),
            }
        )
        mlflow.log_dict(result, "slo_simulation_result.json")
        mlflow.log_artifact(str(args.config), artifact_path="config")
        mlflow.log_artifact(str(args.log_path), artifact_path="alerts")
        result["mlflow_run_id"] = run.info.run_id
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
