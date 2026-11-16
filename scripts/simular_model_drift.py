"""Compara model drift con etiquetas reales y crea una degradación controlada."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import uuid
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.actividad5.datos_prep import EXPECTED_LABELS, load_dataset
from src.configuracion import load_config
from src.metricas import calculate_supervised_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    rows, _ = load_dataset(args.data)
    texts = [row["text"] for row in rows]
    y_true = [row["airline_sentiment"] for row in rows]
    with args.model.open("rb") as stream:
        model = pickle.load(stream)
    baseline_pred = list(model.predict(texts))
    baseline = calculate_supervised_metrics(y_true, baseline_pred, EXPECTED_LABELS)
    degraded_pred = baseline_pred[:]
    for index in range(0, len(degraded_pred), 2):
        degraded_pred[index] = EXPECTED_LABELS[(EXPECTED_LABELS.index(y_true[index]) + 1) % len(EXPECTED_LABELS)]
    degraded = calculate_supervised_metrics(y_true, degraded_pred, EXPECTED_LABELS)
    config = load_config("drift_alerts")
    execution_id = f"model-drift-simulation-{uuid.uuid4().hex[:12]}"
    result = {"execution_id": execution_id, "execution_type": "controlled_model_drift_simulation", "simulation": {"is_simulation": True, "description": "Las predicciones degradadas se alteran únicamente para demostrar una alerta; las etiquetas y métricas se calculan sobre datos reales."}, "data_path": str(args.data.resolve()), "model_path": str(args.model.resolve()), "baseline": baseline, "degraded_simulated": degraded, "thresholds": {"accuracy_target": 0.75, "f1_macro_target": 0.70}, "model_drift_status": "simulated_degradation_below_accuracy_target" if degraded["accuracy"] < 0.75 else "not_triggered", "alert_config": config}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Actividad8_Observabilidad_Drift")
    with mlflow.start_run(run_name=execution_id) as run:
        mlflow.set_tags({"execution_id": execution_id, "execution_type": result["execution_type"], "simulation": "true", "labels_available": "true"})
        mlflow.log_metrics({"baseline_accuracy": float(baseline["accuracy"]), "baseline_f1_macro": float(baseline["f1_macro"]), "degraded_accuracy": float(degraded["accuracy"]), "degraded_f1_macro": float(degraded["f1_macro"])})
        mlflow.log_dict(result, "model_drift_simulation.json")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
