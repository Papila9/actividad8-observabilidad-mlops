"""Mide inferencia real y simula retrasos controlados para evaluar SLO."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alertas import AlertManager
from src.actividad5.datos_prep import load_dataset
from src.configuracion import load_config
from src.metricas import calculate_window_metrics
from src.slo import evaluate_slos


def measure_prediction_window(model: Any, texts: list[str], *, target_latency_ms: float = 0.0) -> dict[str, Any]:
    """Predice cada texto, mide con reloj monotónico e inyecta un retraso objetivo."""
    latencies_ms: list[float] = []
    errors = 0
    predictions: list[Any] = []
    for text in texts:
        started = time.perf_counter()
        try:
            predictions.append(model.predict([text])[0])
            elapsed_ms = (time.perf_counter() - started) * 1000
            if target_latency_ms > elapsed_ms:
                time.sleep((target_latency_ms - elapsed_ms) / 1000)
            latencies_ms.append((time.perf_counter() - started) * 1000)
        except Exception:
            errors += 1
            latencies_ms.append((time.perf_counter() - started) * 1000)
    return {"metrics": calculate_window_metrics(latencies_ms, error_count=errors), "predictions": predictions, "latencies_ms": latencies_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    alert_config = load_config("drift_alerts")
    execution_id = f"slo-simulation-{uuid.uuid4().hex[:12]}"
    rows, _ = load_dataset(args.data)
    texts = [row["text"] for row in rows[: args.sample_size]]
    with args.model.open("rb") as stream:
        model = pickle.load(stream)

    before = measure_prediction_window(model, texts)
    during = measure_prediction_window(model, texts, target_latency_ms=float(config["latency_p95_target_ms"]) + 106.0)
    after = measure_prediction_window(model, texts)
    windows = [before, during, after]
    evaluated = []
    for window in windows:
        metrics = window["metrics"]
        errors = int(round(float(metrics["error_rate"] or 0) * int(metrics["prediction_volume"])))
        breach = int(float(metrics["latency_p95_ms"] or 0) > float(config["latency_p95_target_ms"]))
        evaluated.append(evaluate_slos(metrics, error_count=errors, evaluation_windows=3, latency_breach_windows=breach, config=config))
    manager = AlertManager(args.log_path, cooldown_seconds=float(alert_config["cooldown_seconds"]))
    alerts = manager.evaluate_slo(evaluated[1], execution_id=execution_id, window_id="during")
    result = {
        "execution_id": execution_id,
        "execution_type": "controlled_slo_real_inference_latency_simulation",
        "simulation": {"is_simulation": True, "description": "Predicciones reales sobre una muestra local; el retraso durante se inyecta de forma controlada y no representa producción."},
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "windows": {"before": evaluated[0], "during": evaluated[1], "after": evaluated[2]},
        "recovery": {"detected_in_window": "during", "recovered_in_window": evaluated[2]["overall_status"] == "compliant", "slo_restored": evaluated[2]["overall_status"] == "compliant"},
        "alerts": alerts,
        "alert_stats": manager.stats(),
        "config": config,
        "log_path": str(args.log_path.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{execution_id}.json"
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Actividad8_Observabilidad_SLO")
    with mlflow.start_run(run_name=execution_id) as run:
        mlflow.set_tags({"execution_id": execution_id, "execution_type": result["execution_type"], "simulation": "true", "phase": "fase_4"})
        mlflow.log_params({"sample_size": len(texts), "target_latency_ms_during": float(config["latency_p95_target_ms"]) + 106.0, "cooldown_seconds": float(alert_config["cooldown_seconds"])})
        mlflow.log_metrics({"before_latency_p95_ms": float(evaluated[0]["slis"]["latency_p95_ms"]), "during_latency_p95_ms": float(evaluated[1]["slis"]["latency_p95_ms"]), "after_latency_p95_ms": float(evaluated[2]["slis"]["latency_p95_ms"])})
        mlflow.log_dict(result, "slo_real_latency_result.json")
        mlflow.log_artifact(str(args.config), artifact_path="config")
        result["mlflow_run_id"] = run.info.run_id
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
