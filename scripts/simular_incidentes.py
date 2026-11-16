"""Simula data drift controlado y registra detección, alertas y MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alertas import AlertManager  # noqa: E402
from src.configuracion import load_config  # noqa: E402
from src.deteccion_drift import (  # noqa: E402
    build_baseline,
    detect_drift,
    load_texts_and_labels,
    save_baseline,
    sha256_file,
)


def build_simulated_window(texts: list[str], sample_size: int, scenario: str) -> tuple[list[str], str]:
    """Construye una ventana controlada sin modificar el archivo de referencia."""
    normal = texts[:sample_size]
    if scenario == "long_text_subset":
        current = sorted(texts, key=len, reverse=True)[:sample_size]
        description = "Subconjunto controlado de los textos más largos; no representa tráfico de producción."
    else:
        current = [f"{text} syntheticdrift syntheticdrift" for text in normal]
        description = "Se añadieron dos tokens controlados a textos reales; no representa tráfico de producción."
    return current, description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--scenario", choices=("token_injection", "long_text_subset"), default="token_injection")
    args = parser.parse_args()
    texts, labels = load_texts_and_labels(args.data)
    baseline_path = ROOT / "evidencias" / "drift" / "baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = build_baseline(texts, labels=labels, source_path=str(args.data), source_sha256=sha256_file(args.data))
        save_baseline(baseline_path, baseline)
    alert_config = load_config("drift_alerts")
    normal = texts[: args.sample_size]
    current, scenario_description = build_simulated_window(texts, args.sample_size, args.scenario)
    window_baseline = build_baseline(normal, labels=labels[: args.sample_size], source_path=str(args.data), source_sha256=sha256_file(args.data))
    thresholds = {key: float(alert_config[key]) for key in ("text_medium", "text_high", "class_medium", "class_high", "model_f1_drop_high", "model_f1_drop_critical")}
    before_detection = detect_drift(window_baseline, normal, thresholds=thresholds)
    detection = detect_drift(window_baseline, current, thresholds=thresholds)
    after_detection = detect_drift(window_baseline, normal, thresholds=thresholds)
    execution_id = f"drift-simulation-{uuid.uuid4().hex[:12]}"
    window_id = f"window-{args.sample_size}-{args.scenario}"
    alert_path = ROOT / "logs" / "drift_alerts.jsonl"
    manager = AlertManager(alert_path, cooldown_seconds=float(alert_config["cooldown_seconds"]))
    manager.record_detection(detection, execution_id=execution_id, window_id=window_id)
    alerts = manager.evaluate(detection, execution_id=execution_id, window_id=window_id)
    result = {
        "execution_id": execution_id,
        "execution_type": "controlled_data_drift_simulation",
        "simulation": {
            "is_simulation": True,
            "description": scenario_description,
            "scenario": args.scenario,
            "source_sha256": sha256_file(args.data),
            "sample_size": len(current),
        },
        "baseline_path": str(baseline_path),
        "window_baseline": {"rows": len(normal), "source_sha256": window_baseline["source"]["sha256"], "note": "Referencia de ventana derivada del mismo archivo; no reemplaza el baseline histórico persistido."},
        "detection": detection,
        "windows": {"before": before_detection, "during": detection, "after": after_detection},
        "recovery": {"detected_in_window": "during", "recovered_in_window": after_detection["data_drift"]["severity"] == "none", "slo_restored": None, "note": "Data drift no tiene SLO de calidad; la recuperación se define por ausencia de señal estadística."},
        "alerts": alerts,
        "alert_stats": manager.stats(),
        "alert_log_path": str(alert_path),
    }
    result_path = ROOT / "evidencias" / "drift" / f"{execution_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Actividad8_Observabilidad_Drift")
    with mlflow.start_run(run_name=execution_id) as run:
        result["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags({"execution_id": execution_id, "simulation": "true", "phase": "fase_3"})
        mlflow.log_params({"sample_size": len(current), "baseline_sha256": baseline["source"]["sha256"], "window_id": window_id, "scenario": args.scenario})
        mlflow.log_metrics({"token_jsd": detection["data_drift"]["token_jsd"], "length_ks": detection["data_drift"]["length_ks"]})
        mlflow.log_dict(result, "drift_simulation_result.json")
        mlflow.log_artifact(str(baseline_path), artifact_path="baseline")
        mlflow.log_artifact(str(alert_path), artifact_path="alerts")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
