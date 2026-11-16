"""Monitorización de ventanas de inferencia con logs, trazas y MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

try:
    from src.metricas import calculate_window_metrics
except ImportError:  # Permite ejecutar el archivo directamente desde src/.
    from metricas import calculate_window_metrics


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    """Serializa cada evento de logging como un objeto JSON independiente."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", {})
        event = {
            "timestamp": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            **payload,
            "message": record.getMessage(),
        }
        return json.dumps(event, ensure_ascii=False, sort_keys=True)


def configure_json_logger(log_path: Path, logger_name: str) -> logging.Logger:
    """Configura un logger JSON aislado para una ejecución."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def load_baseline(path: Path) -> dict[str, float | None]:
    """Carga únicamente métricas históricas declaradas del baseline."""
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["models"]["logistic_regression"]["test_metrics"]
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
    }


@dataclass
class MonitoringSession:
    """Agrupa observaciones de una ejecución y la registra en MLflow."""

    model_name: str
    model_version: str
    output_dir: Path
    baseline_path: Path | None = None
    tracking_uri: str | None = None
    experiment_name: str = "Actividad8_Observabilidad"
    execution_id: str = field(default_factory=lambda: f"exec-{uuid.uuid4().hex[:12]}")
    _observations: list[dict[str, Any]] = field(default_factory=list, init=False)
    _trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    _log_path: Path = field(init=False)
    _trace_path: Path = field(init=False)
    _logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        project_root = self.output_dir.parents[1]
        logs_dir = project_root / "logs"
        traces_dir = project_root / "traces"
        self._log_path = logs_dir / f"{self.execution_id}.jsonl"
        self._trace_path = traces_dir / f"{self.execution_id}.jsonl"
        self._logger = configure_json_logger(self._log_path, f"monitor.{self.execution_id}")
        self._emit("execution_started", {"model": self.model_name, "model_version": self.model_version})

    def _emit(self, event: str, fields: dict[str, Any]) -> None:
        trace_event = {
            "timestamp": utc_now(),
            "execution_id": self.execution_id,
            "event": event,
            **fields,
        }
        self._trace.append(trace_event)
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace_event, ensure_ascii=False, sort_keys=True) + "\n")
        level = logging.ERROR if event == "prediction_error" else logging.INFO
        self._logger.log(level, event, extra={"payload": trace_event})

    def record_prediction(
        self,
        *,
        predicted_label: str | None,
        latency_ms: float,
        true_label: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        vectorization_ms: float | None = None,
        prediction_ms: float | None = None,
    ) -> None:
        """Registra una predicción; true_label es opcional y nunca se infiere."""
        if latency_ms < 0:
            raise ValueError("latency_ms no puede ser negativo")
        trace_id = trace_id or f"trace-{uuid.uuid4().hex[:16]}"
        span_id = span_id or f"span-{uuid.uuid4().hex[:16]}"
        stage_durations = {"vectorization_ms": vectorization_ms, "prediction_ms": prediction_ms}
        if any(value is not None and value < 0 for value in stage_durations.values()):
            raise ValueError("Las duraciones de etapas no pueden ser negativas")
        observation = {
            "predicted_label": predicted_label,
            "true_label": true_label,
            "latency_ms": float(latency_ms),
            "error": error,
            "metadata": metadata or {},
            "trace_id": trace_id,
            "span_id": span_id,
            "stage_durations_ms": stage_durations,
        }
        self._observations.append(observation)
        self._emit(
            "prediction_error" if error else "prediction_recorded",
            {"latency_ms": float(latency_ms), "error": error, "metadata": metadata or {}, "trace_id": trace_id, "span_id": span_id, "stage_durations_ms": stage_durations},
        )

    def finalize(self) -> dict[str, Any]:
        """Calcula la ventana, guarda resultados, registra MLflow y cierra el log."""
        latencies = [item["latency_ms"] for item in self._observations]
        errors = sum(item["error"] is not None for item in self._observations)
        labelled = [item for item in self._observations if item["true_label"] is not None]
        predictions = [item for item in labelled if item["predicted_label"] is not None]
        supervised_available = len(labelled) == len(predictions) and bool(labelled)
        metrics = calculate_window_metrics(
            latencies,
            errors,
            [item["true_label"] for item in predictions] if supervised_available else None,
            [item["predicted_label"] for item in predictions] if supervised_available else None,
            sorted({item["true_label"] for item in predictions}) if supervised_available else None,
        )
        result: dict[str, Any] = {
            "execution_id": self.execution_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metrics": metrics,
            "metric_availability": {
                "inference": True,
                "supervised": supervised_available,
                "supervised_observations": len(predictions),
            },
            "log_path": str(self._log_path),
            "trace_path": str(self._trace_path),
        }
        if self.baseline_path is not None and supervised_available:
            baseline = load_baseline(self.baseline_path)
            result["baseline_comparison"] = {
                name: {
                    "baseline": baseline[name],
                    "observed": metrics[name],
                    "delta": metrics[name] - baseline[name],
                }
                for name in ("accuracy", "f1_macro")
            }
        elif self.baseline_path is not None:
            result["baseline_comparison"] = {"status": "unavailable_without_real_labels"}

        result_path = self.output_dir / f"{self.execution_id}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mlflow_run_id = self._log_to_mlflow(result, result_path)
        result["mlflow_run_id"] = mlflow_run_id
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._emit("execution_finished", {"metrics": metrics, "mlflow_run_id": mlflow_run_id})
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()
        self._logger.handlers.clear()
        return result

    def _log_to_mlflow(self, result: dict[str, Any], result_path: Path) -> str:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.execution_id) as run:
            mlflow.set_tags(
                {
                    "execution_id": self.execution_id,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "observability_phase": "fase_2",
                }
            )
            mlflow.log_params(
                {
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "prediction_volume": len(self._observations),
                }
            )
            for name, value in result["metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(name, float(value))
            result["mlflow_run_id"] = run.info.run_id
            mlflow.log_dict(result, "monitoring_result.json")
            mlflow.log_artifact(str(self._log_path), artifact_path="logs")
            mlflow.log_artifact(str(self._trace_path), artifact_path="traces")
            return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", default="logistic_regression")
    parser.add_argument("--model-version", default="actividad5-baseline")
    args = parser.parse_args()
    session = MonitoringSession(
        model_name=args.model_name,
        model_version=args.model_version,
        output_dir=args.output_dir,
        baseline_path=args.baseline,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(session.finalize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
