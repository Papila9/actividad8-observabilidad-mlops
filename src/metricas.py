"""Cálculo de métricas operativas y de calidad del modelo.

Las métricas de inferencia no requieren etiquetas reales. Accuracy y F1 macro
solo se calculan cuando se proporcionan etiquetas reales para las predicciones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

INFERENCE_METRICS = (
    "latency_mean_ms",
    "latency_p95_ms",
    "error_rate",
    "prediction_volume",
)
SUPERVISED_METRICS = ("accuracy", "f1_macro")


def percentile(values: Sequence[float], percentage: float) -> float | None:
    """Calcula un percentil lineal sin depender de una biblioteca adicional."""
    if not values:
        return None
    if not 0 <= percentage <= 100:
        raise ValueError("percentage debe estar entre 0 y 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calculate_inference_metrics(
    latencies_ms: Sequence[float], error_count: int = 0
) -> dict[str, float | int | None]:
    """Calcula latencia, errores y volumen observables durante inferencia."""
    volume = len(latencies_ms)
    if error_count < 0 or error_count > volume:
        raise ValueError("error_count debe estar entre cero y el volumen")
    mean = sum(float(value) for value in latencies_ms) / volume if volume else None
    return {
        "latency_mean_ms": mean,
        "latency_p95_ms": percentile(latencies_ms, 95),
        "error_rate": error_count / volume if volume else None,
        "prediction_volume": volume,
    }


def calculate_supervised_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, float | None]:
    """Calcula métricas que requieren una etiqueta real por predicción."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    if not y_true:
        return {"accuracy": None, "f1_macro": None}
    kwargs: dict[str, Any] = {"average": "macro", "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, **kwargs)),
    }


def calculate_window_metrics(
    latencies_ms: Sequence[float],
    error_count: int = 0,
    y_true: Sequence[str] | None = None,
    y_pred: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, float | int | None]:
    """Combina métricas de inferencia y, si están disponibles, supervisadas."""
    result: dict[str, float | int | None] = dict(
        calculate_inference_metrics(latencies_ms, error_count)
    )
    if y_true is None and y_pred is None:
        result.update({"accuracy": None, "f1_macro": None})
        return result
    if y_true is None or y_pred is None:
        raise ValueError("y_true y y_pred deben proporcionarse juntos")
    result.update(calculate_supervised_metrics(y_true, y_pred, labels))
    return result
