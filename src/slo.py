"""Evaluación de SLI, SLO y error budgets para una ventana operativa.

Los objetivos son académicos y configurables. No representan un compromiso
de producción ni requieren etiquetas cuando se evalúan indicadores técnicos.
"""

from __future__ import annotations

from typing import Any


DEFAULT_SLO_CONFIG: dict[str, Any] = {
    "evaluation_window": "300s",
    "budget_window": "30d_academic",
    "availability_target": 0.99,
    "latency_p95_target_ms": 500.0,
    "latency_bad_window_fraction": 0.01,
    "accuracy_target": 0.75,
    "f1_macro_target": 0.70,
    "quality_bad_window_fraction": 0.01,
    "burn_rate": {"short_window": "5m", "long_window": "1h", "warning": 2.0, "critical": 4.0},
}


def calculate_burn_rate(*, bad_windows: int, evaluated_windows: int, allowed_fraction: float) -> dict[str, float | None]:
    """Calcula burn rate de una ventana; no inventa consumo si no hay observaciones."""
    if bad_windows < 0 or evaluated_windows < 0 or bad_windows > evaluated_windows:
        raise ValueError("Las ventanas incumplidas deben estar entre cero y el total")
    if evaluated_windows == 0 or allowed_fraction <= 0:
        return {"observed_bad_fraction": None, "burn_rate": None}
    observed = bad_windows / evaluated_windows
    return {"observed_bad_fraction": observed, "burn_rate": observed / allowed_fraction}


def _budget(
    *,
    bad_events: float,
    evaluated_events: float,
    allowed_fraction: float,
) -> dict[str, float | None]:
    if evaluated_events <= 0:
        return {
            "bad_events": bad_events,
            "evaluated_events": evaluated_events,
            "allowed_events": None,
            "budget_consumed_fraction": None,
            "budget_remaining_fraction": None,
        }
    allowed_events = evaluated_events * allowed_fraction
    consumed = bad_events / allowed_events if allowed_events else None
    return {
        "bad_events": bad_events,
        "evaluated_events": evaluated_events,
        "allowed_events": allowed_events,
        "budget_consumed_fraction": consumed,
        "budget_remaining_fraction": None if consumed is None else max(0.0, 1.0 - consumed),
    }


def _slo_entry(
    *,
    target: float,
    actual: float | None,
    comparator: str,
    budget: dict[str, float | None],
    applicable: bool = True,
) -> dict[str, Any]:
    if not applicable or actual is None:
        passed: bool | None = None
    elif comparator == ">=":
        passed = actual >= target
    elif comparator == "<=":
        passed = actual <= target
    else:
        raise ValueError(f"Comparador no soportado: {comparator}")
    return {
        "target": target,
        "actual": actual,
        "comparator": comparator,
        "passed": passed,
        "applicable": applicable and actual is not None,
        "error_budget": budget,
    }


def evaluate_slos(
    metrics: dict[str, Any],
    *,
    error_count: int | None = None,
    evaluation_windows: int = 1,
    latency_breach_windows: int | None = None,
    quality_breach_windows: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calcula SLI, cumplimiento SLO y consumo de error budget.

    ``metrics`` debe provenir de ``calculate_window_metrics``. La calidad se
    evalúa únicamente si accuracy y F1 macro están disponibles. Para latencia
    y calidad, el presupuesto representa la fracción permitida de ventanas
    incumplidas dentro de la ventana de evaluación.
    """
    if evaluation_windows <= 0:
        raise ValueError("evaluation_windows debe ser positivo")
    cfg = {**DEFAULT_SLO_CONFIG, **(config or {})}
    volume = int(metrics.get("prediction_volume") or 0)
    error_rate = metrics.get("error_rate")
    availability = None if error_rate is None else 1.0 - float(error_rate)
    if error_count is None:
        error_count_value = 0.0 if error_rate is None else float(error_rate) * volume
    else:
        if error_count < 0 or error_count > volume:
            raise ValueError("error_count debe estar entre cero y el volumen")
        error_count_value = float(error_count)

    latency_breaches = (
        int(latency_breach_windows)
        if latency_breach_windows is not None
        else int(
            metrics.get("latency_p95_ms") is not None
            and float(metrics["latency_p95_ms"]) > float(cfg["latency_p95_target_ms"])
        )
    )
    quality_available = metrics.get("accuracy") is not None and metrics.get("f1_macro") is not None
    quality_breaches = (
        int(quality_breach_windows)
        if quality_breach_windows is not None
        else int(
            quality_available
            and (
                float(metrics["accuracy"]) < float(cfg["accuracy_target"])
                or float(metrics["f1_macro"]) < float(cfg["f1_macro_target"])
            )
        )
    )

    slis = {
        "availability": availability,
        "latency_mean_ms": metrics.get("latency_mean_ms"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "error_rate": error_rate,
        "prediction_volume": volume,
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
    }
    slos = {
        "availability": _slo_entry(
            target=float(cfg["availability_target"]),
            actual=availability,
            comparator=">=",
            budget=_budget(
                bad_events=error_count_value,
                evaluated_events=volume,
                allowed_fraction=1.0 - float(cfg["availability_target"]),
            ),
        ),
        "latency_p95": _slo_entry(
            target=float(cfg["latency_p95_target_ms"]),
            actual=metrics.get("latency_p95_ms"),
            comparator="<=",
            budget=_budget(
                bad_events=latency_breaches,
                evaluated_events=evaluation_windows,
                allowed_fraction=float(cfg["latency_bad_window_fraction"]),
            ),
        ),
        "accuracy": _slo_entry(
            target=float(cfg["accuracy_target"]),
            actual=metrics.get("accuracy"),
            comparator=">=",
            budget=_budget(
                bad_events=quality_breaches,
                evaluated_events=evaluation_windows,
                allowed_fraction=float(cfg["quality_bad_window_fraction"]),
            ),
            applicable=quality_available,
        ),
        "f1_macro": _slo_entry(
            target=float(cfg["f1_macro_target"]),
            actual=metrics.get("f1_macro"),
            comparator=">=",
            budget=_budget(
                bad_events=quality_breaches,
                evaluated_events=evaluation_windows,
                allowed_fraction=float(cfg["quality_bad_window_fraction"]),
            ),
            applicable=quality_available,
        ),
    }
    latency_budget = slos["latency_p95"]["error_budget"]
    burn = calculate_burn_rate(
        bad_windows=latency_breaches,
        evaluated_windows=evaluation_windows,
        allowed_fraction=float(cfg["latency_bad_window_fraction"]),
    )
    burn_config = cfg.get("burn_rate", DEFAULT_SLO_CONFIG["burn_rate"])
    burn["short_window"] = burn_config["short_window"]
    burn["long_window"] = burn_config["long_window"]
    burn["severity"] = (
        "critical" if burn["burn_rate"] is not None and burn["burn_rate"] >= float(burn_config["critical"])
        else "high" if burn["burn_rate"] is not None and burn["burn_rate"] >= float(burn_config["warning"])
        else "none"
    )
    applicable_statuses = [item["passed"] for item in slos.values() if item["applicable"]]
    return {
        "evaluation_window": cfg["evaluation_window"],
        "budget_window": cfg["budget_window"],
        "slis": slis,
        "slos": slos,
        "burn_rate": {"latency_p95": burn, "source_budget": latency_budget},
        "overall_status": "violated" if any(status is False for status in applicable_statuses) else "compliant",
        "quality_status": "available" if quality_available else "unavailable_without_real_labels",
        "methodology": {
            "availability": "1 - error_rate; error budget counts failed predictions",
            "latency": "p95 of each window; budget counts windows above target",
            "quality": "accuracy and F1 macro only with real labels; budget counts failed labelled windows",
            "volume": "informational SLI; no independent SLO",
        },
    }
