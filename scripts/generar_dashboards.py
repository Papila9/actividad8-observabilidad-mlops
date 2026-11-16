"""Genera dashboards estáticos desde evidencias JSON verificables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save(path: Path, title: str, ylabel: str, labels: list[str], values: list[float], *, threshold: float | None = None, threshold_label: str = "SLO", colors: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, values, color=colors or ["#4c78a8"] * len(values))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if threshold is not None:
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label=f"{threshold_label}: {threshold:g}")
        ax.legend()
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_slo(path: Path, windows: dict) -> None:
    labels = ["antes", "durante", "después"]
    latency = [windows[name]["slis"]["latency_p95_ms"] for name in ("before", "during", "after")]
    availability = [windows[name]["slis"]["availability"] * 100 for name in ("before", "during", "after")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, latency, color=["#54a24b", "#e45756", "#54a24b"])
    axes[0].axhline(500, color="#d62728", linestyle="--", label="SLO p95: 500 ms")
    axes[0].set_title("Latencia p95 por ventana")
    axes[0].set_ylabel("Milisegundos")
    axes[0].legend()
    axes[1].bar(labels, availability, color=["#54a24b", "#e45756", "#54a24b"])
    axes[1].axhline(99, color="#d62728", linestyle="--", label="SLO disponibilidad: 99 %")
    axes[1].set_title("Disponibilidad por ventana")
    axes[1].set_ylabel("Porcentaje")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    for axis, values in ((axes[0], latency), (axes[1], availability)):
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.1f}", ha="center", va="bottom")
    fig.suptitle("Figura 2. SLO: ventanas antes, durante y después")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    output = ROOT / "evidencias" / "dashboards"
    output.mkdir(parents=True, exist_ok=True)
    monitoring = load("evidencias/monitoring/exec-62ae1dd1c2bb.json")
    slo = load("evidencias/slo/slo-simulation-0d1f16ed6596.json")
    drift = load("evidencias/drift/drift-simulation-ca954b05123d.json")
    incidents = load("evidencias/incidentes/incidents-simulation-ec32d80abd13.json")
    lifecycle = load("evidencias/modelos/retraining-simulation-295e2bf4242c.json")

    metrics = monitoring["metrics"]
    save(output / "dashboard_operacion.png", "Figura 1. Operación: ejecución smoke_test", "Milisegundos / tasa / volumen", ["media ms", "p95 ms", "error rate", "volumen"], [metrics["latency_mean_ms"], metrics["latency_p95_ms"], metrics["error_rate"], metrics["prediction_volume"]], threshold=500.0, threshold_label="Referencia p95 (ms)", colors=["#4c78a8", "#f58518", "#e45756", "#54a24b"])

    windows = slo["windows"]
    save_slo(output / "dashboard_slo.png", windows)

    drift_data = drift["detection"]["data_drift"]
    save(output / "dashboard_drift_alertas.png", "Figura 3. Drift y alertas: subconjunto de textos largos", "Estadístico", ["JSD tokens", "KS longitudes"], [drift_data["token_jsd"], drift_data["length_ks"]], threshold=drift_data["thresholds"]["high"], threshold_label="umbral alto", colors=["#b279a2", "#f58518"])

    severities: dict[str, int] = {}
    for incident in incidents["incidents"]:
        severities[incident["severity"]] = severities.get(incident["severity"], 0) + 1
    ordered = [name for name in ("medium", "high", "critical") if name in severities]
    save(output / "dashboard_incidentes.png", "Figura 4. Incidentes simulados por severidad", "Número de incidentes", ordered, [float(severities[name]) for name in ordered], colors=["#f2cf5b", "#f58518", "#e45756"][: len(ordered)])

    stable = lifecycle["stable"]["evaluation"]["metrics"]
    candidate = lifecycle["candidate"]["evaluation"]["metrics"]
    save(output / "dashboard_modelos.png", "Figura 5. Modelo estable frente a candidato", "Métrica supervisada", ["estable accuracy", "candidato accuracy", "estable F1", "candidato F1"], [stable["accuracy"], candidate["accuracy"], stable["f1_macro"], candidate["f1_macro"]], threshold=0.70, threshold_label="SLO F1 mínimo", colors=["#4c78a8", "#e45756", "#4c78a8", "#e45756"])

    summary = {"generated_from": {"monitoring": "evidencias/monitoring/exec-62ae1dd1c2bb.json", "slo": "evidencias/slo/slo-simulation-0d1f16ed6596.json", "drift": "evidencias/drift/drift-simulation-ca954b05123d.json", "incidents": "evidencias/incidentes/incidents-simulation-ec32d80abd13.json", "model_lifecycle": "evidencias/modelos/retraining-simulation-295e2bf4242c.json"}, "thresholds": {"latency_p95_ms": 500.0, "availability": 0.99, "f1_macro": 0.70}, "simulation_boundary": "Las ejecuciones SLO, drift, incidentes y ciclo de modelo son simulaciones controladas; no son producción.", "charts": ["dashboard_operacion.png", "dashboard_slo.png", "dashboard_drift_alertas.png", "dashboard_incidentes.png", "dashboard_modelos.png"]}
    (output / "dashboard_data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
