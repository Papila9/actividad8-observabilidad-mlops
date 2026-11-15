"""Genera dashboards estáticos desde las evidencias JSON existentes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def latest(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No hay evidencia para {directory / pattern}")
    return files[-1]


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, *, colors: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}" if value < 10 else f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "evidencias" / "dashboards"
    output.mkdir(parents=True, exist_ok=True)
    monitoring = json.loads((root / "evidencias" / "monitoring" / "exec-c5fdb1da11f3.json").read_text(encoding="utf-8"))
    drift = json.loads((root / "evidencias" / "drift" / "drift-simulation-5fd90e98ad50.json").read_text(encoding="utf-8"))
    slo = json.loads(latest(root / "evidencias" / "slo", "*.json").read_text(encoding="utf-8"))
    incidents = json.loads(latest(root / "evidencias" / "incidentes", "*.json").read_text(encoding="utf-8"))
    lifecycle = json.loads(latest(root / "evidencias" / "modelos", "retraining-simulation-*.json").read_text(encoding="utf-8"))

    operational = monitoring["metrics"]
    save_bar(
        output / "dashboard_operacion.png",
        ["media ms", "p95 ms", "error rate", "volumen"],
        [float(operational["latency_mean_ms"]), float(operational["latency_p95_ms"]), float(operational["error_rate"]), float(operational["prediction_volume"])],
        "Monitorización operativa - smoke test",
        "Valor observado",
        colors=["#4c78a8", "#f58518", "#e45756", "#54a24b"],
    )

    compliance = slo["scenarios"]["compliant"]
    violation = slo["scenarios"]["violation"]
    save_bar(
        output / "dashboard_slo.png",
        ["disponibilidad cumple", "disponibilidad incumple", "p95 cumple", "p95 incumple"],
        [compliance["slis"]["availability"], violation["slis"]["availability"], compliance["slis"]["latency_p95_ms"], violation["slis"]["latency_p95_ms"]],
        "Cumplimiento e incumplimiento SLO",
        "Disponibilidad / milisegundos",
        colors=["#54a24b", "#e45756", "#54a24b", "#e45756"],
    )

    drift_data = drift["detection"]["data_drift"]
    save_bar(output / "dashboard_drift_alertas.png", ["JSD tokens", "KS longitudes"], [drift_data["token_jsd"], drift_data["length_ks"]], "Data drift y señal dominante", "Estadístico", colors=["#b279a2", "#f58518"])

    alert_counts: dict[str, int] = {}
    for incident in incidents["incidents"]:
        severity = incident["severity"]
        alert_counts[severity] = alert_counts.get(severity, 0) + 1
    save_bar(output / "dashboard_incidentes.png", list(alert_counts), [float(value) for value in alert_counts.values()], "Incidentes simulados por severidad", "Número de incidentes", colors=["#e45756", "#f58518", "#54a24b"][: len(alert_counts)])

    stable = lifecycle["stable"]["evaluation"]["metrics"]
    candidate = lifecycle["candidate"]["evaluation"]["metrics"]
    save_bar(output / "dashboard_modelos.png", ["estable accuracy", "candidato accuracy", "estable F1", "candidato F1"], [stable["accuracy"], candidate["accuracy"], stable["f1_macro"], candidate["f1_macro"]], "Comparación de modelos", "Métrica supervisada", colors=["#4c78a8", "#e45756", "#4c78a8", "#e45756"])

    summary = {
        "generated_from": {
            "monitoring": str((root / "evidencias" / "monitoring" / "exec-c5fdb1da11f3.json").relative_to(root)),
            "drift": str((root / "evidencias" / "drift" / "drift-simulation-5fd90e98ad50.json").relative_to(root)),
            "slo": str(latest(root / "evidencias" / "slo", "*.json").relative_to(root)),
            "incidents": str(latest(root / "evidencias" / "incidentes", "*.json").relative_to(root)),
            "model_lifecycle": str(latest(root / "evidencias" / "modelos", "retraining-simulation-*.json").relative_to(root)),
        },
        "simulation_boundary": "Los incidentes, escenarios SLO y ciclo de modelo están identificados como simulaciones controladas.",
        "charts": ["dashboard_operacion.png", "dashboard_slo.png", "dashboard_drift_alertas.png", "dashboard_incidentes.png", "dashboard_modelos.png"],
    }
    (output / "dashboard_data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
