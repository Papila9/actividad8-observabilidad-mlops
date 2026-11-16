"""Genera el diagrama estático de la arquitectura académica."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "evidencias" / "arquitectura_actividad8.png"
    labels = [
        "Dataset\nreferencia",
        "Preprocesamiento\nTF-IDF",
        "Modelo\nLogReg",
        "Predicciones\nlocales",
        "Monitorización\nSLI y métricas",
        "Logs / trazas\nJSONL + MLflow",
        "Drift / SLO\n/anomalías",
        "Alertas\nseveridad",
        "Runbooks\nincidentes",
        "Reentrenamiento\no rollback",
        "Verificación\nde recuperación",
    ]
    fig, axis = plt.subplots(figsize=(18, 4.5))
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylim(-1, 1)
    axis.axis("off")
    for index, label in enumerate(labels):
        color = "#d9eaf7" if index < 6 else "#fce8b2"
        if index in (8, 9, 10):
            color = "#d9ead3"
        patch = FancyBboxPatch((index - 0.42, -0.35), 0.84, 0.7, boxstyle="round,pad=0.03", facecolor=color, edgecolor="#355c7d", linewidth=1.2)
        axis.add_patch(patch)
        axis.text(index, 0, label, ha="center", va="center", fontsize=8)
        if index < len(labels) - 1:
            axis.annotate("", xy=(index + 0.48, 0), xytext=(index + 0.42, 0), arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.2})
    axis.text(5.5, 0.72, "MLflow: experimentos, parámetros, métricas y artefactos", ha="center", va="center", fontsize=10, color="#7f3c8d", bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f3e5f5", "edgecolor": "#7f3c8d"})
    axis.annotate("", xy=(5.5, 0.38), xytext=(5.5, 0.62), arrowprops={"arrowstyle": "->", "color": "#7f3c8d", "lw": 1.1})
    fig.suptitle("Figura 6. Arquitectura de observabilidad y MLOps (simulaciones controladas)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
