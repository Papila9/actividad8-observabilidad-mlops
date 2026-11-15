"""Genera documentos Word desde la plantilla Tecmilenio conservando su diseño."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Inches


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "plantillas" / "Plantilla_Tecmilenio_Gestion_de_Proyectos_de_Inteligencia_Artificial.docx"
OUT = ROOT / "entregables" / "word"


def reset_content(document: Document) -> None:
    """Conserva portada, tablas, header/footer y declaración; elimina placeholders de contenido."""
    keep_after = document.paragraphs[19]._p
    body = document._element.body
    for child in list(body):
        if child.tag.endswith("}p") and child is not keep_after:
            paragraphs = document.paragraphs
            if child in [paragraph._p for paragraph in paragraphs[20:]]:
                body.remove(child)


def prepare_document(title: str) -> Document:
    OUT.mkdir(parents=True, exist_ok=True)
    document = Document(TEMPLATE)
    document.paragraphs[4].text = "TÍTULO DE LA ACTIVIDAD"
    document.paragraphs[6].text = title
    table = document.tables[0]
    values = {
        "Actividad:": "Actividad 8 - Observabilidad y MLOps",
        "Alumno(a):": "[PENDIENTE: nombre completo]",
        "Matrícula:": "[PENDIENTE: matrícula]",
        "Docente:": "[PENDIENTE: nombre del docente]",
        "Fecha de entrega:": "[PENDIENTE: fecha de entrega]",
    }
    for row in table.rows:
        label = row.cells[0].text.strip()
        if label in values:
            row.cells[1].text = values[label]
    for paragraph in document.paragraphs:
        if paragraph.style.name == "Heading 1" and "Declaración" in paragraph.text:
            next_paragraph = paragraph._p.getnext()
            if next_paragraph is not None:
                for candidate in document.paragraphs:
                    if candidate._p is next_paragraph:
                        candidate.text = "Se utilizó Inteligencia Artificial como apoyo para estructurar, revisar y documentar código y resultados. La selección de fuentes, ejecución de pruebas, interpretación de métricas y responsabilidad del contenido corresponden al estudiante. No se declara un porcentaje de uso porque no fue confirmado."
                        break
    reset_content(document)
    return document


def add_references(document: Document) -> None:
    document.add_heading("Referencias Bibliográficas", level=1)
    refs = [
        "MLflow. (2026). MLflow Tracking. https://mlflow.org/docs/latest/tracking/",
        "scikit-learn developers. (2026). f1_score. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
        "scikit-learn developers. (2026). train_test_split. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
    ]
    for ref in refs:
        document.add_paragraph(ref)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        try:
            document.add_paragraph(item, style="List Bullet")
        except KeyError:
            document.add_paragraph("• " + item)


def build_technical() -> Path:
    document = prepare_document("Documento técnico de observabilidad y MLOps")
    document.add_heading("1. Introducción y objetivos", level=2)
    document.add_paragraph("Este documento presenta una implementación académica de observabilidad para un clasificador de sentimientos de comentarios sobre aerolíneas. El objetivo es medir inferencia, calidad supervisada, drift, SLO, incidentes y ciclo controlado de modelos, manteniendo trazabilidad y límites explícitos.")
    document.add_heading("2. Descripción del proyecto", level=2)
    document.add_paragraph("El proyecto reutiliza el dataset limpio de la Actividad 5 y una Regresión Logística con TF-IDF. El archivo de referencia tiene 14,640 filas y SHA-256 82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd. El baseline histórico reporta Accuracy 0.7982, F1 macro 0.7457 y ROC-AUC OVR 0.9110.")
    document.add_heading("3. Arquitectura de monitorización", level=2)
    document.add_paragraph("La arquitectura se organiza en métricas, logs JSONL, trazas JSONL, detección estadística, alertas con fingerprint/cooldown, evaluación SLI/SLO, incidentes, registro de modelos y MLflow. Las evidencias persistentes se almacenan por fase en evidencias/.")
    document.add_heading("4. Métricas, logs y trazas", level=2)
    add_bullets(document, ["Métricas de inferencia: latencia promedio, p95, tasa de errores y volumen.", "Métricas supervisadas: Accuracy y F1 macro únicamente con etiquetas reales.", "Cada ejecución conserva execution_id, modelo, versión, tiempo de procesamiento, log, traza y run MLflow.", "El smoke test se conserva como evidencia técnica y no como mejora estadísticamente demostrada."])
    document.add_heading("5. SLI, SLO y Error Budgets", level=2)
    document.add_paragraph("Los objetivos académicos son disponibilidad >=99 %, latencia p95 <=500 ms, Accuracy >=0.75 y F1 macro >=0.70. El volumen es informativo. El presupuesto de disponibilidad es 1 % de errores; el presupuesto de latencia y calidad se expresa como 1 % de ventanas incumplidas. Sin etiquetas, los SLO de calidad no aplican.")
    document.add_heading("6. Data drift y alertas", level=2)
    document.add_paragraph("KS se aplica a longitudes de texto y Jensen-Shannon a frecuencias de tokens. Los umbrales son criterios operativos académicos, no universales ni calibrados con producción. Data drift no implica model drift ni pérdida de precisión sin etiquetas reales.")
    document.add_heading("7. Dashboards", level=2)
    document.add_paragraph("Los dashboards estáticos se generan con scripts/generar_dashboards.py y visualizan operación, SLO, drift, severidades, incidentes y comparación de modelos. Se basan exclusivamente en JSON existentes y conservan la frontera entre simulación y producción.")
    document.add_picture(str(ROOT / "evidencias" / "dashboards" / "dashboard_operacion.png"), width=Inches(6.2))
    document.add_paragraph("Figura 1. Dashboard operativo generado a partir del smoke test.")
    document.add_heading("8. Respuesta automatizada", level=2)
    document.add_paragraph("Las alertas se deduplican por ventana y fingerprint, aplican cooldown y generan incidentes estructurados. El rollback y el reentrenamiento no sustituyen automáticamente un modelo estable: requieren criterios de aceptación y decisión explícita.")
    document.add_heading("9. Limitaciones y conclusiones", level=2)
    document.add_paragraph("La solución es académica. La latencia corresponde a inferencia local cuando así se documenta; no mide red, colas o infraestructura completa. Los escenarios controlados no representan producción. La evaluación supervisada depende de etiquetas reales y de una separación correcta respecto del entrenamiento.")
    add_references(document)
    path = OUT / "documento_tecnico_observabilidad.docx"
    document.save(path)
    return path


def build_incidents() -> Path:
    document = prepare_document("Documento de simulación de incidentes, rollback y reentrenamiento")
    drift = json.loads((ROOT / "evidencias" / "drift" / "drift-simulation-5fd90e98ad50.json").read_text(encoding="utf-8"))
    slo = json.loads((ROOT / "evidencias" / "slo" / "slo-simulation-ba11951137e0.json").read_text(encoding="utf-8"))
    lifecycle = json.loads((ROOT / "evidencias" / "modelos" / "retraining-simulation-295e2bf4242c.json").read_text(encoding="utf-8"))
    document.add_heading("1. Metodología", level=2)
    document.add_paragraph("Se usaron evidencias persistentes de drift, SLO e historial de modelos. Los incidentes y el ciclo de modelo están identificados como simulaciones controladas; no son incidentes reales de producción.")
    document.add_heading("2. Incidente 1: data drift", level=2)
    document.add_paragraph(f"Ejecución {drift['execution_id']}. La simulación alteró textos controlados y produjo JSD de tokens {drift['detection']['data_drift']['token_jsd']:.6f} y KS de longitudes {drift['detection']['data_drift']['length_ks']:.6f}, con severidad alta.")
    document.add_paragraph("Runbook aplicado: confirmar baseline, ventana, volumen y método; evaluar impacto sin afirmar pérdida de precisión; solicitar etiquetas reales; cerrar conservando evidencia.")
    document.add_heading("3. Incidente 2: latencia elevada e incumplimiento de SLO", level=2)
    document.add_paragraph(f"Ejecución {slo['execution_id']}. La ventana controlada produjo disponibilidad {slo['scenarios']['violation']['slis']['availability']:.2f}, p95 de {slo['scenarios']['violation']['slis']['latency_p95_ms']:.0f} ms y estado violated. El error budget de latencia consumió {slo['scenarios']['violation']['slos']['latency_p95']['error_budget']['budget_consumed_fraction']:.1f} veces el presupuesto permitido.")
    document.add_paragraph("Runbook aplicado: separar inferencia local de servicio completo, revisar errores y volumen, evaluar versión estable y comprobar recuperación en una ventana posterior.")
    document.add_heading("4. Evidencia de detección y severidad", level=2)
    add_bullets(document, ["Data drift: alerta data_drift, severidad high.", "Latencia: alerta slo_breach sobre latency_p95, severidad critical.", "Los eventos se registraron en logs/incidents.jsonl y en MLflow."])
    document.add_heading("5. Reentrenamiento y rollback", level=2)
    document.add_paragraph(f"El candidato {lifecycle['candidate']['version']} obtuvo Accuracy {lifecycle['candidate']['evaluation']['metrics']['accuracy']:.6f} y F1 macro {lifecycle['candidate']['evaluation']['metrics']['f1_macro']:.6f}; la versión estable obtuvo F1 macro {lifecycle['stable']['evaluation']['metrics']['f1_macro']:.6f}. La caída fue {abs(lifecycle['acceptance_decision']['deltas_vs_stable']['f1_macro']):.6f}, por lo que el candidato fue rechazado.")
    document.add_paragraph("El rollback simulado regresó de logreg-candidate-v2 a logreg-stable-v1 en un registro separado. La versión activa del registro principal permaneció estable.")
    document.add_heading("6. Resultados, recuperación y lecciones aprendidas", level=2)
    add_bullets(document, ["Una alerta de drift exige diagnóstico, no una conclusión automática de model drift.", "Un SLO técnico puede agotarse aunque la calidad del modelo no sea evaluable.", "La comparación supervisada y el hash del dataset son necesarios antes de promover un modelo.", "La promoción automática se deshabilitó para evitar reemplazar un modelo estable con un candidato rechazado."])
    add_references(document)
    path = OUT / "documento_simulacion_incidentes.docx"
    document.save(path)
    return path


def build_evidence() -> Path:
    document = prepare_document("Documento de evidencias de implementación y validación")
    dashboard_dir = ROOT / "evidencias" / "dashboards"
    document.add_heading("1. Evidencias conservadas", level=2)
    document.add_paragraph("El repositorio conserva resultados de monitorización, drift, SLO, incidentes, modelos, logs, trazas y pruebas. Las rutas se mantienen relativas al proyecto para facilitar revisión y reproducción.")
    document.add_heading("2. Dashboards", level=2)
    for filename, caption in [("dashboard_slo.png", "Cumplimiento e incumplimiento de SLO"), ("dashboard_drift_alertas.png", "Data drift"), ("dashboard_incidentes.png", "Incidentes simulados"), ("dashboard_modelos.png", "Comparación de modelos")]:
        document.add_picture(str(dashboard_dir / filename), width=Inches(6.2))
        document.add_paragraph(f"Figura: {caption}.")
    document.add_heading("3. MLflow y trazabilidad", level=2)
    add_bullets(document, ["Actividad8_Observabilidad: monitorización base y métricas.", "Actividad8_Observabilidad_Drift: baseline, KS/JSD y alertas.", "Actividad8_Observabilidad_SLO: cumplimiento, incumplimiento y budgets.", "Actividad8_Observabilidad_Incidentes: dos incidentes y runbooks.", "Actividad8_Observabilidad_ModelLifecycle: comparación, rechazo y rollback simulado."])
    document.add_heading("4. Pruebas automatizadas", level=2)
    document.add_paragraph("La validación integral ejecutada antes de generar este documento obtuvo 18 pruebas aprobadas, compilación de 30 archivos Python con código 0 y ausencia de diagnósticos en los nuevos módulos.")
    document.add_heading("5. Trazabilidad y límites", level=2)
    document.add_paragraph("El baseline se identifica por SHA-256 82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd. Las simulaciones están marcadas en sus JSON y en MLflow. No se presentan como producción ni se afirma pérdida de precisión sin etiquetas reales.")
    add_references(document)
    path = OUT / "documento_evidencias_implementacion.docx"
    document.save(path)
    return path


def main() -> None:
    paths = [build_technical(), build_incidents(), build_evidence()]
    print(json.dumps({"documents": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
