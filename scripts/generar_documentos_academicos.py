"""Genera los tres documentos académicos desde la plantilla Tecmilenio."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.shared import Inches

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "plantillas" / "Plantilla_Tecmilenio_Gestion_de_Proyectos_de_Inteligencia_Artificial.docx"
OUT = ROOT / "entregables" / "word"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def first_timestamp(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        return "No disponible en el registro"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            return str(value.get("timestamp", "No disponible en el registro"))
    return "No disponible en el registro"


def prepare_document(title: str) -> Document:
    OUT.mkdir(parents=True, exist_ok=True)
    document = Document(TEMPLATE)
    document.paragraphs[4].text = "TÍTULO DE LA ACTIVIDAD"
    document.paragraphs[6].text = title
    academic = document.tables[0]
    values = {
        "Materia:": "Gestión de Proyectos de Inteligencia Artificial",
        "Actividad:": "Actividad 8 – Observabilidad y MLOps",
        "Alumno(a):": "Paola Quinto Martínez",
        "Matrícula:": "03039650",
        "Docente:": "Luis Ariel Vázquez Piña",
        "Fecha de entrega:": "15 de noviembre de 2026",
    }
    for row in academic.rows:
        label = row.cells[0].text.strip()
        for key, value in values.items():
            if label.startswith(key):
                row.cells[1].text = value

    ai = document.tables[1]
    ai_rows = [
        ("ChatGPT / OpenAI", "☒ Sí    ☐ No", "Análisis, redacción, revisión y orientación."),
        ("Codex", "☒ Sí    ☐ No", "Programación, depuración, pruebas y documentación."),
        ("Claude", "☒ Sí    ☐ No", "Revisión crítica del repositorio y documentos."),
        ("Gemini", "☐ Sí    ☒ No", "No utilizado."),
        ("GitHub Copilot", "☐ Sí    ☒ No", "No utilizado."),
    ]
    for row, values_row in zip(ai.rows[1:], ai_rows):
        row.cells[0].text, row.cells[1].text, row.cells[2].text = values_row

    marker = next(p._p for p in document.paragraphs if p.text.strip() == "Desarrollo de Contenido")
    body = document._element.body
    for child in list(body):
        if child is marker:
            continue
        if child.tag.endswith("}sectPr"):
            continue
        if marker is not None and list(body).index(child) > list(body).index(marker):
            body.remove(child)
    return document


def add_references(document: Document) -> None:
    document.add_heading("Referencias bibliográficas", level=1)
    refs = [
        "MLflow. (s. f.). MLflow Tracking. https://mlflow.org/docs/latest/tracking/",
        "Scikit-learn developers. (s. f.). f1_score. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
        "Scikit-learn developers. (s. f.). train_test_split. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
    ]
    for reference in refs:
        document.add_paragraph(reference)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        try:
            document.add_paragraph(item, style="List Bullet")
        except KeyError:
            document.add_paragraph("• " + item)


def add_traceability_table(document: Document) -> None:
    rows = [
        ["SLI", "SLO", "Objetivo de negocio", "Condición de alerta", "Severidad", "Runbook", "Acción ejecutada", "Evidencia"],
        ["Latencia p95", "≤ 500 ms", "Respuesta oportuna", "p95 > 500 ms en ventana", "Crítica", "Latencia/SLO", "Escalamiento y rollback controlado", "evidencias/slo/slo-simulation-0d1f16ed6596.json"],
        ["Disponibilidad", "≥ 99 %", "Servicio disponible", "Disponibilidad < 99 %", "Crítica", "Disponibilidad", "Alerta verificada; recuperación controlada", "evidencias/slo/slo-simulation-ba11951137e0.json"],
        ["Tasa de errores", "Diagnóstico; sin SLO independiente", "Identificar fallas", "error_rate > 0", "Media/alta según impacto", "Errores", "Registro y diagnóstico", "evidencias/monitoring/exec-62ae1dd1c2bb.json"],
        ["Accuracy/F1 macro", "Accuracy ≥ 0.75; F1 ≥ 0.70", "Calidad de clasificación", "Métrica supervisada bajo objetivo", "Alta/crítica", "Calidad/model drift", "Candidato rechazado; estable conservado", "evidencias/modelos/retraining-simulation-295e2bf4242c.json"],
        ["Data drift JSD/KS", "Diagnóstico; no es SLO de calidad", "Detectar cambio de entradas", "Señal supera umbral calibrado", "Media/alta", "Data drift", "Reentrenamiento y evaluación supervisada", "evidencias/drift/drift-simulation-ca954b05123d.json"],
    ]
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value


def build_technical() -> Path:
    document = prepare_document("Documento técnico de observabilidad y MLOps")
    document.add_heading("1. Introducción y objetivos", level=2)
    document.add_paragraph("Este documento describe una implementación académica de observabilidad para un clasificador de sentimientos de comentarios sobre aerolíneas. El objetivo es medir inferencia, calidad supervisada, data drift, SLI/SLO, incidentes y ciclo de vida del modelo con trazabilidad verificable.")
    document.add_heading("2. Descripción del proyecto y modelo", level=2)
    document.add_paragraph("Se reutiliza el dataset Twitter US Airline Sentiment y una Regresión Logística con TF-IDF. El archivo limpio tiene 14,640 filas y SHA-256 82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd. El baseline histórico reporta accuracy 0.7982, F1 macro 0.7457 y ROC-AUC OVR 0.9110.")
    document.add_paragraph("La evaluación histórica tuvo 71 textos compartidos entre train y test. La auditoría agrupada independiente reporta solapamiento 0. La simulación con accuracy 0.9335 usa el modelo estable sobre todo el dataset limpio y, por tanto, no es comparable ni sustituye al baseline histórico.")
    document.add_heading("3. Desarrollo de las siete fases", level=2)
    add_bullets(document, [
        "Fase 1: auditoría, trazabilidad del proyecto base y conservación del baseline.",
        "Fase 2: métricas de inferencia y supervisadas, logs, trazas y MLflow.",
        "Fase 3: baseline de distribución, JSD/KS, alertas, deduplicación y cooldown.",
        "Fase 4: SLI, SLO, ventanas, Error Budgets y burn rate.",
        "Fase 5: clasificación de incidentes y runbooks ejecutables.",
        "Fase 6: versionamiento, comparación, rechazo de candidatos y rollback con SHA-256.",
        "Fase 7: dashboards, evidencias académicas y validación integral.",
    ])
    document.add_heading("4. Arquitectura de observabilidad", level=2)
    document.add_paragraph("El flujo es Dataset → Preprocesamiento → Modelo → Predicciones → Monitorización → Métricas/Logs/Trazas → Detección de anomalías → Alertas → Runbooks → Reentrenamiento o rollback → Verificación de recuperación. MLflow registra parámetros, métricas y artefactos. Las ejecuciones son por lotes y controladas; no se representa un servicio productivo.")
    architecture = ROOT / "evidencias" / "arquitectura_actividad8.png"
    if architecture.exists():
        document.add_picture(str(architecture), width=Inches(6.2))
        document.add_paragraph("Figura 6. Arquitectura académica de observabilidad y MLOps.")
    document.add_heading("5. Métricas, logs, trazas y MLflow", level=2)
    add_bullets(document, [
        "Inferencia: latencia promedio, latencia p95, tasa de errores y volumen de predicciones.",
        "Supervisión: accuracy y F1 macro únicamente cuando existen etiquetas reales.",
        "Trazabilidad: execution_id, versión del modelo, trace_id, span_id, duración de vectorización y predicción.",
        "La ejecución exec-62ae1dd1c2bb registró media 1.4166 ms, p95 2.2355 ms, error rate 0, volumen 50, accuracy 0.84 y F1 macro 0.7771.",
    ])
    document.add_heading("6. SLI, SLO y Error Budget", level=2)
    document.add_paragraph("Los objetivos académicos son disponibilidad ≥99 %, latencia p95 ≤500 ms, accuracy ≥0.75 y F1 macro ≥0.70. Error Budget se calcula como fracción permitida de eventos incumplidos; la ventana de evaluación usada es 300 segundos y la ventana conceptual de presupuesto es 30d_academic. No son compromisos de producción.")
    add_traceability_table(document)
    document.add_heading("7. Data drift, calibración y alertas", level=2)
    document.add_paragraph("KS se aplica al estadístico de longitudes y Jensen–Shannon a frecuencias de tokens usando el vocabulario del baseline. La calibración usa 20 ventanas aleatorias del mismo dataset, semilla 42: JSD p95 0.0222648, JSD p99 0.0226836, KS p95 0.0845676 y KS p99 0.0890010. Estos valores son criterios operativos iniciales, no universales ni calibrados con producción. Data drift no implica model drift ni pérdida de precisión sin etiquetas.")
    document.add_paragraph("La simulación de textos largos reportó JSD 0.1566179, KS 0.98 y severidad alta. Está documentada en evidencias/drift/drift-simulation-ca954b05123d.json.")
    document.add_heading("8. Respuesta automatizada y ciclo de modelos", level=2)
    document.add_paragraph("Ante data drift se ejecuta reentrenamiento y evaluación. El candidato obtuvo accuracy 0.7739071 y F1 macro 0.725802 frente a la versión estable con accuracy 0.7981557 y F1 macro 0.7457003; fue rechazado por la caída de F1. Ante latencia o disponibilidad se ejecuta escalamiento y rollback, verificando SHA-256 y carga del artefacto. No se promueve automáticamente un candidato rechazado.")
    document.add_heading("9. Limitaciones", level=2)
    add_bullets(document, ["Las ejecuciones son simulaciones académicas por lotes, no producción ni tiempo real.", "La latencia medida es inferencia local; no incluye red, colas, I/O o infraestructura completa.", "Los umbrales de drift requieren recalibración con tráfico histórico representativo.", "La calidad del modelo requiere etiquetas reales y un protocolo de evaluación sin fuga de información."])
    document.add_heading("10. Conclusiones", level=2)
    document.add_paragraph("Se logró una cadena reproducible de observabilidad que conecta métricas, trazas, drift, alertas, SLO, incidentes y ciclo de vida de modelos. Las evidencias demuestran que el sistema calcula latencias desde predicciones reales, detecta un cambio controlado de distribución, deduplica alertas, rechaza un candidato inferior y valida un rollback mediante hash y carga del modelo. Para operar en producción sería necesario desplegar instrumentación de servicio, almacenar ventanas históricas representativas, definir responsables y políticas de cambio, calibrar SLO con datos reales y agregar monitoreo continuo de disponibilidad y recuperación.")
    add_references(document)
    path = OUT / "actividad8_tecnico.docx"
    document.save(path)
    return path


def build_incidents() -> Path:
    document = prepare_document("Documento de simulación de incidentes, runbooks y recuperación")
    drift = load("evidencias/drift/drift-simulation-ca954b05123d.json")
    slo = load("evidencias/slo/slo-simulation-0d1f16ed6596.json")
    availability = load("evidencias/slo/slo-simulation-ba11951137e0.json")
    lifecycle = load("evidencias/modelos/retraining-simulation-295e2bf4242c.json")
    model_drift = load("evidencias/drift/model-drift-simulation.json")
    responses = [json.loads(line) for line in (ROOT / "logs/alert_responses_second_tanda.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    document.add_heading("1. Metodología", level=2)
    document.add_paragraph("Se analizaron evidencias JSON/JSONL y runs de MLflow generados por simulaciones controladas. No se presentan como incidentes de producción. Cada incidente inicia abierto y solo se cierra después de la acción y verificación registradas.")
    document.add_heading("2. Incidente de data drift", level=2)
    document.add_paragraph(f"Ejecución {drift['execution_id']}; detección en ventana during. El subconjunto controlado de textos largos produjo JSD {drift['detection']['data_drift']['token_jsd']:.7f}, KS {drift['detection']['data_drift']['length_ks']:.2f} y severidad {drift['detection']['data_drift']['severity']}. No se afirma pérdida de precisión porque esta detección no contiene etiquetas reales.")
    document.add_paragraph("Runbook: confirmar baseline y ventana, revisar volumen y método, evaluar impacto sin inferir model drift, solicitar etiquetas y ejecutar reentrenamiento controlado. La respuesta registró estado inicial open y cierre controlado después de la evaluación.")
    document.add_heading("3. Incidente de latencia y disponibilidad", level=2)
    document.add_paragraph(f"La ejecución {slo['execution_id']} midió predicciones reales. En la ventana during se obtuvo p95 {slo['windows']['during']['slis']['latency_p95_ms']:.3f} ms frente al SLO de 500 ms; la ventana after obtuvo p95 {slo['windows']['after']['slis']['latency_p95_ms']:.3f} ms. La recuperación de latencia queda demostrada por la medición posterior, no solo por cargar el archivo.")
    document.add_paragraph(f"La evidencia histórica controlada de disponibilidad reporta {availability['scenarios']['violation']['slis']['availability']:.2f}, inferior al SLO de 0.99, en {availability['execution_id']}. Este dato se conserva separado de la ejecución de latencia.")
    document.add_paragraph("La respuesta automatizada ejecutó escalamiento y rollback a logreg-stable-v1; el SHA-256 del artefacto fue verificado y el modelo cargó correctamente. La medición posterior específica de recuperación está en la ejecución rollback_recovery_measurement, con p95 2.6838 ms, error rate 0 y 50 predicciones.")
    document.add_heading("4. Model drift supervisado simulado", level=2)
    document.add_paragraph(f"La simulación {model_drift['execution_id']} utilizó etiquetas reales y alteró predicciones de forma controlada. Reportó accuracy {model_drift['baseline']['accuracy']:.7f} antes de la alteración y {model_drift['degraded_simulated']['accuracy']:.7f} después. Es una simulación explícita, no una medición productiva.")
    document.add_heading("5. Reentrenamiento, rechazo y rollback", level=2)
    document.add_paragraph(f"El candidato tuvo accuracy {lifecycle['candidate']['evaluation']['metrics']['accuracy']:.7f} y F1 macro {lifecycle['candidate']['evaluation']['metrics']['f1_macro']:.7f}; la versión estable tuvo accuracy {lifecycle['stable']['evaluation']['metrics']['accuracy']:.7f} y F1 macro {lifecycle['stable']['evaluation']['metrics']['f1_macro']:.7f}. La decisión fue rechazar el candidato por caída de F1; el estable no fue reemplazado.")
    document.add_paragraph("El registro de respuestas conserva detección, clasificación, acción correctiva, verificación y cierre. Las fechas y horas se encuentran en logs/alert_responses_second_tanda.jsonl y logs/incidents_second_tanda.jsonl.")
    document.add_heading("6. Resultados y lecciones aprendidas", level=2)
    add_bullets(document, ["El drift se debe diagnosticar antes de atribuir pérdida de calidad.", "El incumplimiento de latencia se comprobó con mediciones reales y una ventana posterior.", "La disponibilidad y la latencia provienen de ejecuciones diferentes y no se mezclan.", "Un candidato que empeora la métrica estable debe rechazarse aunque cumpla mínimos absolutos."])
    add_references(document)
    path = OUT / "actividad8_incidentes.docx"
    document.save(path)
    return path


def build_evidence() -> Path:
    document = prepare_document("Documento de evidencias de implementación y validación")
    document.add_heading("1. Índice de evidencias", level=2)
    rows = [
        ["Fase", "Archivo o figura", "Resultado demostrado", "Ruta relativa"],
        ["1", "Baseline", "SHA-256 y distribución de referencia", "evidencias/drift/baseline.json"],
        ["2", "Smoke test", "Métricas, logs, trazas y MLflow", "evidencias/monitoring/exec-62ae1dd1c2bb.json"],
        ["3", "Calibración", "20 ventanas y umbrales empíricos", "evidencias/drift/calibracion_20_ventanas.json"],
        ["3", "Drift realista simulado", "Subconjunto de textos largos con alerta alta", "evidencias/drift/drift-simulation-ca954b05123d.json"],
        ["4", "SLO", "Ventanas antes/durante/después y Error Budget", "evidencias/slo/slo-simulation-0d1f16ed6596.json"],
        ["5", "Incidentes", "Runbooks, estados y acciones", "evidencias/incidentes/incidents-simulation-ec32d80abd13.json"],
        ["6", "Modelo", "Candidato rechazado y rollback", "evidencias/modelos/retraining-simulation-295e2bf4242c.json"],
        ["7", "Dashboards", "Cinco visualizaciones verificables", "evidencias/dashboards/"],
        ["7", "Arquitectura", "Flujo completo de observabilidad", "evidencias/arquitectura_actividad8.png"],
        ["7", "Pruebas", "37 aprobadas, 0 fallidas", "tests/"],
    ]
    table = document.add_table(rows=len(rows), cols=4)
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    document.add_heading("2. Dashboards", level=2)
    for filename, caption in [("dashboard_operacion.png", "Operación"), ("dashboard_slo.png", "SLO y ventanas"), ("dashboard_drift_alertas.png", "Drift y alertas"), ("dashboard_incidentes.png", "Incidentes"), ("dashboard_modelos.png", "Modelos")]:
        image = ROOT / "evidencias" / "dashboards" / filename
        document.add_picture(str(image), width=Inches(6.2))
        document.add_paragraph(f"{caption}: {filename}. La figura conserva unidades y umbrales del resultado fuente.")
    document.add_heading("3. MLflow y trazabilidad", level=2)
    add_bullets(document, [
        "Monitorización: run 8215195010b94883b2b734c3d9cd7d56.",
        "Drift realista: run 362b5523d02c4cec8faf0c4633630963.",
        "SLO: run 1c05aa318a0549bf98d881bfae4afe91.",
        "Incidentes: run 0e69edec0ed24a669108561f269f2a52.",
        "Model drift: run 2a595a9bf6264b1dab31260cba717b94.",
        "La interfaz web local de MLflow no se capturó; los metadatos, métricas y artefactos son consultables en mlruns y en los JSON de evidencia.",
    ])
    document.add_heading("4. Validación", level=2)
    document.add_paragraph("La validación ejecutada obtuvo 37 pruebas aprobadas y 0 fallidas, compilación correcta de src/scripts/tests, 0 JSON/JSONL inválidos, SHA-256 coincidente entre el dataset real y el baseline, y git diff --check sin errores.")
    document.add_heading("5. Declaración y límites", level=2)
    document.add_paragraph("ChatGPT: sí, análisis, redacción, revisión y orientación. Codex: sí, programación, depuración, pruebas y documentación. Claude: sí, revisión crítica del repositorio y documentos. Gemini: no. GitHub Copilot: no. No se declara porcentaje de uso de IA porque no existe una base verificable.")
    add_references(document)
    path = OUT / "actividad8_evidencias.docx"
    document.save(path)
    return path


def main() -> None:
    paths = [build_technical(), build_incidents(), build_evidence()]
    print(json.dumps({"documents": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
