# Fase 5 — Incidentes y runbooks

## Clasificación

| Incidente | Señal | ¿Etiquetas necesarias? | Respuesta inicial |
|---|---|---|---|
| Data drift | KS/JSD sobre textos | No para detectar; sí para calidad | Confirmar ventana, baseline y solicitar etiquetas |
| Incumplimiento de SLO | p95, disponibilidad o error rate | No | Separar inferencia local de servicio completo |
| Incremento de errores | `error_rate` | No | Clasificar errores y cuantificar solicitudes afectadas |
| Degradación de calidad | Accuracy/F1 bajo objetivo | Sí | Detener promoción y evaluar candidato |

## Runbook común

Cada incidente se registra con `incident_id`, `execution_id`, `window_id`, severidad, alerta de origen y estado.

1. **Detección:** confirmar la alerta, timestamp, ventana y fingerprint.
2. **Diagnóstico:** revisar volumen, logs, trazas, versión del modelo y configuración.
3. **Impacto:** cuantificar solicitudes, latencia, errores y, solo con etiquetas, calidad.
4. **Respuesta:** aplicar la acción específica sin declarar producción real.
5. **Recuperación:** verificar una ventana posterior dentro del objetivo o evaluación supervisada aceptable.
6. **Cierre:** conservar evidencia, causa, decisión y responsable académico.

## Runbook de data drift

Confirmar KS/JSD frente a [evidencias/drift/baseline.json](../evidencias/drift/baseline.json), revisar tokenizador, duplicados y tamaño de ventana. Data drift no implica model drift ni pérdida de precisión. Si hay etiquetas reales, calcular Accuracy/F1; sin ellas, mantener la calidad como no disponible.

## Runbook de latencia/SLO

Confirmar p95 contra 500 ms y consumo de error budget. Revisar que la medición corresponda a inferencia local y no atribuir el retraso a red o servicio sin instrumentación. Evaluar rollback solo si existe una versión estable registrada.

## Runbook de incremento de errores

Confirmar `error_rate`, volumen y tipos de excepción en los logs estructurados. Separar errores de entrada, inferencia y persistencia. Cuantificar solicitudes afectadas, detener promociones mientras el presupuesto esté agotado y verificar una ventana posterior dentro del objetivo antes de cerrar.

## Runbook de degradación de calidad

Confirmar que cada predicción tenga etiqueta real, que el conjunto esté separado del entrenamiento y que la versión esté identificada. Calcular Accuracy y F1 macro, comparar contra los objetivos y la versión estable, rechazar candidatos que no cumplan criterios y registrar la decisión en MLflow. Sin etiquetas, el incidente queda no evaluable como model drift.

## Evidencia

[scripts/simular_incidentes_fase5.py](../scripts/simular_incidentes_fase5.py) deriva dos incidentes de las evidencias de drift y SLO. El resultado se identifica como simulación controlada y se registra en JSON/JSONL y MLflow.
