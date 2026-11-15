# SLI, SLO y Error Budgets - Fase 4

## Alcance

Esta fase formaliza objetivos operativos académicos para las ventanas de inferencia del clasificador. Los SLO no constituyen un compromiso de producción: no fueron calibrados con tráfico histórico, disponibilidad de infraestructura ni acuerdos con usuarios.

## SLI

| SLI | Fórmula o medición | Etiquetas reales | Uso |
|---|---|---|---|
| Disponibilidad observada | `1 - error_rate` | No | Proxy de solicitudes procesadas correctamente en la ventana |
| Latencia promedio | `sum(latency_ms) / volumen` | No | Tendencia de tiempo de inferencia |
| Latencia p95 | Percentil lineal 95 de latencias | No | Protección contra colas y casos lentos |
| Tasa de errores | `errores / volumen` | No | Fallas técnicas observadas |
| Volumen | Número de predicciones | No | Contexto; no tiene SLO independiente |
| Accuracy | `aciertos / etiquetas` | Sí | Calidad supervisada |
| F1 macro | Promedio macro de F1 | Sí | Calidad supervisada con clases desbalanceadas |

La disponibilidad implementada es un proxy de la ventana de inferencia, no una medición completa de uptime de red, balanceador o infraestructura. Accuracy y F1 macro quedan como `unavailable_without_real_labels` cuando no existen etiquetas reales.

## SLO académicos

| Indicador | Objetivo inicial | Justificación académica |
|---|---:|---|
| Disponibilidad | >= 99 % | Umbral sencillo para demostrar presupuesto de fallos del 1 % |
| Latencia p95 | <= 500 ms | Límite operativo solicitado y verificable en inferencia local |
| Accuracy | >= 0.75 | Conserva un nivel mínimo cercano al baseline histórico sin afirmar equivalencia |
| F1 macro | >= 0.70 | Protege el desempeño agregado entre clases |

La tasa de errores se conserva como SLI derivado de disponibilidad. El volumen es informativo. En un servicio real estos objetivos requerirían tráfico histórico, definición del punto de medición, ventanas de negocio, contratos y revisión de costos.

## Error budgets

La ventana de evaluación de cada evidencia es de `300s`; el presupuesto académico se expresa como una ventana conceptual de `30d_academic` para facilitar la demostración.

- Disponibilidad: presupuesto permitido = `1 - 0.99 = 0.01` de solicitudes fallidas. Consumo = `errores_observados / (volumen * 0.01)`.
- Latencia: presupuesto permitido = `1 %` de ventanas con p95 superior a 500 ms. Consumo = `ventanas_incumplidas / (ventanas_evaluadas * 0.01)`.
- Calidad: presupuesto permitido = `1 %` de ventanas etiquetadas por debajo del objetivo. Consumo = `ventanas_de_calidad_incumplida / (ventanas_etiquetadas * 0.01)`. Sin etiquetas, el SLO no aplica y no consume presupuesto.
- Presupuesto restante = `max(0, 1 - consumo)`.

Un consumo menor que 1 conserva presupuesto; un consumo igual o mayor que 1 lo agota. La implementación marca una alerta `high` ante un incumplimiento que aún no agota presupuesto y `critical` cuando el consumo calculado alcanza o supera 1.

## Integración con alertas y MLflow

`AlertManager.evaluate_slo` reutiliza fingerprints, deduplicación y cooldown, pero usa claves propias para no interferir con las alertas de drift. Los eventos se escriben como `slo_alert` en JSONL. La simulación registra parámetros, métricas, resultado y configuración en el experimento `Actividad8_Observabilidad_SLO`.

## Limitaciones

Los escenarios son simulaciones controladas. La latencia mide el tiempo suministrado a la ventana de inferencia local, no un servicio completo. Un incumplimiento técnico no demuestra model drift. La calidad solo puede evaluarse cuando las etiquetas reales de la ventana estén disponibles.
