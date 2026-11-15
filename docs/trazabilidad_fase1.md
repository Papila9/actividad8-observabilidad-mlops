# Trazabilidad de la Fase 1

## Alcance

El proyecto nuevo se creó como `actividad8-observabilidad-mlops/`, separado de la carpeta de la Actividad 5. La fuente original no fue modificada.

## Baseline histórico reutilizado

La Regresión Logística de la partición original reportó:

| Métrica | Valor archivado |
|---|---:|
| Accuracy | 0.798155737704918 |
| F1 macro | 0.7457002530158143 |
| ROC-AUC OVR | 0.9110245729780063 |

Estos valores fueron copiados desde los archivos existentes; no son resultados generados durante la Fase 1. La fuente de datos limpia tiene 14,640 filas y SHA-256 `82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd`.

## Insumos reutilizados

- Código de preparación, entrenamiento, auditoría y MLflow en `src/actividad5/`.
- Resultados históricos seleccionados en `evidencias/baseline/`.
- Pruebas heredadas en `tests/actividad5/`.
- Plantilla Word institucional en `docs/plantillas/`.

Los hashes de los archivos fuente originales y copiados se conservarán en el registro de la entrega de esta fase. El dataset original no se copia al proyecto nuevo.

## Dependencias

No se agregaron dependencias. Se conservaron las versiones declaradas por la Actividad 5 en `requirements.txt`. Las dependencias para drift, dashboards o alertas solo se evaluarán cuando exista una necesidad concreta y se documentará su justificación.

## Resultado de la Fase 2

Se añadieron métricas, logs estructurados, trazas JSONL, comparación contra el baseline y registro ampliado en MLflow. La ejecución persistente de validación está marcada como `smoke_test` y no sustituye al baseline.

## Resultado de la Fase 3

Se añadieron baseline de distribución, detección KS/Jensen-Shannon, separación de data drift/class shift/model drift, alertas con severidad, deduplicación y cooldown, simulación controlada y registro en MLflow.

## Pendientes de fases posteriores

Todavía no existen rollback, reentrenamiento, dashboards ni runbooks de respuesta. La formalización SLI/SLO/error budgets queda reservada para la Fase 4.
