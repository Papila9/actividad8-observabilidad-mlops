# Índice de evidencias

Todas las simulaciones están identificadas como tales y no representan producción.

| Evidencia | Ubicación |
|---|---|
| Baseline histórico | `baseline/` |
| Monitorización y smoke test | `monitoring/` y `../logs/exec-c5fdb1da11f3.jsonl` |
| Drift y alertas | `drift/` y `../logs/drift_alerts.jsonl` |
| SLO y error budgets | `slo/` y `../logs/slo_alerts.jsonl` |
| Incidentes y runbooks | `incidentes/` y `../logs/incidents.jsonl` |
| Modelos, comparación y rollback | `modelos/` |
| Dashboards estáticos | `dashboards/` |

Los runs y artefactos de MLflow se encuentran en `../mlruns/`. El índice no sustituye los JSON originales: cada resultado conserva su `execution_id`, hashes, parámetros o `mlflow_run_id` cuando aplica.
