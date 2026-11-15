# Cierre de la Fase 3

## Auditoría final de consistencia

El archivo real utilizado para construir el baseline es `datos/datos_limp/Tweets_limpios.csv` en el workspace de la Actividad 5. Su SHA-256 definitivo es:

`82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd`

Este valor coincide con `evidencias/drift/baseline.json`, `config/trazabilidad.json` y `docs/trazabilidad_fase1.md`. La diferencia detectada no correspondía a un hash alternativo del archivo: este cierre no conservaba explícitamente el hash. Se agregó aquí sin modificar la arquitectura ni los artefactos de la Actividad 5.

Los umbrales JSD y KS son criterios operativos iniciales para esta implementación académica. No son umbrales universales ni fueron calibrados con tráfico histórico de producción. KS aplicado a longitudes resume el texto en una sola variable y puede ser sensible al tamaño de la ventana; JSD aplicada a frecuencias depende del tokenizador, del vocabulario retenido, de `__other__` y del tamaño de la ventana. Por ello, una señal de drift no se interpreta como pérdida de precisión sin etiquetas reales.

## Componentes

- `src/deteccion_drift.py`: baseline, KS de longitudes, JSD de tokens y clases, y estados explícitos para model drift.
- `src/alertas.py`: severidades, fingerprint, deduplicación por ventana y cooldown de 300 segundos.
- `scripts/crear_baseline_drift.py`: construcción reproducible del baseline.
- `scripts/simular_incidentes.py`: simulación controlada, evidencias y registro MLflow.
- `config/drift_alerts.json`: umbrales y justificación de severidades.

## Resultado observado

La simulación `drift-simulation-5fd90e98ad50` utilizó 300 textos reales del dataset limpio y añadió dos tokens controlados por texto. El resultado está marcado como `controlled_data_drift_simulation`.

- Token JSD: `0.2085327843`.
- Longitud KS: `0.3767896175`.
- Señal dominante: `0.3767896175`.
- Severidad: `high`.
- Alertas: una `data_drift` de severidad alta.
- Class shift: no evaluado, porque la ventana simulada no proporcionó etiquetas.
- Model drift: no evaluado; data drift no se interpreta como pérdida de precisión.

## MLflow

- Experimento: `Actividad8_Observabilidad_Drift`.
- Run: `193fed723ff94a84b5aebedcd47cb537`.
- Estado: `FINISHED`.
- Artefactos: `drift_simulation_result.json`, `baseline/` y `alerts/`.

## Validación

Diez pruebas unitarias de las Fases 2 y 3 pasaron: métricas, monitor, detección estadística, severidades, deduplicación y cooldown. La compilación Python terminó con código 0 y no se detectaron problemas en los archivos nuevos.

## Límite de la fase

SLI, SLO y error budgets no fueron implementados en esta fase. Quedan pendientes para la Fase 4 junto con dashboards, rollback, reentrenamiento y runbooks.
