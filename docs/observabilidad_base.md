# Observabilidad base — Fase 2

## Métricas

Las métricas de inferencia se calculan en cada ventana sin etiquetas reales:

- `latency_mean_ms`: latencia promedio.
- `latency_p95_ms`: percentil 95 de latencia.
- `error_rate`: errores divididos entre el volumen de predicciones.
- `prediction_volume`: número de predicciones registradas.

Las métricas supervisadas requieren una etiqueta real asociada a la predicción:

- `accuracy`.
- `f1_macro`.

Cuando no existen etiquetas, las métricas supervisadas se registran como `null` y no se comparan contra el baseline. Esto evita presentar una estimación como resultado observado.

## Trazabilidad

Cada ventana tiene un `execution_id`, nombre del modelo, versión del modelo, archivos JSONL de log y traza, resultado JSON y `mlflow_run_id`.

## Hashes y separación de datos

Se conservan dos archivos de datos distintos y no deben confundirse:

| Uso | Archivo | SHA-256 |
|---|---|---|
| Baseline histórico de Fase 1 | `../datos/datos_limp/Tweets_limpios.csv`, documentado en `evidencias/baseline/metricas_experimentos.json` | `82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd` |
| Smoke test de Fase 2 | `../datos/datos_ini/Tweets.csv`, documentado en `evidencias/monitoring/exec-c5fdb1da11f3.json` | `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e` |

El baseline usa el CSV limpio de la Actividad 5. El smoke test se ejecutó sobre el CSV original disponible localmente, cuyo hash es diferente porque corresponde a otra representación del dataset. El resultado conserva ambos registros y no reemplaza el baseline.

El procedimiento del smoke test fue:

1. Cargar el CSV original completo.
2. Separar con `train_test_split(test_size=0.2, random_state=42, stratify=labels)`.
3. Entrenar con 11,712 filas.
4. Mantener 2,928 filas fuera del entrenamiento.
5. Evaluar las primeras 300 filas de ese conjunto de prueba.

Por tanto, las 300 observaciones evaluadas no participaron en el ajuste del modelo. La separación es verificable en [scripts/smoke_test_monitor.py](../scripts/smoke_test_monitor.py) y en el resultado JSON, que registra `train_rows: 11712` y `sample_size: 300`.

## Medición de latencia

Para cada una de las 300 observaciones se midió:

```python
start = time.perf_counter()
predicted = model.predict([text])[0]
latency_ms = (time.perf_counter() - start) * 1000
```

La medición incluye la transformación TF-IDF de esa observación y la predicción local del clasificador. Excluye entrenamiento, lectura del archivo, escritura de logs, registro en MLflow, red, serialización HTTP, colas y cualquier infraestructura de servicio. Por ello representa inferencia local del pipeline, no latencia p95 de un servicio completo.

El promedio es la media aritmética de las 300 latencias. El p95 usa interpolación lineal sobre las latencias ordenadas, implementada en `src/metricas.py`.

Los valores observados fueron 3.468952 ms de promedio y 9.572335 ms de p95. Son evidencia técnica de esta ejecución local específica, no un SLO ni una garantía de producción.

## Interpretación del smoke test

El smoke test obtuvo accuracy 0.81 y F1 macro 0.769425 sobre 300 etiquetas reales. La diferencia frente al baseline se conserva únicamente como comparación descriptiva. No se realizó una prueba estadística, intervalo de confianza ni evaluación independiente suficiente para afirmar una mejora generalizable.

## MLflow

Cada ejecución registra:

- Parámetros: modelo, versión y volumen.
- Métricas numéricas disponibles.
- Tags de ejecución y fase.
- Artefacto `monitoring_result.json`.
- Log estructurado y traza JSONL.

La URI de tracking se proporciona explícitamente para que las pruebas y ejecuciones académicas puedan usar un almacén local separado.

El run persistido de esta fase es `795face8e8a84422bdb93b6d2de76869`, bajo el experimento `Actividad8_Observabilidad`. MLflow contiene parámetros, métricas, `monitoring_result.json`, log estructurado y traza.

## Smoke test reproducible

La validación controlada se ejecuta con:

```powershell
python scripts/smoke_test_monitor.py --data <ruta-a-Tweets.csv> --sample-size 300
```

La ejecución usa datos reales suministrados por el usuario, conserva el SHA-256 del archivo, registra etiquetas reales y se identifica como `smoke_test`. Sus métricas no sustituyen el baseline histórico.

## Limitación de esta fase

La instrumentación está preparada para recibir predicciones de un modelo, pero aún no implementa un servicio de inferencia, detección de drift, alertas ni runbooks. Esos elementos pertenecen a fases posteriores.
