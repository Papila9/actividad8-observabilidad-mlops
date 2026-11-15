# Cierre definitivo de la Fase 2

## Evidencia de datos

El baseline histórico utiliza `datos/datos_limp/Tweets_limpios.csv` con SHA-256 `82539f706fdaf80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd`.

El smoke test utiliza `datos/datos_ini/Tweets.csv` con SHA-256 `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`. Son archivos distintos y ambos registros se conservan.

## Separación de entrenamiento y evaluación

El procedimiento aplicó partición estratificada 80/20 con semilla 42: 11,712 filas para entrenamiento y 2,928 para prueba. Las 300 observaciones evaluadas fueron tomadas del conjunto de prueba y no se usaron para entrenar el modelo.

## Latencia

La latencia se midió localmente alrededor de `model.predict([text])` con `time.perf_counter()`. Incluye TF-IDF más clasificación, pero excluye red, servicio, colas, I/O y MLflow. Los valores fueron 3.468952 ms de promedio y 9.572335 ms p95 en 300 predicciones.

## Evidencia conservada

- Resultado: `evidencias/monitoring/exec-c5fdb1da11f3.json`.
- Log: `logs/exec-c5fdb1da11f3.jsonl`.
- Traza: `traces/exec-c5fdb1da11f3.jsonl`.
- MLflow: experimento `Actividad8_Observabilidad`, run `795face8e8a84422bdb93b6d2de76869`.

El smoke test es una validación técnica reproducible y no una demostración estadística de mejora frente al baseline.

## Estado

La Fase 2 queda cerrada. La Fase 3 queda limitada a una propuesta hasta recibir autorización explícita para modificar archivos.
