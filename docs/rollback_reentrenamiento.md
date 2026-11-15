# Fase 6 — Rollback y reentrenamiento controlado

## Versionamiento

El registro JSON contiene `active_version`, `stable_version`, artefacto, SHA-256, métricas, hash de datos y eventos. Los artefactos son archivos pickle generados a partir de pipelines reproducibles; su hash se conserva en el registro.

## Reentrenamiento

El flujo usa el CSV limpio existente, partición estratificada 80/20 y semilla 42. Entrena una versión estable reproducible y un candidato con parámetros declarados. La evaluación usa etiquetas del conjunto de prueba y no se presenta como tráfico de producción.

## Aceptación

El candidato debe cumplir Accuracy >= 0.75, F1 macro >= 0.70 y no reducir F1 macro respecto a la versión estable. `auto_promotion` es `false`: un candidato rechazado no reemplaza la versión activa y un candidato aceptado queda elegible para promoción manual.

## Rollback

`src/rollback.py` permite cambiar el `active_version` a una versión registrada, conserva el evento y exige una razón. La evidencia de Fase 6 usa un registro separado para simular promoción temporal y retorno a la versión estable; no cambia un servicio real.

## Separación con drift

El data drift dispara diagnóstico, no promoción automática ni afirmación de model drift. La degradación de calidad solo se evalúa con etiquetas reales y métricas supervisadas.

## Ejecución

```powershell
$env:MLFLOW_ALLOW_FILE_STORE='true'
python scripts/simular_reentrenamiento.py `
  --data ..\datos\datos_limp\Tweets_limpios.csv `
  --output-dir evidencias\modelos `
  --tracking-uri file:///C:/ruta/al/proyecto/mlruns
```
