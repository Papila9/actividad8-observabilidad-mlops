# Actividad 8 — Observabilidad y MLOps para clasificación de sentimientos

Proyecto académico independiente derivado de la Actividad 5. Implementa observabilidad, detección de drift, SLI/SLO, error budgets, gestión de incidentes, runbooks, versionamiento, reentrenamiento y rollback para un clasificador de sentimientos de comentarios sobre aerolíneas.

El proyecto no representa un servicio desplegado en producción. Las ejecuciones denominadas `smoke_test`, `controlled_*` y `simulation` son validaciones controladas y se conservan como evidencia, no como incidentes reales.

## 1. Objetivo y alcance

El objetivo es demostrar, mediante ejecuciones reproducibles, cómo observar un modelo de clasificación, detectar cambios o incumplimientos, registrar incidentes, comparar modelos y recuperar una versión estable.

El alcance incluye:

- Preparación y trazabilidad del dataset de la Actividad 5.
- Regresión Logística con TF-IDF.
- Métricas operativas y supervisadas.
- Logs estructurados, trazas y MLflow.
- Data drift y alertas con severidad, deduplicación y cooldown.
- SLI, SLO y Error Budgets académicos.
- Runbooks e incidentes simulados.
- Registro de modelos, reentrenamiento controlado y rollback simulado.
- Dashboards estáticos y documentos académicos basados en la plantilla Tecmilenio.

No incluye un endpoint de producción, alta disponibilidad real, monitoreo de red, tráfico en vivo ni publicación automática del modelo.

## 2. Tecnologías y herramientas

| Tecnología | Uso |
|---|---|
| Python 3.13.3 | Implementación y ejecución reproducible |
| pandas / NumPy | Manejo y análisis de datos |
| scikit-learn 1.9.1 | TF-IDF, Regresión Logística, particiones y métricas |
| MLflow 3.17.1 | Experimentos, parámetros, métricas y artefactos |
| matplotlib | Dashboards estáticos |
| python-docx 1.2.0 | Generación de Word desde la plantilla institucional |
| unittest | Pruebas automatizadas |
| JSON / JSONL | Evidencias, logs, trazas y registros |

No se agregaron dependencias funcionales nuevas al proyecto; `python-docx` ya estaba declarado en [requirements.txt](requirements.txt) y fue habilitado en el entorno para generar los documentos.

## 3. Estructura del repositorio

```text
config/                 Configuraciones de drift, SLO, ciclo de modelos y trazabilidad
datos/                  Referencias y archivos .gitkeep; el CSV original no se duplica
docs/                   Documentación técnica, runbooks y plantilla Tecmilenio
entregables/word/       Tres documentos Word académicos
entregables/pdf/        Tres versiones PDF de los documentos académicos
evidencias/             Resultados JSON, modelos, dashboards e índice de evidencias
logs/                   Logs estructurados JSONL
scripts/                Smoke tests, simulaciones y generadores
src/                    Métricas, monitor, drift, alertas, SLO e infraestructura de modelos
tests/                  Pruebas de Fases 2–7 y pruebas heredadas
traces/                 Trazas JSONL de ejecuciones de monitorización
mlruns/                 Almacén local de MLflow, excluido del control de versiones
```

El índice detallado se encuentra en [evidencias/README.md](evidencias/README.md).

## 4. Desarrollo de las siete fases

| Fase | Implementación | Evidencia principal |
|---|---|---|
| 1. Auditoría y base | Proyecto independiente, trazabilidad de archivos, baseline histórico y separación de Actividad 5 | [docs/trazabilidad_fase1.md](docs/trazabilidad_fase1.md), [config/trazabilidad.json](config/trazabilidad.json) |
| 2. Observabilidad base | Métricas, logs, trazas, smoke test y registro ampliado en MLflow | [src/metricas.py](src/metricas.py), [src/monitor.py](src/monitor.py), [evidencias/monitoring/](evidencias/monitoring/) |
| 3. Drift y alertas | Baseline textual, KS, Jensen–Shannon, class shift, model drift condicionado a etiquetas, severidades y cooldown | [docs/deteccion_drift_alertas.md](docs/deteccion_drift_alertas.md), [evidencias/drift/](evidencias/drift/) |
| 4. SLI/SLO | Disponibilidad, latencia, errores, volumen, calidad supervisada y Error Budgets | [docs/sli_slo_error_budgets.md](docs/sli_slo_error_budgets.md), [evidencias/slo/](evidencias/slo/) |
| 5. Incidentes | Clasificación, registro JSONL y runbooks para drift, SLO, errores y calidad | [docs/incidentes_runbooks.md](docs/incidentes_runbooks.md), [evidencias/incidentes/](evidencias/incidentes/) |
| 6. Ciclo de modelos | Versionamiento, comparación, criterios de aceptación, reentrenamiento y rollback simulado | [docs/rollback_reentrenamiento.md](docs/rollback_reentrenamiento.md), [evidencias/modelos/](evidencias/modelos/) |
| 7. Entrega | Dashboards, índice de evidencias y documentos Word académicos | [evidencias/dashboards/](evidencias/dashboards/), [entregables/word/](entregables/word/) |

## 5. Datos y modelo

La fuente local de la Actividad 5 no se copia ni modifica. El archivo limpio usado para el baseline es `../datos/datos_limp/Tweets_limpios.csv` y tiene:

```text
Filas: 14,640
SHA-256: 82539f706fdafc80f93dec7ae1cfc2310c179952f41548ed5c3ddf36effd67dd
```

El modelo principal es una Pipeline de TF-IDF y Regresión Logística con `class_weight="balanced"`, selección de hiperparámetros por F1 macro, partición estratificada 80/20 y semilla 42. La evaluación histórica usa 11,712 filas de entrenamiento y 2,928 de prueba.

El smoke test de Fase 2 utilizó el CSV original `../datos/datos_ini/Tweets.csv`, cuyo SHA-256 es `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`. Ambos hashes se conservan porque representan archivos distintos.

## 6. Métricas y resultados reales

### Baseline histórico

| Métrica | Regresión Logística |
|---|---:|
| Accuracy | 0.798155737704918 |
| F1 macro | 0.7457002530158143 |
| ROC-AUC OVR | 0.9110245729780063 |

Fuente: [evidencias/baseline/metricas_experimentos.json](evidencias/baseline/metricas_experimentos.json).

### Smoke test de monitorización

El smoke test evaluó 300 observaciones del conjunto de prueba, fuera del entrenamiento.

| Métrica | Resultado |
|---|---:|
| Latencia promedio | 3.468952329518894 ms |
| Latencia p95 | 9.572334971744569 ms |
| Tasa de errores | 0.0 |
| Volumen | 300 |
| Accuracy supervisada | 0.81 |
| F1 macro supervisado | 0.7694251460765994 |

La diferencia contra el baseline es descriptiva. No se presenta como mejora estadísticamente demostrada.

Fuente: [evidencias/monitoring/exec-c5fdb1da11f3.json](evidencias/monitoring/exec-c5fdb1da11f3.json).

### Reentrenamiento y comparación

| Versión | Accuracy | F1 macro | Decisión |
|---|---:|---:|---|
| `logreg-stable-v1` | 0.798155737704918 | 0.7457002530158143 | Estable |
| `logreg-candidate-v2` | 0.7739071038251366 | 0.7258019946741494 | Rechazado |

El candidato cumplió los mínimos absolutos, pero redujo F1 macro en `0.01989825834166492`; por ello no se promocionó automáticamente.

## 7. Observabilidad

Las métricas disponibles durante inferencia son latencia promedio, latencia p95, tasa de errores y volumen. Accuracy y F1 macro requieren etiquetas reales y se registran como no disponibles cuando no existen.

Cada ejecución conserva `execution_id`, modelo, versión, métricas, tiempos, errores, rutas de log/traza y `mlflow_run_id`. Los logs y trazas son JSONL para facilitar consulta posterior.

La latencia del smoke test se midió alrededor de `model.predict([text])`; incluye TF-IDF y clasificación local, pero excluye red, colas, HTTP, I/O y MLflow. No es latencia de un servicio completo.

## 8. Data drift y alertas

Se utilizan:

- KS sobre longitudes de texto.
- Jensen–Shannon sobre frecuencias de tokens.
- Jensen–Shannon sobre distribución de clases cuando existen etiquetas.
- Model drift únicamente cuando existen etiquetas reales y métricas supervisadas.

Umbrales operativos académicos:

| Señal | Media | Alta |
|---|---:|---:|
| JSD de tokens | 0.10 | 0.20 |
| KS de longitudes | 0.10 | 0.20 |
| JSD de clases | 0.10 | 0.20 |

La simulación de drift produjo JSD `0.20853278431872024`, KS `0.3767896174863389` y una alerta `data_drift` de severidad alta. Los umbrales no son universales ni fueron calibrados con tráfico histórico de producción.

Fuente: [evidencias/drift/drift-simulation-5fd90e98ad50.json](evidencias/drift/drift-simulation-5fd90e98ad50.json).

## 9. SLI, SLO y Error Budgets

| SLI/SLO | Objetivo académico |
|---|---:|
| Disponibilidad observada | >= 99 % |
| Latencia p95 | <= 500 ms |
| Accuracy con etiquetas | >= 0.75 |
| F1 macro con etiquetas | >= 0.70 |
| Volumen | Informativo, sin SLO independiente |

La ventana de evaluación es de 300 segundos y el presupuesto conceptual es `30d_academic`. El presupuesto de disponibilidad es 1 % de solicitudes fallidas; latencia y calidad permiten 1 % de ventanas incumplidas.

La simulación de SLO contiene:

| Escenario | Disponibilidad | p95 | Estado |
|---|---:|---:|---|
| Cumplimiento | 1.00 | 471 ms | `compliant` |
| Incumplimiento | 0.98 | 606 ms | `violated` |

En el escenario de incumplimiento se agotaron los presupuestos de disponibilidad y latencia. Accuracy y F1 quedaron no disponibles porque la simulación no proporcionó etiquetas reales.

Fuente: [evidencias/slo/slo-simulation-ba11951137e0.json](evidencias/slo/slo-simulation-ba11951137e0.json) y [docs/sli_slo_error_budgets.md](docs/sli_slo_error_budgets.md).

## 10. Incidentes y simulaciones

Se implementaron runbooks para data drift, latencia/SLO, incremento de errores y degradación de calidad. Cada runbook cubre detección, diagnóstico, impacto, respuesta, recuperación y cierre.

Se simularon dos incidentes:

1. Data drift, derivado de la evidencia de Fase 3, severidad alta.
2. Latencia elevada/incumplimiento de SLO, con p95 de 606 ms, severidad crítica.

Resultado: [evidencias/incidentes/incidents-simulation-2b4ac90f232f.json](evidencias/incidentes/incidents-simulation-2b4ac90f232f.json). Log: [logs/incidents.jsonl](logs/incidents.jsonl). No son incidentes reales de producción.

## 11. Registro, promoción y rollback de modelos

El registro conserva versión, artefacto, SHA-256, datos utilizados, métricas, estado y eventos. La versión estable activa quedó como `logreg-stable-v1`. El candidato `logreg-candidate-v2` fue rechazado por regresión de F1 macro.

El rollback se probó en un registro separado: se simuló una promoción temporal del candidato y después se regresó a `logreg-stable-v1`. El modelo estable no fue reemplazado automáticamente.

Fuentes: [evidencias/modelos/model_registry.json](evidencias/modelos/model_registry.json), [evidencias/modelos/rollback_simulation_registry.json](evidencias/modelos/rollback_simulation_registry.json) y [evidencias/modelos/retraining-simulation-295e2bf4242c.json](evidencias/modelos/retraining-simulation-295e2bf4242c.json).

## 12. Dashboards

Los gráficos estáticos se regeneran con [scripts/generar_dashboards.py](scripts/generar_dashboards.py):

| Dashboard | Contenido |
|---|---|
| [dashboard_operacion.png](evidencias/dashboards/dashboard_operacion.png) | Latencia promedio, p95, errores y volumen |
| [dashboard_slo.png](evidencias/dashboards/dashboard_slo.png) | Cumplimiento e incumplimiento de SLO |
| [dashboard_drift_alertas.png](evidencias/dashboards/dashboard_drift_alertas.png) | JSD, KS y señal de drift |
| [dashboard_incidentes.png](evidencias/dashboards/dashboard_incidentes.png) | Incidentes simulados por severidad |
| [dashboard_modelos.png](evidencias/dashboards/dashboard_modelos.png) | Comparación del modelo estable y candidato |

El resumen de las fuentes utilizadas está en [evidencias/dashboards/dashboard_data.json](evidencias/dashboards/dashboard_data.json).

## 13. MLflow y evidencias

El almacén local está en [mlruns/](mlruns/) y se excluye del control de versiones. Experimentos principales:

| Experimento | Run |
|---|---|
| `Actividad8_Observabilidad` | `795face8e8a84422bdb93b6d2de76869` |
| `Actividad8_Observabilidad_Drift` | `193fed723ff94a84b5aebedcd47cb537` |
| `Actividad8_Observabilidad_SLO` | `3ae65a91c366499aa784a96c48ea9153` |
| `Actividad8_Observabilidad_Incidentes` | `e42fba1c0dff45069635ef6afccebb47` |
| `Actividad8_Observabilidad_ModelLifecycle` | `dd3260df7874492aa6ab5e183e9c45e7` |

Cada run conserva parámetros, métricas, tags y artefactos conforme a la ejecución correspondiente. MLflow Tracking documenta el registro de parámetros, métricas y artefactos en [su documentación oficial](https://mlflow.org/docs/latest/tracking/).

## 14. Instalación y reproducción

Se requiere Python 3.13.3 o un entorno compatible con las versiones de [requirements.txt](requirements.txt).

```powershell
cd actividad8-observabilidad-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El CSV original permanece en el workspace de la Actividad 5. Para reproducir con la estructura actual:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE='true'
$tracking = 'file:///C:/ruta/absoluta/actividad8-observabilidad-mlops/mlruns'

python scripts/smoke_test_monitor.py --data ..\datos\datos_ini\Tweets.csv --sample-size 300
python scripts/crear_baseline_drift.py --data ..\datos\datos_limp\Tweets_limpios.csv
python scripts/simular_slo.py --config config\slo.json --output-dir evidencias\slo --log-path logs\slo_alerts.jsonl --tracking-uri $tracking
python scripts/simular_incidentes_fase5.py --drift-result evidencias\drift\drift-simulation-5fd90e98ad50.json --slo-result evidencias\slo\slo-simulation-ba11951137e0.json --output-dir evidencias\incidentes --log-path logs\incidents.jsonl --tracking-uri $tracking
python scripts/simular_reentrenamiento.py --data ..\datos\datos_limp\Tweets_limpios.csv --output-dir evidencias\modelos --tracking-uri $tracking
python scripts/generar_dashboards.py
python scripts/generar_documentos_academicos.py
```

Pruebas:

```powershell
python -m unittest discover -s tests -v
```

## 15. Documentos académicos

La plantilla original se conserva en [docs/plantillas/](docs/plantillas/). Los documentos Word se generan con [scripts/generar_documentos_academicos.py](scripts/generar_documentos_academicos.py):

- [actividad8_tecnico.docx](entregables/word/actividad8_tecnico.docx)
- [actividad8_incidentes.docx](entregables/word/actividad8_incidentes.docx)
- [actividad8_evidencias.docx](entregables/word/actividad8_evidencias.docx)

Los campos académicos personales no confirmados permanecen pendientes. Versiones PDF conservadas:

- [actividad8_tecnico.pdf](entregables/pdf/actividad8_tecnico.pdf)
- [actividad8_incidentes.pdf](entregables/pdf/actividad8_incidentes.pdf)
- [actividad8_evidencias.pdf](entregables/pdf/actividad8_evidencias.pdf)

Los seis archivos existen físicamente y tienen tamaño no cero; los Word fueron inspeccionados con `python-docx` conservando dos tablas y dos secciones cada uno.

## 16. Pruebas y validación final

La validación integral más reciente obtuvo:

- 18 pruebas automatizadas aprobadas y 0 fallos.
- 32 archivos Python compilados correctamente, código 0.
- Configuraciones JSON válidas.
- Evidencias requeridas presentes.
- Tres Word válidos con tablas, secciones, encabezados, pies e imágenes.
- Sin diagnósticos en los scripts de dashboards y documentos.

La Actividad 5 no fue modificada. Este repositorio se publicó de forma independiente en GitHub después de la validación final.

## 17. Limitaciones

- Proyecto académico, sin despliegue productivo.
- Las simulaciones controladas no son incidentes reales.
- Data drift no demuestra pérdida de precisión.
- Accuracy y F1 macro requieren etiquetas reales.
- La latencia medida corresponde a inferencia local cuando se indica.
- Los SLO son objetivos académicos iniciales y no compromisos de producción.
- El almacén local de MLflow no reemplaza un servidor de tracking compartido.
- Las versiones PDF actuales se conservan como entregables y deben abrirse visualmente antes de la entrega final.

## 18. Referencias y fuentes consultadas

- MLflow. (2026). *MLflow Tracking*. https://mlflow.org/docs/latest/tracking/
- scikit-learn developers. (2026). *f1_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn developers. (2026). *train_test_split*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- [docs/observabilidad_base.md](docs/observabilidad_base.md)
- [docs/deteccion_drift_alertas.md](docs/deteccion_drift_alertas.md)
- [docs/sli_slo_error_budgets.md](docs/sli_slo_error_budgets.md)
- [docs/incidentes_runbooks.md](docs/incidentes_runbooks.md)
- [docs/rollback_reentrenamiento.md](docs/rollback_reentrenamiento.md)
