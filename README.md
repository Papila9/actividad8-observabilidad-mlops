# Actividad 8 — Observabilidad y MLOps para clasificación de sentimientos

Proyecto derivado de la Actividad 5. Implementa observabilidad, detección de drift, SLI/SLO, error budgets, gestión de incidentes, runbooks, versionamiento, reentrenamiento y rollback para un clasificador de sentimientos de comentarios sobre aerolíneas.

El proyecto no representa un servicio desplegado en producción. Las ejecuciones denominadas `smoke_test`, `controlled_*` y `simulation` son validaciones controladas y se conservan como evidencia, no como incidentes reales.

## Demostración en Google Colab

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Papila9/actividad8-observabilidad-mlops/blob/main/notebooks/actividad8_demo_colab.ipynb)

El notebook [actividad8_demo_colab.ipynb](notebooks/actividad8_demo_colab.ipynb) descarga el repositorio, instala las dependencias, ejecuta las pruebas reproducibles y consulta las evidencias de métricas, SLO, drift, incidentes y ciclo de vida del modelo. El CSV original no se publica; las métricas históricas supervisadas se cargan desde sus evidencias versionadas.

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

Arquitectura: [evidencias/arquitectura_actividad8.png](evidencias/arquitectura_actividad8.png). Dashboards: [operación](evidencias/dashboards/dashboard_operacion.png), [SLO](evidencias/dashboards/dashboard_slo.png), [drift y alertas](evidencias/dashboards/dashboard_drift_alertas.png), [incidentes](evidencias/dashboards/dashboard_incidentes.png) y [modelos](evidencias/dashboards/dashboard_modelos.png). El origen de cada figura está en [dashboard_data.json](evidencias/dashboards/dashboard_data.json).

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
| JSD de tokens | 0.08456762295081968 | 0.08900095628415298 |
| KS de longitudes | 0.08456762295081968 | 0.08900095628415298 |
| JSD de clases | 0.10 | 0.20 |

La evidencia histórica de la Fase 3 conserva JSD `0.20853278431872024` y KS `0.3767896174863389` en [evidencias/drift/drift-simulation-5fd90e98ad50.json](evidencias/drift/drift-simulation-5fd90e98ad50.json). La ejecución corregida más reciente conserva el baseline, usa el vocabulario de referencia y reporta JSD `0.008807901347323788`, KS `0.3833333333333334` y una alerta `data_drift` de severidad alta en [evidencias/drift/drift-simulation-8e21a2f799cf.json](evidencias/drift/drift-simulation-8e21a2f799cf.json). Los umbrales calibrados son criterios operativos académicos; no son universales ni fueron calibrados con tráfico histórico de producción.

La referencia de calibración y la simulación controlada no sustituyen el baseline histórico.

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
| `Actividad8_Observabilidad` | `8215195010b94883b2b734c3d9cd7d56` |
| `Actividad8_Observabilidad_Drift` | `362b5523d02c4cec8faf0c4633630963` |
| `Actividad8_Observabilidad_SLO` | `1c05aa318a0549bf98d881bfae4afe91` |
| `Actividad8_Observabilidad_Incidentes` | `0e69edec0ed24a669108561f269f2a52` |
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
python scripts/simular_slo.py --config config\slo.json --data ..\datos\datos_ini\Tweets.csv --model evidencias\modelos\logreg-stable-v1.pkl --sample-size 10 --output-dir evidencias\slo --log-path logs\slo_real_latency.jsonl --tracking-uri $tracking
python scripts/calibrar_drift.py --data ..\datos\datos_limp\Tweets_limpios.csv --baseline evidencias\drift\baseline.json
python scripts/simular_incidentes.py --data ..\datos\datos_limp\Tweets_limpios.csv --scenario long_text_subset --sample-size 300
python scripts/simular_incidentes_fase5.py --drift-result evidencias\drift\drift-simulation-5fd90e98ad50.json --slo-result evidencias\slo\slo-simulation-ba11951137e0.json --output-dir evidencias\incidentes --log-path logs\incidents.jsonl --tracking-uri $tracking
python scripts/simular_reentrenamiento.py --data ..\datos\datos_limp\Tweets_limpios.csv --output-dir evidencias\modelos --tracking-uri $tracking
python scripts/generar_dashboards.py
python scripts/generar_diagrama_arquitectura.py
python scripts/responder_alertas.py --incidents logs\incidents.jsonl --actions-log logs\alert_responses.jsonl --data ..\datos\datos_limp\Tweets_limpios.csv --registry evidencias\modelos\model_registry.json
python scripts/simular_model_drift.py --data ..\datos\datos_limp\Tweets_limpios.csv --model evidencias\modelos\logreg-stable-v1.pkl --output evidencias\drift\model-drift-simulation.json --tracking-uri $tracking
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

Los datos académicos confirmados se incorporan en la portada: Paola Quinto Martínez, matrícula 03039650, Luis Ariel Vázquez Piña y fecha 15 de noviembre de 2026. Versiones PDF:

- [actividad8_tecnico.pdf](entregables/pdf/actividad8_tecnico.pdf)
- [actividad8_incidentes.pdf](entregables/pdf/actividad8_incidentes.pdf)
- [actividad8_evidencias.pdf](entregables/pdf/actividad8_evidencias.pdf)

Los seis archivos existen físicamente y tienen tamaño no cero; los Word se generan desde la plantilla institucional y conservan portada, tablas, encabezados, pies, logotipo y declaración de uso de IA.

## 16. Pruebas y validación final

La validación integral más reciente obtuvo:

- 37 pruebas automatizadas aprobadas y 0 fallos, incluyendo las pruebas heredadas de compatibilidad de Actividad 5.
- 42 archivos Python identificados y compilados correctamente, código 0.
- Configuraciones JSON válidas.
- Evidencias requeridas presentes.
- Tres Word válidos con tablas, secciones, encabezados, pies e imágenes.
- Sin diagnósticos en los scripts de dashboards y documentos.
- Cinco dashboards regenerados desde evidencias identificadas, con unidades, ventanas antes/durante/después, umbral de latencia p95 de 500 ms y referencia de disponibilidad de 99 %.

La Actividad 5 no fue modificada. Este repositorio se publicó de forma independiente en GitHub después de la validación final.

## 17. Limitaciones

- Proyecto académico, sin despliegue productivo.
- Las simulaciones controladas no son incidentes reales.
- Data drift no demuestra pérdida de precisión.
- Accuracy y F1 macro requieren etiquetas reales.
- La latencia medida corresponde a inferencia local cuando se indica.
- Los SLO son objetivos académicos iniciales y no compromisos de producción.
- El almacén local de MLflow no reemplaza un servidor de tracking compartido.
- Los PDF fueron regenerados con Microsoft Word a partir de los Word actualizados y conservan los seis entregables con tamaño no cero.

## 18. Declaración de uso de IA

ChatGPT: sí, apoyo en análisis, redacción, revisión y orientación. Codex: sí, apoyo en programación, depuración, pruebas y documentación. Claude: sí, revisión crítica del repositorio y documentos. Gemini: no. GitHub Copilot: no. No se declaran porcentajes porque no existe una base verificable para calcularlos.

## 19. Referencias y fuentes consultadas

- MLflow. (s. f.). *MLflow Tracking*. https://mlflow.org/docs/latest/tracking/
- scikit-learn developers. (s. f.). *f1_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn developers. (s. f.). *train_test_split*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- [docs/observabilidad_base.md](docs/observabilidad_base.md)
- [docs/deteccion_drift_alertas.md](docs/deteccion_drift_alertas.md)
- [docs/sli_slo_error_budgets.md](docs/sli_slo_error_budgets.md)
- [docs/incidentes_runbooks.md](docs/incidentes_runbooks.md)
- [docs/rollback_reentrenamiento.md](docs/rollback_reentrenamiento.md)

## Correcciones de observabilidad y recuperación

Las correcciones posteriores a la revisión incorporan:

- [scripts/simular_slo.py](scripts/simular_slo.py) mide cada predicción local con `perf_counter`; el retraso controlado se inyecta únicamente durante la ventana de incidente y el p95 se calcula desde esas mediciones.
- [src/configuracion.py](src/configuracion.py) centraliza la carga de `drift_alerts.json` y `model_lifecycle.json`; los scripts ya no duplican el cooldown ni los criterios de aceptación.
- [scripts/responder_alertas.py](scripts/responder_alertas.py) lee [logs/incidents.jsonl](logs/incidents.jsonl), ejecuta reentrenamiento/evaluación para data drift y escalamiento/rollback para latencia o disponibilidad, registrando timestamp, motivo, resultado y carácter simulado en [logs/alert_responses.jsonl](logs/alert_responses.jsonl).
- [src/slo.py](src/slo.py) calcula consumo de presupuesto y burn rate con ventana corta y larga; la configuración está en [config/slo.json](config/slo.json).
- [src/rollback.py](src/rollback.py) verifica SHA-256, carga el artefacto activo con `load_active_model()` y bloquea la promoción de `rejected_candidate` salvo autorización explícita para una simulación.
- [scripts/simular_model_drift.py](scripts/simular_model_drift.py) compara accuracy/F1 con etiquetas reales y altera predicciones únicamente para una simulación controlada.

Evidencia nueva de ejecución: [evidencias/slo/](evidencias/slo/), [evidencias/drift/model-drift-simulation.json](evidencias/drift/model-drift-simulation.json), [logs/slo_real_latency.jsonl](logs/slo_real_latency.jsonl) y [logs/alert_responses.jsonl](logs/alert_responses.jsonl). Los resultados no representan producción y no reemplazan el baseline original.

## Correcciones

- La calibración de drift usa 20 ventanas aleatorias de 300 filas, semilla 42, y se conserva separada del baseline histórico en [evidencias/drift/calibracion_20_ventanas.json](evidencias/drift/calibracion_20_ventanas.json). Las referencias empíricas observadas fueron JSD p95 `0.02226484478454515`, JSD p99 `0.022683603678210738`, KS p95 `0.08456762295081968` y KS p99 `0.08900095628415298`.
- Las alertas persisten estado en archivos `.state.json` junto al log, respetan el cooldown de [config/drift_alerts.json](config/drift_alerts.json) y exponen detecciones generadas, alertas emitidas y alertas suprimidas.
- Las trazas de predicción incluyen `trace_id`, `span_id`, duración de vectorización y duración de predicción, sin almacenar textos completos.
- Se añadió el escenario controlado `long_text_subset`, que compara una ventana con los 300 textos más largos contra una ventana de referencia del mismo archivo; su evidencia es [evidencias/drift/drift-simulation-ca954b05123d.json](evidencias/drift/drift-simulation-ca954b05123d.json). Reportó JSD `0.1566179042071853`, KS `0.98` y severidad alta. La alerta puede quedar suprimida o emitida según el estado persistido y el cooldown, por lo que el resultado conserva ambas cifras.
- La evaluación histórica original obtuvo accuracy `0.798155737704918` con un split estratificado que tuvo 71 textos compartidos entre train/test. La evaluación agrupada independiente registra solapamiento 0, y la simulación reciente de `0.9335382513661202` evalúa el modelo estable sobre todo el dataset limpio; no reemplaza el baseline histórico ni es una estimación comparable del mismo protocolo.
- Los imports `fuentes.*` permanecen únicamente como compatibilidad heredada dentro de copias de referencia de `src/actividad5`; no se eliminó evidencia histórica ni se modificó el repositorio fuente de Actividad 5.
