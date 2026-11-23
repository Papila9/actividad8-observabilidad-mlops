# Actividad 9 — Escalabilidad, FinOps y gobernanza responsable

**Materia:** Gestión de Proyectos de Inteligencia Artificial  
**Proyecto:** Rediseño conceptual de una solución MLOps para clasificación de sentimientos.  
**Estado:** propuesta académica; **no desplegada en AWS**.

## Objetivo y alcance

La actividad analiza el sistema local de clasificación de sentimientos desarrollado en las Actividades 5 y 8 y propone un rediseño conceptual con escalabilidad, optimización de inferencia, FinOps y gobernanza responsable.

La arquitectura propuesta considera API Gateway, AWS Lambda para inferencia stateless, una alternativa con AWS Fargate y balanceador, CloudWatch, S3, control de acceso, cifrado, presupuestos, monitoreo y revisión humana. No constituye una implementación ni una validación de producción.

## Resultados publicados

### Fairness histórica por aerolínea

La partición histórica contiene 2,928 registros de prueba y comparte 71 textos con entrenamiento, con 74 filas de prueba afectadas. Por tanto, estos resultados no son una evaluación independiente ni deben interpretarse como ausencia de fuga por duplicados.

| Aerolínea | Negativos reales | TP negativos | FN negativos | Recall negativo |
|---|---:|---:|---:|---:|
| American | 384 | 345 | 39 | 89.84 % |
| Delta | 195 | 151 | 44 | 77.44 % |
| Southwest | 244 | 185 | 59 | 75.82 % |
| United | 532 | 473 | 59 | 88.91 % |
| US Airways | 442 | 397 | 45 | 89.82 % |
| Virgin America | 38 | 23 | 15 | 60.53 % |

La brecha máxima observada es de 29.32 puntos porcentuales entre American y Virgin America. Las aerolíneas son segmentos operativos, no atributos demográficos protegidos. Los intervalos de Wilson son descriptivos y no corrigen la contaminación por duplicados.

### Rendimiento local

El benchmark mide `model.predict` local, con 100 repeticiones y 10 calentamientos por lote:

| Tamaño de lote | Latencia p95 | Throughput |
|---:|---:|---:|
| 1 | 1.94 ms | 1,027 predicciones/s |
| 8 | 2.95 ms | 6,221 predicciones/s |
| 32 | 3.61 ms | 14,662 predicciones/s |
| 128 | 7.47 ms | 24,037 predicciones/s |

Son mediciones locales: excluyen HTTP, red, autenticación, arranques en frío, colas, serialización, MLflow y AWS. La memoria RSS se midió antes/después de cada llamada, no como un pico continuo. No son indicadores de capacidad ni SLA en producción.

## Estructura real publicada

```text
actividad9/
├── README.md
├── Actividad9_Reporte_Tecnico.docx
├── Actividad9_Anexo_Tecnico.xlsx
├── Actividad9_Arquitectura_ACTUAL_vs_PROPUESTA.pdf
├── evidencias/
│   ├── Tweets.csv
│   ├── Tweets_limpios.csv
│   ├── dataset_provenance.json
│   ├── fairness_metodologia.json
│   ├── fairness_por_aerolinea.csv
│   ├── integridad_actividades.json
│   ├── rendimiento_local.csv
│   └── rendimiento_metodologia.json
└── scripts/
    ├── measure_activity9.py
    └── verify_integrity.py
```

Los cuatro archivos numéricos son `fairness_por_aerolinea.csv`, `fairness_metodologia.json`, `rendimiento_local.csv` y `rendimiento_metodologia.json`. El manifiesto SHA-256 público está integrado en `evidencias/integridad_actividades.json`; excluye el propio manifiesto y el verificador para evitar autoreferencias circulares.

## Reproducción local

`measure_activity9.py` no funciona de forma autónoma con este repositorio publicado. El script espera, respecto a `actividad9/`, los siguientes archivos locales:

```text
actividad9/
└── inputs/
    ├── Tweets.csv
    ├── Tweets_limpios.csv
    └── logreg-stable-v1.pkl
```

Los CSV publicados están en `evidencias/` para trazabilidad; el modelo estable `.pkl` no se publica. Para ejecutar el script se necesita una copia local autorizada del modelo existente y copias locales de los CSV en `inputs/`. El script escribe resultados en `evidencias/`, por lo que debe ejecutarse en una copia de trabajo separada si se desea preservar las evidencias publicadas.

Dependencias directas: Python, `scikit-learn` y `psutil`; el modelo serializado también requiere versiones compatibles de sus dependencias de scikit-learn. El script registra versiones y hardware cuando se ejecuta. No se debe reentrenar ni sustituir el modelo estable.

`verify_integrity.py` sí puede ejecutarse sobre el paquete publicado y compara cada archivo listado contra el SHA-256 esperado, diferenciando estados `valid`, `missing` y `hash_mismatch`.

## Dataset, licencia y privacidad

Los datos provienen de [Twitter US Airline Sentiment en Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data), cuyo origen indicado es Crowdflower Data for Everyone. La referencia publicada declara la licencia **CC BY-NC-SA 4.0**: requiere atribución, limita el uso comercial y exige conservar la licencia en adaptaciones.

Los CSV contienen el contenido público original del dataset, incluidos nombres de usuario, ubicaciones y textos de tweets. No se añadieron datos personales durante el procesamiento. La publicación de estos archivos debe mantenerse dentro de los fines académicos autorizados y revisarse antes de cualquier uso comercial.

## FinOps y gobernanza

El anexo contiene escenarios didácticos para 100, 500 y 2,000 usuarios, sensibilidad de memoria y duración, y supuestos de equilibrio. No son precios AWS cotizados ni facturas reales; el costo variable de la alternativa con contenedores es hipotético y no representa el mecanismo nativo de facturación de Fargate.

Los controles propuestos incluyen minimización de datos, protección de información personal, IAM de mínimo privilegio, trazabilidad, supervisión humana, validación por segmentos, aprobación de cambios, alertas y rollback probado.

## Limitaciones y siguientes pasos

1. Repetir la evaluación con partición agrupada por texto, sin fuga por duplicados, y contrastes estadísticos apropiados.
2. Ejecutar un piloto aislado y medir p50/p95/p99 HTTP, throughput, errores, CPU, memoria, arranques en frío y costos.
3. Cotizar servicios en una región AWS definida y revisar licencia y privacidad antes de un despliegue comercial.
4. Validar supervisión, alertas y recuperación antes de producción.

Trabajo académico derivado del proyecto de observabilidad MLOps de la [Actividad 8](https://github.com/Papila9/actividad8-observabilidad-mlops). Las mediciones se realizaron en una carpeta local independiente sobre copias del dataset y del modelo. Las comprobaciones de integridad no detectaron cambios en archivos críticos de la Actividad 8; no constituyen una auditoría forense exhaustiva.
