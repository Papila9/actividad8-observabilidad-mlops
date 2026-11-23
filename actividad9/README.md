# Actividad 9 — Escalabilidad, FinOps y gobernanza responsable

**Materia:** Gestión de Proyectos de Inteligencia Artificial  
**Proyecto:** Rediseño conceptual de una solución MLOps para clasificación de sentimientos.  
**Estado:** propuesta académica; **no desplegada en AWS**.

## Objetivo

Analizar el sistema local de clasificación de sentimientos desarrollado en las Actividades 5 y 8 y proponer un rediseño en la nube que contemple escalabilidad vertical y horizontal, optimización de inferencia, control de costos (FinOps) y gobernanza responsable.

## Alcance y arquitectura

- **Sistema actual (local):** datos de comentarios sobre aerolíneas, pipeline TF-IDF + regresión logística, scripts de inferencia, métricas, logs, trazas, MLflow y simulaciones de alertas, drift y rollback.
- **Diseño propuesto (no implementado):** API Gateway, AWS Lambda para inferencia stateless y versionada; evaluación de AWS Fargate + balanceador para carga sostenida; CloudWatch, S3, control de acceso, cifrado, presupuestos, monitoreo y revisión humana.
- **Escalabilidad:** ajuste de memoria/vCPU (vertical) y concurrencia o número de tareas (horizontal), sujetos a pruebas de carga y costos regionales.

## Evidencia y resultados

### Evaluación histórica por aerolínea

Se reconstruyó el conjunto de prueba histórico de **2,928 registros** y se calculó el recall de la clase negativa:

| Aerolínea | Negativos reales | TP negativos | FN negativos | Recall negativo |
|---|---:|---:|---:|---:|
| American | 384 | 345 | 39 | 89.84 % |
| Delta | 195 | 151 | 44 | 77.44 % |
| Southwest | 244 | 185 | 59 | 75.82 % |
| United | 532 | 473 | 59 | 88.91 % |
| US Airways | 442 | 397 | 45 | 89.82 % |
| Virgin America | 38 | 23 | 15 | 60.53 % |

La **brecha máxima observada fue de 29.32 puntos porcentuales**, entre American y Virgin America. Los grupos son **segmentos operativos**, no atributos demográficos protegidos. La partición histórica comparte **71 textos** entre entrenamiento y prueba, con **74 filas de prueba afectadas**; por tanto, **no es un holdout independiente**. Los intervalos de Wilson al 95 % son descriptivos y no corrigen esta limitación.

### Rendimiento local

Benchmark aislado de `model.predict`, con 100 repeticiones por lote y 10 calentamientos:

| Tamaño de lote | Latencia p95 | Throughput |
|---:|---:|---:|
| 1 | 1.94 ms | 1,027 predicciones/s |
| 8 | 2.95 ms | 6,221 predicciones/s |
| 32 | 3.61 ms | 14,662 predicciones/s |
| 128 | 7.47 ms | 24,037 predicciones/s |

**Importante:** las mediciones son locales; excluyen HTTP, red, autenticación, arranques en frío y AWS. No son indicadores de capacidad ni SLA en producción. La memoria RSS fue medida antes/después, no como pico continuo.

## FinOps y gobernanza

El anexo contiene escenarios didácticos para **100, 500 y 2,000 usuarios**, sensibilidad de memoria y duración, y un punto de equilibrio basado en supuestos. **No son precios AWS cotizados ni facturas reales**; el costo variable por solicitud de la alternativa con contenedores es hipotético y no representa el mecanismo nativo de facturación de Fargate.

Los controles propuestos incluyen minimización de datos, protección de información personal, IAM de mínimo privilegio, trazabilidad, supervisión humana, validación por segmentos, aprobación de cambios, alertas y rollback probado. La licencia **CC BY-NC-SA 4.0** del dataset requiere revisión antes de cualquier uso comercial.

## Archivos

La estructura sugerida es:

```text
Entregables/
  Actividad9_Reporte_Tecnico.docx
  Actividad9_Anexo_Tecnico.xlsx
  Actividad9_Arquitectura_ACTUAL_vs_PROPUESTA.pdf
```

Los nombres reales pueden variar. Las evidencias técnicas originales deben conservarse separadas y **no** incluir datasets completos, modelos, secretos ni información personal.

## Limitaciones y siguientes pasos

1. Repetir la evaluación con partición agrupada por texto, sin fuga por duplicados, y contrastes estadísticos apropiados.
2. Ejecutar un piloto aislado y medir p50/p95/p99 HTTP, throughput, errores, CPU, memoria, arranques en frío y costos.
3. Cotizar servicios en una región AWS definida y revisar la licencia y privacidad antes de un despliegue comercial.
4. Validar mecanismos de supervisión, alertas y recuperación antes de producción.

## Trazabilidad

Trabajo académico derivado del proyecto de observabilidad MLOps de la Actividad 8: [repositorio de referencia](https://github.com/Papila9/actividad8-observabilidad-mlops). Las mediciones adicionales se realizaron en una carpeta local independiente, sobre copias del dataset y del modelo. Las comprobaciones de integridad reportadas no detectaron cambios en archivos críticos de la Actividad 8; no constituyen una auditoría forense exhaustiva.

**Declaración de alcance:** esta documentación no afirma que el rediseño esté desplegado ni que exista una validación de producción.
