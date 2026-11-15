# Detección de drift y alertas — Fase 3

## Separación conceptual

- **Data drift:** cambia la distribución de las entradas, aquí textos y longitudes. Puede detectarse sin etiquetas reales.
- **Class distribution shift:** cambia la distribución de etiquetas reales observadas. Requiere etiquetas de la ventana actual y un baseline de etiquetas.
- **Model drift:** disminuye el desempeño del modelo respecto al baseline. Requiere etiquetas reales, predicciones y evaluación supervisada.

Data drift no implica automáticamente pérdida de precisión. Si no hay etiquetas, `model_drift` queda como `unavailable_without_real_labels`.

## Metodología estadística

1. **Longitud de texto:** estadístico de dos muestras Kolmogorov-Smirnov (KS), comparando caracteres por texto.
2. **Tokens:** divergencia de Jensen-Shannon (JSD) sobre frecuencias de tokens normalizadas. Se conservan los 500 tokens principales y una categoría `__other__`.
3. **Clases:** JSD entre conteos de clases del baseline y la ventana actual, solo cuando existen etiquetas.

El baseline se construye desde los datos limpios de la Actividad 5. Conserva hash, tamaño, longitudes, frecuencias agregadas y conteos de clase; no conserva el texto original.

## Umbrales académicos

| Señal | Medio | Alto |
|---|---:|---:|
| JSD de tokens | 0.10 | 0.20 |
| KS de longitudes | 0.10 | 0.20 |
| JSD de clases | 0.10 | 0.20 |

Los umbrales son operativos y académicos, no valores universales. El KS se registra como estadístico y no como p-value; no se afirma significancia estadística. La JSD depende del tokenizador, del tamaño de ventana y del límite de tokens.

## Severidades

- **Media:** una señal supera 0.10; se requiere observación y confirmación.
- **Alta:** una señal supera 0.20 o una distribución de clases cambia fuertemente; requiere investigación prioritaria.
- **Crítica:** reservada para model drift confirmado con etiquetas reales y caída de F1 macro de al menos 0.10. Data drift aislado nunca genera crítica automáticamente.

No se incluyen SLI, SLO ni error budgets; se formalizarán en la Fase 4.

## Deduplicación, ventanas y cooldown

Cada evaluación tiene `window_id`. La misma ventana no genera dos alertas. Las alertas con el mismo fingerprint se silencian durante 300 segundos. El fingerprint combina tipo de señal, severidad y motivo.

## Simulación controlada

La ejecución `drift-simulation-5fd90e98ad50` añadió dos tokens controlados a 300 textos reales. Resultado:

- JSD de tokens: `0.2085327843`.
- KS de longitud: `0.3767896175`.
- Señal dominante: `0.3767896175`.
- Severidad: `high`.
- Alertas: una alerta `data_drift` de severidad `high`.
- Model drift: no evaluable porque no se proporcionaron etiquetas de desempeño.

La evidencia está en [evidencias/drift/drift-simulation-5fd90e98ad50.json](../evidencias/drift/drift-simulation-5fd90e98ad50.json), [evidencias/drift/baseline.json](../evidencias/drift/baseline.json) y [logs/drift_alerts.jsonl](../logs/drift_alerts.jsonl).
