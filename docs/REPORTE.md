# Detección de EPP con visión artificial: reporte técnico (SIN COMPLETAR)

> Plantilla para completar. 

**Autor:** Edison Serrano
**Versión del modelo:** 1.0.0
**Fecha:** 09/10/2026

---

## 0. Resumen ejecutivo

_<5–8 líneas: problema, enfoque, modelo elegido, métricas clave (mAP@0.5:0.95, F1 de eventos), limitaciones principales.>_

| Indicador                          | Valor  |
| ---------------------------------- | ------ |
| mAP@0.5 (test)                     | _<>_ |
| mAP@0.5:0.95 (test)                | _<>_ |
| F1 de predicciones vs ground truth | _<>_ |
| F1 de eventos                      | _<>_ |
| FPS de inferencia (hardware)       | _<>_ |

---

## 1. Contexto y alcance

- Objetivo del sistema.
- Qué entra (imágenes/videos, ROI) y qué sale (`predictions.jsonl`, `events.jsonl`, `metrics.json`).
- Fuera de alcance.

## 2. Datos

> **Evidencia:** `reports/eda/` (generado por `python scripts/eda.py --root src/data/kaggle_dataset --out reports/eda`). Procedencia en `docs/DATASET.md`.

### 2.1 Procedencia

- Dataset(s): nombre, URL, licencia, versión, fecha de descarga.
- Videos de inferencia: fuente, duración, cámara simulada (`CAM_01`, …).
- Ground truth propio: cuántos frames, quién etiquetó, herramienta.

### 2.2 Validación del dataset (`dataset_report.json`)

| Chequeo                                   | Resultado |
| ----------------------------------------- | --------- |
| Imágenes corruptas / ilegibles           | _<>_    |
| Duplicados (hash perceptual)              | _<>_    |
| Imagen sin etiqueta / etiqueta sin imagen | _<>_    |
| Anotaciones vacías                       | _<>_    |
| Cajas fuera de límites                   | _<>_    |
| Clases inconsistentes                     | _<>_    |

### 2.3 Análisis exploratorio

- Imágenes por partición, objetos por clase, resoluciones, desbalance.
- Figuras: `reports/eda/*.png`.
- Hallazgos relevantes y cómo se mitigan.

### 2.4 Particiones

- Estrategia (por fuente/video para evitar fuga), proporciones, semilla.
- Verificación de ausencia de fuga (duplicados entre particiones = 0).

### 2.5 Aumentos de datos

| Aumento | Parámetro | Justificación |
| ------- | ---------- | -------------- |
| _<>_  | _<>_     | _<>_         |

## 3. Modelo y entrenamiento

### 3.1 Decisiones técnicas

| Decisión    | Alternativas                                                                                     | Elección y razón                                                         |
| ------------ | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| Arquitectura | YOLOv8n, RT-DETR, RF-DETR, Faster R-CNN                                                          | YOLOv8n (ligero y rápido);                                                |
| Clases       | 5 clases del dataset (helmet, no_helmet, no_vest, person, vest) vs. 3 clases + inferir ausencias | Se conservan las 5; la estrategia de decisión se compara en la sección 6 |
| Tracker      | ByteTrack, BoT-SORT                                                                              | _<>_                                                                     |
| Licencia     | AGPL vs. Apache 2.0                                                                              | _<>_                                                                     |

### 3.2 Configuración (`model_metadata.json`)

| Parámetro                         | Valor  |
| ---------------------------------- | ------ |
| Pesos iniciales                    | _<>_ |
| Versión del dataset               | _<>_ |
| Tamaño de entrada                 | _<>_ |
| Épocas / batch size               | _<>_ |
| Optimizador / LR                   | _<>_ |
| Semilla                            | _<>_ |
| Hardware / tiempo de entrenamiento | _<>_ |

### 3.3 Selección del mejor checkpoint

- Criterio (p. ej., mAP@0.5:0.95 en validación, desempate por recall de `persona`).
- Trazabilidad de experimentos (MLflow): tabla de corridas comparadas.

## 4. Evaluación offline (`metrics.json`)

> **Evidencia:** `reports/offline/metrics.json`, `reports/offline/confusion_matrix.png`, `reports/offline/precision_recall_curve.png` (generados por `python scripts/train.py`); `models/model_metadata.json`.

- Tabla global y por clase: Precision, Recall, F1, AP@0.5, AP@0.5:0.95.
- Figuras: `confusion_matrix.png`, `precision_recall_curve.png`.
- Velocidad: tiempo medio de inferencia (ms) y FPS.
- **Errores comentados**

| Imagen | Tipo (FP/FN) | Clase  | Causa probable | Mejora propuesta |
| ------ | ------------ | ------ | -------------- | ---------------- |
| _<>_ | _<>_       | _<>_ | _<>_         | _<>_           |

## 5. Pipeline de inferencia

> **Código:** `src/ppe/pipeline.py`, `scripts/inference.py`. **Salidas:** `outputs/latest/`.

- Diagrama de flujo: entrada → validación → modelo → umbrales/NMS → tracking → reglas → persistencia.
- Entradas soportadas: imagen, carpeta, video.
- Manejo de errores por archivo sin detener el lote.
- Parámetros configurables (`configs/inference.yaml`): umbral de confianza, IoU NMS, tamaño de entrada, etc.
- Logs: ubicación y formato.
- Comando de ejecución:

```bash
python scripts/inference.py --source <carpeta de imágenes>
```

### 5.1 Imágenes con resolución distinta a la de entrenamiento

- Modo por defecto: `letterbox` (reescala conservando la proporción y añade bordes); las cajas vuelven a coordenadas originales.
- Resoluciones evaluadas: _<lista de tamaños de entrada y fuente>_.

| Modo de adaptación | Resolución de entrada | mAP@0.5 | F1 predicciones | F1 eventos | ms/imagen |
| ------------------- | ---------------------- | ------- | --------------- | ---------- | --------- |
| letterbox           | _<>_                 | _<>_  | _<>_          | _<>_     | _<>_    |
| resize              | _<>_                 | _<>_  | _<>_          | _<>_     | _<>_    |
| none                | _<>_                 | _<>_  | _<>_          | _<>_     | _<>_    |

Conclusión: _<qué modo se eligió y por qué; efecto sobre objetos pequeños>_.

## 6. Lógica de negocio

### 6.1 Reglas

| Condición                      | Resultado          |
| ------------------------------- | ------------------ |
| Persona fuera de ROI            | Sin evento         |
| Persona sin chaleco             | Severidad media    |
| Persona sin casco               | Severidad alta     |
| Persona sin casco y sin chaleco | Severidad crítica |

### 6.2 Asociación persona–EPP

**Estrategia de decisión** (`decision.strategy`): comparar `association`, `explicit` e `hybrid` sobre el mismo ground truth.

| Estrategia  | Precision eventos | Recall eventos | F1 eventos | Duplicados | Omitidos |
| ----------- | ----------------- | -------------- | ---------- | ---------- | -------- |
| association | _<>_            | _<>_         | _<>_     | _<>_     | _<>_   |
| explicit    | _<>_            | _<>_         | _<>_     | _<>_     | _<>_   |
| hybrid      | _<>_            | _<>_         | _<>_     | _<>_     | _<>_   |

- Método (intersección sobre área del EPP, zonas del bbox), umbrales y justificación.
- Casos límite: EPP ocluido, persona parcialmente visible, EPP en la mano.

### 6.3 ROI

- Formato del polígono en YAML, punto de referencia del bbox (pies), librería.

### 6.4 Persistencia temporal y duplicados

- N frames / segundos requeridos, tolerancia a parpadeos.
- Cooldown por `(track_id, event_type)`, fusión ante cambio de ID.

### 6.5 Severidad configurable (`configs/rules.yaml`)


## 7. Persistencia

> **Evidencia:** `outputs/latest/predictions.jsonl`, `events.jsonl`, `images_index.jsonl` (formatos en el README, sección 5).

- Esquema de `predictions.jsonl` (campos, tipos, `schema_version`).
- Esquema de `events.jsonl`.
- Ejemplo de línea de cada archivo.
- Validación de que cada línea es JSON válido.

## 8. Desempeño de la solución

### 8.1 Predicciones vs `ground_truth.jsonl`

> **Evidencia:** `reports/evaluation/prediction_metrics.json`, `errors.jsonl`, `confusion_matrix_predictions.png`, `pr_curve_predictions.png`, `error_examples/`. Comando: ver README, paso 5.

- Tamaño, criterio de selección y limitaciones del ground truth.
- Emparejamiento: IoU configurable (valor usado: _<>_).
- Tabla por clase: TP, FP, FN, Precision, Recall, F1; mAP cuando aplique.
- Imágenes sin detecciones y anotaciones sin predicción: explicación.
- Errores representativos y causas.

### 8.2 Eventos

> **Evidencia:** `reports/evaluation/event_metrics.json`.
>
> | Métrica                                | Valor  |
> | --------------------------------------- | ------ |
> | Precision / Recall / F1                 | _<>_ |
> | Eventos duplicados / tasa de duplicidad | _<>_ |
> | Eventos omitidos                        | _<>_ |
> | Latencia condición → evento (s)       | _<>_ |

- Desglose por cámara, fuente o escenario.

### 8.3 Monitoreo operacional (`metrics.json` de ejecución)

> **Evidencia:** `outputs/latest/metrics.json`.
>
> | Dimensión    | Indicadores                                             | Valor  |
> | ------------- | ------------------------------------------------------- | ------ |
> | Volumen       | archivos, videos, frames, detecciones, eventos          | _<>_ |
> | Rendimiento   | tiempo total, ms/frame, FPS, archivos fallidos          | _<>_ |
> | Calidad       | confianza por clase, descartes, distribución de clases | _<>_ |
> | Segmentación | por clase, cámara, fuente, periodo                     | _<>_ |
> | Trazabilidad  | modelo, hash de config, fecha, semilla                  | _<>_ |

### 8.4 Selección de muestras y nuevo ciclo

> **Evidencia:** `reports/next_cycle_samples.csv` (`python scripts/select_samples.py ...`).

- Criterios: baja confianza, FP/FN, sin detección, alta confianza (control de calidad).
- Estratificación por clase, cámara y condición visual.
- Salida: `reports/next_cycle_samples.csv` y número de muestras por categoría.

## 9. Pruebas

| Módulo            | Casos cubiertos                | Resultado |
| ------------------ | ------------------------------ | --------- |
| ROI                | dentro, fuera, borde           | _<>_    |
| Asociación        | EPP presente, ausente, ambiguo | _<>_    |
| Deduplicación     | mismo track, cambio de ID      | _<>_    |
| Reglas / severidad | 4 combinaciones de la tabla    | _<>_    |

```bash
pytest -q
```

## 10. Reproducibilidad

1. Entorno: Python _<versión>_, `pip install -r requirements.txt`.
2. Datos: instrucciones de descarga y estructura esperada.
3. Entrenamiento: `python -m ppe.train --config configs/train.yaml`.
4. Evaluación: `python -m ppe.evaluate ...`.
5. Inferencia y eventos: ver sección 5.
6. Semillas y determinismo: qué se fija y qué no.

## 11. Limitaciones, riesgos y trabajo futuro

- Tamaño y sesgo del ground truth.
- Pocas cámaras / sesgo de dominio (iluminación, ángulo, distancia).
- Oclusiones, EPP de colores atípicos, cambios de ID del tracker.
- Mejoras: más datos, reentrenamiento con el nuevo ciclo, detección de deriva, despliegue.

## 12. Estructura de la entrega (.zip)

Ver la estructura completa y la ubicación de cada entregable en el [`README.md`](../README.md#4-dónde-está-cada-entregable).
Se genera con `python scripts/package_delivery.py --out entrega --max-mb 20`.

> Entrega: un solo archivo comprimido adjunto al correo, sin enlaces a Drive, GitHub u otros servicios. Si excede el límite, dividir en partes identificadas (`parte1de2.zip`, …).

## Anexos

- A. Registro de decisiones (`docs/DECISIONS.md`).
- B. Tabla de experimentos.
- C. Guía de etiquetado del ground truth.
