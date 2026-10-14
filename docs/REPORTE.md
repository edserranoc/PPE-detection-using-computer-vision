# Detección de EPP con visión artificial: reporte técnico

> Documento en elaboración: las secciones 0 a 4 ya incluyen los resultados del EDA, del entrenamiento y de la evaluación offline; faltan las de inferencia, reglas y desempeño de la solución. Cada sección indica el archivo del que salen los números. Decisiones ya tomadas en el código: [`DECISIONS.md`](DECISIONS.md). Dataset: [`DATASET.md`](DATASET.md).

**Autor:** Edison Serrano
**Versión del modelo:** 1.0.0
**Fecha:** 

---

## 0. Resumen ejecutivo

Se entrenó un detector YOLOv8n (fine-tuning desde COCO, 20 épocas) sobre el dataset PPE de Kaggle (4 060 imágenes, 30 064 objetos, 5 clases: `person`, `helmet`, `no_helmet`, `vest`, `no_vest`). En la partición de prueba alcanza mAP@0.5 = 0.865 y mAP@0.5:0.95 = 0.558 (F1 global 0.849), con unos 73 FPS en preprocesado + inferencia + NMS. Las clases de ausencia de EPP (`no_helmet`, `no_vest`) son las más difíciles (AP@0.5 de 0.79 y 0.83), justo las que alimentan los eventos de severidad alta y crítica. El EDA no encontró problemas de integridad, pero sí **posible fuga entre particiones** (72 imágenes de val/test, el 8.9 %, tienen un gemelo casi idéntico en train), por lo que las métricas de val/test pueden ser optimistas. El entrenamiento **no ha convergido** (el mejor checkpoint es el de la última época y el mAP@0.5:0.95 sigue subiendo).

_<Añadir 2–3 líneas con el desempeño de las predicciones y de los eventos (secciones 6 y 8) cuando estén evaluados.>_

| Indicador                                                 | Valor                                     |
| --------------------------------------------------------- | ----------------------------------------- |
| mAP@0.5 (test, offline)                                   | 0.865                                     |
| mAP@0.5:0.95 (test, offline)                              | 0.558                                     |
| P / R / F1 global (test, offline)                         | 0.899 / 0.804 / 0.849                     |
| FPS preprocesado + inferencia + NMS (lote de Ultralytics) | 72.7 (98.8 sólo inferencia) — hardware: |
| F1 de predicciones vs ground truth (pipeline)             | _<>_                                    |
| F1 de eventos                                             | _<>_                                    |
| FPS del pipeline completo                                 | _<>_                                    |

---

## 1. Contexto y alcance

- Objetivo del sistema.
- Qué entra (imágenes/videos, ROI) y qué sale (`predictions.jsonl`, `events.jsonl`, `metrics.json`).
- Fuera de alcance.

## 2. Datos

> **Evidencia:** `reports/eda/dataset_report.json` y figuras en `reports/eda/` (`python scripts/eda.py --root src/data/kaggle_dataset --out reports/eda`).

### 2.1 Procedencia

| Campo    | Valor                                                                              |
| -------- | ---------------------------------------------------------------------------------- |
| Dataset  | Personal Protective Equipment (PPE) Dataset, Kaggle (`ndomalau`)                 |
| URL      | https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset |
| Formato  | YOLO, exportación tipo Roboflow (nombres`…_jpg.rf.<hash>.jpg`)                 |
| Tamaño  | 4 060 imágenes · 30 064 objetos · 5 clases                                      |
| Licencia | _Attribution 4.0 International (CC BY 4.0)<verificar en Kaggle></verificar>_     |

### 2.2 Validación del dataset (`dataset_report.json`)

| Chequeo                                                                              | train                | val   | test  |
| ------------------------------------------------------------------------------------ | -------------------- | ----- | ----- |
| Imágenes corruptas o ilegibles                                                      | 0                    | 0     | 0     |
| Imágenes sin etiqueta / etiquetas huérfanas                                        | 0 / 0                | 0 / 0 | 0 / 0 |
| Anotaciones vacías                                                                  | 0                    | 0     | 0     |
| Cajas fuera de límites / degeneradas / clases fuera de rango / líneas mal formadas | 0                    | 0     | 0     |
| Duplicados exactos (md5)                                                             | 0 en todo el dataset |       |       |

El dataset está limpio a nivel de archivos y etiquetas. No hay imágenes sin objetos (fondo): todas las imágenes tienen al menos una anotación.

### 2.3 Análisis exploratorio

| Partición | Imágenes | %    | Objetos | Objetos/imagen (media · mediana · p95 · máx) | Objetos pequeños* |
| ---------- | --------- | ---- | ------- | ------------------------------------------------ | ------------------ |
| train      | 3 248     | 80 % | 24 184  | 7.4 · 6 · 18 · 187                            | 21.7 %             |
| val        | 406       | 10 % | 3 095   | 7.6 · 6 · 18 · 55                             | 17.2 %             |
| test       | 406       | 10 % | 2 785   | 6.9 · 6 · 18 · 63                             | 20.2 %             |

\* Objeto menor de 32×32 px a 640×640. Todas las imágenes miden 640×640 (una sola resolución).

| Clase     | train | val   | test | Total  | % del total |
| --------- | ----- | ----- | ---- | ------ | ----------- |
| person    | 8 259 | 1 042 | 954  | 10 255 | 34.1 %      |
| no_vest   | 4 912 | 649   | 513  | 6 074  | 20.2 %      |
| helmet    | 4 068 | 531   | 529  | 5 128  | 17.1 %      |
| no_helmet | 3 993 | 523   | 393  | 4 909  | 16.3 %      |
| vest      | 2 952 | 350   | 396  | 3 698  | 12.3 %      |

Figuras: `reports/eda/objects_per_class.png`, `bbox_area_per_class.png`, `objects_per_image.png`, `box_centers_heatmap.png`, `resolutions.png`, `samples_with_boxes.png`.

Hallazgos:

- **Desbalance leve:** razón máximo/mínimo de 2.8:1 (`person` frente a `vest`). No requiere sobremuestreo, pero conviene reportar las métricas por clase.
- **Objetos pequeños:** entre el 17 % y el 22 % de los objetos miden menos de 32×32 px (el área mediana de una caja es el 1.5 % de la imagen). Los cascos son los más afectados; sube el riesgo de omisiones en imágenes de mayor resolución reescaladas.
- **Escenas concurridas:** la mediana es de 6 objetos por imagen, pero hay imágenes con hasta 187 (train). Esto exige cuidar el límite de detecciones (`max_det`) y el NMS.
- **Una sola resolución (640×640):** el dataset no permite medir por sí solo la robustez a otras resoluciones; por eso se evalúa con versiones reescaladas del conjunto de prueba (sección 5.1).
- **Las clases `no_helmet` y `no_vest` son explícitas.** Falta comprobar con el notebook 01 (sección 7) qué fracción de las personas tiene su cabeza/torso anotados, porque eso decide la estrategia de decisión.

### 2.4 Particiones y fuga de información

El dataset llega con una partición 80/10/10 (3 248 / 406 / 406 imágenes). El EDA encontró que **no es independiente**:

| Chequeo                                                        | Resultado                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| Duplicados exactos entre particiones                           | 0                                                           |
| Misma imagen fuente (patrón`.rf.`) en particiones distintas | 24 grupos                                                   |
| Casi-duplicados (dHash, Hamming ≤ 2)                          | 88 grupos (222 imágenes), de los que 52 cruzan particiones |
| Imágenes de val/test con un gemelo en train                   | **72 de 812 (8.9 %)**                                 |

Los nombres (por ejemplo `IMG_3103_mp4-17`) indican que muchas imágenes son **fotogramas extraídos de videos**, por lo que fotogramas vecinos del mismo video caen en particiones distintas.

- **Implicación:** las métricas de val y test pueden estar infladas, y el test es menos independiente de lo que parece.
- **Decisión:** se conserva la partición original del dataset y se cuantifica el efecto: se reportan las métricas sobre todo el test y también excluyendo las imágenes con gemelo en train (`reports/eda/leakage_images.csv`, `evaluate_predictions.py --exclude-images`). _<Resultado: métricas con y sin gemelos: …>_
- **Alternativa descartada por costo:** volver a dividir agrupando por video/imagen fuente (exige reentrenar).

### 2.5 Aumentos de datos

| Aumento                           | Valor                                         | Justificación                                                           |
| --------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ |
| HSV (tono / saturación / brillo) | 0.015 / 0.7 / 0.4                             | Cubre cambios de iluminación y de cámara                               |
| Volteo horizontal                 | 0.5                                           | Las escenas son simétricas                                              |
| Volteo vertical                   | 0                                             | Un casco no aparece bajo los pies; rompería la asociación cabeza/torso |
| Rotación / cizalla               | 0 / 0                                         | Personas de pie y cámaras fijas                                         |
| Traslación / escala              | 0.1 / 0.5                                     | Variación de posición y de distancia a la cámara                      |
| Mosaic                            | 1.0 (se desactiva en las últimas 10 épocas) | Más contexto y más objetos pequeños por imagen                        |
| MixUp                             | 0                                             | Mezclar imágenes confunde las clases de ausencia                        |

## 3. Modelo y entrenamiento

### 3.1 Decisiones técnicas

| Decisión       | Alternativas                            | Elección y razón                                                                                                                       |
| --------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Arquitectura    | YOLOv8n, RT-DETR, RF-DETR, Faster R-CNN | YOLOv8n: ligero (≈10 ms por imagen), fácil de reproducir y desplegar. Licencia AGPL-3.0 de Ultralytics: revisar si el uso es comercial |
| Pesos iniciales | Aleatorios, COCO                        | `yolov8n.pt` preentrenado en COCO (ya conoce la clase persona)                                                                         |
| Clases          | 5 clases vs 3 + inferir ausencias       | Se conservan las 5; la estrategia de decisión se decide con el experimento de la sección 6.2                                           |
| Épocas         | 20 vs 100                               | 20 por costo de cómputo (≈58 min);**insuficiente** (ver 3.3)                                                                     |

### 3.2 Configuración (`runs/train/yolov8n/args.yaml`)

| Parámetro               | Valor                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modelo / pesos iniciales | YOLOv8n /`yolov8n.pt` (COCO)                                                                                                                     |
| Versión del dataset     | 4 060 imágenes (3 248 / 406 / 406); huella sha256:_<no registrada; ejecutar una versión nueva de `train.py` guarda `model_metadata.json`>_ |
| Tamaño de entrada       | 640                                                                                                                                                |
| Épocas / paciencia      | 20 / 20                                                                                                                                            |
| Batch                    | 8                                                                                                                                                  |
| Optimizador / lr0 / lrf  | AdamW / 0.001 / 0.01                                                                                                                               |
| Weight decay / warmup    | 0.0005 / 3 épocas                                                                                                                                 |
| Precisión mixta (AMP)   | sí                                                                                                                                                |
| Semilla / determinista   | 42 / sí                                                                                                                                           |
| Tiempo de entrenamiento  | 3 471 s (57.9 min; 174 s por época)                                                                                                               |
| Hardware                 | _<completar: GPU/CPU, RAM>_                                                                                                                      |
| Versiones                | _<ultralytics, torch, python>_                                                                                                                   |

### 3.3 Selección del mejor checkpoint y curvas

- **Criterio:** `fitness = 0.1·mAP@0.5 + 0.9·mAP@0.5:0.95` en validación, evaluado al final de cada época (criterio de Ultralytics), porque premia la calidad de localización que necesita la asociación persona↔EPP.
- **Resultado:** el mejor checkpoint es el de la **época 20** (la última): mAP@0.5:0.95 = 0.568 en validación. La paciencia (20) nunca se activó.

| Época             | 1     | 5     | 10    | 15    | 20    |
| ------------------ | ----- | ----- | ----- | ----- | ----- |
| mAP@0.5 (val)      | 0.656 | 0.782 | 0.853 | 0.879 | 0.894 |
| mAP@0.5:0.95 (val) | 0.343 | 0.435 | 0.514 | 0.548 | 0.568 |

- **Sin sobreajuste:** pérdida de caja final train/val de 1.143 / 1.176 (diferencia de 0.03), ambas decrecientes.
- **Sin converger:** el mAP@0.5:0.95 ganó +0.008 en las últimas 3 épocas. Entrenar más épocas (50–100) debería mejorar sobre todo `no_helmet` y `no_vest`. _<Si se reentrena, anotar aquí la nueva versión y sus métricas.>_
- Gráfica: `runs/train/yolov8n/results.png`. 

## 4. Evaluación offline (`metrics.json`)

> **Evidencia:** `reports/offline/metrics_val.json`, `metrics_test.json` (generados por `scripts/train.py`); gráficas de Ultralytics en `runs/train/yolov8n/` (`eval_test/confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`).

**Cómo leer las cifras:** Ultralytics evalúa con `conf = 0.001`; los mAP usan todas las detecciones, y precision/recall/F1 corresponden al umbral de confianza que maximiza F1 (promedio entre clases). No son las cifras del umbral de operación del pipeline (`configs/inference.yaml`), que se miden en la sección 8.

| Partición     | Precision       | Recall          | F1              | mAP@0.5         | mAP@0.5:0.95    |
| -------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| val            | 0.903           | 0.833           | 0.867           | 0.897           | 0.566           |
| **test** | **0.899** | **0.804** | **0.849** | **0.865** | **0.558** |

Por clase en **test** (entre paréntesis, val):

| Clase     | Precision     | Recall        | F1            | AP@0.5        | AP@0.5:0.95   |
| --------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| person    | 0.924 (0.902) | 0.854 (0.888) | 0.888 (0.895) | 0.894 (0.936) | 0.669 (0.651) |
| vest      | 0.946 (0.954) | 0.846 (0.885) | 0.893 (0.918) | 0.922 (0.945) | 0.637 (0.666) |
| helmet    | 0.916 (0.936) | 0.828 (0.814) | 0.870 (0.870) | 0.888 (0.901) | 0.559 (0.566) |
| no_vest   | 0.849 (0.856) | 0.769 (0.815) | 0.807 (0.835) | 0.829 (0.864) | 0.486 (0.471) |
| no_helmet | 0.861 (0.867) | 0.723 (0.763) | 0.786 (0.812) | 0.790 (0.837) | 0.440 (0.475) |

**Velocidad** (lote de Ultralytics, hardware: ):

| Etapa               | val (ms) | test (ms)                              |
| ------------------- | -------- | -------------------------------------- |
| Preprocesado        | 1.23     | 1.13                                   |
| Inferencia          | 10.45    | 10.12                                  |
| Postprocesado (NMS) | 2.32     | 2.49                                   |
| **Total**     | 14.00    | **13.75**                        |
| FPS                 | 71.4     | **72.7** (98.8 sólo inferencia) |

No incluye lectura de disco, reglas de negocio ni escritura de JSONL; el FPS del pipeline completo sale de `outputs/latest/metrics.json`.

**Observaciones:**

1. **Las clases de ausencia son las más débiles:** `no_helmet` (AP@0.5 0.79, recall 0.72) y `no_vest` (0.83, recall 0.77), frente a 0.89–0.92 de `person`, `helmet` y `vest`. El recall de `no_helmet` implica que alrededor de 1 de cada 4 cabezas sin casco no se detecta, y esas son las que generan eventos de severidad alta.
2. **Test por debajo de val** en todas las clases salvo en AP@0.5:0.95 de `person` y `no_vest` (mAP@0.5: 0.865 frente a 0.897). Val se usa para elegir el checkpoint y comparte gemelos con train, lo que lo vuelve más optimista; la mayor caída es en `no_helmet` (−0.047 de AP@0.5) y `person` (−0.042).
3. **Localización mejorable:** la brecha entre mAP@0.5 (0.865) y mAP@0.5:0.95 (0.558) indica cajas poco ajustadas, coherente con un modelo pequeño y sin converger; afecta la asociación por intersección sobre el área del EPP.
4. **Falta decidir el umbral de operación** con la curva F1 por clase (`BoxF1_curve.png`). 
5. **Matriz de confusión y curva PR:** _<insertar `confusion_matrix.png` y `BoxPR_curve.png` y comentar qué clases se confunden entre sí (p. ej. `helmet` ↔ `no_helmet`).>_

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

- **Fuga entre particiones:** 8.9 % de las imágenes de val/test tienen un gemelo en train (fotogramas de los mismos videos); las métricas son optimistas. Mitigación: reportar con y sin esas imágenes; trabajo futuro: re-dividir por video.
- **Entrenamiento corto (20 épocas):** el modelo no ha convergido; las clases `no_helmet` y `no_vest` aún tienen margen.
- **Una sola resolución en el dataset (640×640):** la robustez a otras resoluciones se mide con versiones reescaladas del test (sección 5.1), no con datos reales de otra cámara.
- **Sesgo de dominio:** _<escenas, ángulos, iluminación y cámaras no representados>_.
- **Métricas offline en el umbral que maximiza F1:** no equivalen a las del umbral de operación.
- **Verdad terreno de eventos derivada de las anotaciones**, no revisada por una persona independiente.
- Hardware, versiones de librerías y huella del dataset no quedaron registrados en el primer entrenamiento (la versión actual de `train.py` ya los guarda).
- Trabajo futuro: más épocas, `imgsz` mayor para objetos pequeños, modelo mayor (YOLOv8s), tracker con re-identificación, calibración de umbrales por clase.

## 12. Estructura de la entrega (.zip)

Ver la estructura completa y la ubicación de cada entregable en el [`README.md`](../README.md#4-dónde-está-cada-entregable).
Se genera con `python scripts/package_delivery.py --out entrega --max-mb 20`.

> Entrega: un solo archivo comprimido adjunto al correo, sin enlaces a Drive, GitHub u otros servicios. Si excede el límite, dividir en partes identificadas (`parte1de2.zip`, …).

## Anexos

- A. Registro de decisiones (`docs/DECISIONS.md`).
- B. Tabla de experimentos.
- C. Guía de etiquetado del ground truth.
