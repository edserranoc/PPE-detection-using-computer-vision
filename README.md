# Detección de EPP con visión artificial

---

**Autor:** Edison Serrano
**Versión del modelo:** 1.0.0
**Fecha:**  08/10/2026
**Empresa:** SURA

## Base de Datos

Recurso Libre Acceso:  [www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset/data](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset/data)

Información detallada: docs/dataset.md

## Información de las decisiones tomadas

docs/DECISIONS.md

## Estructura de la entrega (.zip)

```
ppe-vision/
├── configs/        # train.yaml, inference.yaml, rules.yaml 
├── src/
│   ├── data/       # validación, EDA, splits, augmentations
│   ├── ppe/
│   ├── train.py    # semilla, metadata, MLflow
│   ├── eda.py
│   ├── evaluate.py # P/R/F1, mAP, PR curve, confusión, FPS
│   └── inference/  # pipeline, NMS/dedup, logging, errores por archivo 
├── tests/          # ROI, asociación, dedup, severidad
├── docs/           # DECISIONS.md, reglas de negocio
├── models/			# Best model.pt, model_metadata.json
├── reports/		# EDA, metrics
├── runs/			# yolov8n training with 20 and 2 epochs
├── weights/	
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

Configuración

Toda la configuración está separada del código. Cualquier valor se puede sobrescribir por línea de comandos:

```bash
python src/train.py     --set train.epochs=20 train.batch=8
```

El archivo de Inference no se completó

```
python src/inference.py --source <carpeta> --set thresholds.conf_default=0.4 preprocessing.resize_mode=resize
python src/inference.py --source <carpeta> --set-rules decision.strategy=explicit
```

Cada corrida de inferencia guarda la configuración efectiva y su hash en `outputs/<run_id>/run_config.yaml`.

## `train.yaml`

| Clave                            | Descripción                                                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `data.root` / `data.yaml`    | Carpeta del dataset (`src/data/kaggle_dataset`) y `data.yaml` (se busca solo si es `null`)                            |
| `data.val_fraction_if_missing` | Si no hay`valid/`, se crea desde `train/` agrupando por imagen fuente                                                   |
| `model.weights`                | Pesos iniciales (`yolov8n.pt`, preentrenado en COCO)                                                                      |
| `train.*`                      | `imgsz`, `epochs`, `batch`, `optimizer`, `lr0`, `lrf`, `patience`, `seed`, `deterministic`, `device`... |
| `augment.*`                    | Aumentos de datos con su justificación en comentarios                                                                      |
| `output.*`                     | Dónde guardar corridas,`models/best_model.pt` y `reports/offline/`                                                     |
| `eval.conf`, `eval.iou`      | Umbrales de la evaluación de Ultralytics (conf bajo para curvas PR completas)                                              |
| `tracking.mlflow`              | Activa MLflow (`pip install mlflow`)                                                                                      |

## `inference.yaml` (No se implementó)

| Clave                                                   | Descripción                                                                              |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `model.weights`, `model.version`, `model.imgsz`   | Checkpoint, versión escrita en cada registro y tamaño de entrada (el del entrenamiento) |
| `input.mode`                                          | `independent` (imágenes aisladas) o `sequence` (frames ordenados de una cámara)     |
| `input.camera_id`, `input.fps`, `input.source_id` | Identificadores y reloj para`timestamp_seconds = frame_id / fps` (modo `sequence`)    |
| `preprocessing.resize_mode`                           | `letterbox` (bordes) · `resize` (estira) · `none`                                 |
| `thresholds.conf_default`, `thresholds.per_class`   | Umbral de confianza global y por clase                                                    |
| `thresholds.model_nms_iou`, `thresholds.dedup_iou`  | NMS del modelo y control extra de detecciones duplicadas por clase                        |
| `output.save_annotated`, `output.annotate`          | Imágenes anotadas (`events` o `all`)                                                 |
| `output.mirror_latest`                                | Copia los resultados a`outputs/latest/`                                                 |

## `rules.yaml`    (No se implementó)

| Clave                                                                | Descripción                                                                                    |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `classes`, `output_names`                                        | Rol → clase del modelo, y nombre mostrado en`class_name` (`persona`, `casco`...)         |
| `decision.strategy`                                                | `association` · `explicit` · `hybrid` (ver `docs/DECISIONS.md`)                       |
| `association.min_ioa`, `association.zones`                       | Fracción del EPP dentro de la persona y franjas verticales cabeza/torso                        |
| `roi.*`                                                            | Polígonos normalizados (0–1), punto de referencia (pies), zonas por cámara (`roi.cameras`) |
| `tracking.*`                                                       | Tracker IoU (sólo modo`sequence`)                                                            |
| `persistence.sequence` / `.independent`                          | `min_frames`, `min_seconds`, `max_gap_frames` por modo                                    |
| `events.cooldown_seconds`, `merge_window_seconds`, `merge_iou` | Control de eventos duplicados                                                                   |
| `events.types`                                                     | Condición →`event_type` y `severity` (configurable)                                       |

# License

This project is licensed under the MIT license. See the LICENSE file for details.
