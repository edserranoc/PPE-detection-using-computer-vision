# Decisiones técnicas y reglas de negocio

Cada decisión indica la alternativa descartada y el archivo donde se implementa. Las filas con `_<>_` se completan
con los resultados experimentales.

## 1. Modelo y datos

| Decisión                                                                                     | Alternativas                                           | Justificación                                                                                                                                                       | Dónde                   |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| YOLOv8n, fine-tuning desde COCO                                                               | RT-DETR, RF-DETR, Faster R-CNN                         | Ligero, rápido en CPU/GPU modesta y con un ecosistema que simplifica entrenamiento y exportación. Licencia AGPL-3.0 de Ultralytics: revisar si el uso es comercial | `scripts/train.py`     |
| Se conservan las 5 clases del dataset                                                         | 3 clases (persona, casco, chaleco) e inferir ausencias | Permite comparar detección explícita de ausencias contra asociación                                                                                               | `configs/rules.yaml`   |
| Mejor checkpoint por fitness de Ultralytics (0.1·mAP@0.5 + 0.9·mAP@0.5:0.95 en validación) | Sólo mAP@0.5, o recall de`person`                   | Premia la calidad de localización, relevante para la asociación persona↔EPP                                                                                       | `scripts/train.py`     |
| Particiones sin fuga: agrupar por imagen fuente (`.rf.`) si hay que crear `val`           | División aleatoria por imagen                         | Las copias aumentadas de una misma foto no deben caer en particiones distintas                                                                                       | `train.py`, `eda.py` |
| Aumentos: sin volteo vertical ni rotación;`fliplr`, HSV, escala, traslación y mosaic      | Aumentos agresivos por defecto                         | Cámaras fijas, personas de pie; HSV cubre iluminación; mosaic ayuda con cascos pequeños                                                                           | `configs/train.yaml`   |

## 2. Lógica de negocio

### 2.1 Estrategia de decisión (`decision.strategy`)

Para cada persona se calcula la mayor confianza de cada EPP asociado: `helmet`, `no_helmet`, `vest`, `no_vest`.

| Estrategia           | Falta el EPP cuando…                                      | Riesgo                                                     |
| -------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| `association`      | no hay EPP positivo asociado                               | Falsas alarmas si el detector omite un casco visible       |
| `explicit`         | hay un`no_*` asociado con más confianza que el positivo | Omite incumplimientos si el detector no «ve» la ausencia |
| `hybrid` (defecto) | no hay positivo asociado con confianza ≥ a la del`no_*` | Intermedio; se calibra con`min_ioa` y los umbrales       |

Experimento: _<F1 de eventos por estrategia con `--set-rules decision.strategy=...`>_.

### 3.2 Asociación persona ↔ EPP

- Intersección sobre el **área del EPP** ≥ `min_ioa` (0.5), no IoU: un casco es diminuto frente a una persona.
- El centro del EPP debe caer en la franja esperada de la persona: cabeza (0–40 % de la altura) para casco y
  `no_helmet`; torso (15–80 %) para chaleco y `no_vest`.
- Cada EPP se asigna a una sola persona (mayor intersección; empate → persona más pequeña).
- Implementación: `src/ppe/rules/association.py`.

### 3.3 ROI

Polígonos normalizados en `rules.yaml`, configurables por cámara. La persona está dentro si el punto inferior
central de su caja (los pies) cae dentro; el borde cuenta como dentro. Se implementó el punto en polígono sin
dependencias externas (`src/ppe/rules/geometry.py`). Persona fuera del ROI → no genera evento.

### 3.4 Persistencia temporal

| Modo            | Parámetros                                                            | Razón                                          |
| --------------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| `sequence`    | ≥ 5 frames y ≥ 1 s de condición continua; tolera huecos de 2 frames | Evita eventos por parpadeos del detector        |
| `independent` | 1 frame                                                                | Una imagen aislada no tiene dimensión temporal |

`frame_id` es la posición en el orden natural de nombres de archivo; un archivo ilegible consume su posición (el hueco lo absorbe la tolerancia).

### 3.5 Duplicados

1. **Detecciones:** NMS por clase (`thresholds.dedup_iou`) además del NMS del modelo.
2. **Eventos:** un solo evento por condición continua (mismo fuente, `track_id` y tipo).
3. **Cooldown:** el mismo (fuente, track, tipo) no se repite antes de `cooldown_seconds`.
4. **Cambio de `track_id`:** eventos del mismo tipo con IoU de cajas ≥ `merge_iou` dentro de `merge_window_seconds` se fusionan.
5. Escalada (p. ej. de «sin casco» a «sin casco y sin chaleco») se considera un evento nuevo de mayor severidad.


## 4. MLOps

- Configuración versionada y separada del código; hash de configuración en cada corrida.
- Trazabilidad: `model_metadata.json`, `run_config.yaml`, `inference.log`, `metrics.json → traceability`.
