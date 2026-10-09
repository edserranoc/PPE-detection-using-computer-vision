# Detección de EPP con visión artificial

---

**Autor:** Edison Serrano
**Versión del modelo:** 1.0.0
**Fecha:**  08/10/2026
**Empresa:** SURA


## Base de Datos 

Recurso Libre Acceso:  [www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset/data](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset/data)



## 12. Estructura de la entrega (.zip)

```
ppe-vision/
├── configs/        # train.yaml, inference.yaml, rules.yaml (ROI, severidad, umbrales)
├── src/ppe/
│   ├── data/       # validación, EDA, splits, augmentations
│   ├── train.py    # semilla, metadata, MLflow
│   ├── evaluate.py # P/R/F1, mAP, PR curve, confusión, FPS
│   ├── inference/  # pipeline, NMS/dedup, logging, errores por archivo
│   ├── rules/      # roi.py, association.py, persistence.py, events.py
│   ├── persistence/# writers JSONL versionados
│   ├── monitoring/ # metrics.json operacional
│   └── active_learning.py
├── tests/          # ROI, asociación, dedup, severidad
├── notebooks/      # EDA, errores, resultados
├── docs/           # DECISIONS.md, reglas de negocio
├── requirements.txt
└── README.md
```



# License

This project is licensed under the MIT license. See the LICENSE file for details.
