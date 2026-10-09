# Base de Datos - Personal Protective Equipment (PPE) Dataset

![img](../../reports/eda/samples_with_boxes.png)

## Procedencia

| Campo             | Valor                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| Nombre            | Personal Protective Equipment (PPE) Dataset                                                                           |
| Autor / fuente    | `ndomalau` en Kaggle                                                                                                |
| URL               | https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset                                    |
| Descarga          | `kaggle datasets download -d ndomalau/personal-protective-equipment-ppe-dataset -p src/data/kaggle_dataset --unzip` |
| Licencia          | Attribution 4.0 International (CC BY 4.0)                                                                             |
| Fecha de descarga | 08/10/2025                                                                                                            |
| Formato           | YOLO (`images/` + `labels/` por partición, )                                                                     |
| Resolución       | 640x640                                                                                                               |

## Clases

`['helmet', 'no_helmet', 'no_vest', 'person', 'vest']` (nc = 5). Las clases `no_helmet` y `no_vest` son
ausencias anotadas explícitamente; la clase `person` es la que se evalúa contra el ROI.

## Estructura esperada

```
src/data/kaggle_dataset/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

Las rutas `../train/images` del `data.yaml` no resuelven desde esa carpeta; los scripts buscan automáticamente
`<partición>/images`.

## Limitaciones

- _<Cámaras/escenas representadas, iluminación, ángulos, número de personas por imagen.>_
- _<Sesgo de dominio frente a las cámaras de producción.>_
- La verdad terreno de eventos derivada de anotaciones no es una revisión humana independiente.

## Hallazgos del EDA

Ubicación:

- Análisis: reports/eda/dataset_report.json
- Visualizaciones: reports/eda
