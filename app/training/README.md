# D01-04 — TrainingConfig, factoría CNN y preprocessing compartido

`TrainingConfig` (Pydantic v2, serializable) fija los siete parámetros del
ticket más la arquitectura ratificada en el protocolo de D01-03 (issue #33):
ResNet18 preentrenada. `build_model(config)` construye el modelo con la cabeza
y las capas congeladas/entrenables correspondientes; `preprocessing.py` expone
el mismo transform base para training y evaluación/inferencia, con augmentation
solo en train.

```python
from training.config import TrainingConfig
from training.model import build_model, describe_trainable_layers
from training.preprocessing import preprocess_image

config = TrainingConfig(seed=7, trainable_layers="last_block", hidden_layers=1)
model = build_model(config)
describe_trainable_layers(model)  # capas congeladas/entrenables, para evidencia
```

## Class map

El índice de salida es alfabético sobre `crops.models.FROZEN_CLASSES`
(`cat=0, dog=1`), no el `category_id` de COCO (dog=3, cat=4) — así el
clasificador no depende de un detalle de ingesta ajeno a su contrato.

## Qué no hace (todavía)

- No entrena ni carga el dataset real de crops (eso es un ticket posterior).
- No registra en MLflow ni publica en S3 (Hannah, D01-0X).
- No define el manifest 70/20/10 (P3-06, D02+) — esto usa `image_size`/
  `augmentation` como hiperparámetros, no un split real.