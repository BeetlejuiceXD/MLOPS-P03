"""Class map del clasificador (D01-04): índice <-> nombre de clase.

Orden alfabético sobre `crops.models.FROZEN_CLASSES` (congeladas en el protocolo
de D01-03, issue #33) — NO el `category_id` interno de COCO, que daría dog=0/
cat=1 (dog=3, cat=4 en el COCO crudo). Se fija así para que el índice de salida
del clasificador sea independiente de un detalle de ingesta ajeno al contrato
del modelo.
"""

from crops.models import FROZEN_CLASSES

CLASS_MAP: dict[str, int] = {name: index for index, name in enumerate(sorted(FROZEN_CLASSES))}
INDEX_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)


def class_map() -> dict[str, int]:
    """Copia defensiva; el dict del módulo nunca se muta desde fuera."""
    return dict(CLASS_MAP)
