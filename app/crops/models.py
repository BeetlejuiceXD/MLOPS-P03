"""Contratos del motor de crops (D01-07 / P3-04).

No es un contrato JSON v1.0 congelado (`presentation.contracts`): este ticket entrega
el motor probado con fixtures, todavía sin acreditar el dataset real ni una ubicación
de persistencia (`data/crops/` está excluido de Git — ver `tests/_repo_hygiene_rules.py`
de D01-06 — hasta que se decida su seguimiento, mismo criterio que `splits/README.md`
antes de P2-45 con `splits.json`).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CropExclusionReason = Literal[
    "degenerate_bbox",
    "out_of_bounds",
    "missing_image",
    "unknown_category",
]

# Clases del clasificador, congeladas en el protocolo de D01-03 (issue #33): una
# categoría COCO adicional (p. ej. "horse"), aunque esté bien declarada, no es una
# clase del contrato de P3 y no debe producir un crop.
FROZEN_CLASSES: frozenset[str] = frozenset({"cat", "dog"})


class CropModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Crop(CropModel):
    """Una muestra del clasificador: una caja COCO válida ya convertida a recorte.

    `crop_id` es hoy el mismo valor que `annotation_id`: una caja válida produce
    exactamente un crop (P3-04), así que reutilizar el id ya único evita inventar
    un esquema nuevo. Se expone como campo propio para no acoplar a los llamadores
    a esa igualdad si en el futuro un crop deja de ser 1:1 con su anotación.
    """

    crop_id: Annotated[int, Field(ge=0)]
    image_id: Annotated[int, Field(ge=0)]
    annotation_id: Annotated[int, Field(ge=0)]
    category_id: Annotated[int, Field(ge=0)]
    category_name: Annotated[str, Field(min_length=1)]
    # [x, y, width, height] tal cual el COCO original, en punto flotante — procedencia.
    bbox_original: Annotated[list[float], Field(min_length=4, max_length=4)]
    # [x0, y0, x1, y1] en píxeles enteros, tras floor/ceil, dentro de la imagen real.
    bbox_pixels: Annotated[list[int], Field(min_length=4, max_length=4)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class CropExclusion(CropModel):
    """Una anotación que no produjo un crop, con su motivo auditable."""

    annotation_id: Annotated[int, Field(ge=0)]
    image_id: Annotated[int, Field(ge=0)]
    reason: CropExclusionReason
    detail: Annotated[str, Field(min_length=1)]


class CropResult(CropModel):
    """Salida completa de `generate_crops`: nunca se pierde una anotación en silencio."""

    crops: list[Crop] = Field(default_factory=list)
    exclusions: list[CropExclusion] = Field(default_factory=list)
