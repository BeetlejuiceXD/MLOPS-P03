"""Tier 1 — combina los lotes reales de anotaciones en un solo CocoDataset.

Cada `annotations-lote-*.json` en `data/raw/annotations/` es autocontenido
(ver app/ingestion/README.md); este módulo es el único punto donde se
juntan antes de validarse como un solo documento. Si dos lotes reintroducen
el bug de [[project_dataset_id_collision_bug]] (ids repetidos entre
archivos), `CocoDataset.model_validate` lo rechaza aquí, no río abajo.
"""

import json
from pathlib import Path

from ingestion.models import CocoDataset


def load_dataset(annotations_dir: Path) -> CocoDataset:
    """Lee y valida todos los `*.json` de `annotations_dir` como un solo dataset.

    Las categorías se deduplican por id, quedándose con la primera aparición.
    Si dos lotes usan el mismo id de categoría para
    nombres distintos, es una colisión real (no una simple repetición del
    seeder) y se rechaza aquí en vez de mezclar silenciosamente anotaciones
    de una clase con el nombre de otra.
    """
    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No se encontraron lotes de anotaciones en {annotations_dir}")

    images = []
    annotations = []
    categories_by_id: dict[int, dict] = {}
    categories_source: dict[int, Path] = {}
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        images.extend(raw["images"])
        annotations.extend(raw["annotations"])
        for category in raw["categories"]:
            existing = categories_by_id.get(category["id"])
            if existing is not None and existing["name"] != category["name"]:
                raise ValueError(
                    f"category id={category['id']} tiene nombres distintos: "
                    f"'{existing['name']}' en {categories_source[category['id']]} vs "
                    f"'{category['name']}' en {path}"
                )
            categories_by_id.setdefault(category["id"], category)
            categories_source.setdefault(category["id"], path)

    return CocoDataset.model_validate(
        {
            "images": images,
            "annotations": annotations,
            "categories": list(categories_by_id.values()),
        }
    )


def load_image_bytes(coco: CocoDataset, images_dir: Path) -> dict[int, bytes]:
    """Lee del disco los binarios de cada imagen declarada en `coco`.

    Compartido por `presentation.gate` y `presentation.release`: ambos
    necesitan los mismos bytes por `image_id` para correr pHash. Falla rápido
    (`FileNotFoundError`) ante un archivo ausente: para esos dos consumidores,
    un archivo faltante es un problema de todo el dataset, no de una imagen
    aislada — no hay una noción de "excluir una imagen y seguir".
    """
    return {image.id: (images_dir / image.file_name).read_bytes() for image in coco.images}


def load_image_contents(coco: CocoDataset, images_dir: Path) -> dict[int, bytes]:
    """Como `load_image_bytes`, pero tolera archivos ausentes/ilegibles: los omite
    del resultado en vez de lanzar.

    Lo usa `presentation.crops_report` (D02-02): el motor de crops
    (`crops.engine.generate_crops`) trata un `image_id` sin entrada en
    `image_contents` como `missing_image` y excluye esa anotación puntual, en
    vez de abortar la generación completa — justo lo contrario del criterio de
    `load_image_bytes` arriba.
    """
    contents: dict[int, bytes] = {}
    for image in coco.images:
        try:
            contents[image.id] = (images_dir / image.file_name).read_bytes()
        except OSError:
            continue
    return contents
