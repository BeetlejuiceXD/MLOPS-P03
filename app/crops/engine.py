"""Motor de crops COCO deterministas (D01-07 / P3-04).

Convierte cada anotación en un `Crop` (una muestra de un objeto, con etiqueta e
IDs) o la excluye con un motivo auditable. Opera sobre el COCO crudo (`dict`),
antes de `ingestion.models.CocoDataset.model_validate`: esa validación es
estricta y rechaza el lote *completo* ante una sola caja degenerada o una
categoría desconocida (ver `ingestion/README.md`), mientras que aquí cada
anotación se evalúa de forma independiente — una caja mala no descarta las
demás, y ninguna se pierde en silencio: termina en `crops` o en `exclusions`.

Sigue el patrón de `analyzers.duplicates`/`splits.stratified`: recibe los bytes
de imagen ya cargados (`image_contents`), no rutas de archivo ni clientes de
MinIO/S3. Es una función pura, sin I/O.
"""

from io import BytesIO
from math import ceil, floor, isfinite

from PIL import Image, UnidentifiedImageError

from crops.models import FROZEN_CLASSES, Crop, CropExclusion, CropExclusionReason, CropResult


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _real_pixel_size(data: bytes) -> tuple[int, int] | None:
    """Tamaño real decodificado, o `None` si los bytes no son una imagen válida.

    `Image.open` solo lee la cabecera (rápido, perezoso): un JPEG con la cabecera
    intacta pero los píxeles truncados reporta un `.size` correcto sin que eso
    signifique que el binario es utilizable. `image.load()` fuerza la
    decodificación completa y es lo que de verdad detecta el truncamiento
    (`OSError`, con `ImageFile.LOAD_TRUNCATED_IMAGES` en su valor por defecto).
    """
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.size
    except (UnidentifiedImageError, OSError):
        return None


def generate_crops(
    coco: dict,
    image_contents: dict[int, bytes],
    *,
    allowed_categories: frozenset[str] | None = None,
) -> CropResult:
    """Genera un `Crop` por anotación válida de `coco`, o la excluye con motivo.

    `coco` trae `images`/`annotations`/`categories` en la forma cruda de un COCO
    (no requiere pasar `CocoDataset.model_validate`). `image_contents` cubre los
    `image_id` cuyo binario sí pudo recuperarse; los que faltan, o cuyos bytes no
    decodifican como imagen, producen `missing_image` — igual que un `image_id`
    huérfano que no está declarado en `coco["images"]`.

    `allowed_categories` son las clases del clasificador (por defecto
    `FROZEN_CLASSES`, congeladas en #33): una categoría COCO adicional, aunque
    esté bien declarada (p. ej. "horse"), es `unknown_category` si su nombre no
    pertenece a este conjunto — el motor no produce crops de clases fuera del
    contrato, aunque el dataset las incluya.

    Los límites de la caja se verifican contra el tamaño real de la imagen
    decodificada, no contra `width`/`height` declarados en el COCO: unos
    metadatos desactualizados no deben validar una caja que en realidad se sale
    de la imagen. Una caja parcialmente fuera se excluye igual que una caja
    totalmente fuera (D01-03: no se recorta al borde, para no alterar la
    geometría anotada).

    Los resultados salen ordenados por `annotation_id` (D01-03): se procesa
    `coco["annotations"]` en ese orden, así que tanto `crops` como `exclusions`
    quedan ascendentes, sin depender del orden de entrada del COCO.
    """
    allowed = FROZEN_CLASSES if allowed_categories is None else allowed_categories
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    declared_images = {image["id"] for image in coco["images"]}

    crops: list[Crop] = []
    exclusions: list[CropExclusion] = []

    def exclude(
        annotation_id: int, image_id: int, reason: CropExclusionReason, detail: str
    ) -> None:
        exclusions.append(
            CropExclusion(
                annotation_id=annotation_id, image_id=image_id, reason=reason, detail=detail
            )
        )

    for annotation in sorted(coco["annotations"], key=lambda item: item["id"]):
        annotation_id = annotation["id"]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]

        category_name = categories.get(category_id)
        if category_name is None:
            exclude(
                annotation_id,
                image_id,
                "unknown_category",
                f"category_id={category_id} no está declarada en 'categories'",
            )
            continue
        if category_name not in allowed:
            exclude(
                annotation_id,
                image_id,
                "unknown_category",
                f"category_id={category_id} (name={category_name!r}) no pertenece a las "
                f"clases congeladas del clasificador {sorted(allowed)!r}",
            )
            continue

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(_finite_number(value) for value in bbox)
        ):
            exclude(
                annotation_id,
                image_id,
                "degenerate_bbox",
                f"bbox debe tener 4 valores numéricos finitos [x, y, width, height]; "
                f"se recibió {bbox!r}",
            )
            continue

        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            exclude(
                annotation_id,
                image_id,
                "degenerate_bbox",
                f"width={width} y height={height} deben ser positivos",
            )
            continue

        if image_id not in declared_images:
            exclude(
                annotation_id,
                image_id,
                "missing_image",
                f"image_id={image_id} no está declarado en 'images' (huérfano)",
            )
            continue

        data = image_contents.get(image_id)
        if data is None:
            exclude(
                annotation_id,
                image_id,
                "missing_image",
                f"image_id={image_id}: sin binario recuperado (¿falta `dvc pull` o el archivo?)",
            )
            continue

        real_size = _real_pixel_size(data)
        if real_size is None:
            exclude(
                annotation_id,
                image_id,
                "missing_image",
                f"image_id={image_id}: el binario recuperado no es una imagen decodificable",
            )
            continue

        real_width, real_height = real_size
        if x < 0 or y < 0 or x + width > real_width or y + height > real_height:
            exclude(
                annotation_id,
                image_id,
                "out_of_bounds",
                f"bbox=[{x}, {y}, {width}, {height}] excede la imagen real "
                f"{real_width}x{real_height} (incluye estar parcialmente fuera)",
            )
            continue

        # floor(x) < x + width (width > 0), y x + width <= real_width ya está garantizado
        # arriba: ceil(x + width) siempre queda por encima de floor(x), así que el crop
        # mide al menos 1 px por lado. `Crop.width`/`Crop.height` (`Field(gt=0)`) son la
        # red de seguridad del modelo si algún día esta garantía deja de sostenerse.
        x0, y0 = floor(x), floor(y)
        x1, y1 = ceil(x + width), ceil(y + height)

        crops.append(
            Crop(
                crop_id=annotation_id,
                image_id=image_id,
                annotation_id=annotation_id,
                category_id=category_id,
                category_name=category_name,
                bbox_original=[float(x), float(y), float(width), float(height)],
                bbox_pixels=[x0, y0, x1, y1],
                width=x1 - x0,
                height=y1 - y0,
            )
        )

    return CropResult(crops=crops, exclusions=exclusions)
