"""D01-07 — motor de crops COCO deterministas (P3-04).

Casos conocidos con imágenes sintéticas reales (no simuladas): cada assert de
geometría se verifica contra los píxeles reales de la imagen decodificada, no
contra `width`/`height` declarados en el COCO. `CocoDataset.model_validate`
(ingestion/models.py) rechazaría un lote entero con una sola caja degenerada;
`generate_crops` recibe el COCO crudo (`dict`) precisamente para no tener esa
propiedad — cada anotación se acepta o se excluye de forma independiente.
"""

import glob
import io
import json
import math
import os

import pytest
from crops.engine import generate_crops
from crops.models import CropResult
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _image_bytes(width=64, height=64, seed=0):
    """Imagen JPEG real de tamaño arbitrario, con estructura (no un cuadro sólido)."""
    image = Image.new("RGB", (width, height), color=(20, 20, 20))
    x0 = (seed * 11) % max(width - 20, 1)
    y0 = (seed * 17) % max(height - 20, 1)
    image.paste((220, 220, 220), (x0, y0, x0 + min(20, width - x0), y0 + min(20, height - y0)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _annotation(id_, image_id, category_id, bbox):
    _x, _y, w, h = bbox
    return {
        "id": id_,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [],
    }


def _coco(images, annotations, categories=None):
    categories = categories or [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}]
    return {"images": images, "annotations": annotations, "categories": categories}


def _one_image_dataset(bbox, *, image_id=1, category_id=3, width=64, height=64, seed=0):
    coco = _coco(
        images=[
            {"id": image_id, "file_name": f"img{image_id}.jpg", "width": width, "height": height}
        ],
        annotations=[_annotation(1, image_id, category_id, bbox)],
    )
    contents = {image_id: _image_bytes(width, height, seed)}
    return coco, contents


def _only_exclusion_reason(result: CropResult) -> str:
    assert result.crops == []
    assert len(result.exclusions) == 1
    return result.exclusions[0].reason


# --- crop válido: geometría, redondeo y procedencia -------------------------------


def test_valid_integer_bbox_produces_crop_with_expected_pixel_geometry():
    coco, contents = _one_image_dataset([10, 5, 20, 30], category_id=4)

    result = generate_crops(coco, contents)

    assert result.exclusions == []
    assert len(result.crops) == 1
    crop = result.crops[0]
    assert crop.crop_id == 1
    assert crop.annotation_id == 1
    assert crop.image_id == 1
    assert crop.category_id == 4
    assert crop.category_name == "cat"
    assert crop.bbox_original == [10.0, 5.0, 20.0, 30.0]
    assert crop.bbox_pixels == [10, 5, 30, 35]
    assert crop.width == 20
    assert crop.height == 30


def test_fractional_bbox_rounds_floor_for_origin_and_ceil_for_extent():
    # x0=floor(10.2)=10, y0=floor(5.9)=5, x1=ceil(10.2+20.1)=31, y1=ceil(5.9+30.05)=36
    coco, contents = _one_image_dataset([10.2, 5.9, 20.1, 30.05])

    result = generate_crops(coco, contents)

    crop = result.crops[0]
    assert crop.bbox_pixels == [10, 5, 31, 36]
    assert crop.width == 21
    assert crop.height == 31


def test_crop_pixel_box_matches_the_real_decoded_image_region():
    """Verificación contra píxeles reales: la región que produce bbox_pixels
    existe de verdad dentro de la imagen decodificada, no solo en los metadatos."""
    width, height = 80, 60
    coco, contents = _one_image_dataset([12.4, 8.6, 15.3, 10.2], width=width, height=height)

    crop = generate_crops(coco, contents).crops[0]
    x0, y0, x1, y1 = crop.bbox_pixels

    with Image.open(io.BytesIO(contents[1])) as image:
        assert image.size == (width, height)
        region = image.crop((x0, y0, x1, y1))
    assert region.size == (crop.width, crop.height)
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def test_two_annotations_on_the_same_image_produce_two_independent_crops():
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[
            _annotation(1, 1, 3, [0, 0, 10, 10]),
            _annotation(2, 1, 4, [20, 20, 10, 10]),
        ],
    )
    contents = {1: _image_bytes()}

    result = generate_crops(coco, contents)

    assert result.exclusions == []
    assert {c.crop_id for c in result.crops} == {1, 2}
    assert {c.category_name for c in result.crops} == {"dog", "cat"}


def test_generation_is_deterministic_for_the_same_input():
    coco, contents = _one_image_dataset([1, 2, 3, 4])

    first = generate_crops(coco, contents)
    second = generate_crops(coco, contents)

    assert first == second


# --- orden determinista por annotation_id (D01-03, revisión de PR #48) -----------


def test_crops_and_exclusions_are_ordered_by_annotation_id_regardless_of_input_order():
    """El COCO trae las anotaciones en cualquier orden (id 30 antes que 5, etc.):
    la salida siempre queda ascendente por annotation_id, mezclando válidas y
    excluidas, sin depender del orden en que aparecen en 'annotations'."""
    coco = _coco(
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
        ],
        annotations=[
            _annotation(30, 1, 3, [0, 0, 10, 10]),  # válida, id alto, primero en la entrada
            _annotation(5, 1, 4, [-1, 0, 10, 10]),  # excluida (out_of_bounds), id bajo
            _annotation(17, 2, 3, [0, 0, 10, 10]),  # válida, id intermedio
            _annotation(2, 2, 999, [0, 0, 10, 10]),  # excluida (unknown_category), id más bajo
        ],
    )
    contents = {1: _image_bytes(seed=1), 2: _image_bytes(seed=2)}

    result = generate_crops(coco, contents)

    assert [c.annotation_id for c in result.crops] == [17, 30]
    assert [e.annotation_id for e in result.exclusions] == [2, 5]


def test_reordering_the_input_annotations_does_not_change_the_result():
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[
            _annotation(9, 1, 3, [0, 0, 10, 10]),
            _annotation(1, 1, 4, [10, 10, 10, 10]),
            _annotation(5, 1, 42, [0, 0, 10, 10]),  # unknown_category
        ],
    )
    contents = {1: _image_bytes()}
    shuffled = _coco(images=coco["images"], annotations=list(reversed(coco["annotations"])))

    assert generate_crops(coco, contents) == generate_crops(shuffled, contents)


# --- bordes: exactamente en el límite es válido ------------------------------------


def test_bbox_touching_the_real_image_border_exactly_is_valid():
    coco, contents = _one_image_dataset([54, 34, 10, 30], width=64, height=64)  # 54+10=64, 34+30=64

    result = generate_crops(coco, contents)

    assert result.exclusions == []
    assert result.crops[0].bbox_pixels == [54, 34, 64, 64]


# --- out_of_bounds: total y PARCIALMENTE fuera se excluyen (no se recortan) --------


def test_bbox_one_pixel_beyond_the_real_width_is_excluded_not_clipped():
    coco, contents = _one_image_dataset([60, 0, 5, 5], width=64, height=64)  # 60+5=65 > 64

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "out_of_bounds"


def test_bbox_partially_outside_on_the_bottom_edge_is_excluded():
    coco, contents = _one_image_dataset([0, 60, 5, 10], width=64, height=64)  # 60+10=70 > 64

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "out_of_bounds"


def test_bbox_fully_outside_the_real_image_is_excluded():
    coco, contents = _one_image_dataset([100, 100, 10, 10], width=64, height=64)

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "out_of_bounds"


def test_negative_origin_is_out_of_bounds():
    coco, contents = _one_image_dataset([-1, 0, 10, 10])

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "out_of_bounds"


def test_bounds_are_checked_against_real_pixels_not_declared_coco_dimensions():
    """El COCO declara 200x200 pero la imagen real decodificada es 64x64: manda lo real."""
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 200, "height": 200}],
        annotations=[_annotation(1, 1, 3, [50, 50, 100, 100])],  # cabría en 200x200, no en 64x64
    )
    contents = {1: _image_bytes(64, 64)}

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "out_of_bounds"


# --- degenerate_bbox ----------------------------------------------------------------


@pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [0, 0, -5, 10], [0, 0, 10, 0], [0, 0, 10, -5]])
def test_non_positive_width_or_height_is_degenerate(bbox):
    coco, contents = _one_image_dataset(bbox)

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "degenerate_bbox"


@pytest.mark.parametrize("bbox", [[0, 0, 10], [0, 0, 10, 10, 10], "not-a-list"])
def test_malformed_bbox_shape_is_degenerate(bbox):
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": bbox,
                "area": 1.0,
                "iscrowd": 0,
                "segmentation": [],
            }
        ],
    )
    contents = {1: _image_bytes()}

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "degenerate_bbox"


def test_non_finite_bbox_value_is_degenerate():
    coco, contents = _one_image_dataset([0, 0, math.inf, 10])

    reason = _only_exclusion_reason(generate_crops(coco, contents))

    assert reason == "degenerate_bbox"


def test_tiny_positive_width_still_rounds_to_at_least_one_pixel():
    """floor(x) < ceil(x + width) siempre que width > 0: nunca colapsa a 0 px,
    ni en el caso límite de un ancho positivo casi nulo."""
    coco, contents = _one_image_dataset([10.0, 10.0, 1e-9, 5])

    crop = generate_crops(coco, contents).crops[0]

    assert crop.width >= 1
    assert crop.bbox_pixels[2] > crop.bbox_pixels[0]


# --- missing_image --------------------------------------------------------------


def test_annotation_without_recovered_image_bytes_is_missing_image():
    coco, _ = _one_image_dataset([0, 0, 10, 10])

    result = generate_crops(coco, {})  # sin bytes para image_id=1

    reason = _only_exclusion_reason(result)
    assert reason == "missing_image"
    assert "sin binario recuperado" in result.exclusions[0].detail


def test_annotation_referencing_an_undeclared_image_id_is_missing_image():
    """image_id huérfano: ni siquiera está en coco['images']. Se prueba con bytes
    disponibles bajo ese mismo id (image_contents no exige que esté declarado) para
    aislar esta regla de la de "sin binario recuperado": sin la comprobación contra
    `coco['images']`, este caso resolvería (mal) como válido."""
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[_annotation(1, 999, 3, [0, 0, 10, 10])],  # image_id huérfano
    )
    contents = {1: _image_bytes(), 999: _image_bytes()}  # bytes SÍ disponibles para 999

    result = generate_crops(coco, contents)

    reason = _only_exclusion_reason(result)
    assert reason == "missing_image"
    assert "no está declarado" in result.exclusions[0].detail


def test_undecodable_image_bytes_are_missing_image():
    coco, _ = _one_image_dataset([0, 0, 10, 10])

    reason = _only_exclusion_reason(generate_crops(coco, {1: b"no es una imagen"}))

    assert reason == "missing_image"


def test_truncated_image_with_a_readable_header_is_missing_image_not_valid():
    """Un JPEG al que le faltan los últimos bytes conserva una cabecera legible
    (`Image.open(...).size` da el tamaño correcto) pero sus píxeles no son
    decodificables: no debe colarse como crop válido solo porque el tamaño
    "parece" correcto (revisión de PR #48)."""
    full = _image_bytes()
    truncated = full[:-20]
    assert len(truncated) < len(full)
    with Image.open(io.BytesIO(truncated)) as header_only:
        assert header_only.size == (64, 64)  # la cabecera sí es legible
        with pytest.raises(OSError):
            header_only.load()  # pero los píxeles no decodifican completos
    coco, _ = _one_image_dataset([0, 0, 10, 10])

    reason = _only_exclusion_reason(generate_crops(coco, {1: truncated}))

    assert reason == "missing_image"


# --- unknown_category -------------------------------------------------------------


def test_annotation_with_undeclared_category_is_unknown_category():
    coco, contents = _one_image_dataset([0, 0, 10, 10], category_id=999)

    result = generate_crops(coco, contents)

    reason = _only_exclusion_reason(result)
    assert reason == "unknown_category"
    assert "no está declarada" in result.exclusions[0].detail


def test_unknown_category_is_reported_even_without_image_bytes():
    """unknown_category no depende de poder decodificar la imagen."""
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[_annotation(1, 1, 999, [0, 0, 10, 10])],
    )

    reason = _only_exclusion_reason(generate_crops(coco, {}))

    assert reason == "unknown_category"


def test_a_declared_category_outside_the_frozen_classes_is_unknown_category():
    """ "horse" está bien declarada en 'categories' (a diferencia de category_id=999
    del test anterior), pero no pertenece a las clases congeladas del clasificador
    en #33 (cat/dog): no debe producir un crop válido (revisión de PR #48)."""
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[_annotation(1, 1, 7, [0, 0, 10, 10])],
        categories=[{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}, {"id": 7, "name": "horse"}],
    )
    contents = {1: _image_bytes()}

    result = generate_crops(coco, contents)

    reason = _only_exclusion_reason(result)
    assert reason == "unknown_category"
    assert "horse" in result.exclusions[0].detail


def test_allowed_categories_can_be_overridden_for_a_different_class_contract():
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        annotations=[_annotation(1, 1, 7, [0, 0, 10, 10])],
        categories=[{"id": 7, "name": "horse"}],
    )
    contents = {1: _image_bytes()}

    result = generate_crops(coco, contents, allowed_categories=frozenset({"horse"}))

    assert result.exclusions == []
    assert result.crops[0].category_name == "horse"


# --- mezcla: válidos y excluidos conviven, nada se pierde en silencio -------------


def test_mixed_dataset_keeps_valid_crops_and_reports_every_exclusion_with_cause():
    coco = _coco(
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
        ],
        annotations=[
            _annotation(1, 1, 3, [0, 0, 10, 10]),  # válida
            _annotation(2, 1, 4, [0, 0, -1, 10]),  # degenerate_bbox
            _annotation(3, 2, 3, [200, 200, 10, 10]),  # out_of_bounds
            _annotation(4, 3, 3, [0, 0, 10, 10]),  # missing_image (image_id 3 no declarado)
            _annotation(5, 1, 42, [0, 0, 10, 10]),  # unknown_category
        ],
    )
    contents = {1: _image_bytes(seed=1), 2: _image_bytes(seed=2)}

    result = generate_crops(coco, contents)

    assert len(coco["annotations"]) == len(result.crops) + len(result.exclusions)
    assert {c.annotation_id for c in result.crops} == {1}
    assert {(e.annotation_id, e.reason) for e in result.exclusions} == {
        (2, "degenerate_bbox"),
        (3, "out_of_bounds"),
        (4, "missing_image"),
        (5, "unknown_category"),
    }
    assert all(e.detail for e in result.exclusions)


def test_empty_annotations_produce_empty_result_not_an_error():
    coco = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}], annotations=[]
    )

    result = generate_crops(coco, {1: _image_bytes()})

    assert result.crops == []
    assert result.exclusions == []


# --- contra el dataset real recuperado de DVC/S3 ----------------------------------

real_data = pytest.mark.skipif(
    not os.path.isdir(os.path.join(REPO_ROOT, "data", "raw", "images")),
    reason="datos reales no recuperados (dvc pull -r prod)",
)


@real_data
def test_real_v0_1_1_produces_zero_exclusions_and_matches_manual_audit():
    images, annotations, categories = [], [], {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "data", "raw", "annotations", "*.json"))):
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
        images += raw["images"]
        annotations += raw["annotations"]
        for category in raw["categories"]:
            categories.setdefault(category["id"], category)
    coco = {"images": images, "annotations": annotations, "categories": list(categories.values())}
    images_dir = os.path.join(REPO_ROOT, "data", "raw", "images")
    contents = {}
    for image in images:
        with open(os.path.join(images_dir, image["file_name"]), "rb") as file:
            contents[image["id"]] = file.read()

    result = generate_crops(coco, contents)

    assert result.exclusions == []
    assert len(result.crops) == 668
    per_class = {}
    for crop in result.crops:
        per_class[crop.category_name] = per_class.get(crop.category_name, 0) + 1
    assert per_class == {"cat": 343, "dog": 325}
    originals = {}
    for crop in result.crops:
        originals.setdefault(crop.category_name, set()).add(crop.image_id)
    assert {name: len(ids) for name, ids in originals.items()} == {"cat": 301, "dog": 300}
