"""D02-04 — generador/validador del manifest P3 (grupos indivisibles, proporciones
sobre crops). Prueba `manifest.generator` en aislamiento, con `Crop` sintéticos —
no requiere datos reales ni el motor de D01-07."""

import pytest
from crops.models import Crop
from manifest.generator import (
    ManifestValidationError,
    generate_manifest,
    verify_manifest_assignment,
)
from manifest.models import load_manifest_config
from pydantic import ValidationError

from splits.models import SplitsConfig


def _crop(crop_id, image_id, category_name, category_id=None):
    category_id = category_id if category_id is not None else (3 if category_name == "dog" else 4)
    return Crop(
        crop_id=crop_id,
        image_id=image_id,
        annotation_id=crop_id,
        category_id=category_id,
        category_name=category_name,
        bbox_original=[0.0, 0.0, 10.0, 10.0],
        bbox_pixels=[0, 0, 10, 10],
        width=10,
        height=10,
    )


def _crops_one_per_image(n_cat, n_dog, start_id=1):
    crops = []
    crop_id = start_id
    for _ in range(n_cat):
        crops.append(_crop(crop_id, crop_id, "cat"))
        crop_id += 1
    for _ in range(n_dog):
        crops.append(_crop(crop_id, crop_id, "dog"))
        crop_id += 1
    return crops


def _config(**overrides):
    values = {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42, **overrides}
    return SplitsConfig(**values)


def _all_ids(result):
    return sorted(i for ids in result.assignments.values() for i in ids)


# --- caso base: cobertura, exclusividad, determinismo -------------------------------


def test_covers_every_crop_exactly_once():
    crops = _crops_one_per_image(60, 40)

    result = generate_manifest(crops, _config(), duplicate_pairs=())

    assert _all_ids(result) == sorted(c.crop_id for c in crops)
    seen = set()
    for ids in result.assignments.values():
        assert not seen & set(ids)
        seen |= set(ids)


def test_same_seed_produces_the_same_assignment():
    crops = _crops_one_per_image(60, 40)

    first = generate_manifest(crops, _config(seed=42), duplicate_pairs=())
    second = generate_manifest(crops, _config(seed=42), duplicate_pairs=())

    assert first.assignments == second.assignments
    assert first.originals == second.originals
    assert first.groups == second.groups


def test_real_manifest_yaml_default_config_loads_and_runs():
    """El config por defecto (`manifest/manifest.yaml`, 70/20/10, seed 42, #33)
    corre igual que cualquier otro; no está codificado a mano en los tests."""
    crops = _crops_one_per_image(70, 60)

    result = generate_manifest(crops, load_manifest_config(), duplicate_pairs=())

    assert set(_all_ids(result)) == {c.crop_id for c in crops}


def test_ratios_are_approximately_respected_on_a_balanced_dataset():
    crops = _crops_one_per_image(350, 350)  # 700 crops, similar de escala a v0.1.1

    result = generate_manifest(crops, _config(), duplicate_pairs=())

    total = sum(len(ids) for ids in result.assignments.values())
    for name, target in (("train", 0.7), ("val", 0.2), ("test", 0.1)):
        assert abs(len(result.assignments[name]) / total - target) <= 0.05


# --- grupos indivisibles: mismo original, y duplicados/near-duplicates -------------


def test_all_crops_of_the_same_original_stay_in_the_same_split():
    crops = _crops_one_per_image(20, 20)
    # La imagen 1 (ya "cat") gana 2 crops más: 3 crops del mismo original.
    crops.append(_crop(1001, 1, "cat"))
    crops.append(_crop(1002, 1, "dog"))

    result = generate_manifest(crops, _config(), duplicate_pairs=())

    owners = {name for name, ids in result.assignments.items() if {1, 1001, 1002} & set(ids)}
    assert len(owners) == 1  # las tres cajas de la imagen 1 quedaron en la misma partición
    (owner,) = owners
    assert {1, 1001, 1002} <= set(result.assignments[owner])


def test_near_duplicate_images_are_grouped_indivisibly():
    crops = _crops_one_per_image(20, 20)  # imágenes cat: 1..20 ; imágenes dog: 21..40
    # Las imágenes 5 y 25 son near-duplicates entre sí (par pHash, no mismo id).
    pairs = ({"image_id_a": 5, "image_id_b": 25, "hamming_distance": 2, "similarity": 0.97},)

    result = generate_manifest(crops, _config(), duplicate_pairs=pairs)

    # Comprobación directa sobre `result.groups` (no sobre a dónde los mandó el
    # costo): sin la unión, 5 y 25 serían grupos de 1 elemento cada uno; unidos,
    # deben aparecer juntos en el MISMO grupo de 2 elementos.
    group_with_5 = next(group for group in result.groups if 5 in group)
    assert group_with_5 == (5, 25)

    owner_5 = next(name for name, ids in result.assignments.items() if 5 in ids)
    owner_25 = next(name for name, ids in result.assignments.items() if 25 in ids)
    assert owner_5 == owner_25


def test_duplicate_pair_referencing_an_image_outside_the_crop_universe_is_ignored():
    """Una imagen sin ningún crop válido (ya excluida por D01-07) no participa del
    manifest: un par que la mencione no debe fallar ni inventarle un grupo (caso sin
    ninguna cadena hacia un nodo real — ver los tests de transitividad para el caso
    con cadena, que sí debe unir a través de ella)."""
    crops = _crops_one_per_image(20, 20)
    pairs = ({"image_id_a": 5, "image_id_b": 99999, "hamming_distance": 0, "similarity": 1.0},)

    result = generate_manifest(crops, _config(), duplicate_pairs=pairs)

    assert 99999 not in {i for group in result.groups for i in group}


# --- transitividad de near-duplicates, incluso con un nodo intermedio sin crops ----


def test_transitive_near_duplicates_a_b_c_are_grouped_together():
    """A≈B y B≈C (pares distintos, sin un par directo A≈C): las tres deben terminar
    en el mismo grupo por transitividad, aunque las tres tengan crops válidos."""
    crops = _crops_one_per_image(20, 20)  # imágenes cat: 1..20 ; imágenes dog: 21..40
    pairs = (
        {"image_id_a": 5, "image_id_b": 25, "hamming_distance": 2, "similarity": 0.97},
        {"image_id_a": 25, "image_id_b": 10, "hamming_distance": 2, "similarity": 0.97},
    )

    result = generate_manifest(crops, _config(), duplicate_pairs=pairs)

    group = next(g for g in result.groups if 5 in g)
    assert group == (5, 10, 25)


def test_transitivity_survives_when_the_intermediate_image_has_no_valid_crops():
    """A≈B≈C, pero B fue excluida por D01-07/D02-02 (0 crops válidos): sin
    considerar a B como nodo puente, la unión A-B y B-C se ignoraría por completo
    (B no está en `image_ids`) y A/C terminarían en grupos — y posiblemente
    particiones — distintos, aunque en la realidad son casi-duplicados. La cadena
    debe seguir uniendo A y C a través de B, y B (sin crops) no debe aparecer en
    ningún grupo ni asignación del manifest (revisión de Heri en PR #54)."""
    crops = _crops_one_per_image(20, 20)  # imágenes cat: 1..20 ; imágenes dog: 21..40
    a, b, c = 5, 999, 25  # b=999 no tiene ningún crop: no aparece en `crops`
    pairs = (
        {"image_id_a": a, "image_id_b": b, "hamming_distance": 1, "similarity": 0.98},
        {"image_id_a": b, "image_id_b": c, "hamming_distance": 1, "similarity": 0.98},
    )

    result = generate_manifest(crops, _config(), duplicate_pairs=pairs)

    group_with_a = next(g for g in result.groups if a in g)
    assert group_with_a == (a, c)  # A y C en el mismo grupo; B no participa (no tiene crops)
    assert b not in {i for group in result.groups for i in group}

    owner_a = next(name for name, ids in result.assignments.items() if a in ids)
    owner_c = next(name for name, ids in result.assignments.items() if c in ids)
    assert owner_a == owner_c  # A y C en la MISMA partición, no repartidos por la cadena rota
    assert b not in {i for ids in result.assignments.values() for i in ids}


def test_crops_of_the_same_original_count_together_when_sizing_splits():
    """El tamaño que se balancea es CROPS, no imágenes (#33/D02-04): una imagen con
    varios crops debe pesar como varios crops al medir proporciones, no como 1.

    Una imagen "pesada" (30 crops de "dog": un solo grupo indivisible de tamaño 30),
    29 imágenes "dog" de 1 crop y 60 imágenes "cat" de 1 crop: 119 crops en total,
    con "cat" y "dog" repartidos en muchos grupos pequeños además del pesado, para
    que la presencia de ambas clases en val/test no dependa de dónde caiga ese
    grupo grande. Sin peso por crops, esa imagen pesaría 1 (no 30) y el tamaño real
    de cada partición no se acercaría al objetivo."""
    heavy_crops = [_crop(1000 + i, 1, "dog") for i in range(30)]
    light_dog_crops = [_crop(2000 + i, 2000 + i, "dog") for i in range(29)]
    light_cat_crops = [_crop(3000 + i, 3000 + i, "cat") for i in range(60)]
    crops = heavy_crops + light_dog_crops + light_cat_crops

    result = generate_manifest(
        crops, _config(train=0.6, val=0.2, test=0.2, seed=42), duplicate_pairs=()
    )

    # El grupo pesado (30 crops de la imagen 1) nunca queda partido entre particiones.
    heavy_ids = {c.crop_id for c in heavy_crops}
    counts_per_split = {name: len(heavy_ids & set(ids)) for name, ids in result.assignments.items()}
    assert sorted(counts_per_split.values()) == [0, 0, 30]

    # El tamaño de cada partición se acerca al objetivo medido en crops (119 total);
    # si el grupo pesado pesara 1 imagen en vez de 30 crops, el error sería enorme.
    total = sum(len(ids) for ids in result.assignments.values())
    assert total == len(crops)
    for name, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(len(result.assignments[name]) / total - target) <= 0.1


# --- casos negativos ----------------------------------------------------------------


def test_a_class_with_only_one_crop_cannot_appear_in_both_val_and_test():
    """Con una sola unidad indivisible de "cat" en todo el dataset, esa unidad solo
    puede caer en una partición: al menos otra queda sin "cat" y debe fallar."""
    crops = [_crop(1, 1, "cat"), *_crops_one_per_image(0, 40, start_id=100)]

    with pytest.raises(ManifestValidationError, match="cat"):
        generate_manifest(crops, _config(), duplicate_pairs=())


def test_no_crops_raises_instead_of_producing_an_empty_manifest():
    with pytest.raises(ManifestValidationError):
        generate_manifest([], _config(), duplicate_pairs=())


def test_fewer_than_three_groups_raises():
    crops = _crops_one_per_image(1, 1)  # 2 imágenes = 2 grupos, no alcanzan para 3 splits no vacíos

    with pytest.raises(ManifestValidationError, match="at least three independent"):
        generate_manifest(crops, _config(), duplicate_pairs=())


def test_malformed_duplicate_pair_raises():
    crops = _crops_one_per_image(20, 20)
    pairs = ({"image_id_a": 1, "wrong_key": 2},)

    with pytest.raises(ManifestValidationError, match="image_id_a"):
        generate_manifest(crops, _config(), duplicate_pairs=pairs)


def test_duplicate_pair_referencing_the_same_image_twice_raises():
    crops = _crops_one_per_image(20, 20)
    pairs = ({"image_id_a": 1, "image_id_b": 1, "hamming_distance": 0, "similarity": 1.0},)

    with pytest.raises(ManifestValidationError, match="distinct"):
        generate_manifest(crops, _config(), duplicate_pairs=pairs)


def test_invalid_ratios_are_rejected_even_if_the_caller_bypassed_pydantic():
    """`SplitsConfig` no es `frozen`; `generate_manifest` revalida al entrar (mismo
    criterio defensivo que `splits.stratified.split_dataset`)."""
    crops = _crops_one_per_image(20, 20)
    config = _config()
    config.train = (
        5.0  # SplitsConfig no valida en la asignación (validate_assignment no está activo).
    )

    with pytest.raises(ValidationError):
        generate_manifest(crops, config, duplicate_pairs=())


# --- verify_manifest_assignment: fuga deliberada y corrupción de la asignación -----


def test_verify_rejects_a_group_split_across_two_partitions():
    crop_ids = [1, 2, 3, 4, 5, 6]
    groups = [(1, 2), (3,), (4, 5, 6)]
    # El grupo (1, 2) queda partido entre train y val: fuga.
    assignments = {"train": (1, 3), "val": (2, 4, 5), "test": (6,)}
    crops_per_class = {name: {"cat": 1, "dog": 1} for name in ("train", "val", "test")}

    with pytest.raises(ManifestValidationError, match="Leakage"):
        verify_manifest_assignment(
            assignments,
            crop_ids=crop_ids,
            groups=groups,
            classes=["cat", "dog"],
            crops_per_class=crops_per_class,
        )


def test_verify_rejects_a_crop_id_assigned_twice():
    assignments = {"train": (1, 2), "val": (2, 3), "test": (4,)}
    crops_per_class = {name: {"cat": 1, "dog": 1} for name in ("train", "val", "test")}

    with pytest.raises(ManifestValidationError, match="more than once"):
        verify_manifest_assignment(
            assignments,
            crop_ids=[1, 2, 3, 4],
            groups=[(1,), (2,), (3,), (4,)],
            classes=["cat", "dog"],
            crops_per_class=crops_per_class,
        )


def test_verify_rejects_incomplete_coverage():
    assignments = {"train": (1,), "val": (2,), "test": (3,)}
    crops_per_class = {name: {"cat": 1, "dog": 1} for name in ("train", "val", "test")}

    with pytest.raises(ManifestValidationError, match="does not cover"):
        verify_manifest_assignment(
            assignments,
            crop_ids=[1, 2, 3, 4],  # el 4 nunca aparece en assignments
            groups=[(1,), (2,), (3,), (4,)],
            classes=["cat", "dog"],
            crops_per_class=crops_per_class,
        )


def test_verify_rejects_an_empty_split():
    assignments = {"train": (1, 2, 3), "val": (), "test": ()}
    crops_per_class = {name: {"cat": 1, "dog": 1} for name in ("train", "val", "test")}

    with pytest.raises(ManifestValidationError, match="non-empty"):
        verify_manifest_assignment(
            assignments,
            crop_ids=[1, 2, 3],
            groups=[(1,), (2,), (3,)],
            classes=["cat", "dog"],
            crops_per_class=crops_per_class,
        )


def test_verify_rejects_a_class_missing_from_val_or_test():
    assignments = {"train": (1, 2), "val": (3,), "test": (4,)}
    groups = [(1,), (2,), (3,), (4,)]
    crops_per_class = {
        "train": {"cat": 1, "dog": 1},
        "val": {"cat": 0, "dog": 1},  # "cat" ausente en val
        "test": {"cat": 1, "dog": 0},
    }

    with pytest.raises(ManifestValidationError, match="cat"):
        verify_manifest_assignment(
            assignments,
            crop_ids=[1, 2, 3, 4],
            groups=groups,
            classes=["cat", "dog"],
            crops_per_class=crops_per_class,
        )
