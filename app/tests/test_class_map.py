"""D01-04 — class map: orden alfabético sobre FROZEN_CLASSES, no sobre category_id."""

from crops.models import FROZEN_CLASSES

from training.class_map import CLASS_MAP, INDEX_TO_CLASS, NUM_CLASSES, class_map


def test_class_map_covers_exactly_the_frozen_classes():
    assert set(CLASS_MAP) == FROZEN_CLASSES


def test_class_map_is_alphabetical_not_coco_category_id_order():
    # category_id real: dog=3, cat=4 (release_sources.yaml) — si siguiera ese
    # orden, dog sería 0. Se fija alfabético para no acoplar al COCO.
    assert CLASS_MAP == {"cat": 0, "dog": 1}


def test_index_to_class_is_the_inverse_mapping():
    assert {index: name for name, index in CLASS_MAP.items()} == INDEX_TO_CLASS


def test_num_classes_matches_map_length():
    assert NUM_CLASSES == len(CLASS_MAP) == 2


def test_class_map_function_returns_a_defensive_copy():
    mapping = class_map()
    mapping["cat"] = 99
    assert CLASS_MAP["cat"] == 0
