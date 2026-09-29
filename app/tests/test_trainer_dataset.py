"""D02-03 — fixture del trainer: sin fuga entre particiones, ambas clases
presentes, tamaño identificado y orden determinista por seed."""

from trainer.dataset import build_fixture_dataset


def test_no_group_leaks_across_splits():
    dataset = build_fixture_dataset(seed=1)
    train = dataset.group_ids("train")
    val = dataset.group_ids("val")
    test = dataset.group_ids("test")

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_both_classes_present_in_every_split():
    dataset = build_fixture_dataset(seed=1)
    for split in ("train", "val", "test"):
        labels = {sample.label for sample in getattr(dataset, split)}
        assert labels == {"cat", "dog"}


def test_fixture_sizes_are_identified():
    dataset = build_fixture_dataset(seed=1)
    assert (len(dataset.train), len(dataset.val), len(dataset.test)) == (16, 8, 4)


def test_same_seed_gives_identical_order():
    first = build_fixture_dataset(seed=42)
    second = build_fixture_dataset(seed=42)
    assert [s.sample_id for s in first.train] == [s.sample_id for s in second.train]


def test_different_seed_changes_order():
    first = build_fixture_dataset(seed=1)
    second = build_fixture_dataset(seed=2)
    assert [s.sample_id for s in first.train] != [s.sample_id for s in second.train]
