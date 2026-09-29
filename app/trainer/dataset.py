"""Fixture del trainer (D02-03): sintético y propio, sin depender de `crops/` ni
`manifest/` (D02-04/D03-01 quedan fuera a propósito — #43: "no usar datos que
puedan terminar en el frozen test oficial"). Cada grupo imita el concepto de
original+duplicado indivisible de P2/P3: sus muestras siempre caen en la misma
partición (ver `test_trainer_dataset.py`, patrón de `test_cross_split_leakage.py`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw

from training.class_map import CLASS_MAP

GROUPS_PER_CLASS = {"train": 4, "val": 2, "test": 1}
SAMPLES_PER_GROUP = 2  # "original" + "duplicado"


@dataclass(frozen=True)
class TrainingSample:
    sample_id: int
    group_id: int
    label: str
    image: Image.Image


@dataclass(frozen=True)
class TrainingDataset:
    train: tuple[TrainingSample, ...]
    val: tuple[TrainingSample, ...]
    test: tuple[TrainingSample, ...]

    def group_ids(self, split: str) -> frozenset[int]:
        return frozenset(sample.group_id for sample in getattr(self, split))


def _synthetic_image(seed: int) -> Image.Image:
    """64x64 con estructura real (no color plano): un rectángulo en una posición
    determinada por `seed` — mismo criterio que otros fixtures del repo."""
    image = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    x0, y0 = (seed * 13) % 40, (seed * 19) % 40
    draw.rectangle([x0, y0, x0 + 20, y0 + 20], fill=(225, 225, 225))
    return image


def build_fixture_dataset(seed: int) -> TrainingDataset:
    """28 muestras (14 grupos x 2): 16 train, 8 val, 4 test, cat/dog en cada
    partición. Determinista por `seed` — independiente de la seed 42 del
    manifest oficial (#33): esta seed gobierna solo el fixture."""
    rng = random.Random(seed)
    group_id = sample_id = 0
    splits: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}

    for label in sorted(CLASS_MAP):  # ["cat", "dog"]: orden determinista
        for split, n_groups in GROUPS_PER_CLASS.items():
            for _ in range(n_groups):
                image_seed = rng.randrange(1_000_000)
                for _variant in range(SAMPLES_PER_GROUP):
                    splits[split].append(
                        TrainingSample(sample_id, group_id, label, _synthetic_image(image_seed))
                    )
                    sample_id += 1
                group_id += 1

    for samples in splits.values():
        rng.shuffle(samples)  # el orden de muestras también depende de la seed

    return TrainingDataset(tuple(splits["train"]), tuple(splits["val"]), tuple(splits["test"]))
