"""D04-03 — integridad de la matriz OFAT congelada: 12 filas, cada eje variado
al menos una vez, y cada fila produce un TrainingConfig válido."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from campaign.matrix import MATRIX, to_training_config_kwargs
from training.config import TrainingConfig


def test_matrix_has_exactly_twelve_rows():
    assert len(MATRIX) == 12


def test_row_indices_are_1_to_12_without_gaps_or_duplicates():
    assert [row.index for row in MATRIX] == list(range(1, 13))


def test_every_row_produces_a_valid_training_config():
    for row in MATRIX:
        config = TrainingConfig(**to_training_config_kwargs(row))
        assert config.seed == row.seed


def test_first_row_is_the_baseline():
    assert MATRIX[0].change == "Baseline"


@pytest.mark.parametrize(
    "field",
    [
        "trainable_layers",
        "learning_rate",
        "optimizer",
        "batch_size",
        "max_epochs",
        "image_size",
        "hidden_layers",
        "dropout",
        "seed",
    ],
)
def test_every_axis_is_varied_at_least_once_versus_the_baseline(field):
    baseline = MATRIX[0]
    assert any(getattr(row, field) != getattr(baseline, field) for row in MATRIX[1:])


def test_variance_replicas_only_change_seed_versus_baseline():
    baseline = MATRIX[0]
    for row in MATRIX[10:12]:  # filas 11 y 12
        assert row.seed != baseline.seed
        for field in (
            "trainable_layers",
            "learning_rate",
            "optimizer",
            "batch_size",
            "max_epochs",
            "image_size",
            "hidden_layers",
            "dropout",
        ):
            assert getattr(row, field) == getattr(baseline, field)


def test_no_two_rows_are_identical_configurations():
    configs = [to_training_config_kwargs(row) for row in MATRIX]
    assert len({tuple(sorted(c.items())) for c in configs}) == len(configs)


def test_pretrained_and_architecture_are_fixed_not_varied():
    for row in MATRIX:
        kwargs = to_training_config_kwargs(row)
        assert kwargs["architecture"] == "resnet18"
        assert kwargs["pretrained"] is True
        assert kwargs["hidden_dim"] == 128


def test_an_invalid_value_in_a_row_is_rejected_not_silently_clamped():
    kwargs = to_training_config_kwargs(MATRIX[0]) | {"batch_size": 1000}
    with pytest.raises(ValidationError):
        TrainingConfig(**kwargs)
