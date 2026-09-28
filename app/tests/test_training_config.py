"""D01-04 — límites de TrainingConfig: los siete parámetros y serialización."""

import pytest
from pydantic import ValidationError

from training.config import TrainingConfig

BASE = {"seed": 42}


def _config(**overrides):
    return TrainingConfig(**{**BASE, **overrides})


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_optimizer_accepts_known_values(optimizer):
    assert _config(optimizer=optimizer).optimizer == optimizer


def test_optimizer_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _config(optimizer="rmsprop")


@pytest.mark.parametrize("value", [8, 64])
def test_batch_size_accepts_boundaries(value):
    assert _config(batch_size=value).batch_size == value


@pytest.mark.parametrize("value", [7, 65])
def test_batch_size_rejects_outside_boundaries(value):
    with pytest.raises(ValidationError):
        _config(batch_size=value)


@pytest.mark.parametrize("value", [10, 100])
def test_max_epochs_accepts_boundaries(value):
    assert _config(max_epochs=value).max_epochs == value


@pytest.mark.parametrize("value", [9, 101])
def test_max_epochs_rejects_outside_boundaries(value):
    with pytest.raises(ValidationError):
        _config(max_epochs=value)


@pytest.mark.parametrize("value", [1e-5 + 1e-9, 1e-2])
def test_learning_rate_accepts_boundaries(value):
    assert _config(learning_rate=value).learning_rate == value


@pytest.mark.parametrize("value", [1e-5, 1e-2 + 1e-6])
def test_learning_rate_rejects_outside_boundaries(value):
    with pytest.raises(ValidationError):
        _config(learning_rate=value)


@pytest.mark.parametrize("value", [128, 256])
def test_image_size_accepts_boundaries(value):
    assert _config(image_size=value).image_size == value


@pytest.mark.parametrize("value", [127, 257])
def test_image_size_rejects_outside_boundaries(value):
    with pytest.raises(ValidationError):
        _config(image_size=value)


@pytest.mark.parametrize("value", [0, 1])
def test_hidden_layers_accepts_known_values(value):
    assert _config(hidden_layers=value).hidden_layers == value


def test_hidden_layers_rejects_other_values():
    with pytest.raises(ValidationError):
        _config(hidden_layers=2)


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_dropout_accepts_boundaries(value):
    assert _config(dropout=value).dropout == value


@pytest.mark.parametrize("value", [-0.01, 0.51])
def test_dropout_rejects_outside_boundaries(value):
    with pytest.raises(ValidationError):
        _config(dropout=value)


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _config(made_up_field=True)


def test_seed_is_required():
    with pytest.raises(ValidationError):
        TrainingConfig()


def test_default_config_is_valid():
    config = _config()
    assert config.architecture == "resnet18"
    assert config.trainable_layers == "last_block"


def test_config_round_trips_through_json():
    config = _config(batch_size=32, learning_rate=3e-4, hidden_layers=1)
    restored = TrainingConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_hidden_dim_defaults_to_128():
    assert _config(hidden_layers=1).hidden_dim == 128


def test_hidden_dim_128_is_accepted_with_hidden_layer():
    assert _config(hidden_layers=1, hidden_dim=128).hidden_dim == 128


@pytest.mark.parametrize("value", [1, 32, 127, 129, 256])
def test_hidden_dim_other_than_128_is_rejected_with_hidden_layer(value):
    with pytest.raises(ValidationError, match="128"):
        _config(hidden_layers=1, hidden_dim=value)
