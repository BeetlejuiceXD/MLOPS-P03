"""D01-04 — factoría de CNN: forward pass, capas congeladas y determinismo."""

import pytest
import torch

from training.class_map import NUM_CLASSES
from training.config import TrainingConfig
from training.model import build_model, describe_trainable_layers

BASE = {"seed": 7, "pretrained": False}  # pretrained=False: tests no descargan pesos


def _config(**overrides):
    return TrainingConfig(**{**BASE, **overrides})


def test_forward_pass_output_shape_matches_num_classes():
    config = _config(image_size=128)
    model = build_model(config)
    model.eval()

    batch = torch.randn(4, 3, config.image_size, config.image_size)
    with torch.no_grad():
        output = model(batch)

    assert output.shape == (4, NUM_CLASSES)


@pytest.mark.parametrize("hidden_layers", [0, 1])
def test_forward_pass_works_with_both_head_shapes(hidden_layers):
    config = _config(hidden_layers=hidden_layers, hidden_dim=32)
    model = build_model(config)
    model.eval()

    batch = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        output = model(batch)

    assert output.shape == (2, NUM_CLASSES)


def test_head_only_freezes_everything_except_fc():
    report = describe_trainable_layers(build_model(_config(trainable_layers="head_only")))

    assert all(name.startswith("fc.") for name in report["trainable_layers"])
    assert report["trainable_param_count"] > 0
    assert report["frozen_param_count"] > 0


def test_last_block_unfreezes_layer4_and_fc_only():
    report = describe_trainable_layers(build_model(_config(trainable_layers="last_block")))

    assert all(
        name.startswith("fc.") or name.startswith("layer4") for name in report["trainable_layers"]
    )
    assert any(name.startswith("layer4") for name in report["trainable_layers"])
    assert report["frozen_param_count"] > 0


def test_full_leaves_every_parameter_trainable():
    report = describe_trainable_layers(build_model(_config(trainable_layers="full")))

    assert report["frozen_param_count"] == 0
    assert report["trainable_param_count"] == report["total_param_count"]


def test_same_seed_produces_identical_head_initialization():
    config = _config(seed=123)

    first_head = dict(build_model(config).fc.state_dict())
    second_head = dict(build_model(config).fc.state_dict())

    for key in first_head:
        assert torch.equal(first_head[key], second_head[key])


def test_different_seed_changes_head_initialization():
    first = build_model(_config(seed=1))
    second = build_model(_config(seed=2))

    assert not torch.equal(first.fc[-1].weight, second.fc[-1].weight)
