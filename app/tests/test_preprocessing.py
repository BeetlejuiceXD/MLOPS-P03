"""D01-04 — preprocessing: shape determinista y augmentation solo en train."""

import torch
from PIL import Image

from training.config import TrainingConfig
from training.preprocessing import build_eval_transform, build_train_transform, preprocess_image

BASE = {"seed": 1}


def _config(**overrides):
    return TrainingConfig(**{**BASE, **overrides})


def _sample_image(size=(64, 64)):
    return Image.new("RGB", size, color=(120, 60, 200))


def test_eval_transform_resizes_to_configured_image_size():
    config = _config(image_size=160)
    tensor = preprocess_image(_sample_image(), config, train=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 160, 160)


def test_eval_transform_is_deterministic_across_calls():
    config = _config(image_size=128)
    image = _sample_image()

    first = preprocess_image(image, config, train=False)
    second = preprocess_image(image, config, train=False)

    assert torch.equal(first, second)


def test_train_transform_without_augmentation_matches_eval_transform():
    config = _config(augmentation=False, image_size=128)
    image = _sample_image()

    train_tensor = preprocess_image(image, config, train=True)
    eval_tensor = preprocess_image(image, config, train=False)

    assert torch.equal(train_tensor, eval_tensor)


def test_train_transform_with_augmentation_adds_more_steps_than_eval():
    config = _config(augmentation=True, image_size=128)

    assert len(build_train_transform(config).transforms) > len(
        build_eval_transform(config).transforms
    )
