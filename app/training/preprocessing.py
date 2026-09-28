"""Preprocessing compartido (D01-04): mismo transform en training y evaluación/
inferencia (protocolo D01-03: augmentation solo en train).

Funciones puras sobre imágenes ya decodificadas (PIL.Image); no abren archivos
ni tocan MinIO/S3 — mismo criterio de acoplamiento que `crops.engine`.
"""

import torch
from PIL import Image
from torchvision import transforms

from training.config import TrainingConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(config: TrainingConfig) -> transforms.Compose:
    """Determinista: resize + normalización ImageNet. Se usa igual en validation,
    test e inferencia; nunca cambia entre llamadas con la misma config."""
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(config: TrainingConfig) -> transforms.Compose:
    """Agrega augmentation solo si `config.augmentation` es True (D01-03).
    Sin augmentation, es idéntico al de evaluación."""
    if not config.augmentation:
        return build_eval_transform(config)
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image, config: TrainingConfig, *, train: bool) -> torch.Tensor:
    """Aplica el transform correspondiente a una sola imagen PIL."""
    transform = build_train_transform(config) if train else build_eval_transform(config)
    return transform(image.convert("RGB"))
