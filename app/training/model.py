"""Factoría de CNN (D01-04): ResNet18 + cabeza configurable.

`build_model` es la única función que construye el modelo; `describe_trainable_layers`
declara explícitamente qué queda congelado/entrenable (evidencia del PR).
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from training.class_map import NUM_CLASSES
from training.config import TrainingConfig

_LAST_BLOCK_PREFIX = "layer4"


def _build_head(in_features: int, config: TrainingConfig) -> nn.Module:
    if config.hidden_layers == 0:
        return nn.Sequential(nn.Dropout(config.dropout), nn.Linear(in_features, NUM_CLASSES))
    return nn.Sequential(
        nn.Linear(in_features, config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, NUM_CLASSES),
    )


def _apply_trainable_layers(model: nn.Module, config: TrainingConfig) -> None:
    """`fc` (la cabeza, ya reemplazada) siempre queda entrenable."""
    if config.trainable_layers == "full":
        return
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            continue
        if config.trainable_layers == "last_block" and name.startswith(_LAST_BLOCK_PREFIX):
            continue
        parameter.requires_grad = False


def build_model(config: TrainingConfig) -> nn.Module:
    """La seed fija la inicialización de la cabeza nueva (el backbone preentrenado
    no depende de la seed; solo `fc` se inicializa aleatoriamente)."""
    torch.manual_seed(config.seed)

    weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = resnet18(weights=weights)

    model.fc = _build_head(model.fc.in_features, config)
    _apply_trainable_layers(model, config)
    return model


def describe_trainable_layers(model: nn.Module) -> dict[str, object]:
    """Reporte auditable de capas congeladas/entrenables ('declarar pesos/capas')."""
    frozen, trainable = [], []
    frozen_params = trainable_params = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable.append(name)
            trainable_params += count
        else:
            frozen.append(name)
            frozen_params += count
    return {
        "frozen_layers": frozen,
        "trainable_layers": trainable,
        "frozen_param_count": frozen_params,
        "trainable_param_count": trainable_params,
        "total_param_count": frozen_params + trainable_params,
    }
