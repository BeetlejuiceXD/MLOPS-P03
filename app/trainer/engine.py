"""Trainer reproducible por minibatches (D02-03): forward, loss, backward y paso
del optimizer reales sobre `trainer.dataset` — nunca sobre el manifest oficial ni
el frozen test (custodia de Ale, #33). Integración con manifest congelado y
trainer-worker: D03-03.

No determinismo conocido: en CPU, la suma en paralelo de operaciones (BLAS
multihilo) puede variar en los últimos bits de precisión entre corridas con la
misma seed; el orden de muestras es exacto, los pesos se comparan con tolerancia
(ver `test_trainer_engine.py`). `torch.use_deterministic_algorithms` no se activa
por defecto: ResNet18 usa capas sin implementación determinista garantizada en
todas las versiones de PyTorch; activarlo a ciegas podría romper en CI.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trainer.dataset import TrainingDataset, TrainingSample
from trainer.metrics import EpochMetrics, accuracy_improved, is_better, should_stop
from training.class_map import CLASS_MAP
from training.config import TrainingConfig
from training.model import build_model
from training.preprocessing import preprocess_image


def _normalize_seed(seed: int) -> int:
    """Adapta la seed del run al dominio [0, 2**32) que exige NumPy (y, por
    consistencia entre backends, también random/PyTorch/DataLoader) — sin
    tocar la seed original que queda registrada en TrainingConfig/MLflow.
    Determinista: la misma seed de entrada siempre da el mismo resultado."""
    return seed % 2**32


def seed_everything(seed: int) -> None:
    """Fija random, NumPy y PyTorch. Seed del *run*, independiente de la seed
    42 del manifest (#33)."""
    normalized = _normalize_seed(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)


class _SampleDataset(Dataset):
    def __init__(self, samples: tuple[TrainingSample, ...], config: TrainingConfig, *, train: bool):
        self.samples = samples
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = preprocess_image(sample.image, self.config, train=self.train)
        return image, CLASS_MAP[sample.label]


def _make_loader(
    samples: tuple[TrainingSample, ...], config: TrainingConfig, *, train: bool
) -> DataLoader:
    dataset = _SampleDataset(samples, config, train=train)
    generator = torch.Generator().manual_seed(_normalize_seed(config.seed))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=train, generator=generator)


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "adam":
        return torch.optim.Adam(trainable, **kwargs)
    return torch.optim.SGD(trainable, **kwargs)


def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            batch_predictions = outputs.argmax(dim=1)
            correct += (batch_predictions == labels).sum().item()
            total += labels.size(0)
            predictions.extend(batch_predictions.tolist())
            targets.extend(labels.tolist())
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return total_loss / total, correct / total, macro_f1


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochMetrics, ...]
    best: EpochMetrics
    model: nn.Module
    stopped_early: bool


def train(
    config: TrainingConfig,
    dataset: TrainingDataset,
    *,
    after_step: Callable[[list[int]], None] = lambda _labels: None,
) -> TrainingResult:
    """`after_step` es un gancho de prueba (cuenta pasos, captura orden de
    muestras); producción nunca lo usa — mismo patrón que `Worker.after_epoch`
    en `trainer_worker`."""
    seed_everything(config.seed)
    model = build_model(config)
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(dataset.train, config, train=True)
    val_loader = _make_loader(dataset.val, config, train=False)

    history: list[EpochMetrics] = []
    best: EpochMetrics | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy_metrics: EpochMetrics | None = None  # solo alimenta la paciencia
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            after_step(labels.tolist())

        val_loss, val_accuracy, val_macro_f1 = _evaluate(model, val_loader, criterion)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss / train_total,
            train_accuracy=train_correct / train_total,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_macro_f1,
            learning_rate=config.learning_rate,
        )
        history.append(metrics)

        # Checkpoint: puede desempatar por macro-F1/val_loss (is_better).
        if best is None or is_better(metrics, best):
            best, best_state = metrics, copy.deepcopy(model.state_dict())

        # Paciencia: depende SOLO de val_accuracy (#33) - independiente de
        # cuál metrics quedó como `best` arriba, para que una mejora de
        # macro-F1/val_loss nunca mantenga vivo un entrenamiento cuya
        # accuracy está estancada.
        accuracy_baseline = best_val_accuracy_metrics
        if accuracy_baseline is None or accuracy_improved(metrics, accuracy_baseline):
            best_val_accuracy_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if should_stop(epochs_without_improvement, config.patience):
                stopped_early = True
                break

    model.load_state_dict(best_state)
    return TrainingResult(tuple(history), best, model, stopped_early)
