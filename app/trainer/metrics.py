"""Comparacion de checkpoints por epoca (D02-03): mismo criterio de seleccion
del protocolo de D01-03 (#33) - validation accuracy, desempate por validation
macro-F1 y luego menor validation loss, igualdad a 4 decimales. La verificacion
completa de early stopping/restauracion es de D03-02."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float


def is_better(candidate: EpochMetrics, current_best: EpochMetrics) -> bool:
    """Selecciona qué checkpoint guardar (accuracy -> macro-F1 -> val_loss,
    empate a 4 decimales en cada paso). No controla la paciencia del early
    stopping — eso es `accuracy_improved`, deliberadamente más estricto."""
    if round(candidate.val_accuracy, 4) != round(current_best.val_accuracy, 4):
        return candidate.val_accuracy > current_best.val_accuracy
    if round(candidate.val_macro_f1, 4) != round(current_best.val_macro_f1, 4):
        return candidate.val_macro_f1 > current_best.val_macro_f1
    if round(candidate.val_loss, 4) != round(current_best.val_loss, 4):
        return candidate.val_loss < current_best.val_loss
    # Empate total a 4 decimales en los tres campos. Sin run_id (fuera de
    # alcance de D02-03), no hay como desempatar mas -> se conserva el actual.
    return False


def accuracy_improved(candidate: EpochMetrics, reference: EpochMetrics) -> bool:
    """Unico criterio que alimenta el contador de paciencia (#33: early
    stopping monitorea val_accuracy). Deliberadamente ignora macro-F1 y
    val_loss: que el checkpoint elegido mejore por esos campos no debe
    mantener vivo un entrenamiento cuya accuracy esta estancada."""
    return round(candidate.val_accuracy, 4) > round(reference.val_accuracy, 4)


def should_stop(epochs_without_improvement: int, patience: int) -> bool:
    """`patience` épocas SIN mejora ya agotan la paciencia: con patience=3, la
    3ra época sin mejora dispara la parada (no hace falta una 4ta)."""
    return epochs_without_improvement >= patience
