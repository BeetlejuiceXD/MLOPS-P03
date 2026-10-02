"""D04-03 — Matriz OFAT congelada de 12 configuraciones.

Acta final de #33 (comentario de Esteban:
https://github.com/BeetlejuiceXD/MLOPS-P03/issues/33#issuecomment-5850854067).
Diseño one-factor-at-a-time sobre una fila base (#1): cada fila cambia UN solo
eje respecto al baseline, más dos réplicas de varianza (#11, #12, mismo
baseline con otra seed) — así el efecto de cada hiperparámetro es atribuible.

`weight_decay`, `augmentation`, `pretrained`, `architecture`, `hidden_dim` y
`patience` no son ejes de la matriz (ratificados como constantes en #33); se
fijan en `to_training_config_kwargs`, no se varían por fila.

Esta matriz NO se modifica en D04-03 — el ticket solo la ejecuta.
"""

from __future__ import annotations

from dataclasses import dataclass

PATIENCE = 5
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class CampaignRow:
    index: int
    trainable_layers: str
    learning_rate: float
    optimizer: str
    batch_size: int
    max_epochs: int
    image_size: int
    hidden_layers: int
    dropout: float
    seed: int
    change: str


MATRIX: tuple[CampaignRow, ...] = (
    CampaignRow(1, "last_block", 1e-3, "adam", 16, 30, 224, 0, 0.0, 7, "Baseline"),
    CampaignRow(2, "head_only", 1e-3, "adam", 16, 30, 224, 0, 0.0, 7, "trainable_layers"),
    CampaignRow(3, "last_block", 3e-4, "adam", 16, 30, 224, 0, 0.0, 7, "learning_rate"),
    CampaignRow(4, "last_block", 1e-3, "sgd", 16, 30, 224, 0, 0.0, 7, "optimizer"),
    CampaignRow(5, "last_block", 1e-3, "adam", 32, 30, 224, 0, 0.0, 7, "batch_size"),
    CampaignRow(6, "last_block", 1e-3, "adam", 16, 60, 224, 0, 0.0, 7, "max_epochs"),
    CampaignRow(7, "last_block", 1e-3, "adam", 16, 30, 160, 0, 0.0, 7, "image_size"),
    CampaignRow(8, "last_block", 1e-3, "adam", 16, 30, 224, 1, 0.0, 7, "hidden_layers"),
    CampaignRow(9, "last_block", 1e-3, "adam", 16, 30, 224, 0, 0.3, 7, "dropout"),
    CampaignRow(10, "last_block", 1e-3, "adam", 16, 30, 224, 1, 0.3, 7, "hidden_layers+dropout"),
    CampaignRow(
        11, "last_block", 1e-3, "adam", 16, 30, 224, 0, 0.0, 21, "Réplica de varianza (seed)"
    ),
    CampaignRow(
        12, "last_block", 1e-3, "adam", 16, 30, 224, 0, 0.0, 77, "Réplica de varianza (seed)"
    ),
)


def to_training_config_kwargs(row: CampaignRow) -> dict:
    """Los campos efectivos de `TrainingConfig` para una fila de la matriz."""
    return {
        "architecture": "resnet18",
        "pretrained": True,
        "trainable_layers": row.trainable_layers,
        "image_size": row.image_size,
        "batch_size": row.batch_size,
        "learning_rate": row.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": row.optimizer,
        "max_epochs": row.max_epochs,
        "patience": PATIENCE,
        "augmentation": True,
        "seed": row.seed,
        "hidden_layers": row.hidden_layers,
        "hidden_dim": 128,
        "dropout": row.dropout,
    }
