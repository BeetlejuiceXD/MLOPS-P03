"""TrainingConfig (D01-04): configuración de entrenamiento validada.

Protocolo congelado en D01-03 (issue #33): release v0.1.1, clases cat/dog,
ResNet18 preentrenada. Este módulo solo declara y valida la forma de la
configuración; no entrena ni carga datos (eso es un ticket posterior).
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

HIDDEN_DIM = 128

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    # --- Arquitectura y pesos (ratificado en #33) ---
    architecture: Literal["resnet18"] = "resnet18"
    pretrained: bool = True
    trainable_layers: Literal["head_only", "last_block", "full"] = "last_block"

    # --- Cabeza clasificadora ---
    hidden_layers: Literal[0, 1] = 0
    hidden_dim: int = Field(default=HIDDEN_DIM, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)

    # --- Los siete parámetros de D01-04 ---
    optimizer: Literal["adam", "sgd"] = "adam"
    batch_size: int = Field(default=16, ge=8, le=64)
    max_epochs: int = Field(default=30, ge=10, le=100)
    learning_rate: float = Field(default=1e-3, gt=1e-5, le=1e-2)
    image_size: int = Field(default=224, ge=128, le=256)

    # --- Optimización adicional ratificada en #33 ---
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-2)
    patience: int = Field(default=5, ge=3, le=15)
    augmentation: bool = True

    # --- Reproducibilidad ---
    seed: int
    
    @model_validator(mode="after")
    def _hidden_dim_is_frozen(self) -> Self:
        if self.hidden_layers == 1 and self.hidden_dim != HIDDEN_DIM:
            raise ValueError(
                f"hidden_dim debe ser {HIDDEN_DIM} cuando hidden_layers=1 (protocolo #33)"
            )
        return self
