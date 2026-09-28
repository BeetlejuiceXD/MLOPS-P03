"""Mutation testing específico de D01-04: TrainingConfig, class map, modelo y preprocessing.

Cada mutante debe ser KILLED por los tests del módulo. Reutiliza `run_mutation` del
script del quality gate (restaura el archivo en `finally`; un error de pytest o un
objetivo no encontrado exactamente una vez es ERROR, nunca un mutante muerto).
"""

from __future__ import annotations

from dataclasses import dataclass

from run_gate_mutation import APP_ROOT, KILLED, SURVIVED, run_mutation

CFG, CLS, MODEL, PRE = (
    ("tests/test_training_config.py",),
    ("tests/test_class_map.py",),
    ("tests/test_model.py",),
    ("tests/test_preprocessing.py",),
)


@dataclass(frozen=True)
class Mutant:
    name: str
    target: str
    original: str
    mutant: str
    tests: tuple[str, ...]


def _m(name, target, original, mutant, tests):
    return Mutant(name, target, original, mutant, tests)


BS = "batch_size: int = Field(default=16, ge=8, le=64)"
EP = "max_epochs: int = Field(default=30, ge=10, le=100)"
LR = "learning_rate: float = Field(default=1e-3, gt=1e-5, le=1e-2)"
IS = "image_size: int = Field(default=224, ge=128, le=256)"
DO = "dropout: float = Field(default=0.0, ge=0.0, le=0.5)"
C = "training/config.py"

MUTANTS = (
    _m("batch_size limite superior", C, BS, BS.replace("le=64", "le=65"), CFG),
    _m("batch_size limite inferior", C, BS, BS.replace("ge=8", "ge=7"), CFG),
    _m("max_epochs limite superior", C, EP, EP.replace("le=100", "le=101"), CFG),
    _m("max_epochs limite inferior", C, EP, EP.replace("ge=10", "ge=9"), CFG),
    _m("learning_rate exclusivo -> inclusivo", C, LR, LR.replace("gt=1e-5", "ge=1e-5"), CFG),
    _m("learning_rate limite superior", C, LR, LR.replace("le=1e-2", "le=2e-2"), CFG),
    _m("image_size limite inferior", C, IS, IS.replace("ge=128", "ge=127"), CFG),
    _m("image_size limite superior", C, IS, IS.replace("le=256", "le=257"), CFG),
    _m("dropout limite superior", C, DO, DO.replace("le=0.5", "le=0.6"), CFG),
    _m("dropout limite inferior", C, DO, DO.replace("ge=0.0", "ge=-0.1"), CFG),
    _m("optimizer acepta desconocido", C, 'optimizer: Literal["adam", "sgd"] = "adam"',
       'optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"', CFG),
    _m("hidden_layers acepta 2", C, "hidden_layers: Literal[0, 1] = 0",
       "hidden_layers: Literal[0, 1, 2] = 0", CFG),
    _m("hidden_dim: != -> ==", C, "self.hidden_dim != HIDDEN_DIM", "self.hidden_dim == HIDDEN_DIM", CFG),
    _m("hidden_dim: solo si hidden_layers=0", C, "self.hidden_layers == 1 and",
       "self.hidden_layers == 0 and", CFG),
    _m("extra=forbid -> ignore", C, 'extra="forbid"', 'extra="ignore"', CFG),
    _m("class map orden invertido", "training/class_map.py", "sorted(FROZEN_CLASSES)",
       "sorted(FROZEN_CLASSES, reverse=True)", CLS),
    _m("ultimo bloque = layer3", "training/model.py", '_LAST_BLOCK_PREFIX = "layer4"',
       '_LAST_BLOCK_PREFIX = "layer3"', MODEL),
    _m("fc deja de ser entrenable", "training/model.py", 'if name.startswith("fc."):',
       'if not name.startswith("fc."):', MODEL),
    _m("full ya no es full", "training/model.py", 'if config.trainable_layers == "full":',
       'if config.trainable_layers == "head_only":', MODEL),
    _m("salida de la cabeza +1", "training/model.py", "nn.Linear(config.hidden_dim, NUM_CLASSES),",
       "nn.Linear(config.hidden_dim, NUM_CLASSES + 1),", MODEL),
    _m("sin ReLU en la cabeza", "training/model.py", "nn.ReLU(inplace=True),", "nn.Identity(),", MODEL),
    _m("seed ignorada", "training/model.py", "torch.manual_seed(config.seed)",
       "torch.manual_seed(0)", MODEL),
    _m("media ImageNet alterada", "training/preprocessing.py",
       "IMAGENET_MEAN = (0.485, 0.456, 0.406)", "IMAGENET_MEAN = (0.5, 0.456, 0.406)", PRE),
    _m("std ImageNet alterada", "training/preprocessing.py",
       "IMAGENET_STD = (0.229, 0.224, 0.225)", "IMAGENET_STD = (0.3, 0.224, 0.225)", PRE),
    _m("augmentation invertida", "training/preprocessing.py", "if not config.augmentation:",
       "if config.augmentation:", PRE),
    _m("train/eval invertidos", "training/preprocessing.py", "if train else", "if not train else", PRE),
    _m("imagen no se convierte a RGB", "training/preprocessing.py", 'image.convert("RGB")',
       'image.convert("L")', PRE),
)


def main() -> int:
    outcomes = [
        (m.name, run_mutation(APP_ROOT / m.target, m.original.encode(), m.mutant.encode(),
                              list(m.tests), APP_ROOT))
        for m in MUTANTS
    ]
    print("\nResumen de mutantes (D01-04)")
    for name, code in outcomes:
        label = "KILLED" if code == KILLED else "SURVIVED" if code == SURVIVED else "ERROR"
        print(f"{label:9} {name}")
    return 0 if all(code == KILLED for _, code in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())