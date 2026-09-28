"""Mutation testing propio de D01-04 (#34): TrainingConfig, factoría CNN/heads y preprocessing.

Independiente del mutation test del quality gate. Para cada mutante: aplica el cambio
en una línea concreta, corre solo los tests del área, registra QUÉ tests fallaron y
restaura el archivo en `finally`. KILLED = pytest corrió y falló >=1 test (sin errores
de colección); SURVIVED = todo pasó; ERROR = objetivo no encontrado exactamente una vez
o pytest roto (nunca cuenta como muerto).

Uso (desde la raíz del repo):  cd app && uv run python ../.github/scripts/run_training_mutations.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

CFG = ["tests/test_training_config.py"]
CLS = ["tests/test_class_map.py"]
MODEL = ["tests/test_model.py"]
PRE = ["tests/test_preprocessing.py"]


@dataclass(frozen=True)
class Mutant:
    area: str
    name: str
    target: str
    original: str
    mutant: str
    tests: list[str]


BS = "batch_size: int = Field(default=16, ge=8, le=64)"
EP = "max_epochs: int = Field(default=30, ge=10, le=100)"
LR = "learning_rate: float = Field(default=1e-3, gt=1e-5, le=1e-2)"
IS = "image_size: int = Field(default=224, ge=128, le=256)"
DO = "dropout: float = Field(default=0.0, ge=0.0, le=0.5)"
C, M, P = "training/config.py", "training/model.py", "training/preprocessing.py"

MUTANTS = [
    Mutant(
        "TrainingConfig", "batch_size: máximo 64 -> 65", C, BS, BS.replace("le=64", "le=65"), CFG
    ),
    Mutant("TrainingConfig", "batch_size: mínimo 8 -> 7", C, BS, BS.replace("ge=8", "ge=7"), CFG),
    Mutant(
        "TrainingConfig",
        "max_epochs: máximo 100 -> 101",
        C,
        EP,
        EP.replace("le=100", "le=101"),
        CFG,
    ),
    Mutant("TrainingConfig", "max_epochs: mínimo 10 -> 9", C, EP, EP.replace("ge=10", "ge=9"), CFG),
    Mutant(
        "TrainingConfig",
        "learning_rate: cota inferior exclusiva -> inclusiva",
        C,
        LR,
        LR.replace("gt=1e-5", "ge=1e-5"),
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "learning_rate: máximo 1e-2 -> 2e-2",
        C,
        LR,
        LR.replace("le=1e-2", "le=2e-2"),
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "image_size: mínimo 128 -> 127",
        C,
        IS,
        IS.replace("ge=128", "ge=127"),
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "image_size: máximo 256 -> 257",
        C,
        IS,
        IS.replace("le=256", "le=257"),
        CFG,
    ),
    Mutant(
        "TrainingConfig", "dropout: máximo 0.5 -> 0.6", C, DO, DO.replace("le=0.5", "le=0.6"), CFG
    ),
    Mutant(
        "TrainingConfig", "dropout: mínimo 0.0 -> -0.1", C, DO, DO.replace("ge=0.0", "ge=-0.1"), CFG
    ),
    Mutant(
        "TrainingConfig",
        "optimizer: acepta 'rmsprop'",
        C,
        'optimizer: Literal["adam", "sgd"] = "adam"',
        'optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"',
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "hidden_layers: acepta 2",
        C,
        "hidden_layers: Literal[0, 1] = 0",
        "hidden_layers: Literal[0, 1, 2] = 0",
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "hidden_dim: HIDDEN_DIM 128 -> 64",
        C,
        "HIDDEN_DIM = 128",
        "HIDDEN_DIM = 64",
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "hidden_dim: != -> == (acepta solo lo prohibido)",
        C,
        "self.hidden_dim != HIDDEN_DIM",
        "self.hidden_dim == HIDDEN_DIM",
        CFG,
    ),
    Mutant(
        "TrainingConfig",
        "hidden_dim: validación solo con hidden_layers=0",
        C,
        "self.hidden_layers == 1 and",
        "self.hidden_layers == 0 and",
        CFG,
    ),
    Mutant(
        "TrainingConfig", "extra='forbid' -> 'ignore'", C, 'extra="forbid"', 'extra="ignore"', CFG
    ),
    Mutant("TrainingConfig", "strict=True -> False", C, "strict=True", "strict=False", CFG),
    Mutant(
        "Class map",
        "orden alfabético invertido",
        "training/class_map.py",
        "sorted(FROZEN_CLASSES)",
        "sorted(FROZEN_CLASSES, reverse=True)",
        CLS,
    ),
    Mutant(
        "Modelo/heads",
        "último bloque entrenable: layer4 -> layer3",
        M,
        '_LAST_BLOCK_PREFIX = "layer4"',
        '_LAST_BLOCK_PREFIX = "layer3"',
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "fc deja de ser entrenable",
        M,
        'if name.startswith("fc."):',
        'if not name.startswith("fc."):',
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "'full' ya no deja todo entrenable",
        M,
        'if config.trainable_layers == "full":',
        'if config.trainable_layers == "head_only":',
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "cabeza 512->128->2 con salida +1",
        M,
        "nn.Linear(config.hidden_dim, NUM_CLASSES),",
        "nn.Linear(config.hidden_dim, NUM_CLASSES + 1),",
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "cabeza sin ReLU (Identity)",
        M,
        "nn.ReLU(inplace=True),",
        "nn.Identity(),",
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "cabeza sin capa oculta con salida +1",
        M,
        "nn.Linear(in_features, NUM_CLASSES))",
        "nn.Linear(in_features, NUM_CLASSES + 1))",
        MODEL,
    ),
    Mutant(
        "Modelo/heads",
        "seed ignorada (siempre 0)",
        M,
        "torch.manual_seed(config.seed)",
        "torch.manual_seed(0)",
        MODEL,
    ),
    Mutant(
        "Preprocessing",
        "media ImageNet alterada",
        P,
        "IMAGENET_MEAN = (0.485, 0.456, 0.406)",
        "IMAGENET_MEAN = (0.5, 0.456, 0.406)",
        PRE,
    ),
    Mutant(
        "Preprocessing",
        "std ImageNet alterada",
        P,
        "IMAGENET_STD = (0.229, 0.224, 0.225)",
        "IMAGENET_STD = (0.3, 0.224, 0.225)",
        PRE,
    ),
    Mutant(
        "Preprocessing",
        "augmentation invertida",
        P,
        "if not config.augmentation:",
        "if config.augmentation:",
        PRE,
    ),
    Mutant("Preprocessing", "train/eval invertidos", P, "if train else", "if not train else", PRE),
    Mutant(
        "Preprocessing",
        "imagen no se convierte a RGB",
        P,
        'image.convert("RGB")',
        'image.convert("L")',
        PRE,
    ),
]


def run_pytest(tests: list[str]) -> tuple[int, list[str], int]:
    """Devuelve (returncode, tests fallidos, errores)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        env = {**os.environ, "PYTHONPYCACHEPREFIX": str(Path(tmp) / "pyc")}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
                *tests,
            ],
            cwd=APP,
            env=env,
            check=False,
            capture_output=True,
        )
        failed, errors = [], 0
        if report.is_file():
            for case in ET.parse(report).getroot().iter("testcase"):
                if case.find("failure") is not None:
                    failed.append(f"{case.get('classname', '').split('.')[-1]}::{case.get('name')}")
                if case.find("error") is not None:
                    errors += 1
        return proc.returncode, failed, errors


def main() -> int:
    rc, failed, errors = run_pytest(sorted({t for m in MUTANTS for t in m.tests}))
    if rc != 0:
        print(f"ERROR: los tests ya fallan SIN mutante (rc={rc}); no se puede evaluar.")
        return 2
    print(f"Base limpia: tests en verde. Mutantes: {len(MUTANTS)}\n")
    results, exit_code = [], 0
    for m in MUTANTS:
        path = APP / m.target
        source = path.read_bytes()
        if source.count(m.original.encode()) != 1:
            results.append((m, "ERROR", ["objetivo no encontrado exactamente una vez"]))
            exit_code = 2
            continue
        try:
            path.write_bytes(source.replace(m.original.encode(), m.mutant.encode()))
            rc, failed, errors = run_pytest(m.tests)
        finally:
            path.write_bytes(source)
        if errors or rc not in (0, 1):
            status, exit_code = "ERROR", 2
        elif failed:
            status = "KILLED"
        else:
            status, exit_code = "SURVIVED", max(exit_code, 1)
        results.append((m, status, failed))
        print(f"{status:9} [{m.area}] {m.name}  ->  {len(failed)} test(s)")

    print("\n| # | Área | Mutación | Resultado | Tests que la detectaron |\n|---|---|---|---|---|")
    for i, (m, status, failed) in enumerate(results, 1):
        shown = "<br>".join(f"`{t}`" for t in failed[:4]) + (
            f"<br>(+{len(failed) - 4} más)" if len(failed) > 4 else ""
        )
        print(f"| {i} | {m.area} | {m.name} | {status} | {shown or '—'} |")
    killed = sum(1 for _, s, _ in results if s == "KILLED")
    print(f"\nKILLED {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())