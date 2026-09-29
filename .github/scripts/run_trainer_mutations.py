"""Mutation testing de D02-03: metrics.py y engine.py del trainer.

Uso: cd app && uv run python ../.github/scripts/run_trainer_mutations.py
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
ENGINE_TESTS = ["tests/test_trainer_engine.py"]
METRICS_TESTS = ["tests/test_trainer_metrics.py"]
DATASET_TESTS = ["tests/test_trainer_dataset.py"]


@dataclass(frozen=True)
class Mutant:
    name: str
    target: str
    original: str
    mutant: str
    tests: list[str]


E, M, D = "trainer/engine.py", "trainer/metrics.py", "trainer/dataset.py"

MUTANTS = [
    # (M01 original, "> -> >=" en is_better, se quitó: el guard de arriba
    # (round(...) != round(...)) ya garantiza que esa línea solo corre con
    # valores distintos, así que ">" y ">=" son equivalentes ahí — no hay test
    # capaz de matarla sin cambiar el propio guard, y cambiarlo no es un bug.)
    Mutant(
        "tolerancia de empate: 4 -> 2 decimales",
        M,
        "if round(candidate.val_accuracy, 4) != round(current_best.val_accuracy, 4):",
        "if round(candidate.val_accuracy, 2) != round(current_best.val_accuracy, 2):",
        METRICS_TESTS,
    ),
    Mutant(
        "desempate macro-F1 invertido",
        M,
        "return candidate.val_macro_f1 > current_best.val_macro_f1",
        "return candidate.val_macro_f1 < current_best.val_macro_f1",
        METRICS_TESTS,
    ),
    Mutant(
        "early stopping: > en vez de >=",
        M,
        "return epochs_without_improvement >= patience",
        "return epochs_without_improvement > patience",
        METRICS_TESTS,
    ),
    Mutant(
        "optimizer incluye parametros congelados",
        E,
        "trainable = [p for p in model.parameters() if p.requires_grad]",
        "trainable = list(model.parameters())",
        ENGINE_TESTS,
    ),
    Mutant(
        "shuffle ignora seed",
        E,
        "generator = torch.Generator().manual_seed(_normalize_seed(config.seed))",
        "generator = torch.Generator().manual_seed(0)",
        ENGINE_TESTS,
    ),
    Mutant("train siempre sin shuffle", E, "shuffle=train,", "shuffle=False,", ENGINE_TESTS),
    Mutant(
        "optimizer siempre sgd",
        E,
        'if config.optimizer == "adam":',
        'if config.optimizer == "unknown":',
        ENGINE_TESTS,
    ),
    Mutant(
        "weight_decay ignorado",
        E,
        '"weight_decay": config.weight_decay}',
        '"weight_decay": 0.0}',
        ENGINE_TESTS,
    ),
    Mutant(
        "tamano de grupos train/val/test alterado",
        D,
        'GROUPS_PER_CLASS = {"train": 4, "val": 2, "test": 1}',
        'GROUPS_PER_CLASS = {"train": 3, "val": 3, "test": 1}',
        DATASET_TESTS,
    ),
]


def run_pytest(tests: list[str]) -> tuple[int, list[str], int]:
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
    rc, _, _ = run_pytest(sorted({t for m in MUTANTS for t in m.tests}))
    if rc != 0:
        print(f"ERROR: los tests ya fallan SIN mutante (rc={rc}).")
        return 2

    exit_code = 0
    for i, m in enumerate(MUTANTS, 1):
        path = APP / m.target
        source = path.read_bytes()
        if source.count(m.original.encode()) != 1:
            print(f"ERROR    [{i:02d}] {m.name}: objetivo no encontrado exactamente una vez")
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
        tests_str = ", ".join(failed) if failed else "—"
        print(f"{status:9} [{i:02d}] {m.name}  ->  {tests_str}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
