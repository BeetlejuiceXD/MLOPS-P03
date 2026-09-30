"""Mutation testing de D03-02 (#59): early stopping y mejor checkpoint.

Para cada mutante: aplica el cambio en `app/trainer/engine.py` o `app/trainer/metrics.py`
(el texto original debe aparecer exactamente una vez), corre los tests del área,
registra QUÉ tests fallaron y CÓMO, y restaura el archivo en `finally`.

Clasificación de cada test que falla (del reporte JUnit de pytest):
- `aserción`: veredicto explícito del test: `AssertionError` (o `assert ...`), `pytest.fail(...)` o
  `pytest.raises` que no se cumplió (`Failed: ...`).
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = pytest corrió sin errores de colección/setup y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección/setup o
  pytest roto (rc distinto de 0/1). Nunca cuenta como muerto.

La base sin mutantes debe estar en verde. Sale con 0 solo si todos quedan KILLED.

Uso (desde la raíz del repo; ~15-20 min en total):
    cd app && uv run python ../.github/scripts/run_early_stopping_mutations.py
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
ENGINE, METRICS = "trainer/engine.py", "trainer/metrics.py"
TESTS = ["tests/test_trainer_early_stopping.py", "tests/test_trainer_metrics.py"]


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    target: str
    original: str
    mutant: str


MUTANTS = [
    Mutant(
        "E01",
        "no restaura el mejor estado al terminar",
        ENGINE,
        "    model.load_state_dict(best_state)\n",
        "    pass\n",
    ),
    Mutant(
        "E02",
        "mejor estado sin copiar (referencia a los pesos vivos)",
        ENGINE,
        "best, best_state = metrics, copy.deepcopy(model.state_dict())",
        "best, best_state = metrics, model.state_dict()",
    ),
    Mutant(
        "E03",
        "la paciencia se reinicia con macro-F1/loss (is_better)",
        ENGINE,
        "if accuracy_baseline is None or accuracy_improved(metrics, accuracy_baseline):",
        "if accuracy_baseline is None or is_better(metrics, accuracy_baseline):",
    ),
    Mutant(
        "E04",
        "el contador de paciencia no se reinicia al mejorar",
        ENGINE,
        "            epochs_without_improvement = 0\n",
        "            pass\n",
    ),
    Mutant(
        "E05",
        "parada con > patience en vez de >=",
        METRICS,
        "return epochs_without_improvement >= patience",
        "return epochs_without_improvement > patience",
    ),
    Mutant(
        "E06",
        "mejora de accuracy sin redondear a 4 decimales",
        METRICS,
        "return round(candidate.val_accuracy, 4) > round(reference.val_accuracy, 4)",
        "return candidate.val_accuracy > reference.val_accuracy",
    ),
    Mutant(
        "E07",
        "desempate por val_loss sin redondear a 4 decimales",
        METRICS,
        "if round(candidate.val_loss, 4) != round(current_best.val_loss, 4):",
        "if candidate.val_loss != current_best.val_loss:",
    ),
    Mutant(
        "E08",
        "checkpoint solo por accuracy (sin desempate macro-F1/loss)",
        ENGINE,
        "if best is None or is_better(metrics, best):",
        "if best is None or accuracy_improved(metrics, best):",
    ),
    Mutant(
        "E09",
        "best reportado = última época",
        ENGINE,
        "return TrainingResult(tuple(history), best, model, stopped_early)",
        "return TrainingResult(tuple(history), history[-1], model, stopped_early)",
    ),
]


@dataclass(frozen=True)
class Failure:
    test: str
    kind: str  # "aserción" o "excepción <Tipo>"


def _kind(message: str) -> str:
    """Solo la primera línea: pytest reporta un `assert` simple como "assert 5 == 4"
    (sin el prefijo AssertionError) y a veces con varias líneas de detalle."""
    lines = message.strip().splitlines()
    first = lines[0] if lines else ""
    if first.startswith(("AssertionError", "assert ", "Failed: ")):
        return "aserción"
    return f"excepción {first.split(':', 1)[0].rsplit('.', 1)[-1] or '?'}"


def run_pytest() -> tuple[int, list[Failure], int]:
    """Devuelve (returncode, tests fallidos con su tipo, errores de colección/setup)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        env = {
            **os.environ,
            "PYTHONPYCACHEPREFIX": str(Path(tmp) / "pyc"),
            "PYTHONIOENCODING": "utf-8",
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
                *TESTS,
            ],
            cwd=APP,
            env=env,
            check=False,
            capture_output=True,
        )
        failures, errors = [], 0
        if report.is_file():
            for case in ET.parse(report).getroot().iter("testcase"):
                failure = case.find("failure")
                if failure is not None:
                    failures.append(
                        Failure(case.get("name", "?"), _kind(failure.get("message", "")))
                    )
                if case.find("error") is not None:
                    errors += 1
        else:
            errors += 1
        return proc.returncode, failures, errors


def _detection(failures: list[Failure]) -> str:
    kinds: dict[str, int] = {}
    for f in failures:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    return ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items()))


def main() -> int:
    rc, failures, errors = run_pytest()
    if rc != 0 or failures or errors:
        print(f"ERROR: los tests ya fallan SIN mutante (rc={rc}); no se puede evaluar.")
        return 2
    print(f"Base limpia: tests en verde. Mutantes: {len(MUTANTS)}\n")

    results, exit_code = [], 0
    for m in MUTANTS:
        path = APP / m.target
        source = path.read_bytes()
        if source.count(m.original.encode()) != 1:
            results.append((m, "ERROR", [], "objetivo no encontrado exactamente una vez"))
            exit_code = 2
            print(f"ERROR     {m.id} {m.name}: objetivo no encontrado exactamente una vez")
            continue
        try:
            path.write_bytes(source.replace(m.original.encode(), m.mutant.encode()))
            rc, failures, errors = run_pytest()
        finally:
            path.write_bytes(source)
        if errors or rc not in (0, 1):
            status, note, exit_code = "ERROR", f"rc={rc}, errores de colección/setup={errors}", 2
        elif any(f.kind == "aserción" for f in failures):
            status, note = "KILLED", _detection(failures)
        elif failures:
            status, note = "KILLED (solo excepción)", _detection(failures)
            exit_code = max(exit_code, 1)
        else:
            status, note, exit_code = "SURVIVED", "—", max(exit_code, 1)
        results.append((m, status, failures, note))
        print(f"{status:9} {m.id} {m.name}  ->  {len(failures)} test(s): {note}")

    print(
        "\n| # | Archivo | Mutación | Resultado | Tipo de detección | Tests que la detectan |"
        "\n|---|---|---|---|---|---|"
    )
    for m, status, failures, note in results:
        asserted = [f.test for f in failures if f.kind == "aserción"]
        others = [f.test for f in failures if f.kind != "aserción"]
        shown = asserted[:3] + others[: max(0, 3 - len(asserted))]
        tests = "<br>".join(f"`{t}`" for t in shown)
        if len(failures) > len(shown):
            tests += f"<br>(+{len(failures) - len(shown)} más)"
        print(f"| {m.id} | `{m.target}` | {m.name} | {status} | {note} | {tests or '—'} |")
    killed = sum(1 for _, s, _, _ in results if s == "KILLED")
    print(f"\nKILLED por aserción {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
