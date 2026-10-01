"""Mutation testing de D03-05 (#62): motor de métricas y espejo de `evaluation_response`.

Trabaja en una COPIA AISLADA: copia a un directorio temporal solo lo que necesitan
los tests (`evaluation/`, `presentation/`, `analyzers/`, los dos archivos de tests,
`pyproject.toml` y los fixtures compartidos de `evaluation_response`) y aplica cada
mutante ahí. El árbol de trabajo del repo nunca se modifica.

Para cada mutante: el texto original debe aparecer exactamente una vez; se corre la
batería, se registra QUÉ tests fallaron y CÓMO (reporte JUnit) y se restaura la copia.

Clasificación de cada test que falla:
- `aserción`: `AssertionError` (o `assert ...`), `pytest.fail(...)` o un
  `pytest.raises` que no se cumplió (`Failed: ...`).
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = sin errores de colección/setup y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección/setup o
  pytest roto (rc distinto de 0/1). Nunca cuenta como muerto.

La base sin mutantes debe estar en verde. Sale con 0 solo si todos quedan KILLED.

Uso (desde la raíz del repo; ~2-3 min):
    cd app && uv run python ../.github/scripts/run_evaluation_metrics_mutations.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
ENGINE, CONTRACT = "evaluation/metrics.py", "presentation/contracts.py"
TESTS = ["tests/test_evaluation_metrics.py", "tests/test_evaluation_contract.py"]
COPIED_PACKAGES = ["evaluation", "presentation", "analyzers"]
FIXTURES = Path("contracts/p3/fixtures/evaluation_response")


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    target: str
    original: str
    mutant: str


MUTANTS = [
    # --- Motor de métricas -----------------------------------------------------------
    Mutant(
        "M01",
        "matriz transpuesta (filas = predichas)",
        ENGINE,
        "counts[index[real]][index[predicted]] += 1",
        "counts[index[predicted]][index[real]] += 1",
    ),
    Mutant(
        "M02",
        "precision dividida entre support",
        ENGINE,
        "precision=_ratio(tp, predicted),",
        "precision=_ratio(tp, support),",
    ),
    Mutant(
        "M03",
        "recall dividido entre predichas",
        ENGINE,
        "recall=_ratio(tp, support),",
        "recall=_ratio(tp, predicted),",
    ),
    Mutant(
        "M04",
        "baseline con la clase minoritaria",
        ENGINE,
        "majority_baseline_accuracy=max(c.support for c in per_class) / n,",
        "majority_baseline_accuracy=min(c.support for c in per_class) / n,",
    ),
    Mutant(
        "M05",
        "macro-F1 ponderado por support",
        ENGINE,
        "macro_f1=sum(c.f1 for c in per_class) / len(per_class),",
        "macro_f1=sum(c.f1 * c.support for c in per_class) / n,",
    ),
    Mutant(
        "M06",
        "macro-F1 ignora clases sin soporte",
        ENGINE,
        "macro_f1=sum(c.f1 for c in per_class) / len(per_class),",
        "macro_f1=sum(c.f1 for c in per_class) / sum(1 for c in per_class if c.support),",
    ),
    Mutant(
        "M07",
        "accuracy redondeada a 4 decimales",
        ENGINE,
        "accuracy=correct / n,",
        "accuracy=round(correct / n, 4),",
    ),
    Mutant(
        "M08",
        "F1 por clase redondeado a 4 decimales",
        ENGINE,
        "f1=_ratio(2 * tp, predicted + support),",
        "f1=round(_ratio(2 * tp, predicted + support), 4),",
    ),
    Mutant(
        "M09",
        "0/0 vale 1 en vez de 0",
        ENGINE,
        "return 0.0 if denominator == 0 else numerator / denominator",
        "return 1.0 if denominator == 0 else numerator / denominator",
    ),
    Mutant(
        "M10",
        "orden de clases según aparición en la entrada",
        ENGINE,
        "index = {label: i for i, label in enumerate(LABELS)}",
        "index = {label: i for i, label in enumerate(dict.fromkeys([*y_true, *LABELS]))}",
    ),
    Mutant(
        "M11",
        "umbral con > en vez de >=",
        ENGINE,
        "return correct * ACCEPTANCE_ACCURACY.denominator >= ACCEPTANCE_ACCURACY.numerator * total",
        "return correct * ACCEPTANCE_ACCURACY.denominator > ACCEPTANCE_ACCURACY.numerator * total",
    ),
    Mutant(
        "M12",
        "umbral sobre accuracy redondeada a 2 decimales",
        ENGINE,
        "return correct * ACCEPTANCE_ACCURACY.denominator >= ACCEPTANCE_ACCURACY.numerator * total",
        "return round(correct / total, 2) >= 0.85",
    ),
    Mutant(
        "M13",
        "aceptación del reporte con accuracy redondeada",
        ENGINE,
        "return meets_acceptance(self.correct, self.n_test)",
        "return round(self.accuracy, 2) >= 0.85",
    ),
    Mutant(
        "M14",
        "sin validar conteos imposibles",
        ENGINE,
        "if total <= 0 or not 0 <= correct <= total:",
        "if total <= 0:",
    ),
    Mutant(
        "M15",
        "sin validar longitudes distintas",
        ENGINE,
        "if len(y_true) != len(y_pred):",
        "if False:",
    ),
    Mutant(
        "M16",
        "sin validar etiquetas desconocidas",
        ENGINE,
        "unknown = {repr(v) for v in values if not isinstance(v, str) or v not in LABELS}",
        "unknown = set()",
    ),
    # --- Espejo del contrato -----------------------------------------------------------
    Mutant(
        "C01",
        "evaluated_at igual al cierre se acepta",
        CONTRACT,
        "if _parse_timestamp(self.evaluated_at) <= _parse_timestamp(self.selection.closed_at):",
        "if _parse_timestamp(self.evaluated_at) < _parse_timestamp(self.selection.closed_at):",
    ),
    Mutant(
        "C02",
        "sin regla de suma = n_test",
        CONTRACT,
        "if sum(map(sum, rows)) != self.n_test:",
        "if False:",
    ),
    Mutant(
        "C03",
        "accuracy con tolerancia 1e-4 en vez de exacta",
        CONTRACT,
        "if not _close(self.metrics.accuracy, trace / self.n_test, EVALUATION_EXACT_TOLERANCE):",
        "if not _close(self.metrics.accuracy, trace / self.n_test, 0.2):",
    ),
    Mutant(
        "C04",
        "sin regla support = suma de la fila",
        CONTRACT,
        "if stats.support != supports[i]:",
        "if False:",
    ),
    Mutant(
        "C05",
        "precision esperada calculada por filas",
        CONTRACT,
        "precision = 0 if predicted == 0 else tp / predicted",
        "precision = 0 if supports[i] == 0 else tp / supports[i]",
    ),
    Mutant(
        "C06",
        "sin regla macro-F1 = promedio",
        CONTRACT,
        "if not _close(self.metrics.macro_f1, macro, EVALUATION_REPORTED_METRIC_TOLERANCE):",
        "if False:",
    ),
    Mutant(
        "C07",
        "baseline con la clase minoritaria",
        CONTRACT,
        "majority = max(supports) / self.n_test",
        "majority = min(supports) / self.n_test",
    ),
    Mutant(
        "C08",
        "sin exigir métricas de cada clase",
        CONTRACT,
        'raise ValueError(f"Faltan métricas de la clase {label}")',
        "continue",
    ),
    Mutant(
        "C09",
        "clases sin exigir exactamente cat/dog",
        CONTRACT,
        "return len(classes) == len(MANIFEST_CLASSES) and set(classes) == set(MANIFEST_CLASSES)",
        "return set(classes) <= set(MANIFEST_CLASSES)",
    ),
]


@dataclass(frozen=True)
class Failure:
    test: str
    kind: str  # "aserción" o "excepción <Tipo>"


def _kind(message: str) -> str:
    """Solo la primera línea: pytest reporta un `assert` simple como "assert 5 == 4"."""
    lines = message.strip().splitlines()
    first = lines[0] if lines else ""
    if first.startswith(("AssertionError", "assert ", "Failed: ")):
        return "aserción"
    return f"excepción {first.split(':', 1)[0].rsplit('.', 1)[-1] or '?'}"


def build_isolated_copy(root: Path) -> Path:
    """`root/app/...` + `root/contracts/...`: los tests ubican los fixtures con
    `parents[2]`, igual que en el repo."""
    app = root / "app"
    for package in COPIED_PACKAGES:
        shutil.copytree(APP / package, app / package, ignore=shutil.ignore_patterns("__pycache__"))
    (app / "tests").mkdir(parents=True)
    for name in ["tests/__init__.py", *TESTS]:
        shutil.copy2(APP / name, app / name)
    shutil.copy2(APP / "pyproject.toml", app / "pyproject.toml")
    shutil.copytree(REPO / FIXTURES, root / FIXTURES)
    return app


def run_pytest(app: Path, tmp: Path) -> tuple[int, list[Failure], int]:
    """Devuelve (returncode, tests fallidos con su tipo, errores de colección/setup)."""
    report = tmp / "junit.xml"
    report.unlink(missing_ok=True)
    env = {
        **os.environ,
        "PYTHONPYCACHEPREFIX": str(tmp / "pyc"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
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
        cwd=app,
        env=env,
        check=False,
        capture_output=True,
    )
    failures, errors = [], 0
    if report.is_file():
        for case in ET.parse(report).getroot().iter("testcase"):
            failure = case.find("failure")
            if failure is not None:
                failures.append(Failure(case.get("name", "?"), _kind(failure.get("message", ""))))
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
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        app = build_isolated_copy(tmp / "copy")
        print(f"Copia aislada: {app}")

        rc, failures, errors = run_pytest(app, tmp)
        if rc != 0 or failures or errors:
            print(f"ERROR: los tests ya fallan SIN mutante (rc={rc}); no se puede evaluar.")
            return 2
        print(f"Base limpia: tests en verde. Mutantes: {len(MUTANTS)}\n")

        results, exit_code = [], 0
        for m in MUTANTS:
            path = app / m.target
            source = path.read_bytes()
            if source.count(m.original.encode()) != 1:
                results.append((m, "ERROR", [], "objetivo no encontrado exactamente una vez"))
                exit_code = 2
                print(f"ERROR     {m.id} {m.name}: objetivo no encontrado exactamente una vez")
                continue
            try:
                path.write_bytes(source.replace(m.original.encode(), m.mutant.encode()))
                rc, failures, errors = run_pytest(app, tmp)
            finally:
                path.write_bytes(source)
            if errors or rc not in (0, 1):
                status, note = "ERROR", f"rc={rc}, errores de colección/setup={errors}"
                exit_code = 2
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
