"""Mutation testing de D05-05 (#88): estados de Evaluation (blocked/pending/ready/fallo).

Tres suites, cada una en una COPIA AISLADA en un directorio temporal; el árbol de trabajo
del repo nunca se modifica:

- `py`: copia `evaluation/`, `presentation/`, `analyzers/`, los tests del contrato
  `evaluation_response`, del motor de métricas y del productor, `pyproject.toml`,
  `backend/src` (el test del productor compara columnas con las migraciones),
  `backend/tests/fixtures` y los fixtures compartidos.
- `ts`: copia `backend/src`, los tests de evaluación, contratos y selección,
  `package.json`, `tsconfig.json`, `backend/tests/fixtures`, los fixtures compartidos y el
  espejo `frontend/src/p3/contracts.ts`; `node_modules` se enlaza.
- `fe`: copia `frontend/src`, los tests de la página Evaluation y de contratos,
  `tests/p3-fixtures.ts`, `package.json`, `tsconfig.json` y `vitest.config.ts`;
  `node_modules` se enlaza (junction en Windows, symlink en Linux), no se copia.

Para cada mutante: el texto original debe aparecer exactamente una vez; se corre la
batería de su suite con reporte JUnit, se registra QUÉ tests fallaron y CÓMO, y se
restaura la copia.

Clasificación de cada test que falla:
- `aserción`: `AssertionError` (o `assert ...`), `pytest.fail(...)` / `Failed: ...`, o un
  `expect` de vitest que no se cumplió (incluido un `findBy*` que no encuentra el estado).
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = sin errores de colección y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección o runner roto.
  Nunca cuenta como muerto.

Las tres bases sin mutantes deben estar en verde. Sale con 0 solo si todos quedan KILLED.

Uso (desde la raíz del repo, con `npm ci` hecho en backend/ y frontend/; ~5-8 min):
    cd app && uv run python ../.github/scripts/run_evaluation_states_mutations.py
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
BACKEND = REPO / "backend"
FRONTEND = REPO / "frontend"

PY_CONTRACT, PY_METRICS, PY_PRODUCER = (
    "presentation/contracts.py",
    "evaluation/metrics.py",
    "evaluation/producer.py",
)
PY_TESTS = [
    "tests/test_evaluation_contract.py",
    "tests/test_evaluation_metrics.py",
    "tests/test_evaluation_producer.py",
]
PY_PACKAGES = ["evaluation", "presentation", "analyzers"]

SERVICE, TS_CONTRACT = "src/logic/evaluation.service.ts", "src/logic/p3.contracts.ts"
TS_TESTS = [
    "tests/evaluation.test.ts",
    "tests/p3-contracts.test.ts",
    "tests/model-selection.test.ts",
]

PAGE, API, FE_CONTRACT = (
    "src/p3/pages/Evaluation.tsx",
    "src/p3/api.ts",
    "src/p3/contracts.ts",
)
FE_TESTS = ["tests/p3-evaluation.test.tsx", "tests/p3-contracts.test.ts"]


@dataclass(frozen=True)
class Mutant:
    id: str
    suite: str  # "py" | "ts" | "fe"
    name: str
    target: str
    original: str
    mutant: str


@dataclass(frozen=True)
class Failure:
    test: str
    kind: str


MUTANTS = [
    # --- API (backend) --------------------------------------------------------------------
    Mutant(
        "T01",
        "ts",
        "cerrada sin resultado vuelve a 404 en vez de pending",
        SERVICE,
        "if (!stored) return { evaluation: pendingEvaluation(closed, namespace), predictions: null };",
        "if (!stored) throw new NotFoundError(missingDetail(namespace));",
    ),
    Mutant(
        "T02",
        "ts",
        "pending siempre se declara official",
        SERVICE,
        "    namespace,\n    reason: 'evaluation_missing',",
        "    namespace: 'official',\n    reason: 'evaluation_missing',",
    ),
    Mutant(
        "T03",
        "ts",
        "pending con un candidato fijo en vez del cierre persistido",
        SERVICE,
        "      candidate_run_id: closed.candidate_run_id,\n      metric: 'val_accuracy',",
        "      candidate_run_id: 'a'.repeat(32),\n      metric: 'val_accuracy',",
    ),
    Mutant(
        "T04",
        "ts",
        "pending con un manifest fijo en vez del cierre persistido",
        SERVICE,
        "    manifest_hash: closed.manifest_hash,\n    detail: missingDetail(namespace),",
        "    manifest_hash: 'd'.repeat(64),\n    detail: missingDetail(namespace),",
    ),
    Mutant(
        "T05",
        "ts",
        "no comprueba el namespace de la evaluación guardada",
        SERVICE,
        "if (evaluation.namespace !== namespace) {",
        "if (false) {",
    ),
    Mutant(
        "T06",
        "ts",
        "la exportación sin resultado no es 404",
        SERVICE,
        "if (!predictions) throw new NotFoundError(missingDetail(namespace));",
        "if (!predictions) return predictions as never;",
    ),
    Mutant(
        "T07",
        "ts",
        "acepta una evaluación guardada que no está en ready",
        SERVICE,
        "if (!evaluation.success || evaluation.data.state !== 'ready') {",
        "if (!evaluation.success) {",
    ),
    # --- Contrato compartido (backend) ----------------------------------------------------
    Mutant(
        "C01",
        "ts",
        "ready sin namespace obligatorio",
        TS_CONTRACT,
        "    namespace: evaluationNamespaceSchema,\n    selection: evaluationSelectionSchema,\n    manifest_hash: sha256Schema,\n    evaluated_at",
        "    namespace: evaluationNamespaceSchema.optional(),\n    selection: evaluationSelectionSchema,\n    manifest_hash: sha256Schema,\n    evaluated_at",
    ),
    Mutant(
        "C02",
        "ts",
        "local_test aceptado como namespace de Evaluation",
        TS_CONTRACT,
        "const evaluationNamespaceSchema = z.enum(evaluationNamespaces);",
        "const evaluationNamespaceSchema = z.enum([...evaluationNamespaces, 'local_test']);",
    ),
    Mutant(
        "C03",
        "ts",
        "pending admite campos extra (resultados)",
        TS_CONTRACT,
        "const evaluationPendingSchema = z.strictObject({",
        "const evaluationPendingSchema = z.object({",
    ),
    Mutant(
        "C04",
        "ts",
        "pending fuera de la unión",
        TS_CONTRACT,
        "  evaluationBlockedSchema,\n  evaluationPendingSchema,\n  evaluationReadySchema,",
        "  evaluationBlockedSchema,\n  evaluationReadySchema,",
    ),
    # --- Página (frontend) ----------------------------------------------------------------
    Mutant(
        "F01",
        "fe",
        "nunca avisa que un resultado synthetic es de prueba",
        PAGE,
        '{data.namespace === "synthetic" && (',
        "{false && (",
    ),
    Mutant(
        "F02",
        "fe",
        "avisa de prueba también en official",
        PAGE,
        '{data.namespace === "synthetic" && (',
        "{true && (",
    ),
    Mutant(
        "F03",
        "fe",
        "el badge siempre dice official",
        PAGE,
        "      {namespace}\n    </span>",
        "      official\n    </span>",
    ),
    Mutant(
        "F04",
        "fe",
        "pending se muestra como bloqueado",
        PAGE,
        'if (data.state === "pending") {\n            return (\n              <div',
        'if (data.state === "pending") {\n            return (\n              <StatePanel variant="blocked" title="El test sigue cerrado." />\n            );\n            return (\n              <div',
    ),
    Mutant(
        "F05",
        "fe",
        "el 503 no muestra su motivo",
        API,
        'useValidatedFetch("/evaluation", evaluationResponseSchema, WITH_REASON);',
        'useValidatedFetch("/evaluation", evaluationResponseSchema);',
    ),
    Mutant(
        "F06",
        "fe",
        "sin la procedencia (manifest) en ready",
        PAGE,
        'Manifest <span className="font-mono">{shortHash(data.manifest_hash)}</span>',
        "Manifest",
    ),
    Mutant(
        "F07",
        "fe",
        "espejo del portal sin pending",
        FE_CONTRACT,
        "  evaluationBlockedSchema,\n  evaluationPendingSchema,\n  evaluationReadySchema,",
        "  evaluationBlockedSchema,\n  evaluationReadySchema,",
    ),
    Mutant(
        "F08",
        "fe",
        "espejo del portal acepta local_test",
        FE_CONTRACT,
        "const evaluationNamespaceSchema = z.enum(evaluationNamespaces);",
        'const evaluationNamespaceSchema = z.enum([...evaluationNamespaces, "local_test"]);',
    ),
    # --- Python ---------------------------------------------------------------------------
    Mutant(
        "P01",
        "py",
        "ready sin namespace obligatorio",
        PY_CONTRACT,
        '    state: Literal["ready"]\n    namespace: EvaluationNamespace\n',
        '    state: Literal["ready"]\n    namespace: EvaluationNamespace | None = None\n',
    ),
    Mutant(
        "P02",
        "py",
        "local_test aceptado como namespace de Evaluation",
        PY_CONTRACT,
        'EvaluationNamespace = Literal["official", "synthetic"]',
        'EvaluationNamespace = Literal["official", "synthetic", "local_test"]',
    ),
    Mutant(
        "P03",
        "py",
        "pending fuera de la unión",
        PY_CONTRACT,
        "    EvaluationBlocked | EvaluationPending | EvaluationReady, Field(",
        "    EvaluationBlocked | EvaluationReady, Field(",
    ),
    Mutant(
        "P04",
        "py",
        "el productor declara siempre official",
        PY_PRODUCER,
        "        namespace=namespace,\n        candidate_run_id=selection.candidate_run_id,",
        '        namespace="official",\n        candidate_run_id=selection.candidate_run_id,',
    ),
    Mutant(
        "P05",
        "py",
        "el motor ignora el namespace recibido",
        PY_METRICS,
        '"namespace": namespace,',
        '"namespace": "synthetic",',
    ),
]


def _py_kind(message: str) -> str:
    """Solo la primera línea: pytest reporta un `assert` simple como "assert 5 == 4"."""
    lines = message.strip().splitlines()
    first = lines[0] if lines else ""
    if first.startswith(("AssertionError", "assert ", "Failed: ")):
        return "aserción"
    return f"excepción {first.split(':', 1)[0].rsplit('.', 1)[-1] or '?'}"


def _ts_kind(failure: ET.Element) -> str:
    kind = failure.get("type", "")
    return "aserción" if kind == "AssertionError" else f"excepción {kind or '?'}"


def _link(target: Path, link: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def _unlink(link: Path) -> None:
    """Quita el enlace a node_modules SIN tocar su destino antes de borrar la copia."""
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def build_isolated_copy(root: Path) -> tuple[Path, Path, Path]:
    """`root/app`, `root/backend`, `root/frontend` y `root/contracts`: los tests resuelven
    fixtures y migraciones con rutas relativas a la raíz, igual que en el repo."""
    app = root / "app"
    for package in PY_PACKAGES:
        shutil.copytree(
            APP / package, app / package, ignore=shutil.ignore_patterns("__pycache__")
        )
    (app / "tests").mkdir(parents=True)
    for name in ["tests/__init__.py", *PY_TESTS]:
        shutil.copy2(APP / name, app / name)
    shutil.copy2(APP / "pyproject.toml", app / "pyproject.toml")

    backend = root / "backend"
    shutil.copytree(BACKEND / "src", backend / "src")
    (backend / "tests").mkdir(parents=True)
    for name in TS_TESTS:
        shutil.copy2(BACKEND / name, backend / name)
    shutil.copytree(BACKEND / "tests" / "fixtures", backend / "tests" / "fixtures")
    for name in ("package.json", "tsconfig.json"):
        shutil.copy2(BACKEND / name, backend / name)
    shutil.copytree(
        REPO / "contracts" / "p3" / "fixtures", root / "contracts" / "p3" / "fixtures"
    )
    _link(BACKEND / "node_modules", backend / "node_modules")

    # El portal completo (src): p3-contracts.test.ts del backend lee además su espejo.
    frontend = root / "frontend"
    shutil.copytree(FRONTEND / "src", frontend / "src")
    (frontend / "tests").mkdir(parents=True)
    for name in [*FE_TESTS, "tests/p3-fixtures.ts"]:
        shutil.copy2(FRONTEND / name, frontend / name)
    for name in ("package.json", "tsconfig.json", "vitest.config.ts"):
        shutil.copy2(FRONTEND / name, frontend / name)
    _link(FRONTEND / "node_modules", frontend / "node_modules")
    return app, backend, frontend


def _parse(report: Path, kind) -> tuple[list[Failure], int]:
    failures, errors = [], 0
    if not report.is_file():
        return failures, 1
    for case in ET.parse(report).getroot().iter("testcase"):
        failure = case.find("failure")
        if failure is not None:
            failures.append(Failure(case.get("name", "?"), kind(failure)))
        if case.find("error") is not None:
            errors += 1
    return failures, errors


def run_pytest(app: Path, tmp: Path) -> tuple[int, list[Failure], int]:
    report = tmp / "junit-py.xml"
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
            *PY_TESTS,
        ],
        cwd=app,
        env=env,
        check=False,
        capture_output=True,
    )
    failures, errors = _parse(report, lambda f: _py_kind(f.get("message", "")))
    return proc.returncode, failures, errors


def run_vitest(
    cwd: Path, tmp: Path, tests: list[str], name: str
) -> tuple[int, list[Failure], int]:
    report = tmp / f"junit-{name}.xml"
    report.unlink(missing_ok=True)
    node = shutil.which("node")
    if node is None:
        raise SystemExit("ERROR: no se encontró `node` en el PATH.")
    proc = subprocess.run(
        [
            node,
            "node_modules/vitest/vitest.mjs",
            "run",
            *tests,
            "--reporter=junit",
            f"--outputFile={report}",
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    failures, errors = _parse(report, _ts_kind)
    return proc.returncode, failures, errors


def _detection(failures: list[Failure]) -> str:
    kinds: dict[str, int] = {}
    for f in failures:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    return ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items()))


def main() -> int:
    # La consola de Windows (cp1252) no imprime `≠` ni `→`: salida siempre en UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    for package in (BACKEND, FRONTEND):
        if not (package / "node_modules").is_dir():
            print(f"ERROR: falta {package.name}/node_modules (corre `npm ci` ahí).")
            return 2
    tmp = Path(tempfile.mkdtemp())
    app, backend, frontend = build_isolated_copy(tmp / "copy")
    runners = {
        "py": lambda: run_pytest(app, tmp),
        "ts": lambda: run_vitest(backend, tmp, TS_TESTS, "ts"),
        "fe": lambda: run_vitest(frontend, tmp, FE_TESTS, "fe"),
    }
    roots = {"py": app, "ts": backend, "fe": frontend}
    results, exit_code = [], 0
    try:
        print(f"Copia aislada: {tmp / 'copy'}")
        for suite, run in runners.items():
            rc, failures, errors = run()
            if rc != 0 or failures or errors:
                print(f"ERROR: la suite {suite} ya falla SIN mutante (rc={rc}).")
                return 2
        print(
            f"Bases limpias (py, ts y fe) en verde. Mutantes: {len(MUTANTS)}\n",
            flush=True,
        )

        for m in MUTANTS:
            path = roots[m.suite] / m.target
            source = path.read_bytes()
            if source.count(m.original.encode()) != 1:
                results.append(
                    (m, "ERROR", [], "objetivo no encontrado exactamente una vez")
                )
                exit_code = 2
                print(
                    f"ERROR     {m.id} {m.name}: objetivo no encontrado exactamente una vez"
                )
                continue
            try:
                path.write_bytes(source.replace(m.original.encode(), m.mutant.encode()))
                rc, failures, errors = runners[m.suite]()
            finally:
                path.write_bytes(source)
            if errors or rc not in (0, 1):
                status, note = "ERROR", f"rc={rc}, errores de colección={errors}"
                exit_code = 2
            elif any(f.kind == "aserción" for f in failures):
                status, note = "KILLED", _detection(failures)
            elif failures:
                status, note = "KILLED (solo excepción)", _detection(failures)
                exit_code = max(exit_code, 1)
            else:
                status, note, exit_code = "SURVIVED", "—", max(exit_code, 1)
            results.append((m, status, failures, note))
            print(
                f"{status:9} {m.id} {m.name}  ->  {len(failures)} test(s): {note}",
                flush=True,
            )
    finally:
        # Nunca borrar a través de un enlace: si alguno no se pudo quitar, se deja la copia.
        removed = True
        for link in (backend / "node_modules", frontend / "node_modules"):
            try:
                if os.path.lexists(link):
                    _unlink(link)
            except OSError as error:
                removed = False
                print(
                    f"AVISO: no se quitó el enlace {link} ({error}); la copia queda en {tmp}."
                )
        if removed:
            shutil.rmtree(tmp, ignore_errors=True)

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
        print(
            f"| {m.id} | `{m.target}` | {m.name} | {status} | {note} | {tests or '—'} |"
        )
    killed = sum(1 for _, s, _, _ in results if s == "KILLED")
    print(f"\nKILLED por aserción {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
