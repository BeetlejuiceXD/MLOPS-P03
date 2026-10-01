"""Mutation testing de D04-05 (#74): productor de la evaluación, API y exportación.

Dos suites, cada una en una COPIA AISLADA en un directorio temporal; el árbol de trabajo
del repo nunca se modifica:

- `py`: copia `evaluation/`, `presentation/`, `analyzers/`, los tests del productor y del
  contrato `evaluation_predictions`, `pyproject.toml`, las migraciones del backend (el
  test compara columnas), `backend/tests/fixtures` y los fixtures compartidos.
- `ts`: copia `backend/src`, los tests de evaluación y de contratos, `package.json`,
  `tsconfig.json`, `backend/tests/fixtures`, los fixtures compartidos y el espejo
  `frontend/src/p3/contracts.ts` (lo compara el test de contratos); `node_modules` se
  enlaza (junction en Windows, symlink en Linux), no se copia ni se modifica.

Para cada mutante: el texto original debe aparecer exactamente una vez; se corre la
batería de su suite con reporte JUnit, se registra QUÉ tests fallaron y CÓMO, y se
restaura la copia.

Clasificación de cada test que falla:
- `aserción`: `AssertionError` (o `assert ...`), `pytest.fail(...)` / `Failed: ...`, o un
  `expect` de vitest que no se cumplió.
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = sin errores de colección y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección o runner roto.
  Nunca cuenta como muerto.

Las dos bases sin mutantes deben estar en verde. Sale con 0 solo si todos quedan KILLED.

Uso (desde la raíz del repo, con `npm ci` hecho en backend/; ~3-5 min):
    cd app && uv run python ../.github/scripts/run_evaluation_api_mutations.py
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

PRODUCER, STORE, CONTRACT = (
    "evaluation/producer.py",
    "evaluation/store.py",
    "presentation/contracts.py",
)
PY_TESTS = ["tests/test_evaluation_producer.py", "tests/test_evaluation_predictions_contract.py"]
PY_PACKAGES = ["evaluation", "presentation", "analyzers"]

SERVICE, ROUTES, TS_CONTRACT = (
    "src/logic/evaluation.service.ts",
    "src/ui/evaluation.routes.ts",
    "src/logic/p3.contracts.ts",
)
TS_TESTS = ["tests/evaluation.test.ts", "tests/p3-contracts.test.ts"]


@dataclass(frozen=True)
class Mutant:
    id: str
    suite: str  # "py" | "ts"
    name: str
    target: str
    original: str
    mutant: str


@dataclass(frozen=True)
class Failure:
    test: str
    kind: str


MUTANTS = [
    # --- Productor: compatibilidad de los datos ------------------------------------------
    Mutant(
        "E01",
        "py",
        "namespace desconocido aceptado",
        PRODUCER,
        "    if namespace not in NAMESPACES:",
        "    if False:",
    ),
    Mutant(
        "E02",
        "py",
        "predicciones de otro run aceptadas",
        PRODUCER,
        "    if model_run_id != selection.candidate_run_id:",
        "    if False:",
    ),
    Mutant(
        "E03",
        "py",
        "partición de otro manifest aceptada",
        PRODUCER,
        "    if partition.manifest_hash != selection.manifest_hash:",
        "    if False:",
    ),
    Mutant(
        "E04",
        "py",
        "test_split_hash sin verificar",
        PRODUCER,
        "    if frozen_test_split_hash(list(partition.crop_ids)) != partition.test_split_hash:",
        "    if False:",
    ),
    Mutant(
        "E05",
        "py",
        "crops repetidos aceptados",
        PRODUCER,
        "    if len(set(ids)) != len(ids):",
        "    if False:",
    ),
    Mutant(
        "E06",
        "py",
        "crops faltantes aceptados",
        PRODUCER,
        "    if missing or extra:",
        "    if extra:",
    ),
    Mutant(
        "E07",
        "py",
        "crops ajenos a la partición aceptados",
        PRODUCER,
        "    if missing or extra:",
        "    if missing:",
    ),
    Mutant(
        "E08",
        "py",
        "evaluated_at igual al cierre aceptado",
        PRODUCER,
        "    if evaluated_at <= closed_at:",
        "    if evaluated_at < closed_at:",
    ),
    Mutant(
        "E09",
        "py",
        "sin truncar al milisegundo",
        PRODUCER,
        "    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)",
        "    return moment",
    ),
    Mutant(
        "E10",
        "py",
        "exportación sin ordenar por crop_id",
        PRODUCER,
        "    ordered = sorted(samples, key=lambda sample: sample.crop_id)",
        "    ordered = samples",
    ),
    Mutant(
        "E11",
        "py",
        "predicción inválida no se traduce en rechazo",
        PRODUCER,
        "    except (ValidationError, MetricsInputError) as error:",
        "    except MetricsInputError as error:",
    ),
    Mutant(
        "E12",
        "py",
        "matriz de la exportación con filas = predicha",
        PRODUCER,
        "        rows[index[sample.true_class]][index[sample.predicted_class]] += 1",
        "        rows[index[sample.predicted_class]][index[sample.true_class]] += 1",
    ),
    # --- Guarda de D04-04 y persistencia -------------------------------------------------
    Mutant(
        "E13",
        "py",
        "productor sin guarda de selección cerrada",
        PRODUCER,
        "    selection = store.closed_selection()\n",
        "    selection = ClosedSelection(model_run_id, datetime(2000, 1, 1, tzinfo=UTC), "
        "partition.manifest_hash)\n",
    ),
    Mutant(
        "E14",
        "py",
        "guarda acepta selección no cerrada",
        STORE,
        '        if row is None or row.status != "closed":',
        "        if row is None:",
    ),
    Mutant(
        "E15",
        "py",
        "cierre sin closed_at aceptado",
        STORE,
        ' or not reference.get("manifest_hash") or not row.closed_at:',
        ' or not reference.get("manifest_hash"):',
    ),
    Mutant(
        "E16",
        "py",
        "la evaluación oficial se sobrescribe",
        STORE,
        '                if record.namespace != "official":',
        "                if True:",
    ),
    Mutant(
        "E17",
        "py",
        "la corrida sintética no reemplaza la anterior",
        STORE,
        '                if record.namespace != "official":',
        "                if False:",
    ),
    # --- Contrato Python `evaluation_predictions` ---------------------------------------
    Mutant(
        "E18",
        "py",
        "crop_id repetido aceptado (contrato)",
        CONTRACT,
        "            if sample.crop_id <= previous.crop_id:",
        "            if sample.crop_id < previous.crop_id:",
    ),
    Mutant(
        "E19",
        "py",
        "n_test ≠ número de predicciones (contrato)",
        CONTRACT,
        "        if len(self.predictions) != self.n_test:",
        "        if False:",
    ),
    Mutant(
        "E20",
        "py",
        "falta la probabilidad de una clase (contrato)",
        CONTRACT,
        "            if len(probabilities) != len(self.classes)"
        " or set(probabilities) != set(self.classes):",
        "            if False:",
    ),
    Mutant(
        "E21",
        "py",
        "probabilidades que no suman ~1 (contrato)",
        CONTRACT,
        "            if not _close(sum(probabilities.values()), 1, PROBABILITY_SUM_TOLERANCE):",
        "            if False:",
    ),
    Mutant(
        "E22",
        "py",
        "predicted_class ≠ argmax (contrato)",
        CONTRACT,
        "            if probabilities[sample.predicted_class] != max(probabilities.values()):",
        "            if False:",
    ),
    Mutant(
        "E45",
        "py",
        "evaluated_at con fecha imposible aceptado (contrato)",
        CONTRACT,
        "            _parse_timestamp(value)",
        "            pass",
    ),
    # --- API: guardas -------------------------------------------------------------------
    Mutant(
        "E23",
        "ts",
        "lee la evaluación antes de la guarda",
        SERVICE,
        "    const closed = await guard.requireClosed();\n"
        "    const stored = await repo.read(namespace);\n",
        "    const stored = await repo.read(namespace);\n"
        "    const closed = await guard.requireClosed();\n",
    ),
    Mutant(
        "E24",
        "ts",
        "GET /evaluation sin estado blocked",
        SERVICE,
        "      if (blocked) return blocked;\n",
        "",
    ),
    # --- API: coherencia de lo guardado ---------------------------------------------------
    Mutant(
        "E25",
        "ts",
        "namespace de la fila sin verificar",
        SERVICE,
        "  if (predictions.namespace !== namespace) {",
        "  if (false) {",
    ),
    Mutant(
        "E26",
        "ts",
        "candidato de la evaluación sin verificar",
        SERVICE,
        "    evaluation.selection.candidate_run_id !== closed.candidate_run_id ||",
        "    false ||",
    ),
    Mutant(
        "E27",
        "ts",
        "candidato de la exportación sin verificar",
        SERVICE,
        "    predictions.candidate_run_id !== closed.candidate_run_id\n",
        "    false\n",
    ),
    Mutant(
        "E28",
        "ts",
        "closed_at sin verificar",
        SERVICE,
        "  if (!sameInstant(evaluation.selection.closed_at, closed.closed_at)) {",
        "  if (false) {",
    ),
    Mutant(
        "E29",
        "ts",
        "manifest de la evaluación sin verificar",
        SERVICE,
        "    evaluation.manifest_hash !== closed.manifest_hash ||",
        "    false ||",
    ),
    Mutant(
        "E30",
        "ts",
        "manifest de la exportación sin verificar",
        SERVICE,
        "    predictions.manifest_hash !== closed.manifest_hash\n",
        "    false\n",
    ),
    Mutant(
        "E31",
        "ts",
        "evaluated_at sin verificar",
        SERVICE,
        "  if (!sameInstant(evaluation.evaluated_at, predictions.evaluated_at)) {",
        "  if (false) {",
    ),
    Mutant(
        "E32",
        "ts",
        "orden de clases sin verificar",
        SERVICE,
        "  if (labels.join() !== predictions.classes.join()) {",
        "  if (false) {",
    ),
    Mutant(
        "E33",
        "ts",
        "test_split_hash sin verificar",
        SERVICE,
        "  if (testSplitHash(ids) !== predictions.test_split_hash) {",
        "  if (false) {",
    ),
    Mutant(
        "E34",
        "ts",
        "matriz reconstruida sin comparar",
        SERVICE,
        "  if (JSON.stringify(rebuilt) !== JSON.stringify(evaluation.confusion_matrix.rows)) {",
        "  if (false) {",
    ),
    Mutant(
        "E35",
        "ts",
        "evaluación blocked guardada aceptada",
        SERVICE,
        "    if (!evaluation.success || evaluation.data.state !== 'ready') {",
        "    if (!evaluation.success) {",
    ),
    Mutant(
        "E36",
        "ts",
        "test_split_hash con orden lexicográfico",
        SERVICE,
        "[...cropIds].sort((a, b) => a - b)",
        "[...cropIds].sort()",
    ),
    Mutant(
        "E37",
        "ts",
        "matriz con filas = predicha",
        SERVICE,
        "    const row = rows[labels.indexOf(sample.true_class)];\n"
        "    const column = labels.indexOf(sample.predicted_class);\n",
        "    const row = rows[labels.indexOf(sample.predicted_class)];\n"
        "    const column = labels.indexOf(sample.true_class);\n",
    ),
    Mutant(
        "E38",
        "ts",
        "CSV sin hashes por fila",
        SERVICE,
        "      ...meta,\n",
        "",
    ),
    # --- Rutas --------------------------------------------------------------------------
    Mutant(
        "E39",
        "ts",
        "format desconocido aceptado",
        ROUTES,
        "      if (format !== 'json' && format !== 'csv') {",
        "      if (false) {",
    ),
    Mutant(
        "E40",
        "ts",
        "CSV sin descarga como archivo",
        ROUTES,
        "      res.attachment(`p3-evaluation-predictions-${predictions.namespace}.csv`);\n",
        "",
    ),
    # --- Contrato TS `evaluation_predictions` -------------------------------------------
    Mutant(
        "E41",
        "ts",
        "crop_id repetido aceptado (contrato TS)",
        TS_CONTRACT,
        "      if (previous && sample.crop_id <= previous.crop_id) {",
        "      if (previous && sample.crop_id < previous.crop_id) {",
    ),
    Mutant(
        "E42",
        "ts",
        "n_test ≠ número de predicciones (contrato TS)",
        TS_CONTRACT,
        "    if (exported.predictions.length !== exported.n_test) {",
        "    if (false) {",
    ),
    Mutant(
        "E43",
        "ts",
        "predicted_class ≠ argmax (contrato TS)",
        TS_CONTRACT,
        "      if (sample.probabilities[sample.predicted_class] !== Math.max(...values)) {",
        "      if (false) {",
    ),
    Mutant(
        "E44",
        "ts",
        "namespace libre (contrato TS)",
        TS_CONTRACT,
        "    namespace: z.enum(evaluationNamespaces),",
        "    namespace: z.string(),",
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


def build_isolated_copy(root: Path) -> tuple[Path, Path]:
    """`root/app`, `root/backend` y `root/contracts`: los tests resuelven fixtures y
    migraciones con rutas relativas a la raíz, igual que en el repo."""
    app = root / "app"
    for package in PY_PACKAGES:
        shutil.copytree(APP / package, app / package, ignore=shutil.ignore_patterns("__pycache__"))
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
    shutil.copytree(REPO / "contracts" / "p3" / "fixtures", root / "contracts" / "p3" / "fixtures")
    # p3-contracts.test.ts compara el contrato del backend con su espejo del portal.
    mirror = Path("frontend/src/p3/contracts.ts")
    (root / mirror).parent.mkdir(parents=True)
    shutil.copy2(REPO / mirror, root / mirror)
    _link(BACKEND / "node_modules", backend / "node_modules")
    return app, backend


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


def run_vitest(backend: Path, tmp: Path) -> tuple[int, list[Failure], int]:
    report = tmp / "junit-ts.xml"
    report.unlink(missing_ok=True)
    node = shutil.which("node")
    if node is None:
        raise SystemExit("ERROR: no se encontró `node` en el PATH.")
    proc = subprocess.run(
        [
            node,
            "node_modules/vitest/vitest.mjs",
            "run",
            *TS_TESTS,
            "--reporter=junit",
            f"--outputFile={report}",
        ],
        cwd=backend,
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
    if not (BACKEND / "node_modules").is_dir():
        print("ERROR: falta backend/node_modules (corre `npm ci` en backend/).")
        return 2
    tmp = Path(tempfile.mkdtemp())
    app, backend = build_isolated_copy(tmp / "copy")
    runners = {"py": lambda: run_pytest(app, tmp), "ts": lambda: run_vitest(backend, tmp)}
    roots = {"py": app, "ts": backend}
    results, exit_code = [], 0
    try:
        print(f"Copia aislada: {tmp / 'copy'}")
        for suite, run in runners.items():
            rc, failures, errors = run()
            if rc != 0 or failures or errors:
                print(f"ERROR: la suite {suite} ya falla SIN mutante (rc={rc}).")
                return 2
        print(f"Bases limpias (py y ts) en verde. Mutantes: {len(MUTANTS)}\n", flush=True)

        for m in MUTANTS:
            path = roots[m.suite] / m.target
            source = path.read_bytes()
            if source.count(m.original.encode()) != 1:
                results.append((m, "ERROR", [], "objetivo no encontrado exactamente una vez"))
                exit_code = 2
                print(f"ERROR     {m.id} {m.name}: objetivo no encontrado exactamente una vez")
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
            print(f"{status:9} {m.id} {m.name}  ->  {len(failures)} test(s): {note}", flush=True)
    finally:
        # Nunca borrar a través del enlace: si no se pudo quitar, se deja la copia.
        link = backend / "node_modules"
        try:
            if os.path.lexists(link):
                _unlink(link)
        except OSError as error:
            print(f"AVISO: no se quitó el enlace {link} ({error}); la copia queda en {tmp}.")
        else:
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
        print(f"| {m.id} | `{m.target}` | {m.name} | {status} | {note} | {tests or '—'} |")
    killed = sum(1 for _, s, _, _ in results if s == "KILLED")
    print(f"\nKILLED por aserción {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
