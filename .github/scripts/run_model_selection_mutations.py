"""Mutation testing de D04-04 (#73): selección por validation, estado y guardas del test.

Trabaja en una COPIA AISLADA: copia a un directorio temporal `backend/src`, el test
`backend/tests/model-selection.test.ts`, `package.json`, `tsconfig.json` y los fixtures
de `contracts/p3/fixtures`; `node_modules` se enlaza (junction en Windows, symlink en
Linux), no se copia ni se modifica. El árbol de trabajo del repo nunca se toca.

Para cada mutante: el texto original debe aparecer exactamente una vez; se corre vitest
con reporte JUnit, se registra QUÉ tests fallaron y CÓMO, y se restaura la copia.

Clasificación de cada test que falla (atributo `type` del JUnit de vitest):
- `aserción`: `AssertionError` (un `expect` que no se cumplió).
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = sin errores de colección y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección o vitest roto.
  Nunca cuenta como muerto.

La base sin mutantes debe estar en verde. Sale con 0 solo si todos quedan KILLED.

Uso (desde la raíz del repo, con `npm ci` hecho en backend/; ~3-5 min):
    python .github/scripts/run_model_selection_mutations.py
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
BACKEND = REPO / "backend"
SELECTION = "src/logic/model-selection.ts"
SERVICE = "src/logic/model-selection.service.ts"
ROUTES = "src/ui/model-selection.routes.ts"
TEST = "tests/model-selection.test.ts"


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    target: str
    original: str
    mutant: str


MUTANTS = [
    # --- Ranking -----------------------------------------------------------------------
    Mutant(
        "S01",
        "menor val_accuracy primero",
        SELECTION,
        "at4(b.val_accuracy) - at4(a.val_accuracy) ||",
        "at4(a.val_accuracy) - at4(b.val_accuracy) ||",
    ),
    Mutant(
        "S02",
        "val_accuracy sin igualdad a 4 decimales",
        SELECTION,
        "at4(b.val_accuracy) - at4(a.val_accuracy) ||",
        "b.val_accuracy - a.val_accuracy ||",
    ),
    Mutant(
        "S03",
        "sin desempate por macro-F1",
        SELECTION,
        "at4(b.val_macro_f1) - at4(a.val_macro_f1) ||",
        "",
    ),
    Mutant(
        "S04",
        "macro-F1 sin igualdad a 4 decimales",
        SELECTION,
        "at4(b.val_macro_f1) - at4(a.val_macro_f1) ||",
        "b.val_macro_f1 - a.val_macro_f1 ||",
    ),
    Mutant(
        "S05",
        "gana la MAYOR val_loss",
        SELECTION,
        "at4(a.val_loss) - at4(b.val_loss) ||",
        "at4(b.val_loss) - at4(a.val_loss) ||",
    ),
    Mutant(
        "S06",
        "val_loss sin igualdad a 4 decimales",
        SELECTION,
        "at4(a.val_loss) - at4(b.val_loss) ||",
        "a.val_loss - b.val_loss ||",
    ),
    Mutant(
        "S07",
        "gana el MAYOR run_id",
        SELECTION,
        "    compareText(a.run_id, b.run_id)\n  );",
        "    compareText(b.run_id, a.run_id)\n  );",
    ),
    Mutant(
        "S08",
        "accuracy de la última época en vez del mejor checkpoint",
        SELECTION,
        "val_accuracy: run.summary.best_val_accuracy,",
        "val_accuracy: run.history[run.history.length - 1]?.val_accuracy ?? 0,",
    ),
    # --- Elegibilidad --------------------------------------------------------------------
    Mutant(
        "S09",
        "acepta runs no FINISHED con resumen",
        SELECTION,
        "if (run.status !== 'FINISHED' || run.summary === null) {",
        "if (run.summary === null) {",
    ),
    Mutant(
        "S10",
        "la fila de la matriz ignora la seed",
        SELECTION,
        "keys.every((key) => entry.config[key] === params[key]),",
        "keys.every((key) => key === 'seed' || entry.config[key] === params[key]),",
    ),
    Mutant(
        "S11",
        "la fila de la matriz ignora ejes no barridos (patience)",
        SELECTION,
        "keys.every((key) => entry.config[key] === params[key]),",
        "keys.every((key) => key === 'patience' || entry.config[key] === params[key]),",
    ),
    Mutant(
        "S12",
        "sin comprobar el manifest congelado",
        SELECTION,
        "if (tags.manifest_hash !== reference.manifest_hash) {",
        "if (false) {",
    ),
    Mutant(
        "S13",
        "release comprobado sin dvc_release_hash",
        SELECTION,
        "tags.dvc_release !== reference.dataset_version ||\n"
        "    tags.dvc_release_hash !== reference.dvc_release_hash",
        "tags.dvc_release !== reference.dataset_version",
    ),
    Mutant(
        "S14",
        "sin comprobar sha256(images:annotations)",
        SELECTION,
        "if (tags.dvc_release_hash !== expectedHash) {",
        "if (false) {",
    ),
    Mutant(
        "S15",
        "fila repetida: cuenta la mejor, no la más temprana",
        SELECTION,
        "Date.parse(a.start_time) - Date.parse(b.start_time) || compareText(a.run_id, b.run_id),",
        "compareRanked(a, b),",
    ),
    Mutant(
        "S16",
        "fila repetida: cuentan todas las corridas",
        SELECTION,
        "if (first) ranking.push(first);",
        "ranking.push(...candidates);",
    ),
    Mutant(
        "S17",
        "cierre exige MÁS de 10 filas",
        SELECTION,
        "ready_to_close: campaignRows.length >= MIN_COMPARABLE_RUNS,",
        "ready_to_close: campaignRows.length > MIN_COMPARABLE_RUNS,",
    ),
    Mutant(
        "S18",
        "run_id repetido se ignora",
        SELECTION,
        "if (seen.has(id)) throw new ValidationError(`run_id repetido en la fuente: ${id}`);",
        "if (seen.has(id)) continue;",
    ),
    # --- Estado y guardas ----------------------------------------------------------------
    Mutant(
        "V01",
        "propose no se niega de entrada tras el cierre",
        SERVICE,
        "if ((await repo.read()).status === 'closed') {\n"
        "        throw new ConflictError('La selección ya está cerrada: no se vuelve a seleccionar.');",
        "if (false) {\n"
        "        throw new ConflictError('La selección ya está cerrada: no se vuelve a seleccionar.');",
    ),
    Mutant(
        "V02",
        "propose guarda aunque no haya candidato",
        SERVICE,
        "if (outcome.candidate === null) {",
        "if (false) {",
    ),
    Mutant(
        "V03",
        "close no exige el run_id del candidato",
        SERVICE,
        "if (proposed.candidate.run_id !== runId.data) {",
        "if (false) {",
    ),
    Mutant(
        "V04",
        "close no exige 10 filas comparables",
        SERVICE,
        "if (!proposed.ready_to_close) {",
        "if (false) {",
    ),
    Mutant(
        "V05",
        "close no detecta que la campaña cambió",
        SERVICE,
        "if (current.outcome_hash !== proposed.outcome_hash) {",
        "if (false) {",
    ),
    Mutant(
        "V06",
        "close sin validar el formato del run_id",
        SERVICE,
        "if (!runId.success) {",
        "if (false) {",
    ),
    Mutant(
        "V07",
        "close ignora el resultado de la escritura condicional",
        SERVICE,
        "if (!(await repo.close(proposed.outcome_hash, clock()))) {",
        "if ((await repo.close(proposed.outcome_hash, clock())) && false) {",
    ),
    Mutant(
        "V08",
        "propose acepta un manifest no congelado",
        SERVICE,
        "if (!parsed.data.frozen) {",
        "if (false) {",
    ),
    Mutant(
        "V09",
        "requireClosed acepta un candidato preparatorio",
        SERVICE,
        "if (status !== 'closed') {",
        "if (status === 'open') {",
    ),
    Mutant(
        "V10",
        "Evaluation se desbloquea con un candidato preparatorio",
        SERVICE,
        "if ((await repo.read()).status === 'closed') return null;",
        "if ((await repo.read()).status !== 'open') return null;",
    ),
    # --- API ----------------------------------------------------------------------------
    Mutant(
        "R01",
        "GET /evaluation cerrada responde 200 sin evaluación oficial",
        ROUTES,
        "if (blocked) return blocked;",
        "return blocked ?? { state: 'ready' };",
    ),
    Mutant(
        "R02",
        "POST /selection/close lee otro campo del body",
        ROUTES,
        "?.candidate_run_id),",
        "?.run_id),",
    ),
]


@dataclass(frozen=True)
class Failure:
    test: str
    kind: str  # "aserción" o "excepción <Tipo>"


def _link(target: Path, link: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def build_isolated_copy(root: Path) -> Path:
    """`root/backend/...` + `root/contracts/p3/fixtures`: los tests resuelven los fixtures
    con `path.resolve('../contracts/p3/fixtures')`, igual que en el repo."""
    backend = root / "backend"
    shutil.copytree(BACKEND / "src", backend / "src")
    (backend / "tests").mkdir(parents=True)
    shutil.copy2(BACKEND / TEST, backend / TEST)
    for name in ("package.json", "tsconfig.json"):
        shutil.copy2(BACKEND / name, backend / name)
    shutil.copytree(
        REPO / "contracts" / "p3" / "fixtures", root / "contracts" / "p3" / "fixtures"
    )
    _link(BACKEND / "node_modules", backend / "node_modules")
    return backend


def _unlink(link: Path) -> None:
    """Quita el enlace a node_modules SIN tocar su destino antes de borrar la copia."""
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def run_vitest(backend: Path, report: Path) -> tuple[int, list[Failure], int]:
    """Devuelve (returncode, tests fallidos con su tipo, errores de colección)."""
    report.unlink(missing_ok=True)
    node = shutil.which("node")
    if node is None:
        raise SystemExit("ERROR: no se encontró `node` en el PATH.")
    proc = subprocess.run(
        [
            node,
            "node_modules/vitest/vitest.mjs",
            "run",
            TEST,
            "--reporter=junit",
            f"--outputFile={report}",
        ],
        cwd=backend,
        check=False,
        capture_output=True,
    )
    failures, errors = [], 0
    if report.is_file():
        for case in ET.parse(report).getroot().iter("testcase"):
            failure = case.find("failure")
            if failure is not None:
                kind = failure.get("type", "")
                failures.append(
                    Failure(
                        case.get("name", "?"),
                        "aserción"
                        if kind == "AssertionError"
                        else f"excepción {kind or '?'}",
                    )
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
    if not (BACKEND / "node_modules").is_dir():
        print("ERROR: falta backend/node_modules (corre `npm ci` en backend/).")
        return 2
    tmp = Path(tempfile.mkdtemp())
    backend = build_isolated_copy(tmp / "copy")
    report = tmp / "junit.xml"
    try:
        print(f"Copia aislada: {backend}")
        rc, failures, errors = run_vitest(backend, report)
        if rc != 0 or failures or errors:
            print(
                f"ERROR: los tests ya fallan SIN mutante (rc={rc}); no se puede evaluar."
            )
            return 2
        print(f"Base limpia: tests en verde. Mutantes: {len(MUTANTS)}\n", flush=True)

        results, exit_code = [], 0
        for m in MUTANTS:
            path = backend / m.target
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
                rc, failures, errors = run_vitest(backend, report)
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
        # Nunca borrar a través del enlace: si no se pudo quitar, se deja la copia.
        link = backend / "node_modules"
        try:
            if os.path.lexists(link):
                _unlink(link)
        except OSError as error:
            print(
                f"AVISO: no se quitó el enlace {link} ({error}); la copia queda en {tmp}."
            )
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
        print(
            f"| {m.id} | `{m.target}` | {m.name} | {status} | {note} | {tests or '—'} |"
        )
    killed = sum(1 for _, s, _, _ in results if s == "KILLED")
    print(f"\nKILLED por aserción {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
