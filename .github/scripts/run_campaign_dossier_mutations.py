"""Mutation testing de D05-02 (#85): expediente de campaña, representante cronológico,
intentos pendientes y guardas del cierre.

Trabaja en una COPIA AISLADA: copia a un directorio temporal `backend/src`, los tests
`backend/tests/model-selection.test.ts` y `backend/tests/campaign-dossier.test.ts` con su
helper `tests/selection-runs.ts`, `package.json`, `tsconfig.json` y los fixtures de
`contracts/p3/fixtures`; `node_modules` se enlaza (junction en Windows, symlink en Linux),
no se copia ni se modifica. El árbol de trabajo del repo nunca se toca.

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

Uso (desde la raíz del repo, con `npm ci` hecho en backend/; ~4-6 min):
    python .github/scripts/run_campaign_dossier_mutations.py
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
DOSSIER = "src/logic/campaign-dossier.ts"
SERVICE = "src/logic/model-selection.service.ts"
ROUTES = "src/ui/model-selection.routes.ts"
TESTS = ["tests/model-selection.test.ts", "tests/campaign-dossier.test.ts"]
HELPER = "tests/selection-runs.ts"


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    target: str
    original: str
    mutant: str


MUTANTS = [
    # --- Selección (D04-04 ampliada) --------------------------------------------------
    Mutant(
        "S01",
        "FINISHED sin checkpoint verificado sigue siendo elegible",
        SELECTION,
        "if (!run.campaign_eligible) {",
        "if (false) {",
    ),
    Mutant(
        "S02",
        "un intento en curso de la matriz no bloquea el cierre",
        SELECTION,
        "campaignRows.length >= MIN_COMPARABLE_RUNS && pending === 0,",
        "campaignRows.length >= MIN_COMPARABLE_RUNS,",
    ),
    Mutant(
        "S03",
        "SCHEDULED no cuenta como intento en curso",
        SELECTION,
        "pending: run.status === 'RUNNING' || run.status === 'SCHEDULED',",
        "pending: run.status === 'RUNNING',",
    ),
    Mutant(
        "S04",
        "un smoke en curso se toma como intento de la matriz",
        SELECTION,
        "  if (campaignRow === null) {\n    return {\n      eligible: false,\n      reason: 'outside_campaign_matrix',",
        "  if (campaignRow === null && run.status === 'FINISHED') {\n    return {\n      eligible: false,\n      reason: 'outside_campaign_matrix',",
    ),
    # --- Expediente: roles y orden --------------------------------------------------------
    Mutant(
        "D01",
        "un retry se rotula como excluido, no como retry",
        DOSSIER,
        "role: excluded.reason === 'duplicate_campaign_row' ? 'retry' : 'excluded',",
        "role: 'excluded',",
    ),
    Mutant(
        "D02",
        "un intento activo no se marca pendiente",
        DOSSIER,
        "else if (active) verdict = { role: 'pending', reason: null, detail: null };",
        "else if (false) verdict = { role: 'pending', reason: null, detail: null };",
    ),
    Mutant(
        "D03",
        "un job fallido sin run pierde su motivo",
        DOSSIER,
        "verdict = { role: 'excluded', reason: 'job_failed', detail: job.error };",
        "verdict = { role: 'excluded', reason: null, detail: job.error };",
    ),
    Mutant(
        "D04",
        "un run excluido por D04-01 pierde sus motivos",
        DOSSIER,
        "return { role: 'excluded', reason: 'adapter_excluded', detail: adapter.reasons.join('; ') };",
        "return { role: 'excluded', reason: 'adapter_excluded', detail: '' };",
    ),
    Mutant(
        "D05",
        "los intentos no van en orden cronológico",
        DOSSIER,
        "compareText(a.attempt.start_time ?? a.created_at, b.attempt.start_time ?? b.created_at) ||",
        "0 ||",
    ),
    Mutant(
        "D06",
        "la fila sale del request aunque el run entrenó otra config",
        DOSSIER,
        "row: campaignRowOf(run?.params ?? job.config),",
        "row: campaignRowOf(job.config),",
    ),
    Mutant(
        "D07",
        "los jobs controlled cuentan como intentos",
        DOSSIER,
        "if (job.task !== 'training') continue;",
        "if (job.task === 'controlled' && false) continue;",
    ),
    # --- Expediente: conciliación ---------------------------------------------------------
    Mutant(
        "C01",
        "no compara el job_id que declara el run",
        DOSSIER,
        "if (run.tags.job_id !== job.id) {",
        "if (false) {",
    ),
    Mutant(
        "C02",
        "no compara la config del run con el request",
        DOSSIER,
        "if (!sameConfig(job.config, run.params)) {",
        "if (false) {",
    ),
    Mutant(
        "C03",
        "no compara el manifest del job con el del run",
        DOSSIER,
        "if (job.manifest_hash !== run.tags.manifest_hash) {",
        "if (false) {",
    ),
    Mutant(
        "C04",
        "no compara el release del job con el del run",
        DOSSIER,
        "if (job.dataset_version !== run.tags.dvc_release) {",
        "if (false) {",
    ),
    Mutant(
        "C05",
        "no compara el estado del job con el del run",
        DOSSIER,
        "return RUN_STATUS_FOR_JOB[job.status].includes(runStatus)",
        "return true",
    ),
    Mutant(
        "C06",
        "un job cuyo run no está en MLflow no es problema",
        DOSSIER,
        "if (runId !== null && !listedRuns.has(runId)) {",
        "if (false) {",
    ),
    Mutant(
        "C07",
        "los runs sin job desaparecen del expediente",
        DOSSIER,
        "if (runId === null || claimed.has(runId)) continue;",
        "if (true) continue;",
    ),
    Mutant(
        "C08",
        "un job_id repetido no es problema",
        DOSSIER,
        "if ((jobIds.get(job.id) ?? 0) > 1)",
        "if (false)",
    ),
    Mutant(
        "C09",
        "un job fuera de contrato no bloquea",
        DOSSIER,
        "problems: ['el job no cumple el contrato training_job'],",
        "problems: [],",
    ),
    # --- Expediente: bloqueos e identidad ------------------------------------------------
    Mutant(
        "B01",
        "menos de 10 filas aceptadas no bloquea",
        DOSSIER,
        "if (acceptedRows.length < MIN_COMPARABLE_RUNS) {",
        "if (false) {",
    ),
    Mutant(
        "B02",
        "las filas pendientes no bloquean",
        DOSSIER,
        "if (pendingRows.length > 0) {",
        "if (false) {",
    ),
    Mutant(
        "B03",
        "los intentos que no concilian no bloquean",
        DOSSIER,
        "if (unreconciled.length > 0) {",
        "if (false) {",
    ),
    Mutant(
        "B04",
        "ready_to_close ignora los bloqueos del expediente",
        DOSSIER,
        "ready_to_close: outcome.ready_to_close && closeBlockers.length === 0,",
        "ready_to_close: outcome.ready_to_close,",
    ),
    Mutant(
        "B05",
        "la identidad del candidato no trae su job",
        DOSSIER,
        "job_id: tags.job_id,",
        "job_id: 0,",
    ),
    Mutant(
        "B06",
        "la identidad del candidato no trae su checkpoint",
        DOSSIER,
        "checkpoint_sha256: run.checkpoint_sha256,",
        "checkpoint_sha256: null,",
    ),
    # --- Servicio y API ----------------------------------------------------------------------
    Mutant(
        "V01",
        "el cierre ignora el expediente de campaña",
        SERVICE,
        "if (!dossier.ready_to_close) {",
        "if (false) {",
    ),
    Mutant(
        "V02",
        "matches_proposal siempre dice que coincide",
        SERVICE,
        "matches_proposal: proposedHash === null ? null : proposedHash === dossier.outcome_hash,",
        "matches_proposal: proposedHash === null ? null : true,",
    ),
    Mutant(
        "R01",
        "GET /selection/campaign no está montada",
        ROUTES,
        "'/selection/campaign',",
        "'/selection/campaigns',",
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
    for name in (*TESTS, HELPER):
        shutil.copy2(BACKEND / name, backend / name)
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
            *TESTS,
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
    sys.stdout.reconfigure(encoding="utf-8")
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
