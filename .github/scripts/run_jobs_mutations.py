"""Mutation testing de D02-05 (#45), D03-03 (#60) y D03-04 (#61, B1/B2 de #69):
jobs persistentes, trainer-worker, compuerta de training y entrenamiento real sobre fuentes
verificadas.

Para cada mutante: aplica un cambio en una línea concreta (debe aparecer exactamente una
vez), corre SOLO los tests del área, registra QUÉ tests fallaron y restaura el archivo en
`finally`. Resultado por mutante:

  KILLED   = el runner corrió y falló >= 1 test (sin errores de colección/importación).
  SURVIVED = todos los tests pasaron con el mutante: los tests no protegen ese código.
  ERROR    = objetivo no encontrado, o el runner no llegó a ejecutar los tests; NUNCA
             cuenta como detectada.

Antes de mutar se comprueba que los tests pasan sin mutante (si no, no hay nada que medir).

Uso (desde la raíz del repo):
  cd app && uv run python ../.github/scripts/run_jobs_mutations.py --suite app
  python3 .github/scripts/run_jobs_mutations.py --suite backend   # requiere npm ci en backend/
Sin --suite corre ambas. Sale con 0 solo si todos los mutantes quedan KILLED.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKER_TESTS = ["tests/test_trainer_worker.py"]
BACKEND_TESTS = ["tests/training-jobs.test.ts"]
SOURCES_TESTS = ["tests/test_training_sources.py", "tests/test_trainer_worker_training.py"]
P3_SOURCES_TESTS = ["tests/p3-sources.test.ts"]
SMOKE_TESTS = ["tests/test_smoke.py"]


@dataclass(frozen=True)
class Mutant:
    suite: str  # "app" (pytest) o "backend" (vitest)
    area: str
    name: str
    target: str  # relativo a la raíz del repo
    original: str
    mutant: str
    tests: list[str]


STORE, RUNNER = "app/trainer_worker/store.py", "app/trainer_worker/runner.py"
SERVICE = "backend/src/logic/training-jobs.service.ts"
SOURCES = "app/trainer_worker/sources.py"
P3_SOURCES = "backend/src/logic/p3-sources.service.ts"
SMOKE = "app/tracking/smoke.py"

MUTANTS = [
    # --- Las 5 mutaciones citadas en la descripción de #53 ---
    Mutant(
        "app",
        "Claim atómico",
        "UPDATE del claim sin la condición status='queued'",
        STORE,
        '.where(and_(training_jobs.c.id == row.id, training_jobs.c.status == "queued"))',
        ".where(training_jobs.c.id == row.id)",
        WORKER_TESTS,
    ),
    Mutant(
        "app",
        "Recuperación",
        "ignorar el latido vencido (todo latido cuenta como reciente)",
        STORE,
        "if owner == worker_id or (heartbeat is not None and heartbeat > limit):",
        "if owner == worker_id or heartbeat is not None:",
        WORKER_TESTS,
    ),
    Mutant(
        "app",
        "Run huérfano",
        "la recuperación no deja pendiente cerrar el run (FAILED -> nada)",
        STORE,
        '(training_jobs.c.mlflow_run_id.is_(None), None), else_="FAILED"',
        "(training_jobs.c.mlflow_run_id.is_(None), None), else_=None",
        WORKER_TESTS,
    ),
    Mutant(
        "backend",
        "Compuerta training",
        "aceptar un release no aprobado",
        SERVICE,
        "      if (!approved) {",
        "      if (false) {",
        BACKEND_TESTS,
    ),
    Mutant(
        "backend",
        "Compuerta training",
        "aceptar un manifest no congelado",
        SERVICE,
        "      if (!official.frozen) {",
        "      if (false) {",
        BACKEND_TESTS,
    ),
    # --- Protecciones nuevas de la revisión B1 (no cambian el conteo de arriba) ---
    Mutant(
        "app",
        "B1 cierre",
        "UPDATE de éxito sin la condición cancel_requested=false",
        STORE,
        "            conditions.append(training_jobs.c.cancel_requested.is_(False))\n",
        "            pass\n",
        WORKER_TESTS,
    ),
    Mutant(
        "app",
        "B1 parada",
        "no revisar la parada (SIGTERM) antes del COMMIT del estado terminal",
        STORE,
        "            if updated != 1 or (abort_if is not None and abort_if()):\n",
        "            if updated != 1:\n",
        WORKER_TESTS,
    ),
    # --- D03-03: fuentes reales verificadas y entrenamiento real ---
    Mutant(
        "app",
        "D03-03 manifest",
        "aceptar un artefacto que no coincide con el md5 versionado en DVC",
        SOURCES,
        "    if hashlib.md5(content).hexdigest() != declared_md5:",
        "    if False:",
        SOURCES_TESTS,
    ),
    Mutant(
        "app",
        "D03-03 manifest",
        "no recalcular manifest_hash (artefacto adulterado pasa)",
        SOURCES,
        "    if recomputed != manifest.manifest_hash:",
        "    if False:",
        SOURCES_TESTS,
    ),
    Mutant(
        "app",
        "D03-03 identidad",
        "no comparar dvc_release_hash con el release resuelto",
        SOURCES,
        "        or manifest.dvc_release_hash != expected_release_hash",
        "        or False",
        SOURCES_TESTS,
    ),
    Mutant(
        "app",
        "D03-03 partición",
        "no verificar la asignación contra los grupos reales (fuga entre splits)",
        SOURCES,
        "        verify_manifest_assignment(",
        "        (lambda *_a, **_k: None)(",
        SOURCES_TESTS,
    ),
    Mutant(
        "app",
        "D03-03 test",
        "cargar los píxeles de test en el dataset del trainer",
        SOURCES,
        "test=())",
        'test=samples("test"))',
        SOURCES_TESTS,
    ),
    Mutant(
        "app",
        "D03-03 checkpoint",
        "aceptar un checkpoint servido distinto del subido",
        RUNNER,
        "        if served != local_sha:",
        "        if False:",
        SOURCES_TESTS,
    ),
    Mutant(
        "backend",
        "D03-03 API",
        "servir un payload publicado fuera de contrato",
        P3_SOURCES,
        "    if (!parsed.success) {",
        "    if (false) {",
        P3_SOURCES_TESTS,
    ),
    # --- D03-04: verificador del smoke real ---
    Mutant(
        "app",
        "D03-04 smoke",
        "aceptar un model.pt cuyo sha256 no es el del tag",
        SMOKE,
        '    if report.checkpoint_sha256 != run.data.tags.get("checkpoint_sha256"):',
        "    if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 smoke",
        "aceptar métricas de test en el run",
        SMOKE,
        "    if leaked:",
        "    if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 smoke",
        "aceptar un run fuera de p3-cnn-classifier",
        SMOKE,
        "    if experiment != P3_EXPERIMENT:",
        "    if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 smoke",
        "cargar el checkpoint sin exigir que sea del modelo de la config",
        SMOKE,
        "strict=True)",
        "strict=False)",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 smoke",
        "no contrastar training_config.json con la config del job",
        SMOKE,
        '    if stored_config != job["config"]:',
        "    if False:",
        SMOKE_TESTS,
    ),
    # --- D03-04 B1: historial por época y resumen best_* (auditoría de #69) ---
    Mutant(
        "app",
        "D03-04 B1",
        "aceptar épocas duplicadas en el historial",
        SMOKE,
        "        if duplicated:",
        "        if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B1",
        "aceptar un historial que no cubre exactamente 1..epoch",
        SMOKE,
        "        elif sorted(steps) != list(range(1, epoch + 1)):",
        "        elif False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B1",
        "aceptar best_epoch no entero o fuera de las épocas registradas",
        SMOKE,
        "    if not float(raw).is_integer() or not 1 <= int(raw) <= epoch:",
        "    if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B1",
        "contrastar solo best_val_accuracy (no macro-F1 ni loss)",
        SMOKE,
        "    for summary, curve in BEST_SUMMARY:",
        "    for summary, curve in BEST_SUMMARY[:1]:",
        SMOKE_TESTS,
    ),
    # --- D03-04 B2: procedencia contra las fuentes oficiales ---
    Mutant(
        "app",
        "D03-04 B2",
        "aceptar el tag classes que declare el run",
        SMOKE,
        '        "classes": frozen_classes,',
        '        "classes": tags.get("classes"),',
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B2",
        "aceptar el dvc_images_md5 que declare el run",
        SMOKE,
        '            "dvc_images_md5": release["images_md5"],',
        '            "dvc_images_md5": tags.get("dvc_images_md5"),',
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B2",
        "no recalcular dvc_release_hash desde los md5 oficiales",
        SMOKE,
        '        if tags.get("dvc_release_hash") != recomputed:',
        "        if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B2",
        "aceptar un job cuyo release no está aprobado",
        SMOKE,
        "    if release is None:",
        "    if False:",
        SMOKE_TESTS,
    ),
    Mutant(
        "app",
        "D03-04 B2",
        "no contrastar sources.json del checkpoint con el run",
        SMOKE,
        "        if recorded.get(name) != value:",
        "        if False:",
        SMOKE_TESTS,
    ),
]


def _junit_failures(report: Path) -> tuple[list[str], int]:
    failed, errors = [], 0
    if report.is_file():
        for case in ET.parse(report).getroot().iter("testcase"):
            if case.find("failure") is not None:
                classname = case.get("classname", "")
                if not classname.endswith(".ts"):  # pytest: tests.test_x -> test_x
                    classname = classname.split(".")[-1]
                failed.append(f"{classname}::{case.get('name')}")
            if case.find("error") is not None:
                errors += 1
    return failed, errors


def run_tests(suite: str, tests: list[str]) -> tuple[int, list[str], int, bool]:
    """(returncode, tests fallidos, errores, ¿se generó el reporte?)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        if suite == "app":
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
                *tests,
            ]
            env = {**os.environ, "PYTHONPYCACHEPREFIX": str(Path(tmp) / "pyc")}
            cwd = REPO / "app"
        else:
            command = [
                "npx",
                "vitest",
                "run",
                "--reporter=junit",
                f"--outputFile={report}",
                *tests,
            ]
            env, cwd = dict(os.environ), REPO / "backend"
        proc = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True)
        failed, errors = _junit_failures(report)
        return proc.returncode, failed, errors, report.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=["app", "backend"])
    args = parser.parse_args()
    mutants = [m for m in MUTANTS if args.suite in (None, m.suite)]

    for suite in sorted({m.suite for m in mutants}):
        tests = sorted({t for m in mutants if m.suite == suite for t in m.tests})
        rc, failed, errors, _ = run_tests(suite, tests)
        if rc != 0 or failed or errors:
            print(f"ERROR: los tests de {suite} ya fallan SIN mutante (rc={rc}): {failed}")
            return 2
        print(f"Base limpia ({suite}): {', '.join(tests)} en verde.")
    print(f"Mutantes: {len(mutants)}\n")

    results, exit_code = [], 0
    for m in mutants:
        path = REPO / m.target
        source = path.read_bytes()
        if source.count(m.original.encode()) != 1:
            results.append((m, "ERROR", ["objetivo no encontrado exactamente una vez"]))
            exit_code = 2
            continue
        try:
            path.write_bytes(source.replace(m.original.encode(), m.mutant.encode()))
            rc, failed, errors, reported = run_tests(m.suite, m.tests)
        finally:
            path.write_bytes(source)
        if errors or not reported or (rc != 0 and not failed):
            status, exit_code = "ERROR", 2  # colección/importación rota: no es "detectada"
        elif failed:
            status = "KILLED"
        else:
            status, exit_code = "SURVIVED", max(exit_code, 1)
        results.append((m, status, failed))
        print(f"{status:9} [{m.area}] {m.name}  ->  {len(failed)} test(s)")

    print("\n| # | Área | Mutación | Archivo | Resultado | Tests que la detectaron |")
    print("|---|---|---|---|---|---|")
    for i, (m, status, failed) in enumerate(results, 1):
        shown = "<br>".join(f"`{t}`" for t in failed[:4])
        if len(failed) > 4:
            shown += f"<br>(+{len(failed) - 4} más)"
        print(f"| {i} | {m.area} | {m.name} | `{m.target}` | {status} | {shown or '—'} |")
    killed = sum(1 for _, status, _ in results if status == "KILLED")
    print(f"\nKILLED {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
