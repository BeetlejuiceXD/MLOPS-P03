"""Mutation testing de D04-06 (#75): registro de modelos y adaptador del bucket de modelos.

Trabaja en una COPIA AISLADA: copia a un directorio temporal `backend/src`, el test
`backend/tests/model-registry.test.ts`, `package.json`, `tsconfig.json` y los fixtures
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

Uso (desde la raíz del repo, con `npm ci` hecho en backend/; ~2-4 min):
    python .github/scripts/run_model_registry_mutations.py
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
RULES = "src/logic/model-registry.ts"
SERVICE = "src/logic/model-registry.service.ts"
STORAGE = "src/data/storage/model-object.storage.ts"
TEST = "tests/model-registry.test.ts"


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    target: str
    original: str
    mutant: str


MUTANTS = [
    # --- Reglas puras ---------------------------------------------------------------------
    Mutant(
        "R01",
        "clave fuera del prefijo del semver",
        RULES,
        "return `models/${P3_EXPERIMENT}/${semver}/${MODEL_OBJECT_NAME}`;",
        "return `models/${P3_EXPERIMENT}/${MODEL_OBJECT_NAME}`;",
    ),
    Mutant(
        "R02",
        "semver comparado como texto",
        RULES,
        "const diff = (pa[i] ?? 0) - (pb[i] ?? 0);",
        "const diff = String(pa[i]).localeCompare(String(pb[i]));",
    ),
    Mutant(
        "R03",
        "published_at como Date en vez de ISO",
        RULES,
        "entry.published_at === null ? null : entry.published_at.toISOString()",
        "entry.published_at === null ? null : (entry.published_at as unknown as string)",
    ),
    # --- register -------------------------------------------------------------------------
    Mutant(
        "G01",
        "register sin validar el contrato",
        SERVICE,
        "if (!contract.success) {",
        "if (false) {",
    ),
    Mutant(
        "G02",
        "register sin validar size_bytes",
        SERVICE,
        "if (!sizeSchema.safeParse(input.size_bytes).success) {",
        "if (false) {",
    ),
    Mutant(
        "G03",
        "semver duplicado no es conflicto",
        SERVICE,
        "if (!(await repo.insertDraft(entry))) {",
        "if (!(await repo.insertDraft(entry)) && false) {",
    ),
    Mutant(
        "G04",
        "bucket fijo en vez de MODEL_S3_BUCKET",
        SERVICE,
        "s3_bucket: store.bucket,",
        "s3_bucket: 'p3-models',",
    ),
    # --- upload ---------------------------------------------------------------------------
    Mutant(
        "U01",
        "upload acepta versiones no draft",
        SERVICE,
        "    async upload(semver, body) {\n      const entry = await load(semver);\n      requireDraft(entry);",
        "    async upload(semver, body) {\n      const entry = await load(semver);",
    ),
    Mutant(
        "U02",
        "upload reemplaza un VersionId ya registrado",
        SERVICE,
        "if (entry.version_id !== null) {\n        throw new ConflictError(`La versión ${semver} ya se subió",
        "if (false) {\n        throw new ConflictError(`La versión ${semver} ya se subió",
    ),
    Mutant(
        "U03",
        "upload sin comprobar el tamaño",
        SERVICE,
        "if (body.length !== entry.size_bytes) {",
        "if (false) {",
    ),
    Mutant(
        "U04",
        "upload sin comprobar el SHA-256",
        SERVICE,
        "if (actual !== entry.sha256) {\n        return fail(entry, {",
        "if (false) {\n        return fail(entry, {",
    ),
    Mutant(
        "U05",
        "sin VersionId no falla",
        SERVICE,
        "if (versionId === null) {\n        return fail(entry, {",
        "if (false) {\n        return fail(entry, {",
    ),
    Mutant(
        "U06",
        "metadato sha256 con el hash calculado en vez del registrado",
        SERVICE,
        "store.put(entry.s3_key, body, entry.sha256)",
        "store.put(entry.s3_key, body, sha256Hex(Buffer.from('')))",
    ),
    # --- verify / inspect -----------------------------------------------------------------
    Mutant(
        "V01",
        "objeto ausente en head no falla",
        SERVICE,
        "if (head === null) return { reason: 'object_missing', detail: `No existe ${where}` };",
        "if (head === null) return null;",
    ),
    Mutant(
        "V02",
        "VersionId de head sin comprobar",
        SERVICE,
        "if (head.versionId !== versionId) {",
        "if (false) {",
    ),
    Mutant(
        "V03",
        "tamaño guardado sin comprobar",
        SERVICE,
        "if (head.size !== entry.size_bytes) {",
        "if (false) {",
    ),
    Mutant(
        "V04",
        "metadato sha256 sin comprobar",
        SERVICE,
        "if (head.sha256 !== entry.sha256) {",
        "if (false) {",
    ),
    Mutant(
        "V05",
        "get sin objeto no falla",
        SERVICE,
        "if (body === null) return { reason: 'object_missing', detail: `GET de ${where} sin objeto` };",
        "if (body === null) return null;",
    ),
    Mutant(
        "V06",
        "SHA-256 del contenido sin comprobar",
        SERVICE,
        "if (actual !== entry.sha256) {\n      return {",
        "if (false) {\n      return {",
    ),
    Mutant(
        "V07",
        "fallo detectado pero se publica igual",
        SERVICE,
        "if (failure) return fail(entry, failure);",
        "if (failure) void 0;",
    ),
    Mutant(
        "V08",
        "verify sin objeto subido",
        SERVICE,
        "if (versionId === null) {\n        throw new ConflictError(`La versión ${semver} todavía no tiene objeto subido`);",
        "if (versionId === null && false) {\n        throw new ConflictError(`La versión ${semver} todavía no tiene objeto subido`);",
    ),
    Mutant(
        "V09",
        "publicación concurrente ignorada",
        SERVICE,
        "if (!(await repo.markPublished(namespace, semver, versionId, now()))) {",
        "if (!(await repo.markPublished(namespace, semver, versionId, now())) && false) {",
    ),
    Mutant(
        "V10",
        "verify acepta versiones ya published/failed",
        SERVICE,
        "    async verify(semver) {\n      const entry = await load(semver);\n      requireDraft(entry);",
        "    async verify(semver) {\n      const entry = await load(semver);",
    ),
    Mutant(
        "V11",
        "error del storage se trata como objeto ausente",
        SERVICE,
        "throw new ServiceUnavailableError(`Storage de modelos no disponible: ${detail}`);",
        "return null as T;",
    ),
    # --- audit / list ---------------------------------------------------------------------
    Mutant(
        "A01",
        "audit sobre versiones no publicadas",
        SERVICE,
        "if (entry.status !== 'published' || entry.version_id === null) {",
        "if (entry.version_id === null) {",
    ),
    Mutant(
        "A02",
        "audit siempre ok",
        SERVICE,
        "return failure ? { ok: false, ...failure } : { ok: true };",
        "return { ok: true };",
    ),
    Mutant(
        "L01",
        "list sin ordenar por semver",
        SERVICE,
        "rows.sort((a, b) => compareSemver(a.semver, b.semver));",
        "",
    ),
    # --- Adaptador MinIO ------------------------------------------------------------------
    Mutant(
        "M01",
        "put sin metadato sha256",
        STORAGE,
        "'X-Amz-Meta-Sha256': sha256,",
        "",
    ),
    Mutant(
        "M02",
        "head sin VersionId (lee la última versión)",
        STORAGE,
        "client.statObject(bucket, key, { versionId })",
        "client.statObject(bucket, key, {})",
    ),
    Mutant(
        "M03",
        "get sin VersionId (lee la última versión)",
        STORAGE,
        "client.getObject(bucket, key, { versionId })",
        "client.getObject(bucket, key, {})",
    ),
    Mutant(
        "M04",
        "cualquier error se trata como objeto ausente",
        STORAGE,
        "if (isMissing(error)) return null;",
        "return null;",
    ),
    Mutant(
        "M05",
        "NoSuchVersion no cuenta como ausente",
        STORAGE,
        "const MISSING = new Set(['NotFound', 'NoSuchKey', 'NoSuchVersion']);",
        "const MISSING = new Set(['NotFound', 'NoSuchKey']);",
    ),
    Mutant(
        "M06",
        "get devuelve solo el primer fragmento",
        STORAGE,
        "for await (const chunk of stream) chunks.push(Buffer.from(chunk));",
        "for await (const chunk of stream) {\n          chunks.push(Buffer.from(chunk));\n          break;\n        }",
    ),
    Mutant(
        "M07",
        "bucket sin versioning aceptado",
        STORAGE,
        "if (config?.Status !== 'Enabled') {",
        "if (false) {",
    ),
    Mutant(
        "M08",
        "bucket local sin activar versioning",
        STORAGE,
        "await client.setBucketVersioning(bucket, { Status: 'Enabled' });",
        "",
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
