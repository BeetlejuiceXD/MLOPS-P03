"""Mutation testing de D03-01 (#58): auditoría y congelación del manifest P3.

Para cada mutante: aplica el cambio en `app/presentation/manifest_freeze.py` (el texto
original debe aparecer exactamente una vez), corre `tests/test_manifest_freeze.py`
completo, registra QUÉ tests fallaron y CÓMO, y restaura el archivo en `finally`.

Clasificación de cada test que falla (del reporte JUnit de pytest):
- `aserción`: veredicto explícito del test: `AssertionError` (o `assert ...`), `pytest.fail(...)` o
  `pytest.raises` que no se cumplió (`Failed: ...`). El test esperaba un resultado
  concreto (p. ej. un motivo de bloqueo) y el mutante lo cambió.
- `excepción <Tipo>`: el test terminó con una excepción no esperada.

Resultado por mutante:
- KILLED = pytest corrió sin errores de colección/setup y >=1 test falló por aserción.
- KILLED (solo excepción) = falló >=1 test, pero ninguno por aserción.
- SURVIVED = todo pasó.
- ERROR = objetivo no encontrado exactamente una vez, error de colección/setup o
  pytest roto (rc distinto de 0/1). Nunca cuenta como muerto.

El test contra el v0.1.1 real se excluye para que el resultado no dependa de tener
`data/raw` (en CI se salta igual). La base sin mutantes debe estar en verde.

Uso (desde la raíz del repo; ~20-30 min en total):
    cd app && uv run python ../.github/scripts/run_manifest_freeze_mutations.py
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
TARGET = "presentation/manifest_freeze.py"
TESTS = [
    "tests/test_manifest_freeze.py",
    "--deselect",
    "tests/test_manifest_freeze.py::test_real_v0_1_1_freezes_the_d02_04_candidate_unchanged",
]


@dataclass(frozen=True)
class Mutant:
    id: str
    name: str
    original: str
    mutant: str


MUTANTS = [
    Mutant(
        "M01",
        "sin sha256 registrado del artefacto",
        "    if expected_sha256 is not None and _sha256(manifest) != expected_sha256:",
        "    if False:",
    ),
    Mutant("M02", "sin chequeo de grupos que cruzan", "    if crossing:\n", "    if False:\n"),
    Mutant(
        "M03",
        "grupos sin nodos puente",
        '        a, b = find(pair["image_id_a"]), find(pair["image_id_b"])\n',
        '        if not {pair["image_id_a"], pair["image_id_b"]} <= image_ids:\n'
        "            continue\n"
        '        a, b = find(pair["image_id_a"]), find(pair["image_id_b"])\n',
    ),
    Mutant("M04", "sin chequeo de clase ausente", "        if absent:\n", "        if False:\n"),
    Mutant(
        "M05",
        "umbral de originales < -> <=",
        "for name in SPLITS) < min_originals_per_class",
        "for name in SPLITS) <= min_originals_per_class",
    ),
    Mutant(
        "M06",
        "sin tolerancia de proporciones",
        "> MANIFEST_SPLIT_TOLERANCE + 1e-12:",
        "> 1.0:",
    ),
    Mutant(
        "M07",
        "sin recalcular manifest_hash",
        "    if recomputed != summary.manifest_hash:",
        "    if False:",
    ),
    Mutant(
        "M08",
        "sin verificar dvc_release_hash",
        "    if summary.dvc_release_hash != _dvc_release_hash(images_md5, annotations_md5):",
        "    if False:",
    ),
    Mutant(
        "M09",
        "sin summary_mismatch",
        '            raise FreezeBlockedError("summary_mismatch", '
        'f"conteos de {name} no coinciden")',
        "            pass",
    ),
    Mutant(
        "M10", "sin chequeo de cobertura", "    if set(assigned) != set(by_id):", "    if False:"
    ),
    Mutant(
        "M11", "sin candidate_drift", "    if regenerated.manifest != manifest:", "    if False:"
    ),
    Mutant(
        "M12",
        "sin recalcular test_split_hash",
        '    if frozen_test_split_hash(assignments["test"]) != artifact.test_split_hash:',
        "    if False:",
    ),
    Mutant(
        "M13",
        "sin verificar el .dvc",
        "    if _dvc_md5(manifest_path) != hashlib.md5(manifest).hexdigest():",
        "    if False:",
    ),
    Mutant(
        "M14",
        "frozen queda en False al congelar",
        'summary.model_copy(update={"frozen": True})',
        'summary.model_copy(update={"frozen": False})',
    ),
    Mutant(
        "M15",
        "sin md5 DVC del artefacto vs release",
        "    if (artifact.images_md5, artifact.annotations_md5) != (images_md5, annotations_md5):",
        "    if False:",
    ),
    Mutant(
        "M16",
        "sin overlap de originales",
        "            if split_of_image.setdefault(image_id, name) != name:",
        "            if split_of_image.setdefault(image_id, name) != name and False:",
    ),
    Mutant(
        "M17",
        "sin identidad artefacto vs resumen",
        "        if getattr(artifact, key) != getattr(summary, key):",
        "        if False:",
    ),
    Mutant(
        "M18",
        "artefacto sin validar contrato",
        "        artifact = FrozenManifest.model_validate_json(manifest)",
        "        artifact = FrozenManifest.model_construct(**json.loads(manifest))",
    ),
    Mutant(
        "M19",
        "resumen publicado sin exigir frozen: true",
        "    if summary.frozen is not True:",
        "    if False:",
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

    path = APP / TARGET
    results, exit_code = [], 0
    for m in MUTANTS:
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
            status, note, exit_code = (
                "KILLED (solo excepción)",
                _detection(failures),
                max(exit_code, 1),
            )
        else:
            status, note, exit_code = "SURVIVED", "—", max(exit_code, 1)
        results.append((m, status, failures, note))
        print(f"{status:9} {m.id} {m.name}  ->  {len(failures)} test(s): {note}")

    print(
        "\n| # | Mutación | Resultado | Tipo de detección | Tests que la detectan |"
        "\n|---|---|---|---|---|"
    )
    for m, status, failures, note in results:
        asserted = [f.test for f in failures if f.kind == "aserción"]
        others = [f.test for f in failures if f.kind != "aserción"]
        shown = asserted[:3] + others[: max(0, 3 - len(asserted))]
        tests = "<br>".join(f"`{t}`" for t in shown)
        if len(failures) > len(shown):
            tests += f"<br>(+{len(failures) - len(shown)} más)"
        print(f"| {m.id} | {m.name} | {status} | {note} | {tests or '—'} |")
    killed = sum(1 for _, s, _, _ in results if s == "KILLED")
    print(f"\nKILLED por aserción {killed}/{len(results)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
