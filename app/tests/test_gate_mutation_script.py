"""D01-06 (review PR #39): `run_gate_mutation.py` distingue mutación detectada de error.

Solo cuenta como "Mutation killed" que pytest termine con tests fallidos (exit 1)
sin errores de colección, de fixtures ni de ejecución. Un error de pytest debe hacer
fallar el job (exit 2), igual que una mutación que sobrevive (exit 1).

Cada caso monta un proyecto mínimo en `tmp_path` y ejecuta pytest de verdad.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "run_gate_mutation.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="script fuera del checkout")

ORIGINAL = b"return value > 0"
MUTANT = b"return value < 0"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_gate_mutation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


def _project(tmp_path: Path, test_body: str, target_body: str | None = None) -> Path:
    (tmp_path / "target.py").write_text(
        target_body or "def is_positive(value):\n    return value > 0\n", encoding="utf-8"
    )
    (tmp_path / "test_target.py").write_text(textwrap.dedent(test_body), encoding="utf-8")
    return tmp_path / "target.py"


def _run(script: ModuleType, target: Path, mutant: bytes = MUTANT) -> int:
    return script.run_mutation(
        target=target,
        original=ORIGINAL,
        mutant=mutant,
        pytest_args=["test_target.py"],
        cwd=target.parent,
    )


GOOD_TEST = """
    from target import is_positive

    def test_detects_sign():
        assert is_positive(1)
        assert not is_positive(-1)
"""


def test_exit_codes_are_distinct(script: ModuleType) -> None:
    assert (script.KILLED, script.SURVIVED, script.ERROR) == (0, 1, 2)


def test_mutation_detected_by_a_failing_test_is_killed(script: ModuleType, tmp_path) -> None:
    target = _project(tmp_path, GOOD_TEST)
    original_source = target.read_bytes()

    assert _run(script, target) == script.KILLED
    assert target.read_bytes() == original_source, "el archivo mutado no se restauró"


def test_mutation_not_detected_survives(script: ModuleType, tmp_path) -> None:
    weak_test = """
        from target import is_positive

        def test_only_checks_one_value():
            assert is_positive(1) in (True, False)
    """
    target = _project(tmp_path, weak_test)

    assert _run(script, target) == script.SURVIVED


def test_collection_error_is_an_error_not_a_kill(script: ModuleType, tmp_path) -> None:
    """El mutante rompe la sintaxis: pytest no puede ni importar el módulo."""
    target = _project(tmp_path, GOOD_TEST)
    original_source = target.read_bytes()

    assert _run(script, target, mutant=b"return value >") == script.ERROR
    assert target.read_bytes() == original_source


def test_fixture_error_is_an_error_not_a_kill(script: ModuleType, tmp_path) -> None:
    """pytest sale con 1, pero por un error de fixture, no por un assert que falla."""
    test_with_fixture = """
        import pytest
        from target import is_positive

        @pytest.fixture
        def checked():
            if not is_positive(1):
                raise RuntimeError("fixture rota por el mutante")
            return True

        def test_uses_fixture(checked):
            assert checked
    """
    target = _project(tmp_path, test_with_fixture)

    assert _run(script, target) == script.ERROR


def test_failing_baseline_is_an_error(script: ModuleType, tmp_path) -> None:
    """Si los tests ya fallan sin mutar, un fallo con el mutante no prueba nada."""
    broken_test = """
        def test_already_broken():
            assert False
    """
    target = _project(tmp_path, broken_test)

    assert _run(script, target) == script.ERROR


def test_missing_mutation_target_is_an_error(script: ModuleType, tmp_path) -> None:
    target = _project(tmp_path, GOOD_TEST, target_body="def is_positive(value):\n    return 1\n")

    assert _run(script, target) == script.ERROR


def test_real_gate_mutation_is_killed(script: ModuleType) -> None:
    """La mutación real del quality gate sigue detectándose con la nueva clasificación."""
    assert script.main() == script.KILLED
