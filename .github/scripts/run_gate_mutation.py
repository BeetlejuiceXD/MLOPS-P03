"""Run one deterministic mutation against the quality gate.

The mutation must be killed by the gate tests. The source file is restored in
``finally`` so a local run cannot leave the checkout mutated.

Exit codes (D01-06, review PR #39):

* ``0`` KILLED: with the mutant, pytest ran and at least one test *failed*
  (exit 1, failures > 0, errors == 0).
* ``1`` SURVIVED: with the mutant, every test passed.
* ``2`` ERROR: anything else — mutation target not found, the tests already fail
  without the mutant, or pytest itself errored (collection/import error, fixture
  error, usage error, no tests collected, interrupted). A pytest error is never
  counted as a killed mutation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
TARGET = APP_ROOT / "presentation/gate.py"
ORIGINAL = b'if any(check.action == "fail" and not check.passed for check in checks):'
MUTANT = b'if any(check.action == "fail" and check.passed for check in checks):'
GATE_TESTS = ["tests/test_gate.py"]

KILLED = 0
SURVIVED = 1
ERROR = 2

PYTEST_OK = 0
PYTEST_TESTS_FAILED = 1


@dataclass(frozen=True)
class PytestRun:
    returncode: int
    tests: int
    failures: int
    errors: int


def _run_pytest(pytest_args: list[str], cwd: Path) -> PytestRun:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        # Caché de bytecode nueva en cada corrida: si el mutante tiene el mismo tamaño
        # que el original y se escribe en el mismo segundo, Python reutilizaría el .pyc
        # del código sin mutar y la mutación "sobreviviría" sin haberse probado.
        env = {**os.environ, "PYTHONPYCACHEPREFIX": str(Path(tmp) / "pycache")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
                *pytest_args,
            ],
            cwd=cwd,
            env=env,
            check=False,
        )
        tests = failures = errors = 0
        if report.is_file():
            root = ET.parse(report).getroot()
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            for suite in suites:
                tests += int(suite.get("tests", 0))
                failures += int(suite.get("failures", 0))
                errors += int(suite.get("errors", 0))
    return PytestRun(result.returncode, tests, failures, errors)


def classify(run: PytestRun) -> int:
    """Clasifica la corrida de pytest con el mutante aplicado."""
    if run.returncode == PYTEST_OK and run.errors == 0 and run.tests > 0:
        return SURVIVED
    if run.returncode == PYTEST_TESTS_FAILED and run.failures > 0 and run.errors == 0:
        return KILLED
    return ERROR


def run_mutation(
    target: Path,
    original: bytes,
    mutant: bytes,
    pytest_args: list[str],
    cwd: Path,
) -> int:
    source = target.read_bytes()
    if source.count(original) != 1:
        print("Mutation target was not found exactly once.", file=sys.stderr)
        return ERROR

    baseline = _run_pytest(pytest_args, cwd)
    if baseline.returncode != PYTEST_OK or baseline.errors or baseline.tests == 0:
        print(
            "Baseline is not green: the tests fail or error without the mutant "
            f"(pytest exit {baseline.returncode}, failures={baseline.failures}, "
            f"errors={baseline.errors}).",
            file=sys.stderr,
        )
        return ERROR

    target.write_bytes(source.replace(original, mutant))
    try:
        mutated = _run_pytest(pytest_args, cwd)
    finally:
        target.write_bytes(source)

    outcome = classify(mutated)
    if outcome == KILLED:
        print(f"Mutation killed: {mutated.failures} test(s) failed on the broken condition.")
    elif outcome == SURVIVED:
        print("Mutation survived: the tests did not detect the broken condition.")
    else:
        print(
            "Pytest error with the mutant: this is not a killed mutation "
            f"(pytest exit {mutated.returncode}, failures={mutated.failures}, "
            f"errors={mutated.errors}).",
            file=sys.stderr,
        )
    return outcome


def main() -> int:
    return run_mutation(TARGET, ORIGINAL, MUTANT, GATE_TESTS, APP_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
