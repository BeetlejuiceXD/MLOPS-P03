"""D06-01 — Primera y única evaluación oficial del frozen test (custodia: Ale, #33).

Orden estricto, y cada paso solo corre si el anterior pasó:

1. **Preflight** sin abrir el test:
   - `p3_model_selection` está `closed` (guarda de D04-04/D05-08);
   - el acta de D05-08 coincide con lo persistido (run, `outcome_hash`, manifest);
   - en `official` no hay un resultado previo: si lo hay, se detiene para auditarlo y
     nunca se genera otro;
   - el paquete trae exactamente el checkpoint cerrado (SHA del acta, run, mejor época,
     manifest, release y class_map) y reproduce su salida de referencia;
   - el manifest congelado de D03-01 es el versionado en DVC, su contenido da su hash y
     su `test_split_hash` es el del acta.
2. **Intento**: `attempt.json` se escribe antes de cargar un solo crop de test. Si ya
   existe, el intento anterior se audita; no se vuelve a llamar "primera evaluación".
3. **Inferencia**, una vez por crop, tras comprobar que los crops son todos y solo los IDs
   de la partición test y que su etiqueta está en el class_map.
4. **Persistencia** con `produce_evaluation` (motor D03-05 + `EvaluationStore`): mismas
   guardas, `official` se escribe una sola vez.
5. **Auditoría**: se relee lo guardado y se recalcula con scikit-learn, independiente del
   motor; el umbral 0.85 se compara con conteos enteros, sin redondear.

Un resultado por debajo del 85 % se registra igual: la selección sigue cerrada y nada de
aquí la reabre ni la modifica (solo se lee).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from evaluation.producer import (
    ClosedSelection,
    EvaluationRecord,
    EvaluationRefusedError,
    SamplePrediction,
    TestPartition,
    iso_utc,
    produce_evaluation,
)
from evaluation.store import EvaluationStore
from model_package import LoadedPackage, PackageError, load_package
from presentation.contracts import FrozenManifest
from trainer_worker.sources import (
    SourcesNotEligibleError,
    _check_self_consistency,
    _read_frozen_manifest,
)
from training.class_map import CLASS_MAP

ACCURACY_TARGET = 0.85
EXACT_TOLERANCE = 1e-9
REPORTED_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ClosingAct:
    """Lo que declara el acta de D05-08. Se coteja con lo persistido antes de abrir el test."""

    candidate_run_id: str
    checkpoint_sha256: str
    manifest_hash: str
    test_split_hash: str
    outcome_hash: str


@dataclass(frozen=True)
class TestCrop:
    """Un crop de la partición test con su clase real."""

    __test__ = False  # no es una clase de tests de pytest

    crop_id: int
    true_class: str
    image: Image.Image


CropLoader = Callable[[FrozenManifest], Iterable[TestCrop]]


@dataclass(frozen=True)
class Preflight:
    selection: ClosedSelection
    outcome: dict[str, Any]
    package: LoadedPackage
    manifest: FrozenManifest

    def summary(self) -> dict[str, Any]:
        source = self.package.manifest.source
        candidate = self.outcome.get("candidate") or {}
        return {
            "selection": {
                "status": "closed",
                "candidate_run_id": self.selection.candidate_run_id,
                "campaign_row": candidate.get("campaign_row"),
                "best_epoch": candidate.get("best_epoch"),
                "closed_at": iso_utc(self.selection.closed_at),
                "outcome_hash": self.outcome.get("outcome_hash"),
            },
            "package": {
                "package_id": self.package.manifest.package_id,
                "mlflow_run_id": source.mlflow_run_id,
                "checkpoint_sha256": source.checkpoint_sha256,
                "best_epoch": source.best_epoch,
                "class_map": self.package.class_map,
            },
            "manifest": {
                "manifest_version": self.manifest.manifest_version,
                "manifest_hash": self.manifest.manifest_hash,
                "dataset_version": self.manifest.dataset_version,
                "dvc_release_hash": self.manifest.dvc_release_hash,
                "test_split_hash": self.manifest.test_split_hash,
                "n_test": len(self.manifest.assignments.test),
            },
        }


@dataclass(frozen=True)
class OfficialRun:
    record: EvaluationRecord
    preflight: Preflight
    audit: dict[str, Any]
    evidence_dir: Path


def _refuse(reason: str, detail: str) -> EvaluationRefusedError:
    return EvaluationRefusedError(reason, detail)


def _check_act(act: ClosingAct, selection: ClosedSelection, closed: dict[str, Any]) -> None:
    declared = {
        "candidate_run_id": (act.candidate_run_id, selection.candidate_run_id),
        "outcome_hash": (act.outcome_hash, closed.get("outcome_hash")),
        "manifest_hash": (act.manifest_hash, selection.manifest_hash),
    }
    for field, (in_act, persisted) in declared.items():
        if in_act != persisted:
            raise _refuse("act_mismatch", f"{field}: el acta dice {in_act}, persistido {persisted}")


def _load_candidate_package(
    package_dir: Path, act: ClosingAct, selection: ClosedSelection, outcome: dict[str, Any]
) -> LoadedPackage:
    if not Path(package_dir).exists():
        raise _refuse("package_rejected", f"paquete faltante: {package_dir} no existe")
    try:
        package = load_package(package_dir)
        reference = package.check_reference()
    except PackageError as error:
        raise _refuse("package_rejected", str(error)) from error
    source = package.manifest.source
    if source.checkpoint_sha256 != act.checkpoint_sha256:
        raise _refuse(
            "package_rejected",
            f"el paquete trae el checkpoint {source.checkpoint_sha256}; el acta cerró "
            f"{act.checkpoint_sha256}",
        )
    if not reference["matches"]:
        raise _refuse(
            "package_rejected",
            f"no reproduce su salida de referencia (max_abs_diff {reference['max_abs_diff']})",
        )
    candidate = outcome.get("candidate") or {}
    expected = {
        "mlflow_run_id": selection.candidate_run_id,
        "best_epoch": candidate.get("best_epoch", source.best_epoch),
        "manifest_hash": selection.manifest_hash,
        "dvc_release_hash": (outcome.get("reference") or {}).get("dvc_release_hash"),
    }
    for field, value in expected.items():
        if getattr(source, field) != value:
            raise _refuse(
                "package_not_candidate",
                f"{field} del paquete es {getattr(source, field)}; el cierre dice {value}",
            )
    # El class_map ya lo exige `load_package` (orden congelado de #33): llega aquí igual.
    return package


def _read_manifest(
    manifest_path: Path, act: ClosingAct, selection: ClosedSelection, outcome: dict[str, Any]
) -> FrozenManifest:
    try:
        manifest = _read_frozen_manifest(Path(manifest_path))
        _check_self_consistency(manifest)
    except SourcesNotEligibleError as error:
        raise _refuse("manifest_rejected", f"{error.reason}: {error.detail}") from error
    reference = outcome.get("reference") or {}
    expected = {
        "manifest_hash": selection.manifest_hash,
        "dataset_version": reference.get("dataset_version"),
        "dvc_release_hash": reference.get("dvc_release_hash"),
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise _refuse(
                "manifest_mismatch",
                f"{field} del manifest es {getattr(manifest, field)}; el cierre dice {value}",
            )
    if manifest.test_split_hash != act.test_split_hash:
        raise _refuse(
            "test_split_hash_mismatch",
            f"el manifest da {manifest.test_split_hash}; el acta dice {act.test_split_hash}",
        )
    return manifest


def preflight(
    store: EvaluationStore,
    *,
    act: ClosingAct,
    package_dir: Path,
    manifest_path: Path,
    namespace: str = "official",
) -> Preflight:
    """Todo lo que se comprueba antes de abrir el test. No lee crops ni escribe nada."""
    selection = store.closed_selection()
    closed = store.closed_outcome()
    _check_act(act, selection, closed)
    if namespace == "official" and store.read("official") is not None:
        raise _refuse(
            "official_already_recorded",
            "ya hay una evaluación oficial: se audita esa, no se genera otra",
        )
    outcome = {**(closed["outcome"] or {}), "outcome_hash": closed["outcome_hash"]}
    package = _load_candidate_package(Path(package_dir), act, selection, outcome)
    manifest = _read_manifest(Path(manifest_path), act, selection, outcome)
    return Preflight(selection=selection, outcome=outcome, package=package, manifest=manifest)


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _checked_crops(crops: Iterable[TestCrop], partition: TestPartition) -> list[TestCrop]:
    """Todos y solo los IDs de test, sin repetir y con etiqueta del class_map; antes de
    predecir, para que la inferencia corra una sola vez por crop."""
    crops = list(crops)
    ids = [crop.crop_id for crop in crops]
    if len(set(ids)) != len(ids):
        raise _refuse("duplicate_crop_id", "el cargador entregó crops repetidos")
    expected = set(partition.crop_ids)
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra:
        raise _refuse(
            "crop_ids_not_test_split",
            f"faltan {len(missing)}, sobran {len(extra)} respecto a la partición test",
        )
    outside = sorted(crop.crop_id for crop in crops if crop.true_class not in CLASS_MAP)
    if outside:
        raise _refuse(
            "label_outside_class_map", f"crops con clase fuera de {list(CLASS_MAP)}: {outside}"
        )
    return sorted(crops, key=lambda crop: crop.crop_id)


def audit_stored(stored: dict[str, Any]) -> dict[str, Any]:
    """Recalcula desde las predicciones guardadas, con scikit-learn (independiente del motor
    D03-05), y lo contrasta con la evaluación guardada."""
    evaluation, export = stored["evaluation"], stored["predictions"]
    labels = list(export["classes"])
    samples = export["predictions"]
    y_true = [sample["true_class"] for sample in samples]
    y_pred = [sample["predicted_class"] for sample in samples]
    n_test = len(samples)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    correct = sum(matrix[i][i] for i in range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "class_name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(labels)
    ]
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    baseline = max(int(s) for s in support) / n_test

    def close(a: float, b: float, tolerance: float) -> bool:
        return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)

    stored_metrics = evaluation["metrics"]
    stored_per_class = {row["class_name"]: row for row in stored_metrics["per_class"]}
    ids = [sample["crop_id"] for sample in samples]
    checks = {
        "ids_sorted_and_unique": ids == sorted(set(ids)),
        "n_test": evaluation["n_test"] == n_test == export["n_test"],
        "same_candidate": evaluation["selection"]["candidate_run_id"] == export["candidate_run_id"],
        "same_evaluated_at": evaluation["evaluated_at"] == export["evaluated_at"],
        "evaluated_after_close": datetime.fromisoformat(evaluation["evaluated_at"])
        > datetime.fromisoformat(evaluation["selection"]["closed_at"]),
        "confusion_matrix": evaluation["confusion_matrix"]["rows"] == matrix
        and evaluation["confusion_matrix"]["labels"] == labels,
        "accuracy": close(stored_metrics["accuracy"], accuracy, EXACT_TOLERANCE),
        "macro_f1": close(stored_metrics["macro_f1"], macro_f1, REPORTED_TOLERANCE),
        "per_class": set(stored_per_class) == set(labels)
        and all(
            stored_per_class[row["class_name"]]["support"] == row["support"]
            and all(
                close(stored_per_class[row["class_name"]][key], row[key], REPORTED_TOLERANCE)
                for key in ("precision", "recall", "f1")
            )
            for row in per_class
        ),
        "majority_baseline": close(
            evaluation["majority_baseline_accuracy"], baseline, EXACT_TOLERANCE
        ),
        "predicted_is_argmax": all(
            sample["predicted_class"]
            == max(sample["probabilities"], key=sample["probabilities"].__getitem__)
            for sample in samples
        ),
    }
    return {
        "matches": all(checks.values()),
        "checks": checks,
        "n_test": n_test,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": {"labels": labels, "rows": matrix},
        "per_class": per_class,
        "majority_baseline_accuracy": baseline,
        # Umbral de #33 con enteros: correct/n_test >= 0.85 ⇔ correct·100 >= 85·n_test.
        "target": {
            "accuracy": ACCURACY_TARGET,
            "correct": correct,
            "n_test": n_test,
            "met": correct * 100 >= 85 * n_test,
        },
        "error_crop_ids": [
            sample["crop_id"]
            for sample in samples
            if sample["true_class"] != sample["predicted_class"]
        ],
    }


def run_evaluation(
    store: EvaluationStore,
    *,
    act: ClosingAct,
    package_dir: Path,
    manifest_path: Path,
    load_crops: CropLoader,
    out_dir: Path,
    namespace: str = "official",
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    commit: str = "unknown",
) -> OfficialRun:
    checked = preflight(
        store,
        act=act,
        package_dir=package_dir,
        manifest_path=manifest_path,
        namespace=namespace,
    )
    out_dir = Path(out_dir)
    attempt_file = out_dir / "attempt.json"
    if attempt_file.exists():
        raise _refuse(
            "attempt_already_started",
            f"{attempt_file} ya existe: se audita ese intento, no se repite la evaluación",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        attempt_file,
        {
            "started_at": iso_utc(datetime.now(UTC)),
            "namespace": namespace,
            "commit": commit,
            "act": asdict(act),
            "package_dir": str(package_dir),
            "manifest_path": str(manifest_path),
        },
    )
    _write_json(out_dir / "preflight.json", checked.summary())

    partition = TestPartition.from_manifest(checked.manifest)
    crops = _checked_crops(load_crops(checked.manifest), partition)
    samples = []
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for crop in crops:
            image = crop.image.convert("RGB")
            output = checked.package.predict(image)
            samples.append(
                SamplePrediction(
                    crop_id=crop.crop_id,
                    true_class=crop.true_class,
                    predicted_class=output["predicted_class"],
                    probabilities=output["probabilities"],
                )
            )
            line = {
                "crop_id": crop.crop_id,
                "input_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "input_size": list(image.size),
                "true_class": crop.true_class,
                "predicted_class": output["predicted_class"],
                "probabilities": output["probabilities"],
            }
            trace.write(json.dumps(line) + "\n")

    record = produce_evaluation(
        store,
        samples,
        namespace=namespace,
        model_run_id=checked.package.manifest.source.mlflow_run_id,
        partition=partition,
        clock=clock,
    )
    stored = store.read(namespace)
    _write_json(out_dir / "evaluation.json", stored["evaluation"])
    _write_json(out_dir / "predictions.json", stored["predictions"])
    audit = audit_stored(stored)
    _write_json(out_dir / "audit.json", audit)
    if not audit["matches"]:
        failed = [name for name, ok in audit["checks"].items() if not ok]
        raise _refuse("audit_mismatch", f"el recálculo no coincide con lo guardado: {failed}")
    return OfficialRun(record=record, preflight=checked, audit=audit, evidence_dir=out_dir)


def release_test_crops(*, manifest_path: Path, repo_root: Path) -> CropLoader:
    """Cargador real: recorta los crops de la partición test desde el release verificado
    (las mismas fuentes y comprobaciones que Training, D03-03). Solo se llama después del
    preflight completo."""
    from policies.models import load_quality_policy
    from presentation.release_resolver import load_release_sources
    from trainer_worker.sources import _crop_pixels, verify_training_sources

    def load(manifest: FrozenManifest) -> list[TestCrop]:
        try:
            verified = verify_training_sources(
                manifest.dataset_version,
                manifest.manifest_hash,
                manifest_path=manifest_path,
                repo_root=repo_root,
                reports_dir=repo_root / "reports",
                sources=load_release_sources(),
                policy=load_quality_policy(),
            )
        except SourcesNotEligibleError as error:
            raise _refuse("sources_rejected", f"{error.reason}: {error.detail}") from error
        crops = {crop.crop_id: crop for crop in verified.crops}
        return [
            TestCrop(
                crop_id=crop_id,
                true_class=crops[crop_id].category_name,
                image=_crop_pixels(crops[crop_id], verified.images_dir, verified.image_files),
            )
            for crop_id in manifest.assignments.test
        ]

    return load


# --- CLI ----------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.official",
        description="D06-01: evaluación oficial única del frozen test (lee DATABASE_URL).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "comprueba cierre, acta, paquete y manifest; no abre el test"),
        ("run", "preflight + la única evaluación oficial + auditoría"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument(
            "--manifest", type=Path, default=REPO_ROOT / "data" / "p3" / "manifest.json"
        )
        command.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        command.add_argument(
            "--out", type=Path, default=REPO_ROOT / "reports" / "evaluation_p3" / "official"
        )
        for flag in (
            "--candidate-run-id",
            "--checkpoint-sha256",
            "--manifest-hash",
            "--test-split-hash",
            "--outcome-hash",
        ):
            command.add_argument(flag, required=True)
    return parser


def _git_commit() -> str:
    """Commit del código que evalúa: `GIT_COMMIT` (imagen) o el checkout local."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from sqlalchemy import create_engine

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Falta DATABASE_URL", file=sys.stderr)
        return 2
    store = EvaluationStore(create_engine(url, pool_pre_ping=True))
    act = ClosingAct(
        candidate_run_id=args.candidate_run_id,
        checkpoint_sha256=args.checkpoint_sha256,
        manifest_hash=args.manifest_hash,
        test_split_hash=args.test_split_hash,
        outcome_hash=args.outcome_hash,
    )
    try:
        if args.command == "preflight":
            checked = preflight(
                store, act=act, package_dir=args.package, manifest_path=args.manifest
            )
            print(json.dumps(checked.summary(), indent=2, ensure_ascii=False))
            return 0
        result = run_evaluation(
            store,
            act=act,
            package_dir=args.package,
            manifest_path=args.manifest,
            load_crops=release_test_crops(manifest_path=args.manifest, repo_root=args.repo_root),
            out_dir=args.out,
            commit=_git_commit(),
        )
    except EvaluationRefusedError as error:
        print(f"Rechazado — {error.reason}: {error.detail}", file=sys.stderr)
        return 2
    print(json.dumps({"evidence": str(result.evidence_dir), **result.audit["target"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
