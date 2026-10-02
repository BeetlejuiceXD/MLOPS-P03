"""D06-02 — Paquete final y model card oficial del candidato evaluado.

Reutiliza el formato y el constructor de D05-01 tal cual, sin
reconstruirlos: el paquete final usa exactamente el mismo layout de
archivos que el paquete smoke (mismo `PAYLOAD_FILES`, mismo loader para
recargarlo). Lo único que distingue un paquete final de uno smoke es
`kind` en el manifiesto y el contenido de la tarjeta — nunca la ruta de
sus archivos.

Bloqueado por D06-01: sin su resultado real, solo se puede ejercitar este
módulo con un paquete smoke de prueba (ver `tests/`) — nunca declarar un
paquete "final" de verdad sin un `OfficialTestMetrics` real de D06-01.
Nunca se sustituye por métricas synthetic/local_test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_package.build import build_smoke_package
from model_package.format import MODEL_NAME, PackageError, PackageManifest
from model_package.version import ModelVersion


@dataclass(frozen=True)
class OfficialRecord:
    """El acta (D05-08): identidad ya cerrada del run/checkpoint
    seleccionado, independiente de si D06-01 ya entregó resultado."""

    run_id: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class OfficialTestMetrics:
    """El resultado de D06-01 para el MISMO run/checkpoint del acta.
    `run_id` propio aquí (no se asume, se exige) para poder detectar una
    tarjeta armada con métricas de otro run/namespace antes de
    empaquetar — justo el negativo que pide el ticket."""

    run_id: str
    accuracy: float
    macro_f1: float
    loss: float
    test_split_hash: str


LOW_ACCURACY_THRESHOLD = 0.85


def _official_card(
    *,
    run_id: str,
    best_epoch: int | None,
    semver: ModelVersion,
    metrics: OfficialTestMetrics | None,
    limitations: tuple[str, ...],
) -> dict:
    """La tarjeta oficial. Sin `metrics` (D06-01 no ha corrido todavía),
    queda explícitamente pendiente — nunca se rellena con synthetic/
    local_test ni se inventa un número."""
    pending = metrics is None
    low_accuracy = (not pending) and metrics.accuracy < LOW_ACCURACY_THRESHOLD
    return {
        "kind": "official",
        "model": MODEL_NAME,
        "model_version": str(semver),
        "mlflow_run_id": run_id,
        "best_epoch": best_epoch,
        "official_test_metrics": None
        if pending
        else {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "loss": metrics.loss,
            "test_split_hash": metrics.test_split_hash,
        },
        "pending_official_results": pending,
        "low_accuracy_disclosed": low_accuracy,
        "limitations": list(limitations),
        "statement": (
            "PLANTILLA — sin resultados oficiales de D06-01 todavía. No es un artefacto "
            "publicable; no sustituir con métricas synthetic/local_test."
            if pending
            else (
                "Paquete final publicable: el mismo checkpoint seleccionado y evaluado — "
                "no un retrain, no el package smoke usado durante desarrollo."
            )
        ),
    }


def build_official_package(
    checkpoint: Path | str,
    out: Path | str,
    *,
    experiment: str,
    record: OfficialRecord,
    semver: ModelVersion,
    best_epoch: int | None = None,
    metrics: OfficialTestMetrics | None = None,
    limitations: tuple[str, ...] = (),
    created_at: str | None = None,
) -> PackageManifest:
    """Empaqueta el candidato evaluado, usando `record` (el acta) como
    única fuente de verdad sobre qué run/checkpoint es. Si `metrics` trae
    un `run_id` distinto al del acta, se bloquea ANTES de tocar el
    checkpoint — nunca se elige otro checkpoint ni se ignora el desajuste."""
    if metrics is not None and metrics.run_id != record.run_id:
        raise PackageError(
            f"las métricas oficiales son del run {metrics.run_id!r}, "
            f"el acta referencia {record.run_id!r}: no se empaqueta, hay que "
            "corregir el origen, nunca elegir otro checkpoint"
        )

    card = _official_card(
        run_id=record.run_id,
        best_epoch=best_epoch,
        semver=semver,
        metrics=metrics,
        limitations=limitations,
    )
    return build_smoke_package(
        checkpoint,
        out,
        run_id=record.run_id,
        experiment=experiment,
        expected_sha256=record.checkpoint_sha256,
        created_at=created_at,
        kind="official",
        card=card,
    )
