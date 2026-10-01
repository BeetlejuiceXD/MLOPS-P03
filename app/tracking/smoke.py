"""D03-04 — Smoke real Training → MLflow → checkpoint (verificador)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SmokeReport:
    problems: list[str] = field(default_factory=list)
    run_id: str | None = None
    run_status: str | None = None
    epochs_logged: int | None = None
    checkpoint_sha256: str | None = None
    manifest_hash: str | None = None
    loaded_classes: list[str] | None = None
    probabilities_sum: float | None = None


def verify_smoke(job: dict, *, tracking_uri: str, workdir: Path) -> SmokeReport:
    raise NotImplementedError
