"""D04-05 — Acceso del productor a `p3_model_selection` y `p3_evaluation`.

Las tablas las crean las migraciones del backend (Drizzle, 0006 y 0007); aquí se
declaran las mismas columnas para SQLAlchemy Core (`test_evaluation_producer.py` las
compara). La selección solo se LEE: abrirla, proponer y cerrar es del backend (D04-04).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from evaluation.producer import ClosedSelection, EvaluationRecord, EvaluationRefusedError

metadata = MetaData()

# D04-04 (0006_p3_model_selection.sql): un solo registro, id = 1.
p3_model_selection = Table(
    "p3_model_selection",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("status", String(16), nullable=False),
    Column("outcome", Text),
    Column("outcome_hash", String(64)),
    Column("proposed_at", DateTime),
    Column("closed_at", DateTime),
)

# D04-05 (0007_p3_evaluation.sql): una evaluación por namespace.
p3_evaluation = Table(
    "p3_evaluation",
    metadata,
    Column("namespace", String(16), primary_key=True),
    Column("candidate_run_id", String(32), nullable=False),
    Column("evaluated_at", DateTime, nullable=False),
    Column("evaluation", Text, nullable=False),
    Column("predictions", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

SELECTION_ID = 1


def _json(value: Any) -> Any:
    # MariaDB guarda JSON como LONGTEXT; algunos drivers ya lo entregan decodificado.
    return json.loads(value) if isinstance(value, str | bytes) else value


def _naive_utc(moment: datetime) -> datetime:
    """UTC sin zona: MariaDB `timestamp` y los contenedores trabajan en UTC."""
    return moment.astimezone(UTC).replace(tzinfo=None)


class EvaluationStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def closed_selection(self) -> ClosedSelection:
        """MODEL SELECTION CLOSED persistido, o `EvaluationRefusedError` (el test
        sigue bloqueado)."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(p3_model_selection).where(p3_model_selection.c.id == SELECTION_ID)
            ).first()
        if row is None or row.status != "closed":
            raise EvaluationRefusedError(
                "model_selection_open",
                "MODEL SELECTION CLOSED no existe: el frozen test sigue bloqueado",
            )
        outcome = _json(row.outcome) or {}
        candidate = outcome.get("candidate") or {}
        reference = outcome.get("reference") or {}
        if not candidate.get("run_id") or not reference.get("manifest_hash") or not row.closed_at:
            raise EvaluationRefusedError(
                "incomplete_selection", "registro de cierre sin candidato, manifest o fecha"
            )
        return ClosedSelection(
            candidate_run_id=candidate["run_id"],
            closed_at=row.closed_at.replace(tzinfo=UTC),
            manifest_hash=reference["manifest_hash"],
        )

    def save(self, record: EvaluationRecord) -> None:
        """`official` se escribe una sola vez; `synthetic` reemplaza la corrida anterior."""
        values = {
            "namespace": record.namespace,
            "candidate_run_id": record.predictions.candidate_run_id,
            "evaluated_at": _naive_utc(datetime.fromisoformat(record.predictions.evaluated_at)),
            "evaluation": record.evaluation.model_dump_json(),
            "predictions": record.predictions.model_dump_json(),
            "created_at": _naive_utc(datetime.now(UTC)),
        }
        try:
            with self.engine.begin() as conn:
                if record.namespace != "official":
                    conn.execute(
                        p3_evaluation.delete().where(p3_evaluation.c.namespace == record.namespace)
                    )
                conn.execute(p3_evaluation.insert().values(**values))
        except IntegrityError as error:
            raise EvaluationRefusedError(
                "official_already_recorded",
                "la evaluación oficial del frozen test ya existe; no se sobrescribe",
            ) from error

    def read(self, namespace: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(p3_evaluation).where(p3_evaluation.c.namespace == namespace)
            ).first()
        if row is None:
            return None
        return {"evaluation": _json(row.evaluation), "predictions": _json(row.predictions)}
