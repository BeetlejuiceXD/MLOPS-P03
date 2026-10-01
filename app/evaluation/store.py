"""D04-05 — Acceso del productor a `p3_model_selection` y `p3_evaluation` (stub Red)."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text
from sqlalchemy.engine import Engine

metadata = MetaData()

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


class EvaluationStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def closed_selection(self):
        raise NotImplementedError

    def save(self, record) -> None:
        raise NotImplementedError

    def read(self, namespace: str):
        raise NotImplementedError
