#!/usr/bin/env bash
# D06-01 — Primera y única evaluación oficial del frozen test.
#
# Se corre UNA SOLA VEZ en la máquina del stack productor, porque ahí está la MariaDB con
# MODEL SELECTION CLOSED (D05-08, closed_at 2026-10-03T05:01:59.553Z, evidencia en
# reports/selection_p3/cierre-d05-08/).
#
# Uso (desde la raíz del repo, rama d06-01-evaluacion-oficial):
#   ./scripts/d06-01/evaluacion-oficial.sh            # preflight y, si pasa, la corrida
#   ./scripts/d06-01/evaluacion-oficial.sh preflight  # solo el preflight (no abre el test)
#
# Requisitos: stack productor arriba (mariadb publicado en localhost:3306), `uv` (o PYTHON=<intérprete de app>), y
# `dvc pull` de data/p3, data/raw y data/mlflow_snapshot (de ahí sale el checkpoint).
# No reentrena, no hace dvc push y no toca volúmenes.

set -euo pipefail

MODE="${1:-run}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Acta de D05-08 (#91, PR #127): identidad cerrada.
RUN_ID="2d56233c886142b7824e1551b90e8327"
CHECKPOINT_SHA256="0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b"
MANIFEST_HASH="0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43"
TEST_SPLIT_HASH="66068afd57c1d5827a633b321fee0c039db730f05196237464b35e80614f7ea0"
OUTCOME_HASH="16b13f546767fef06beb92afe9853a5b14c201fb321788f00f1f4b40d280b8a7"
ACT=(
  --candidate-run-id "$RUN_ID"
  --checkpoint-sha256 "$CHECKPOINT_SHA256"
  --manifest-hash "$MANIFEST_HASH"
  --test-split-hash "$TEST_SPLIT_HASH"
  --outcome-hash "$OUTCOME_HASH"
)

CHECKPOINT="${CHECKPOINT:-data/mlflow_snapshot/artifacts/mlflow-artifacts/1/$RUN_ID/artifacts/checkpoint}"
PACKAGE="${PACKAGE:-$REPO_ROOT/.d06-01-package}"
OUT="$REPO_ROOT/reports/evaluation_p3/official"
# Intérprete de app/: `uv run python` si hay uv; si no, PYTHON (p. ej. app/.venv/Scripts/python.exe).
if [ -n "${PYTHON:-}" ]; then PY=("$PYTHON"); elif command -v uv >/dev/null; then PY=(uv run python); else PY=(python3); fi

if [ -f "$OUT/attempt.json" ]; then
  echo "ALTO: $OUT/attempt.json ya existe. El intento se audita; no se repite." >&2
  exit 3
fi

if [ -z "${DATABASE_URL:-}" ]; then
  if [ -z "${MARIADB_ROOT_PASSWORD:-}" ]; then
    MARIADB_ROOT_PASSWORD="$(sed -n 's/^MARIADB_ROOT_PASSWORD=//p' .env | head -n 1)"
  fi
  export DATABASE_URL="mysql+pymysql://root:${MARIADB_ROOT_PASSWORD}@127.0.0.1:3306/image_repo"
fi

echo "==> [1/4] Checkpoint del acta"
actual="$(sha256sum "$CHECKPOINT/model.pt" | cut -d' ' -f1)"
if [ "$actual" != "$CHECKPOINT_SHA256" ]; then
  echo "ALTO: $CHECKPOINT/model.pt da $actual, el acta dice $CHECKPOINT_SHA256" >&2
  exit 3
fi
echo "    $actual"

echo "==> [2/4] Paquete del candidato (formato D05-01, fuera de Git)"
if [ ! -d "$PACKAGE" ]; then
  (cd app && "${PY[@]}" - "$REPO_ROOT/$CHECKPOINT" "$PACKAGE" "$RUN_ID" "$CHECKPOINT_SHA256" <<'PY'
import sys
from datetime import UTC, datetime
from pathlib import Path

from model_package import build_smoke_package

checkpoint, out, run_id, sha = sys.argv[1:5]
build_smoke_package(
    Path(checkpoint),
    Path(out),
    run_id=run_id,
    experiment="p3-cnn-classifier",
    expected_sha256=sha,
    created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
)
PY
  )
fi

echo "==> [3/4] Preflight (no abre el test)"
(cd app && "${PY[@]}" -m evaluation.official preflight --package "$PACKAGE" "${ACT[@]}")

if [ "$MODE" = "preflight" ]; then
  exit 0
fi

echo "==> [4/4] Evaluación oficial (una sola vez)"
(cd app && "${PY[@]}" -m evaluation.official run --package "$PACKAGE" "${ACT[@]}")

echo
echo "Listo. Sube la evidencia (sin el paquete):"
echo "  git add reports/evaluation_p3/official && git commit -m 'D06-01: evidencia de la evaluación oficial' && git push"
