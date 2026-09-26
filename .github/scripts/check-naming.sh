#!/usr/bin/env bash
# Comprueba la convención de nombre de rama y de título de PR (ver CONTRIBUTING.md).
#
# Uso: bash .github/scripts/check-naming.sh "<rama>" "<título del PR>"
#
# Lo llama .github/workflows/pr-hygiene.yml, y sirve igual en local antes de abrir el PR:
#   bash .github/scripts/check-naming.sh "$(git branch --show-current)" "D01-06: CI efectiva"
set -u

# Sin esto, en algunas configuraciones regionales [a-z] también acepta mayúsculas.
export LC_ALL=C

BRANCH="${1:-}"
TITLE="${2:-}"

TYPES='feat|fix|test|chore|docs|refactor|ci|build|perf|style'
# Tickets diarios de P3: DNN-NN (día-ticket), p. ej. D01-06.
# d01-06-ci-exclusions · d02-03-d02-04-trainer-mlflow · fix/nombre-corto
TICKET='d[0-9]{2}-[0-9]{2}'
BRANCH_RE="^(${TICKET}(-${TICKET})*-[a-z0-9]+(-[a-z0-9]+)*|($TYPES)/[a-z0-9]+(-[a-z0-9]+)*)\$"
# D01-06: texto · D02-03 D02-04: texto · fix: texto · feat(ui): texto
TITLE_TICKET='D[0-9]{2}-[0-9]{2}'
TITLE_RE="^(${TITLE_TICKET}( ${TITLE_TICKET})*|($TYPES)(\\([a-z0-9-]+\\))?): .+"

status=0

if ! [[ "$BRANCH" =~ $BRANCH_RE ]]; then
  echo "::error title=Nombre de rama::'$BRANCH' no sigue la convención. Usa dNN-NN-descripcion" \
    "(varios tickets: d02-03-d02-04-descripcion) o tipo/descripcion con tipo en ($TYPES)," \
    "todo en minúsculas y con guiones. Ver CONTRIBUTING.md." >&2
  status=1
fi

if ! [[ "$TITLE" =~ $TITLE_RE ]]; then
  echo "::error title=Título del PR::'$TITLE' no sigue la convención. Usa 'DNN-NN: descripción'" \
    "(varios tickets: 'D02-03 D02-04: ...') o 'tipo: descripción' con" \
    "tipo en ($TYPES). Ver CONTRIBUTING.md." >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "La rama y el título cumplen la convención."
fi
exit "$status"
