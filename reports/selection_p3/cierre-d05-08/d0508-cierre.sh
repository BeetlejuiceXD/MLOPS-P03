#!/usr/bin/env bash
# D05-08: cierre formal de la seleccion (MODEL SELECTION CLOSED) en el stack productor (macOS/Linux).
# Verifica TODO antes de cerrar; si algo no cuadra, NO cierra. Nunca reintenta ni fuerza.
# Uso: bash d0508-cierre.sh [http://localhost:3100]
set -u
API="${1:-http://localhost:3100}"
RUN="2d56233c886142b7824e1551b90e8327"
HASH="16b13f546767fef06beb92afe9853a5b14c201fb321788f00f1f4b40d280b8a7"
OUT="d0508-cierre-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
stop() { echo "NO SE CIERRA: $1"; echo "Evidencia en $OUT"; exit 1; }
field() { python3 -c "import json,sys;d=json.load(open('$1'));print($2)" 2>/dev/null; }

echo "== D05-08 verificacion previa $(date -u +%Y-%m-%dT%H:%M:%SZ)"
curl -s "$API/selection"          > "$OUT/1-selection-antes.json"  || stop "no responde $API"
curl -s "$API/selection/campaign" > "$OUT/2-campaign.json"
curl -s "$API/evaluation"         > "$OUT/3-evaluation-antes.json"

STATUS=$(field "$OUT/1-selection-antes.json" "d['status']")
CAND=$(field "$OUT/1-selection-antes.json" "(d.get('candidate') or {}).get('run_id')")
OHASH=$(field "$OUT/1-selection-antes.json" "d.get('outcome_hash')")
READY=$(field "$OUT/2-campaign.json" "d.get('ready_to_close')")
MATCH=$(field "$OUT/2-campaign.json" "d.get('matches_proposal')")
EVAL=$(cat "$OUT/3-evaluation-antes.json")
echo "status=$STATUS candidato=$CAND"
echo "outcome_hash=$OHASH"
echo "ready_to_close=$READY matches_proposal=$MATCH"
echo "evaluation=$EVAL"

[ "$STATUS" = "candidate" ] || stop "la seleccion no esta en candidate ($STATUS)"
[ "$CAND" = "$RUN" ]        || stop "el candidato no es $RUN"
[ "$OHASH" = "$HASH" ]      || stop "outcome_hash distinto al de D05-02"
[ "$READY" = "True" ]       || stop "ready_to_close no es true"
[ "$MATCH" = "True" ]       || stop "matches_proposal no es true"

read -r -p "Todo cuadra. Escribe CERRAR para ejecutar el cierre formal (no se puede deshacer): " OK
[ "$OK" = "CERRAR" ] || stop "cancelado"

CODE=$(curl -s -o "$OUT/4-cierre.json" -w "%{http_code}" -X POST "$API/selection/close" \
  -H "Content-Type: application/json" -d "{\"candidate_run_id\":\"$RUN\"}")
echo "POST /selection/close -> HTTP $CODE"; cat "$OUT/4-cierre.json"; echo
[ "$CODE" = "200" ] || stop "el cierre respondio $CODE; queda abierta. No reintentes"

curl -s "$API/selection"  > "$OUT/5-selection-despues.json"
curl -s "$API/evaluation" > "$OUT/6-evaluation-despues.json"
echo "status=$(field "$OUT/5-selection-despues.json" "d['status']") closed_at=$(field "$OUT/5-selection-despues.json" "d.get('closed_at')")"
echo "== Listo. Comprime y manda la carpeta $OUT a Hannah: zip -r $OUT.zip $OUT"
