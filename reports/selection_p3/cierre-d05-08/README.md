# D05-08 — Evidencia del cierre formal (MODEL SELECTION CLOSED)

Respuestas de la API del backend, tal cual las devolvió el stack productor con el MLflow de la campaña D04-03 (lo opera Esteban). Las capturó `d0508-cierre.sh` (en esta carpeta), que solo hace GET para verificar y un único `POST /selection/close` si todo cuadra. Los archivos no se editaron. Autorizó el cierre el PM (Heri); acta en #91.

| Archivo | Petición | Resultado | SHA-256 del archivo |
|---|---|---|---|
| `1-selection-antes.json` | `GET /selection` | `candidate`, run `2d56233c…`, `outcome_hash` `16b13f54…`, `closed_at: null` | `fed57ab6…` |
| `2-campaign.json` | `GET /selection/campaign` | 12/12 filas, `close_blockers: []`, `unattributed: []`, `ready_to_close: true`, `matches_proposal: true`; candidato job 7, checkpoint `0c6b589b…` | `a054ccfc…` |
| `3-evaluation-antes.json` | `GET /evaluation` | `blocked` / `model_selection_open` | `6fb90f19…` |
| `4-cierre.json` | `POST /selection/close` `{"candidate_run_id":"2d56233c886142b7824e1551b90e8327"}` | HTTP 200, `closed`, **`closed_at: 2026-10-03T05:01:59.553Z`** | `99835a26…` |
| `5-selection-despues.json` | `GET /selection` | idéntico a la respuesta del cierre (mismo SHA-256) | `99835a26…` |
| `6-evaluation-despues.json` | `GET /evaluation` | `pending` / `evaluation_missing`: selección cerrada, evaluación oficial todavía inexistente | `24d22b6d…` |

**Cronología.** Propuesta (D05-02): 2026-10-02T16:56:17.478Z → verificación previa: 2026-10-03T04:30:40Z → cierre: 2026-10-03T05:01:59.553Z. Antes del cierre la evaluación estaba bloqueada y después del cierre no existe ninguna evaluación oficial: ningún resultado del frozen test precede a `closed_at`.

**Identidad cerrada.** Run `2d56233c886142b7824e1551b90e8327` (fila 3, job 7, learning_rate 3e-4, mejor época 5; validation accuracy 0.9552, macro-F1 0.9552, loss 0.1317), checkpoint `0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b`, release v0.1.1 (`dvc_release_hash` `2e7029bd…`), `manifest_hash` `0c03c395…`, `outcome_hash` `16b13f546767fef06beb92afe9853a5b14c201fb321788f00f1f4b40d280b8a7`, igual que la propuesta de D05-02 (`../3-propuesta.json`).

D06-01 (#92) es el primer y único consumidor autorizado del frozen test con esta identidad.
