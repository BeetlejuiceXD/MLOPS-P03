# D05-02 — Evidencia de la selección sobre el MLflow real

Son las respuestas de la API del backend (rama `d05-02-reconciliacion-candidato`) tal cual las devolvió el stack que tiene el MLflow de la campaña de D04-03. Ese stack lo opera Esteban. Los archivos no se editaron. En ningún momento se leyó el frozen test ni se cerró la selección.

| Archivo | Petición | Resultado |
|---|---|---|
| `1-estado-inicial.json` | `GET /selection` | `open`, sin candidato |
| `2-expediente.json` | `GET /selection/campaign` | 12/12 filas aceptadas, sin bloqueos, `ready_to_close: true`, `matches_proposal: true` |
| `3-propuesta.json` | `POST /selection/candidate` | `candidate`, `outcome_hash` `16b13f54…` |
| `4-estado-final.json` | `GET /selection` | idéntico a la propuesta (mismo SHA-256 de archivo) |
| `5-evaluation.json` | `GET /evaluation` | `blocked` / `model_selection_open` |
| — | `GET /evaluation/predictions` | HTTP **409** |

El expediente se capturó después de la propuesta: trae `selection_status: candidate` y confirma que la campaña actual es la misma que se propuso (`matches_proposal: true`).

**Fila 1.** Tiene 5 intentos válidos (jobs 1–5). El representante es el job 1 (run `684c6a694c34408fb07c780045656da8`), por tener el menor `start_time`. Los jobs 2–5 quedan como `retry` con motivo `duplicate_campaign_row`. Los cinco tienen el mismo checkpoint (`6adebe3f…`) y las mismas métricas, porque el entrenamiento es determinista.

**Candidato:** run `2d56233c886142b7824e1551b90e8327`, fila 3 (learning_rate 3e-4), job 7, commit `bb7deb5d`, checkpoint `0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b`. Sus métricas de validation en la mejor época (5): accuracy 0.9552, macro-F1 0.9552 y loss 0.1317.

`backend/tests/campaign-report-replay.test.ts` comprueba que el recálculo desde `reports/campaign_p3.json` da el mismo ranking y el mismo candidato que este expediente.
