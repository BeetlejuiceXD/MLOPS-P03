# D05-03 — Cotejo de Experiments con la campaña y el candidato reales

Respuestas reales del MLflow de la campaña de D04-03, restaurado sin reentrenar con el
snapshot DVC de #104 (`dvc pull -r prod` + `restore.sh`, proyecto aislado
`mlops-p03-restore`, `verify.py` de #108: PASS). Los archivos no se editaron. En ningún
momento se leyó el frozen test ni se cerró la selección.

| Archivo | Petición | Resultado |
|---|---|---|
| `1-experiments-runs.json` | `GET /api/experiments/runs` (adaptador D04-01) | 16 runs de training `FINISHED`, 0 auxiliares |
| `2-selection.json` | `GET /api/selection` (D05-02) | `candidate`, 12 filas en el ranking, 4 en `excluded` (`duplicate_campaign_row`). Igual, campo por campo, a `reports/selection_p3/4-estado-final.json` de D05-02 (`outcome_hash` `16b13f54…`) |
| `3-mlflow-runs-search.json` | MLflow `POST /api/2.0/mlflow/runs/search` | 16 runs del experimento `p3-cnn-classifier` |
| `4-mlflow-candidate-checkpoint.json` | MLflow `GET /api/2.0/mlflow/artifacts/list` (candidato, `checkpoint/`) | `checkpoint/model.pt` de 44 781 003 bytes y 4 archivos más |

## Cotejo por run ID (MLflow ↔ API ↔ selección ↔ UI)

Para cada uno de los 16 runs se compararon `status`, `best_epoch`, `best_val_accuracy`,
`best_val_macro_f1`, `best_val_loss` y `checkpoint_sha256` entre MLflow (3), la API de
Experiments (1) y, en las filas aceptadas, el ranking de la selección (2): **0
discrepancias**. La columna UI es lo que muestra Experiments con esos datos.

| Fila | Job | Run | Mejor época | val acc. | macro-F1 | val loss | checkpoint | Selección (D05-02) | UI (Experiments) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `684c6a694c34408fb07c780045656da8` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | aceptada | Fila 1 |
| 2 | 6 | `a09c7b6e5102491f9dfc2391b88bb9b8` | 17 | 0.9552 | 0.9552 | 0.1498 | `2aa2549e…` | aceptada | Fila 2 |
| 3 | 7 | `2d56233c886142b7824e1551b90e8327` | 5 | 0.9552 | 0.9552 | 0.1317 | `0c6b589b…` | aceptada · **candidato** | Fila 3 · Candidato propuesto |
| 4 | 8 | `379b553913254241a2efece3ed451d2d` | 16 | 0.9403 | 0.9402 | 0.1755 | `e9b36b93…` | aceptada | Fila 4 |
| 5 | 9 | `c2adc19a9fb94b6999937289bf2a1f2c` | 11 | 0.9552 | 0.9551 | 0.1351 | `1e7ab9b0…` | aceptada | Fila 5 |
| 6 | 10 | `6646544b73734ac1abdf413d1b63b3d9` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | aceptada | Fila 6 |
| 7 | 11 | `a8b7545b465b465990fe7e7c594c1798` | 6 | 0.9179 | 0.9177 | 0.3127 | `9ef31508…` | aceptada | Fila 7 |
| 8 | 12 | `5fe9d9d9ca46440b9e591daff45acf35` | 7 | 0.9254 | 0.9252 | 0.1603 | `69e74269…` | aceptada | Fila 8 |
| 9 | 13 | `02397ddadec047809ac0dd2421aeff7b` | 10 | 0.9328 | 0.9325 | 0.1881 | `a6534599…` | aceptada | Fila 9 |
| 10 | 14 | `438cc06bdf32461fa2982963e8be3727` | 5 | 0.9403 | 0.9403 | 0.2152 | `fa92d391…` | aceptada | Fila 10 |
| 11 | 15 | `33715e5e663e4ace82265b3d05b48158` | 4 | 0.9179 | 0.9179 | 0.1597 | `b95de4b7…` | aceptada | Fila 11 |
| 12 | 16 | `bd4ee53765bc4a9c96ff3cb3b1f5330d` | 2 | 0.9403 | 0.9403 | 0.1235 | `55857f4f…` | aceptada | Fila 12 |
| (1) | 2 | `800fa919f5bb4b30a3ffdd1f5e2dcbec` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | excluida: `duplicate_campaign_row` | Reintento de una fila ya aceptada (no cuenta) |
| (1) | 3 | `d7965d2980a645f8b3ec74613ddd66a0` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | excluida: `duplicate_campaign_row` | Reintento de una fila ya aceptada (no cuenta) |
| (1) | 4 | `bafb68badcda4b8f963b51372f8c50f8` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | excluida: `duplicate_campaign_row` | Reintento de una fila ya aceptada (no cuenta) |
| (1) | 5 | `7576d02bb5564dbf94429373e9587a3a` | 7 | 0.9179 | 0.9178 | 0.1590 | `6adebe3f…` | excluida: `duplicate_campaign_row` | Reintento de una fila ya aceptada (no cuenta) |

Conteo en Experiments: **16** runs de training, **12** cuentan para la campaña (las filas
aceptadas por D05-02) y **4** reintentos no cuentan. Antes de D05-03 decía "Elegibles para
campaña: 16"; ese conteo solo se sigue mostrando mientras la selección no ha aceptado filas.

**Candidato:** `2d56233c886142b7824e1551b90e8327`, fila 3, job 7. Experiments lo rotula
"Candidato propuesto" y "Propuesta pendiente de cierre (D05-08)… No es un cierre formal".
Detalle del run: commit `bb7deb5d`, release `v0.1.1`, manifest `p3-v0.1.1-s42`,
`checkpoint_sha256` `0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b`,
historial de 10 épocas y artefacto `checkpoint/model.pt` (42.7 MB = 44 781 003 bytes en MLflow).

## Recarga e indisponibilidad

- Con MLflow detenido (`docker compose -p mlops-p03-restore stop mlflow`), Experiments
  muestra el 503 del adaptador (`mlflow_unavailable…`) con "Reintentar". No muestra runs,
  curvas ni conteos inventados.
- Al levantarlo de nuevo, **Actualizar** vuelve a pedir runs y selección y muestra lo mismo:
  16 / 12 / 4 y el mismo candidato.

Las capturas están en el PR (#100).
