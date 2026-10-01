# D04-04 — Selección por validation y bloqueo del test

El servicio de selección elige el candidato **solo con métricas de validation** del
mejor checkpoint de cada run. Persiste el estado en MariaDB y bloquea cualquier
acceso a resultados del frozen test mientras no exista MODEL SELECTION CLOSED.

- `model-selection.ts`: matriz OFAT, elegibilidad y ranking. Es una función pura.
- `model-selection.service.ts`: estado `open → candidate → closed` y guardas.
- `../data/repositories/model-selection.repository.ts`: tabla `p3_model_selection`
  (migración 0006). Tiene un único registro, creado en `open`, que solo cambia con
  `UPDATE` condicionales.
- `../ui/model-selection.routes.ts`: `GET /selection`, `POST /selection/candidate`,
  `POST /selection/close`. `GET /evaluation` y la exportación por muestra están en
  `evaluation.routes.ts` (D04-05) y usan `requireClosed`/`blockedEvaluation` de aquí.

> Este ticket prueba el mecanismo con runs sintéticos. Un candidato **propuesto** es
> preparatorio y no desbloquea el test. El cierre de la campaña oficial lo declara
> D05-02 con los runs reales; la evaluación oficial es D06-01.

## Qué run es elegible

Se evalúa en este orden, y el primer motivo que falle es el que se registra:

| Motivo de exclusión | Regla |
|---|---|
| `invalid_contract` | No cumple `experimentRunSchema`. Por ejemplo, trae métricas de test o le falta un tag de procedencia. |
| `not_finished` | `status` distinto de `FINISHED`. Un FAILED/KILLED con resumen parcial tampoco cuenta. |
| `outside_campaign_matrix` | La config completa no es idéntica a una de las 12 filas OFAT de #33, **seed incluida**. Así se excluyen el smoke de D03-04 (`max_epochs=10`), los short-runs y cualquier config fuera de la matriz. |
| `manifest_mismatch` | `tags.manifest_hash` ≠ manifest congelado. |
| `release_mismatch` | `dvc_release` o `dvc_release_hash` ≠ los del manifest congelado. |
| `provenance_inconsistent` | `dvc_release_hash` ≠ `sha256("<images_md5>:<annotations_md5>")`. |
| `duplicate_campaign_row` | Ya cuenta otra corrida de esa fila. Solo cuenta la terminada **más temprana**, para que volver a correr una configuración no dé más oportunidades de ganar. |

Lo que la matriz no barre queda fijo en los defaults congelados
(`TRAINING_CONFIG_DEFAULTS`): `pretrained=true`, `weight_decay=1e-4`, `patience=5`,
`augmentation=true`, `hidden_dim=128`.

## Ranking

`val_accuracy` ↓ → `val_macro_f1` ↓ → `val_loss` ↑ → `run_id` ↑.

Cada métrica se compara redondeada a 4 decimales: si dos runs coinciden en el valor
redondeado, decide el siguiente criterio. Las métricas salen del `summary`, es decir
del mejor checkpoint, no de la última época.

## Cierre

`POST /selection/close` con `{ "candidate_run_id": "<32 hex>" }`. Solo cierra si se
cumplen todas estas condiciones:

1. El estado es `candidate` y el run confirmado **es** el candidato propuesto.
2. Hay al menos 10 filas distintas de la matriz comparables.
3. Recalculado con los runs actuales, el `outcome_hash` es el mismo que el propuesto.
   Si la campaña cambió, hay que volver a proponer.
4. La escritura condicional confirma que el estado no cambió entretanto.

`closed` es definitivo: proponer o cerrar otra vez responde 409, y ni siquiera se
consultan los runs.

## Fuente de runs

Los runs llegan con el contrato `experiment_runs_response` del adaptador MLflow de
D04-01. Hasta que ese adaptador se integre, `server.ts` usa
`runsAdapterPendingSource`: proponer y cerrar responden **503** con el motivo, en
vez de inventar runs.

## Tests y mutation testing

```bash
cd backend
npx vitest run tests/model-selection.test.ts
# Persistencia real (solo contra una MariaDB desechable con migraciones; la corre CI):
P3_SELECTION_MARIADB_TEST=1 DATABASE_URL=mysql://... npx vitest run tests/model-selection.mariadb.test.ts
cd .. && python .github/scripts/run_model_selection_mutations.py   # copia aislada
```
