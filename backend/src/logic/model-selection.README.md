# D04-04 — Selección por validation y bloqueo del test

El servicio de selección elige el candidato **solo con métricas de validation** del
mejor checkpoint de cada run. Persiste el estado en MariaDB y bloquea cualquier
acceso a resultados del frozen test mientras no exista MODEL SELECTION CLOSED.

- `model-selection.ts`: matriz OFAT, elegibilidad y ranking. Es una función pura.
- `model-selection.service.ts`: estado `open → candidate → closed` y guardas.
- `../data/repositories/model-selection.repository.ts`: tabla `p3_model_selection`
  (migración 0006). Tiene un único registro, creado en `open`, que solo cambia con
  `UPDATE` condicionales.
- `campaign-dossier.ts` (D05-02): expediente de campaña. Concilia request → job → run de
  cada intento con su fila y explica el ranking; no decide nada nuevo.
- `../ui/model-selection.routes.ts`: `GET /selection`, `GET /selection/campaign`,
  `POST /selection/candidate`, `POST /selection/close`. `GET /evaluation` y la exportación por muestra están en
  `evaluation.routes.ts` (D04-05) y usan `requireClosed`/`blockedEvaluation` de aquí.

> Este ticket prueba el mecanismo con runs sintéticos. Un candidato **propuesto** es
> preparatorio y no desbloquea el test. El cierre de la campaña oficial lo declara
> D05-02 con los runs reales; la evaluación oficial es D06-01.

## Qué run es elegible

Se evalúa en este orden, y el primer motivo que falle es el que se registra:

| Motivo de exclusión | Regla |
|---|---|
| `invalid_contract` | No cumple `experimentRunSchema`. Por ejemplo, trae métricas de test o le falta un tag de procedencia. |
| `outside_campaign_matrix` | La config completa no es idéntica a una de las 12 filas OFAT de #33, **seed incluida**. Así se excluyen el smoke de D03-04 (`max_epochs=10`), los short-runs y cualquier config fuera de la matriz. |
| `not_finished` | `status` distinto de `FINISHED`. Un FAILED/KILLED con resumen parcial tampoco cuenta. |
| `not_campaign_eligible` | D04-01 no lo da por elegible aunque esté FINISHED; por ejemplo, no tiene `checkpoint_sha256` verificado. |
| `manifest_mismatch` | `tags.manifest_hash` ≠ manifest congelado. |
| `release_mismatch` | `dvc_release` o `dvc_release_hash` ≠ los del manifest congelado. |
| `provenance_inconsistent` | `dvc_release_hash` ≠ `sha256("<images_md5>:<annotations_md5>")`. |
| `duplicate_campaign_row` | Ya cuenta otra corrida de esa fila. Solo cuenta la terminada **más temprana**, para que volver a correr una configuración no dé más oportunidades de ganar. |

Lo que la matriz no barre queda fijo en los defaults congelados
(`TRAINING_CONFIG_DEFAULTS`): `pretrained=true`, `weight_decay=1e-4`, `patience=5`,
`augmentation=true`, `hidden_dim=128`.

Mientras algún run **de la matriz** siga `RUNNING` o `SCHEDULED`, `ready_to_close` es
`false`: un intento en curso podría terminar antes que el representante actual, así que
ni el conteo ni el representante son finales. Un smoke en curso no bloquea.

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
4. El expediente de campaña (D05-02) no tiene bloqueos: al menos 10 filas aceptadas,
   ningún intento de la matriz pendiente (job en cola o corriendo, run en curso) y todos
   los intentos concilian.
5. La escritura condicional confirma que el estado no cambió entretanto.

`closed` es definitivo: proponer o cerrar otra vez responde 409, y ni siquiera se
consultan los runs.

## Expediente de campaña (D05-02)

`GET /selection/campaign` arma, con los runs y jobs actuales, una fila por cada
configuración de la matriz. Es de solo lectura, funciona en cualquier estado y no
escribe nada. Cada fila trae:

- **Todos los intentos**, en orden cronológico: los jobs de training (aunque fallaran
  antes de crear su run o sigan en cola) y los runs que ningún job registra.
- El **rol** de cada intento:
  - `representative`: la corrida FINISHED válida más temprana. Es la que decide
    `selectCandidate`, nunca la de mejor métrica.
  - `retry`: otra corrida válida de la misma fila, que se documenta pero no cuenta.
  - `pending`: el intento sigue en curso.
  - `excluded`: no cuenta; trae el motivo.
- La **conciliación** de cada intento. El run tiene que:
  - declarar el `job_id` del job;
  - haber entrenado exactamente la config del request;
  - tener el mismo manifest y release que el job;
  - tener un estado coherente con el del job.

  Cualquier diferencia queda en `problems`.
- El estado de la fila:
  - `accepted`: tiene representante;
  - `pending`: tiene algún intento en curso;
  - `missing`: no tiene representante.

Además trae el ranking y la **identidad completa del candidato**: run, job, commit,
config, `checkpoint_sha256`, release y DVC, manifest y clases. También trae
`close_blockers` y `matches_proposal`, que indica si la campaña sigue siendo la que se
propuso.

Los smoke, los runs auxiliares y los jobs `controlled` no ocupan ninguna fila: los
smoke y los auxiliares quedan en `unattributed` y los jobs `controlled` no se listan.
Un intento que no concilia bloquea el cierre hasta que se resuelva, sin importar en qué
fila esté.

## Fuentes

- Runs: los reales del adaptador MLflow de D04-01 (`experiment_runs_response`, con sus
  `excluded`).
- Jobs: los de `GET /training/jobs` (D02-05).
- Manifest: el congelado de D03-03.

Si alguna fuente cae o no cumple su contrato, la operación responde **503** con el
motivo y no escribe nada.

## Recálculo desde la evidencia de D04-03

`src/cli/replay-campaign-report.ts` pasa `reports/campaign_p3.json` (PR #82) por el
mismo expediente y la misma selección, con la referencia del manifest congelado
(`reports/manifest_p3.json`). No lee MLflow ni el test y no escribe estado. Sirve para
reproducir la propuesta desde los IDs y hashes entregados y para contrastarla con
`GET /selection/campaign` sobre el MLflow real: deben coincidir el ranking y el
candidato. El `outcome_hash` también cubre los runs excluidos (smoke, short-run), que no
están en el reporte, así que solo coincide si MLflow no tiene ninguno.

```bash
cd backend
npx tsx src/cli/replay-campaign-report.ts   # sale con 1 si la campaña no está lista
```

El reporte no trae las curvas por época, así que `history` se rellena con las métricas
del mejor checkpoint. Eso no interviene en la selección, que usa el `summary`.
`tests/campaign-report-replay.test.ts` fija el resultado sobre la evidencia real.

## Tests y mutation testing

```bash
cd backend
npx vitest run tests/model-selection.test.ts tests/campaign-dossier.test.ts tests/campaign-report-replay.test.ts
# Persistencia real (solo contra una MariaDB desechable con migraciones; la corre CI):
P3_SELECTION_MARIADB_TEST=1 DATABASE_URL=mysql://... npx vitest run tests/model-selection.mariadb.test.ts
cd .. && python .github/scripts/run_model_selection_mutations.py   # copia aislada
```
