# D02-03 — Trainer reproducible por minibatches

`trainer.engine.train(config, dataset)` ejecuta forward/loss/backward/optimizer
real por minibatch sobre `trainer.dataset.build_fixture_dataset(seed)` — un
fixture propio, sintético, sin depender de `crops/` ni `manifest/` (D02-04/D03-01
quedan fuera a propósito: nunca se entrena sobre datos que puedan terminar en el
frozen test oficial, custodiado por Ale).

## TrainingConfig efectivo — cómo se usa cada campo

| Campo | Dónde se aplica |
|---|---|
| `architecture`, `pretrained` | `training.model.build_model` |
| `trainable_layers` | `build_model` (congela capas) + `_build_optimizer` (filtra parámetros) |
| `hidden_layers`, `hidden_dim`, `dropout` | `build_model` (cabeza) |
| `optimizer`, `learning_rate`, `weight_decay` | `_build_optimizer` |
| `batch_size` | `_make_loader` |
| `max_epochs`, `patience` | bucle de `train()` (early stopping sobre `val_accuracy`) |
| `augmentation` | `training.preprocessing` vía `train=True/False` en `_make_loader` |
| `seed` | `seed_everything` + init de la cabeza (`build_model`) + shuffle del `DataLoader` |

## Early stopping y mejor checkpoint (D03-02)

Estas son las reglas vigentes. D03-02 las ratificó con secuencias controladas
(`tests/test_trainer_early_stopping.py`) sin cambiar el código:

| Regla | Dónde |
|---|---|
| La paciencia depende **solo** de `val_accuracy`. Una mejora estricta a 4 decimales la reinicia; una mejora solo de macro-F1 o de loss, no. | `metrics.accuracy_improved` |
| Con `patience` épocas seguidas sin mejora, el entrenamiento para en esa misma época (con `patience=3` y la mejor en la 1, para en la 4). | `metrics.should_stop` |
| El mejor checkpoint se elige por accuracy → macro-F1 → menor loss, con igualdad a 4 decimales en cada paso. Si empatan los tres, se queda la época anterior. | `metrics.is_better` |
| Al terminar, con o sin early stopping, el modelo devuelto tiene exactamente los pesos y buffers de la mejor época (`deepcopy` del `state_dict` + `load_state_dict`). | `engine.train` |

**Contrato del resultado (handoff a D03-03).** `train()` devuelve un `TrainingResult` con estos campos:

| Campo | Qué contiene |
|---|---|
| `history` | Un `EpochMetrics` por época corrida, con `epoch` 1..N consecutivo: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `val_macro_f1`, `learning_rate`. Son las curvas train/validation. |
| `best` | Es **la misma entrada** `history[best.epoch - 1]`, sin redondear. `best_epoch` y `best_val_*` se publican de aquí. |
| `model` | Ya restaurado a la mejor época. `torch.save(result.model.state_dict())` es el checkpoint: es lo que sube D02-06 a MLflow y verifica por sha256. |
| `stopped_early` | `True` si la paciencia se agotó antes de `max_epochs`. |

El resultado no incluye métricas de test, y nada aquí elige globalmente la campaña: eso lo hace D05-02.

**Mutation testing:**

```bash
cd app && uv run python ../.github/scripts/run_early_stopping_mutations.py   # ~15-20 min
```

El script aplica 9 mutantes (E01–E09) a `engine.py` y `metrics.py` con el mismo
procedimiento y la misma clasificación que el de D03-01 (ver `app/manifest/README.md`).

## Qué no hace (todavía)

- No consume el manifest oficial ni `crops/` — D03-03 conecta el trainer real
  con el manifest congelado (D03-01) y el `trainer-worker` de jobs (D02-05).
- No selecciona el mejor run de la campaña (D05-02) ni introduce `min_delta`.
- No evalúa ni referencia el frozen test oficial.