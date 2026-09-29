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

## Qué no hace (todavía)

- No consume el manifest oficial ni `crops/` — D03-03 conecta el trainer real
  con el manifest congelado (D03-01) y el `trainer-worker` de jobs (D02-05).
- No implementa la restauración/verificación completa de early stopping — D03-02.
- No evalúa ni referencia el frozen test oficial.