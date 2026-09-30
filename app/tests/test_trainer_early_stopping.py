"""D03-02 — ratificación de early stopping y mejor checkpoint con secuencias
controladas de validation.

`trainer.engine._evaluate` se sustituye por una secuencia conocida de
(val_loss, val_accuracy, val_macro_f1) por época; el entrenamiento (forward,
backward, optimizer) es real, así que los pesos cambian en cada época. En cada
evaluación se guarda el hash de los pesos de ESA época: con eso se comprueba que el
modelo devuelto — y el checkpoint que guarda D02-06 (`torch.save(result.model
.state_dict())`) — son los de la mejor época y no los de la última.

Todo sobre `trainer.dataset` (fixture sintético): nunca manifest oficial ni frozen
test (custodia de Ale, #33).
"""

import hashlib

import pytest
import torch

from trainer.dataset import build_fixture_dataset
from trainer.engine import train
from training.config import TrainingConfig

BASE = {
    "seed": 7,
    "pretrained": False,
    "batch_size": 8,
    "image_size": 128,
}


def _config(**overrides):
    return TrainingConfig(**{**BASE, **overrides})


def _state_hash(state_dict) -> str:
    """sha256 de pesos + buffers (BatchNorm incluido), en orden de clave."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def dataset():
    return build_fixture_dataset(seed=123)


def _scripted(monkeypatch, accuracies, *, f1=None, losses=None):
    """Sustituye `_evaluate` por la secuencia dada y devuelve la lista (que se
    llena durante `train`) con el hash de los pesos evaluados en cada época."""
    f1 = f1 or [0.5] * len(accuracies)
    losses = losses or [0.5] * len(accuracies)
    sequence = list(zip(losses, accuracies, f1, strict=True))
    seen: list[str] = []

    def fake_evaluate(model, loader, criterion):
        if len(seen) == len(sequence):
            pytest.fail(f"el entrenamiento siguió después de la época {len(sequence)}")
        seen.append(_state_hash(model.state_dict()))
        return sequence[len(seen) - 1]

    monkeypatch.setattr("trainer.engine._evaluate", fake_evaluate)
    return seen


def _assert_restored(result, seen):
    """El modelo devuelto tiene EXACTAMENTE los pesos evaluados en la mejor época."""
    assert len(set(seen)) == len(seen), "los pesos deben cambiar en cada época"
    assert _state_hash(result.model.state_dict()) == seen[result.best.epoch - 1]


# --- parada en la época esperada ------------------------------------------------------


def test_constant_accuracy_with_improving_f1_and_loss_exhausts_patience(monkeypatch, dataset):
    """patience=3, accuracy estancada en 0.8 mientras macro-F1 y loss mejoran en cada
    época: solo val_accuracy alimenta la paciencia → se detiene en la época 4."""
    seen = _scripted(
        monkeypatch,
        [0.8] * 10,
        f1=[0.50 + 0.01 * i for i in range(10)],
        losses=[0.50 - 0.01 * i for i in range(10)],
    )

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.stopped_early
    assert [m.epoch for m in result.history] == [1, 2, 3, 4]
    # El checkpoint sí desempata por macro-F1: la mejor es la 4, con sus pesos.
    assert result.best.epoch == 4
    _assert_restored(result, seen)


def test_improvement_resets_patience_and_stops_three_epochs_later(monkeypatch, dataset):
    """Mejora en la época 5 reinicia la paciencia: 6, 7 y 8 sin mejora → para en 8."""
    seen = _scripted(monkeypatch, [0.60, 0.70, 0.70, 0.70, 0.80, 0.80, 0.80, 0.80, 0.99, 0.99])

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.stopped_early
    assert len(result.history) == 8
    assert result.best.epoch == 5  # distinta de la última
    _assert_restored(result, seen)


def test_accuracy_tie_at_four_decimals_does_not_reset_patience(monkeypatch, dataset):
    """0.80004 frente a 0.80001: iguales a cuatro decimales → no es mejora."""
    seen = _scripted(monkeypatch, [0.80001, 0.80004, 0.80003, 0.80004, 0.9, 0.9])

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.stopped_early
    assert len(result.history) == 4
    assert result.best.epoch == 1  # empate total: se conserva la primera
    _assert_restored(result, seen)


def test_deterioration_stops_and_restores_the_first_epoch(monkeypatch, dataset):
    seen = _scripted(monkeypatch, [0.90, 0.85, 0.80, 0.75, 0.70])

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.stopped_early
    assert len(result.history) == 4
    assert result.best.epoch == 1
    assert seen[0] != seen[-1]
    _assert_restored(result, seen)


def test_steady_improvement_runs_all_epochs_without_early_stop(monkeypatch, dataset):
    seen = _scripted(monkeypatch, [0.50 + 0.04 * i for i in range(10)])

    result = train(_config(patience=3, max_epochs=10), dataset)  # max_epochs >= 10

    assert not result.stopped_early
    assert len(result.history) == 10
    assert result.best.epoch == 10
    _assert_restored(result, seen)


# --- desempates del mejor checkpoint -------------------------------------------------


def test_loss_equal_at_four_decimals_keeps_the_earlier_checkpoint(monkeypatch, dataset):
    """Misma accuracy y macro-F1; val_loss 0.40002 (época 2) y 0.40001 (época 3) son
    iguales a cuatro decimales → no reemplazan a la época 2, aunque sea "menor"."""
    seen = _scripted(
        monkeypatch,
        [0.70, 0.80, 0.80, 0.80, 0.80],
        f1=[0.70, 0.80, 0.80, 0.80, 0.80],
        losses=[0.5, 0.40002, 0.40001, 0.40002, 0.40001],
    )

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.stopped_early
    assert len(result.history) == 5
    assert result.best.epoch == 2
    _assert_restored(result, seen)


def test_lower_loss_beyond_four_decimals_moves_the_checkpoint(monkeypatch, dataset):
    seen = _scripted(
        monkeypatch,
        [0.80, 0.80, 0.80, 0.80],
        f1=[0.80, 0.80, 0.80, 0.80],
        losses=[0.4500, 0.4200, 0.4400, 0.4300],
    )

    result = train(_config(patience=3, max_epochs=10), dataset)

    assert result.best.epoch == 2
    _assert_restored(result, seen)


# --- coherencia de resultado, historial y checkpoint ----------------------------------


def test_best_is_exactly_the_history_entry_of_its_epoch(monkeypatch, dataset):
    _scripted(
        monkeypatch,
        [0.60, 0.8123456, 0.70, 0.75, 0.79],
        f1=[0.6, 0.7777777, 0.6, 0.6, 0.6],
        losses=[0.6, 0.3333333, 0.5, 0.5, 0.5],
    )

    result = train(_config(patience=3, max_epochs=10), dataset)

    # Mismos valores sin redondear: best_val_* publicados = history de esa época.
    assert result.best == result.history[result.best.epoch - 1]
    assert (result.best.val_accuracy, result.best.val_macro_f1, result.best.val_loss) == (
        0.8123456,
        0.7777777,
        0.3333333,
    )
    assert [m.epoch for m in result.history] == list(range(1, len(result.history) + 1))


def test_saved_checkpoint_holds_the_best_epoch_weights(monkeypatch, dataset, tmp_path):
    """Lo que D02-06 sube a MLflow (`torch.save(result.model.state_dict())`), al
    recargarlo, son los pesos de la mejor época y no los de la última."""
    seen = _scripted(monkeypatch, [0.70, 0.90, 0.80, 0.80, 0.80])

    result = train(_config(patience=3, max_epochs=10), dataset)
    path = tmp_path / "model.pt"
    torch.save(result.model.state_dict(), path)
    reloaded = torch.load(path, weights_only=True)

    assert result.best.epoch == 2
    assert _state_hash(reloaded) == seen[1]
    assert _state_hash(reloaded) != seen[-1]
