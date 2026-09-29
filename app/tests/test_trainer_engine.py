"""D02-03 - trainer reproducible: pesos actualizados, capas congeladas intactas,
pasos por batch, mismo seed -> mismo orden, augmentation nunca en val/test.
Todo sobre `trainer.dataset` - nunca manifest oficial ni frozen test."""

import hashlib

import pytest
import torch

from trainer.dataset import build_fixture_dataset
from trainer.engine import _build_optimizer, _make_loader, seed_everything, train
from training.config import TrainingConfig
from training.model import build_model

BASE = {
    "seed": 7,
    "pretrained": False,
    "batch_size": 8,
    "max_epochs": 10,
    "image_size": 128,
    "patience": 3,
}


def _config(**overrides):
    return TrainingConfig(**{**BASE, **overrides})


def _param_hash(model, *, trainable: bool) -> str:
    parts = [
        p.detach().numpy().tobytes()
        for name, p in sorted(model.named_parameters())
        if p.requires_grad is trainable
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


@pytest.fixture(scope="module")
def dataset():
    return build_fixture_dataset(seed=123)


def test_trainable_weights_change_after_training(dataset):
    config = _config(trainable_layers="head_only")
    before = _param_hash(build_model(config), trainable=True)

    result = train(config, dataset)

    assert _param_hash(result.model, trainable=True) != before


def test_frozen_weights_are_unchanged_after_training(dataset):
    config = _config(trainable_layers="head_only")
    before = _param_hash(build_model(config), trainable=False)

    result = train(config, dataset)

    assert _param_hash(result.model, trainable=False) == before


def test_step_count_matches_batches_times_epochs_run(dataset):
    config = _config()
    steps = []

    result = train(config, dataset, after_step=lambda labels: steps.append(len(labels)))

    expected_batches_per_epoch = -(-len(dataset.train) // config.batch_size)  # ceil
    assert len(steps) == expected_batches_per_epoch * len(result.history)


def test_same_seed_gives_identical_order_and_close_weights(dataset):
    config = _config()
    order_a, order_b = [], []

    result_a = train(config, dataset, after_step=lambda labels: order_a.append(list(labels)))
    result_b = train(config, dataset, after_step=lambda labels: order_b.append(list(labels)))

    assert order_a == order_b  # orden de muestras: exacto, sin tolerancia
    for p_a, p_b in zip(result_a.model.parameters(), result_b.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b, atol=1e-5)


def test_different_seed_changes_sample_order(dataset):
    order_a, order_b = [], []
    train(_config(seed=1), dataset, after_step=lambda labels: order_a.append(list(labels)))
    train(_config(seed=2), dataset, after_step=lambda labels: order_b.append(list(labels)))

    assert order_a != order_b


def test_augmentation_never_applies_to_val_even_if_enabled(dataset):
    config = _config(augmentation=True)

    first = next(iter(_make_loader(dataset.val, config, train=False)))[0]
    second = next(iter(_make_loader(dataset.val, config, train=False)))[0]

    assert torch.equal(first, second)  # determinista: nunca augmentation en val


# --- Aisladas de `train()`: no dependen de early stopping ni de convergencia,
# así que no quedan confundidas por cuántas épocas corrió cada config. ---


def test_optimizer_type_matches_config():
    adam = _build_optimizer(build_model(_config(optimizer="adam")), _config(optimizer="adam"))
    sgd = _build_optimizer(build_model(_config(optimizer="sgd")), _config(optimizer="sgd"))

    assert isinstance(adam, torch.optim.Adam)
    assert isinstance(sgd, torch.optim.SGD)


def test_optimizer_applies_configured_weight_decay():
    config = _config(weight_decay=0.005)  # rango valido: [0, 1e-2]
    optimizer = _build_optimizer(build_model(config), config)

    assert optimizer.param_groups[0]["weight_decay"] == 0.005


def test_optimizer_only_receives_trainable_parameters():
    config = _config(trainable_layers="head_only")
    model = build_model(config)
    optimizer = _build_optimizer(model, config)

    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_params = {id(p) for p in model.parameters() if p.requires_grad}
    frozen_params = {id(p) for p in model.parameters() if not p.requires_grad}

    assert optimizer_params == trainable_params
    assert not (optimizer_params & frozen_params)


def test_shuffle_order_depends_on_seed(dataset):
    order_a = [labels for _, labels in _make_loader(dataset.train, _config(seed=1), train=True)]
    order_b = [labels for _, labels in _make_loader(dataset.train, _config(seed=2), train=True)]

    assert not all(torch.equal(a, b) for a, b in zip(order_a, order_b, strict=True))


# --- B1: seeds fuera del dominio de NumPy (validas segun TrainingConfig, #33/#51) ---


@pytest.mark.parametrize("seed", [-1, 4_294_967_296])
def test_seed_everything_accepts_seeds_outside_numpy_domain(seed):
    seed_everything(seed)  # no debe lanzar ValueError


@pytest.mark.parametrize("seed", [-1, 4_294_967_296])
def test_shuffle_generator_accepts_seeds_outside_numpy_domain(seed, dataset):
    loader = _make_loader(dataset.train, _config(seed=seed), train=True)
    next(iter(loader))  # no debe lanzar


@pytest.mark.parametrize("seed", [-1, 4_294_967_296])
def test_train_runs_end_to_end_with_out_of_range_seed(seed, dataset):
    train(_config(seed=seed), dataset)  # no debe lanzar ValueError (smoke)


def test_out_of_range_seed_keeps_original_value_on_config():
    # seed_everything/_make_loader normalizan internamente, pero la seed
    # registrada en TrainingConfig (para MLflow/evidencia) no cambia.
    config = _config(seed=-1)
    assert config.seed == -1


# --- B3: la paciencia depende solo de val_accuracy, no de macro-F1/val_loss ---


def test_early_stopping_ignores_macro_f1_and_loss_improvement(monkeypatch, dataset):
    """Reproduce el caso de la auditoria: val_accuracy constante en 0.8
    mientras macro-F1/val_loss mejoran cada epoca no debe reiniciar la
    paciencia (#33: early stopping monitorea val_accuracy)."""
    config = _config(patience=3, max_epochs=10)
    fake_metrics = iter((0.5 - 0.01 * i, 0.8, 0.5 + 0.01 * i) for i in range(10))

    def fake_evaluate(model, loader, criterion):
        return next(fake_metrics)

    monkeypatch.setattr("trainer.engine._evaluate", fake_evaluate)

    result = train(config, dataset)

    assert result.stopped_early
    assert len(result.history) == config.patience + 1
    assert all(m.val_accuracy == 0.8 for m in result.history)
    assert result.best.epoch == len(result.history)
