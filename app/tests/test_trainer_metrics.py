"""D02-03 - desempate de checkpoints: accuracy -> macro-F1 -> val_loss (#33)."""

import pytest

from trainer.metrics import EpochMetrics, is_better, should_stop


def _metrics(**overrides):
    base = dict(
        epoch=1, train_loss=0.5, train_accuracy=0.8, val_loss=0.4,
        val_accuracy=0.8, val_macro_f1=0.8, learning_rate=1e-3,
    )
    return EpochMetrics(**{**base, **overrides})


def test_higher_val_accuracy_wins():
    assert is_better(_metrics(val_accuracy=0.81), _metrics(val_accuracy=0.80))


def test_lower_val_accuracy_loses():
    assert not is_better(_metrics(val_accuracy=0.79), _metrics(val_accuracy=0.80))


def test_tied_accuracy_breaks_on_macro_f1():
    best = _metrics(val_accuracy=0.80, val_macro_f1=0.70)
    assert is_better(_metrics(val_accuracy=0.80, val_macro_f1=0.75), best)
    assert not is_better(_metrics(val_accuracy=0.80, val_macro_f1=0.65), best)


def test_tied_accuracy_and_macro_f1_breaks_on_val_loss():
    best = _metrics(val_accuracy=0.80, val_macro_f1=0.70, val_loss=0.40)
    assert is_better(_metrics(val_accuracy=0.80, val_macro_f1=0.70, val_loss=0.35), best)
    assert not is_better(_metrics(val_accuracy=0.80, val_macro_f1=0.70, val_loss=0.45), best)


def test_tie_tolerance_is_four_decimals_not_coarser():
    # A 4 decimales, 0.801 != 0.804: decide la accuracy sola (candidate gana,
    # sin mirar macro_f1). Con una tolerancia mas gruesa (p. ej. 2 decimales,
    # donde ambas redondean a 0.80), el codigo caeria al desempate de macro_f1
    # y el macro_f1 mas bajo del candidate le haria perder -> resultado opuesto.
    best = _metrics(val_accuracy=0.801, val_macro_f1=0.90)
    candidate = _metrics(val_accuracy=0.804, val_macro_f1=0.10)
    assert is_better(candidate, best)


@pytest.mark.parametrize(("epochs_without_improvement", "expected"), [(2, False), (3, True), (4, True)])
def test_should_stop_boundary_is_exactly_patience(epochs_without_improvement, expected):
    assert should_stop(epochs_without_improvement, patience=3) is expected