"""D02-03 - desempate de checkpoints: accuracy -> macro-F1 -> val_loss (#33)."""

import pytest

from trainer.metrics import EpochMetrics, accuracy_improved, is_better, should_stop


def _metrics(**overrides):
    base = {
        "epoch": 1,
        "train_loss": 0.5,
        "train_accuracy": 0.8,
        "val_loss": 0.4,
        "val_accuracy": 0.8,
        "val_macro_f1": 0.8,
        "learning_rate": 1e-3,
    }
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


@pytest.mark.parametrize(
    ("epochs_without_improvement", "expected"), [(2, False), (3, True), (4, True)]
)
def test_should_stop_boundary_is_exactly_patience(epochs_without_improvement, expected):
    assert should_stop(epochs_without_improvement, patience=3) is expected


# --- B2: el desempate por val_loss tambien respeta 4 decimales ---


def test_val_loss_tie_break_respects_four_decimal_tolerance():
    best = _metrics(val_accuracy=0.80, val_macro_f1=0.80, val_loss=0.40002)
    candidate = _metrics(val_accuracy=0.80, val_macro_f1=0.80, val_loss=0.40001)
    assert not is_better(candidate, best)
    assert not is_better(best, candidate)


def test_val_loss_beyond_four_decimals_still_decides():
    best = _metrics(val_accuracy=0.80, val_macro_f1=0.80, val_loss=0.4500)
    candidate = _metrics(val_accuracy=0.80, val_macro_f1=0.80, val_loss=0.3500)
    assert is_better(candidate, best)


# --- B3: accuracy_improved es el UNICO criterio de paciencia ---


def test_accuracy_improved_true_when_strictly_higher():
    assert accuracy_improved(_metrics(val_accuracy=0.81), _metrics(val_accuracy=0.80))


def test_accuracy_improved_false_when_tied_at_four_decimals():
    assert not accuracy_improved(_metrics(val_accuracy=0.80001), _metrics(val_accuracy=0.80004))


def test_accuracy_improved_false_when_macro_f1_or_loss_improve_but_accuracy_does_not():
    reference = _metrics(val_accuracy=0.80, val_macro_f1=0.50, val_loss=0.50)
    better_secondary_metrics = _metrics(val_accuracy=0.80, val_macro_f1=0.95, val_loss=0.05)
    assert not accuracy_improved(better_secondary_metrics, reference)
