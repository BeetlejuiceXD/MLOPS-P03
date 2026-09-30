"""D03-01 — lectura del manifest P3 congelado para el trainer (solo train/val)."""


class FrozenManifestError(ValueError):
    pass


def load_train_val(train_val_path, record_path):
    raise NotImplementedError
