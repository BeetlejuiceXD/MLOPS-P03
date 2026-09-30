"""D03-01 — lectura del manifest P3 congelado para el trainer (D03-03).

Solo abre `train_val.json`: el frozen test vive en otro archivo (`test.json`,
custodia de Ale, #33) que el trainer no necesita descargar ni leer. Antes de
devolver nada verifica que el registro de congelación diga `frozen: true` y que el
sha256 del archivo sea el registrado, para que el trainer nunca entrene sobre un
manifest distinto del auditado."""

import hashlib
import json
from pathlib import Path


class FrozenManifestError(ValueError):
    """`train_val.json` no es el manifest congelado y auditado."""


def load_train_val(train_val_path: Path, record_path: Path) -> dict:
    """Devuelve el payload de `train_val.json` (identidad + `splits.train`/`splits.val`)
    tras comprobarlo contra `reports/manifest_p3_freeze.json`."""
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if record.get("frozen") is not True:
        raise FrozenManifestError("el registro no corresponde a un manifest congelado")
    data = Path(train_val_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["train_val_sha256"]:
        raise FrozenManifestError("sha256 de train_val.json distinto del registrado")
    payload = json.loads(data)
    if payload["manifest_hash"] != record["manifest_hash"]:
        raise FrozenManifestError("manifest_hash de train_val.json distinto del registrado")
    if set(payload["splits"]) != {"train", "val"}:
        raise FrozenManifestError("train_val.json debe contener solo train y val")
    return payload
