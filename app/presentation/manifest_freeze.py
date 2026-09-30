"""D03-01 — auditoría y congelación del manifest P3 (stub Red)."""


class FreezeBlockedError(ValueError):
    def __init__(self, reason, detail, *, classes_below_minimum=None):
        super().__init__(detail)
        self.reason = reason
        self.classes_below_minimum = classes_below_minimum or []


def freeze_candidate(summary, assignments, **kwargs):
    raise NotImplementedError


def freeze_manifest(version, **kwargs):
    raise NotImplementedError


def write_frozen_manifest(frozen, **kwargs):
    raise NotImplementedError


def audit_frozen_on_disk(version, **kwargs):
    raise NotImplementedError
