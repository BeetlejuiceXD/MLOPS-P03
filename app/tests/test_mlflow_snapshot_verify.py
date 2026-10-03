"""#104 — scripts/mlflow-snapshot/verify.py comprueba el MLflow restaurado REAL.

El fake de abajo responde como la API REST de MLflow (runs/search paginado, get-history,
artifacts/list) y como GET /selection. Se construye a partir del contrato versionado,
y cada test negativo le quita, agrega o altera un dato REAL para comprobar que verify
falla (exit 1) en vez de tomar el dato del contrato.
"""

from __future__ import annotations

import copy
import hashlib
import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mlflow_snapshot_verify", REPO / "scripts/mlflow-snapshot/verify.py"
)
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)

EVIDENCE = json.loads((REPO / "reports/campaign_p3.json").read_text(encoding="utf-8"))
CANDIDATE = verify.CANDIDATE_RUN


class FakeMlflow:
    """MLflow + API restaurados en memoria, coherentes con el contrato."""

    def __init__(self, evidence: dict, page_size: int = 5):
        self.page_size = page_size
        self.runs: dict[str, dict] = {}
        self.jobs: dict[int, dict] = {}
        self.history: dict[tuple[str, str], list[dict]] = {}
        self.artifacts: dict[str, set[str]] = {}
        self.candidate_digest = (verify.CANDIDATE_CHECKPOINT_SHA256, 45_000_000)
        self.selection = {
            "status": "candidate",
            "closed_at": None,
            "candidate": {"run_id": CANDIDATE, "campaign_row": 3},
        }
        for result in evidence["results"]:
            report, details, job = result["report"], result["details"], result["job"]
            run_id = report["run_id"]
            best = report["best_epoch"]
            summary = {
                "val_accuracy": report["best_val_accuracy"],
                "val_macro_f1": report["best_val_macro_f1"],
                "val_loss": details["best_val_loss"],
            }
            for key in verify.HISTORY_METRICS:
                self.history[(run_id, key)] = [
                    {
                        "key": key,
                        "step": step,
                        "value": summary[key] if step == best and key in summary else 0.5,
                    }
                    for step in range(1, report["epochs_logged"] + 1)
                ]
            metrics = {
                "best_epoch": float(best),
                "best_val_accuracy": report["best_val_accuracy"],
                "best_val_macro_f1": report["best_val_macro_f1"],
                "best_val_loss": details["best_val_loss"],
                "val_accuracy": 0.5,
                "train_loss": 0.1,
            }
            self.runs[run_id] = {
                "info": {
                    "run_id": run_id,
                    "status": report["run_status"],
                    "lifecycle_stage": "active",
                    "start_time": details["start_time"],
                },
                "data": {
                    "metrics": [{"key": k, "value": v} for k, v in metrics.items()],
                    "tags": [
                        {"key": "job_id", "value": str(job["id"])},
                        {"key": "checkpoint_sha256", "value": report["checkpoint_sha256"]},
                    ],
                },
            }
            self.artifacts[run_id] = {"checkpoint/model.pt"}
            self.jobs[job["id"]] = {"id": job["id"], "mlflow_run_id": run_id}

    # --- interfaz que usa verify.verify ------------------------------------------
    def mlflow_get(self, path, params):
        if path.endswith("/experiments/get-by-name"):
            assert params == {"experiment_name": "p3-cnn-classifier"}
            return {"experiment": {"experiment_id": "1"}}
        if path.endswith("/metrics/get-history"):
            points = self.history.get((params["run_id"], params["metric_key"]), [])
            start = int(params.get("page_token") or 0)
            page = points[start : start + 4]
            out = {"metrics": page}
            if start + 4 < len(points):
                out["next_page_token"] = str(start + 4)
            return out
        if path.endswith("/artifacts/list"):
            files = self.artifacts.get(params["run_id"])
            if files is None:
                raise OSError("run sin artefactos")
            return {"files": [{"path": p, "is_dir": False} for p in sorted(files)]}
        raise AssertionError(path)

    def mlflow_post(self, path, body):
        assert path.endswith("/runs/search")
        assert body["run_view_type"] == "ALL"
        ordered = list(self.runs.values())
        start = int(body.get("page_token") or 0)
        out = {"runs": ordered[start : start + self.page_size]}
        if start + self.page_size < len(ordered):
            out["next_page_token"] = str(start + self.page_size)
        return out

    def mlflow_sha256(self, run_id, path):
        assert (run_id, path) == (CANDIDATE, "checkpoint/model.pt")
        if path not in self.artifacts.get(run_id, set()):
            raise OSError("404 artefacto")
        return self.candidate_digest

    def api_get(self, path):
        if path == "/training/jobs":
            return {"jobs": list(self.jobs.values())}
        assert path == "/selection"
        return self.selection


def run(fake: FakeMlflow, evidence: dict = EVIDENCE) -> tuple[list[str], list[str]]:
    return verify.verify(fake, evidence)


def run_of(fake: FakeMlflow, row: int, job: int) -> str:
    result = next(r for r in EVIDENCE["results"] if r["row"] == row and r["job"]["id"] == job)
    return result["report"]["run_id"]


ROW1_RETRY = run_of(FakeMlflow(EVIDENCE), 1, 2)


def test_fixture_completo_de_16_runs_pasa_y_pagina():
    fake = FakeMlflow(EVIDENCE, page_size=5)  # 16 runs → 4 páginas
    lines, problems = run(fake)
    assert problems == []
    assert any("16/16 esperados, 0 inesperados" in line for line in lines)
    assert any(
        line.startswith(f"ROW 1: representante={verify.ROW1_REPRESENTATIVE_RUN} reintentos=4")
        for line in lines
    )


def test_main_devuelve_exit_0_y_1(capsys):
    assert verify.main([], http=FakeMlflow(EVIDENCE)) == verify.VERIFIED
    assert "VERIFY: PASS" in capsys.readouterr().out
    fake = FakeMlflow(EVIDENCE)
    fake.selection = {**fake.selection, "status": "closed"}
    assert verify.main([], http=fake) == verify.FAILED
    assert "VERIFY: FAIL" in capsys.readouterr().out


def test_solo_12_representantes_sin_reintentos_falla():
    fake = FakeMlflow(EVIDENCE)
    for result in EVIDENCE["results"]:
        if result["row"] == 1 and result["job"]["id"] != 1:
            del fake.runs[result["report"]["run_id"]]
    _, problems = run(fake)
    assert any("faltan 4 intentos" in p for p in problems)
    assert any("fila 1: 1 intentos reales" in p for p in problems)


def test_un_reintento_faltante_falla():
    fake = FakeMlflow(EVIDENCE)
    del fake.runs[ROW1_RETRY]
    _, problems = run(fake)
    assert any(ROW1_RETRY in p and "faltan 1" in p for p in problems)


def test_reintento_extra_inesperado_falla():
    fake = FakeMlflow(EVIDENCE)
    extra = copy.deepcopy(fake.runs[ROW1_RETRY])
    extra["info"]["run_id"] = "f" * 32
    fake.runs["f" * 32] = extra
    _, problems = run(fake)
    assert any("runs inesperados" in p and "f" * 32 in p for p in problems)


def test_intento_inesperado_que_sustituye_a_uno_esperado_falla():
    fake = FakeMlflow(EVIDENCE)
    impostor = fake.runs.pop(ROW1_RETRY)
    impostor["info"]["run_id"] = "e" * 32
    fake.runs["e" * 32] = impostor
    _, problems = run(fake)
    assert any("faltan 1" in p for p in problems)
    assert any("inesperados" in p for p in problems)


def test_run_no_finished_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.runs[ROW1_RETRY]["info"]["status"] = "FAILED"
    _, problems = run(fake)
    assert any(ROW1_RETRY in p and "status=FAILED" in p for p in problems)


def test_run_borrado_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.runs[ROW1_RETRY]["info"]["lifecycle_stage"] = "deleted"
    _, problems = run(fake)
    assert any("lifecycle_stage=deleted" in p for p in problems)


def test_metrica_test_en_un_reintento_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.runs[ROW1_RETRY]["data"]["metrics"].append({"key": "test_accuracy", "value": 0.9})
    _, problems = run(fake)
    assert any(ROW1_RETRY in p and "test_accuracy" in p for p in problems)


def test_metrica_de_validation_ausente_falla():
    fake = FakeMlflow(EVIDENCE)
    metrics = fake.runs[CANDIDATE]["data"]["metrics"]
    fake.runs[CANDIDATE]["data"]["metrics"] = [m for m in metrics if m["key"] != "best_val_loss"]
    _, problems = run(fake)
    assert any("falta la métrica best_val_loss" in p for p in problems)


def test_metrica_de_validation_distinta_del_contrato_falla():
    fake = FakeMlflow(EVIDENCE)
    for metric in fake.runs[CANDIDATE]["data"]["metrics"]:
        if metric["key"] == "best_val_accuracy":
            metric["value"] += 0.01
    _, problems = run(fake)
    assert any("best_val_accuracy=" in p and "contrato" in p for p in problems)


def test_historial_incompleto_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.history[(CANDIDATE, "val_loss")].pop()
    _, problems = run(fake)
    assert any("historial de val_loss" in p for p in problems)


def test_historial_sin_el_valor_de_best_epoch_falla():
    fake = FakeMlflow(EVIDENCE)
    for point in fake.history[(CANDIDATE, "val_macro_f1")]:
        point["value"] = 0.1
    _, problems = run(fake)
    assert any("val_macro_f1[5]" in p for p in problems)


def test_historial_de_metrica_ausente_falla():
    fake = FakeMlflow(EVIDENCE)
    del fake.history[(ROW1_RETRY, "train_accuracy")]
    _, problems = run(fake)
    assert any(ROW1_RETRY in p and "historial de train_accuracy" in p for p in problems)


def test_artifact_ausente_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.artifacts[ROW1_RETRY] = set()
    _, problems = run(fake)
    assert any(ROW1_RETRY in p and "falta el artefacto" in p for p in problems)


def test_checkpoint_del_candidato_ausente_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.artifacts[CANDIDATE] = set()
    _, problems = run(fake)
    assert any("no se pudo descargar" in p for p in problems)


def test_checkpoint_corrupto_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.candidate_digest = (hashlib.sha256(b"corrupto").hexdigest(), 8)
    _, problems = run(fake)
    assert any("SHA-256 de los bytes" in p for p in problems)


def test_tag_de_checkpoint_correcto_no_basta():
    """El tag dice el SHA correcto pero los bytes no: debe fallar."""
    fake = FakeMlflow(EVIDENCE)
    fake.candidate_digest = (hashlib.sha256(b"otro").hexdigest(), 4)
    tags = {t["key"]: t["value"] for t in fake.runs[CANDIDATE]["data"]["tags"]}
    assert tags["checkpoint_sha256"] == verify.CANDIDATE_CHECKPOINT_SHA256
    _, problems = run(fake)
    assert problems


@pytest.mark.parametrize(
    "selection",
    [
        {"status": "closed", "closed_at": "2026-10-02T00:00:00Z"},
        {"status": "candidate", "closed_at": "2026-10-02T00:00:00Z"},
        {"status": "candidate", "closed_at": None, "candidate": {"run_id": "x", "campaign_row": 3}},
        {
            "status": "candidate",
            "closed_at": None,
            "candidate": {"run_id": CANDIDATE, "campaign_row": 2},
        },
        {"status": "open", "closed_at": None, "candidate": None},
        {"status": "candidate", "candidate": {"run_id": CANDIDATE, "campaign_row": 3}},
    ],
)
def test_seleccion_incorrecta_falla(selection):
    fake = FakeMlflow(EVIDENCE)
    fake.selection = {
        "candidate": {"run_id": CANDIDATE, "campaign_row": 3},
        **selection,
    }
    _, problems = run(fake)
    assert any(p.startswith("selection.") for p in problems)


def test_representante_de_fila_1_no_es_el_mas_temprano_falla():
    fake = FakeMlflow(EVIDENCE)
    fake.runs[ROW1_RETRY]["info"]["start_time"] = 1
    _, problems = run(fake)
    assert any("el intento más temprano real" in p for p in problems)


def test_tag_job_id_equivocado_falla():
    fake = FakeMlflow(EVIDENCE)
    for tag in fake.runs[CANDIDATE]["data"]["tags"]:
        if tag["key"] == "job_id":
            tag["value"] = "8"
    _, problems = run(fake)
    assert any("tag job_id='8'" in p for p in problems)


def test_contrato_sin_reintentos_se_rechaza():
    evidence = copy.deepcopy(EVIDENCE)
    evidence["results"] = [
        r for r in evidence["results"] if not (r["row"] == 1 and r["job"]["id"] != 1)
    ]
    _, problems = run(FakeMlflow(EVIDENCE), evidence)
    assert problems and problems[0].startswith("contrato reports/campaign_p3.json inválido")


def test_mlflow_caido_falla():
    class Down(FakeMlflow):
        def mlflow_get(self, path, params):
            raise OSError("connection refused")

    _, problems = run(Down(EVIDENCE))
    assert any("no se pudo consultar el MLflow" in p for p in problems)


def test_la_descarga_real_calcula_el_sha_de_los_bytes():
    """Http.mlflow_sha256 lee los BYTES de /get-artifact por bloques, no un tag."""
    payload = b"\x00checkpoint" * 300_000
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = verify.Http(f"http://127.0.0.1:{server.server_port}", "http://unused")
        digest, size = client.mlflow_sha256(CANDIDATE, "checkpoint/model.pt")
    finally:
        server.shutdown()
    assert (digest, size) == (hashlib.sha256(payload).hexdigest(), len(payload))
    assert seen["path"] == f"/get-artifact?path=checkpoint%2Fmodel.pt&run_uuid={CANDIDATE}"
