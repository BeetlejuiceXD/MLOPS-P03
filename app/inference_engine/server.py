"""D05-04 — Adaptador HTTP del motor, con el contrato que consume D05-07
(`backend/src/logic/inference-engine.ts`, `INFERENCE_ENGINE_URL`):

    GET  /identity → 200 inference_engine · 503 {error} sin paquete
    POST /predict  → cuerpo = bytes de la imagen, Content-Type image/*
                     200 inference_engine_prediction
                     400/413/415 {error} imagen rechazada
                     500 {error} salida incoherente del modelo · 503 {error} sin paquete
    GET  /health   → 200 {status: ok, model} · 503 {error}

Solo transporte: toda la validación vive en `InferenceEngine`. Biblioteca estándar, sin
dependencias nuevas.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inference_engine.engine import (
    MAX_IMAGE_BYTES,
    EngineError,
    InferenceEngine,
    InferenceFailed,
    InputRejected,
    PackageRejected,
)

log = logging.getLogger("inference_engine")

STATUS_BY_ERROR: dict[type[EngineError], HTTPStatus] = {
    InputRejected: HTTPStatus.BAD_REQUEST,
    PackageRejected: HTTPStatus.SERVICE_UNAVAILABLE,
    InferenceFailed: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def make_handler(engine: InferenceEngine, *, max_bytes: int = MAX_IMAGE_BYTES):
    class Handler(BaseHTTPRequestHandler):
        server_version = "p3-inference-engine"

        def _send(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, error: EngineError) -> None:
            status = STATUS_BY_ERROR.get(type(error), HTTPStatus.INTERNAL_SERVER_ERROR)
            log.warning("%s %s → %d %s: %s", self.command, self.path, status, error.kind, error)
            self._send(status, {"error": str(error)})

        def do_GET(self) -> None:
            try:
                if self.path == "/identity":
                    self._send(HTTPStatus.OK, engine.identity())
                elif self.path == "/health":
                    self._send(HTTPStatus.OK, {"status": "ok", **engine.identity()})
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": f"ruta {self.path} no existe"})
            except EngineError as error:
                self._fail(error)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(HTTPStatus.NOT_FOUND, {"error": f"ruta {self.path} no existe"})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                self._send(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": f"Content-Type {content_type or '(vacío)'}: se espera image/*"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "Content-Length inválido"})
                return
            if length > max_bytes:
                self._send(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": f"imagen de {length} B: el máximo es {max_bytes} B"},
                )
                return
            data = self.rfile.read(length)
            try:
                prediction = engine.predict(data)
            except EngineError as error:
                self._fail(error)
                return
            log.info(
                "predict %s (%s %dx%d) → %s %s run %s",
                prediction.input_sha256[:12],
                prediction.input_format,
                *prediction.input_size,
                prediction.predicted_class,
                prediction.probabilities,
                prediction.model["mlflow_run_id"],
            )
            self._send(HTTPStatus.OK, prediction.contract())

        def log_message(self, format: str, *args) -> None:
            log.debug(format, *args)

    return Handler


def make_server(
    engine: InferenceEngine, host: str, port: int, *, max_bytes: int = MAX_IMAGE_BYTES
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(engine, max_bytes=max_bytes))
