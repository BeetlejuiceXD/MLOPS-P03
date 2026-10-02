"""D05-04 — CLI del motor de inferencia.

    # Proceso limpio: carga el paquete, predice imágenes y escribe JSON en stdout:
    python -m inference_engine predict --package ./smoke-package --image gato.jpg [--image …] \\
        [--expected-sha256 <checkpoint_sha256>]
    # Servicio HTTP para D05-07 (INFERENCE_ENGINE_URL=http://<host>:8090):
    python -m inference_engine serve --package ./smoke-package --port 8090

Códigos de salida: 0 = ok · 1 = paquete rechazado · 3 = alguna imagen rechazada o
inferencia incoherente (las demás se reportan igual).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from inference_engine.engine import EngineError, InferenceEngine, PackageRejected

OK, PACKAGE_REJECTED, PREDICTION_REJECTED = 0, 1, 3


def _predict(args: argparse.Namespace) -> int:
    engine = InferenceEngine.from_package(
        args.package, expected_checkpoint_sha256=args.expected_sha256
    )
    results, code = [], OK
    for path in args.image:
        try:
            prediction = engine.predict(Path(path).read_bytes())
        except EngineError as error:
            results.append(
                {"input": str(path), "error": {"kind": error.kind, "message": str(error)}}
            )
            code = PREDICTION_REJECTED
            continue
        results.append(
            {
                "input": str(path),
                "input_sha256": prediction.input_sha256,
                "input_format": prediction.input_format,
                "input_size": list(prediction.input_size),
                **prediction.contract(),
            }
        )
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"engine": engine.identity(), "predictions": results}, indent=2))
    return code


def _serve(args: argparse.Namespace) -> int:
    from inference_engine.server import make_server

    engine = InferenceEngine.from_package(
        args.package, expected_checkpoint_sha256=args.expected_sha256
    )
    identity = engine.identity()["model"]
    server = make_server(engine, args.host, args.port)
    logging.getLogger("inference_engine").info(
        "motor en http://%s:%d · %s · run %s · sha %s",
        args.host,
        args.port,
        identity["package_id"],
        identity["mlflow_run_id"],
        identity["checkpoint_sha256"],
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return OK


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="D05-04 — motor de inferencia")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("predict", _predict), ("serve", _serve)):
        command = commands.add_parser(name)
        command.add_argument("--package", required=True, type=Path)
        command.add_argument("--expected-sha256")
        command.set_defaults(handler=handler)
        if name == "predict":
            command.add_argument("--image", required=True, action="append", type=Path)
        else:
            command.add_argument("--host", default="0.0.0.0")
            command.add_argument("--port", default=8090, type=int)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except PackageRejected as error:
        print(f"motor sin modelo: {error}", file=sys.stderr)
        return PACKAGE_REJECTED


if __name__ == "__main__":
    sys.exit(main())
