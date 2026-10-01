FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app

RUN useradd --system --no-create-home appuser

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=180
ENV UV_CONCURRENT_DOWNLOADS=2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --no-build

COPY ingestion/ ./ingestion/
COPY analyzers/ ./analyzers/
COPY policies/ ./policies/
COPY splits/ ./splits/
COPY storage/ ./storage/
COPY presentation/ ./presentation/
COPY crops/ ./crops/
COPY training/ ./training/
COPY projections/ ./projections/
COPY copilot/ ./copilot/
COPY tracking/ ./tracking/
COPY trainer_worker/ ./trainer_worker/
COPY trainer/ ./trainer/
COPY manifest/ ./manifest/
# D03-05/D04-05: motor de métricas y productor de la evaluación (lo usa D06-01).
COPY evaluation/ ./evaluation/

# El volumen nombrado reports_data (docker-compose.yml) se inicializa la
# primera vez con el contenido/dueño de este directorio si ya existe en la
# imagen — por eso se crea y se le da dueño ANTES de montar el volumen y de
# bajar a appuser, si no, la primera escritura de gate.py truena con
# PermissionError (el volumen nace root:root). Igual /app/.torch: caché de pesos
# preentrenados del trainer-worker (D03-03, volumen torch_cache).
RUN mkdir -p /app/reports /app/.torch && chown appuser:appuser /app/reports /app/.torch

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

CMD ["python", "-m", "presentation.main"]
