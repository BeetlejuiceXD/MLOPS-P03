/**
 * D04-01 — Cliente de la API REST de MLflow (2.0) para el portal.
 *
 * Solo lectura: experimento por nombre, búsqueda de runs (todas las páginas), un run,
 * historial de una métrica y artefactos. Los artefactos van por el proxy del servidor
 * (`mlflow-artifacts:/…`, igual que el worker en Compose); nunca con URL prefirmada a MinIO.
 *
 * Sin reintentos y con timeout corto: si MLflow no responde, el error sale rápido y
 * explícito (`MlflowUnavailableError`) para que la API conteste 503 con el motivo.
 */

export class MlflowUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MlflowUnavailableError';
  }
}

export interface MlflowRunInfo {
  run_id: string;
  experiment_id: string;
  status: string;
  start_time: number;
  end_time?: number;
  artifact_uri: string;
  lifecycle_stage?: string;
}

export interface MlflowKeyValue {
  key: string;
  value: string;
}

export interface MlflowMetric {
  key: string;
  value: number;
  step: number;
  timestamp?: number;
}

export interface MlflowRun {
  info: MlflowRunInfo;
  data: {
    metrics?: MlflowMetric[];
    params?: MlflowKeyValue[];
    tags?: MlflowKeyValue[];
  };
}

export interface MlflowArtifact {
  /** Ruta relativa a la raíz de artefactos del run. */
  path: string;
  is_dir: boolean;
  file_size?: number;
}

export interface MlflowReader {
  getExperimentIdByName(name: string): Promise<string | null>;
  searchRuns(experimentId: string): Promise<MlflowRun[]>;
  getRun(runId: string): Promise<MlflowRun | null>;
  getMetricHistory(runId: string, key: string): Promise<MlflowMetric[]>;
  listArtifacts(run: MlflowRunInfo, dir?: string): Promise<MlflowArtifact[]>;
  downloadArtifact(run: MlflowRunInfo, path: string): Promise<Response>;
}

interface ClientOptions {
  baseUrl: string;
  timeoutMs?: number;
}

const PAGE_SIZE = 1000;
const HISTORY_PAGE_SIZE = 25000;
const PROXIED_PREFIX = 'mlflow-artifacts:/';

export function createMlflowRestClient({
  baseUrl,
  timeoutMs = 10_000,
}: ClientOptions): MlflowReader {
  const root = baseUrl.replace(/\/+$/, '');

  async function request(path: string, init?: RequestInit): Promise<Response> {
    try {
      return await fetch(`${root}${path}`, { ...init, signal: AbortSignal.timeout(timeoutMs) });
    } catch (error) {
      const cause = error instanceof Error ? (error.cause ?? error) : error;
      const detail = cause instanceof Error ? cause.message : String(cause);
      throw new MlflowUnavailableError(`no se pudo conectar con MLflow en ${root} (${detail})`);
    }
  }

  /** JSON de MLflow; `null` si el recurso no existe (404 RESOURCE_DOES_NOT_EXIST). */
  async function json<T>(path: string, init?: RequestInit): Promise<T | null> {
    const res = await request(path, init);
    const text = await res.text();
    let body: unknown;
    try {
      body = text ? JSON.parse(text) : {};
    } catch {
      throw new MlflowUnavailableError(`MLflow respondió ${res.status} sin JSON en ${path}`);
    }
    const code = (body as { error_code?: string }).error_code;
    if (res.status === 404 || code === 'RESOURCE_DOES_NOT_EXIST') return null;
    if (!res.ok) {
      const message = (body as { message?: string }).message ?? text.slice(0, 200);
      throw new MlflowUnavailableError(`MLflow respondió ${res.status} en ${path}: ${message}`);
    }
    return body as T;
  }

  /** `mlflow-artifacts:/1/<run>/artifacts` → `1/<run>/artifacts`; otro esquema no se sirve. */
  function proxiedRoot(run: MlflowRunInfo): string {
    if (!run.artifact_uri.startsWith(PROXIED_PREFIX)) {
      throw new MlflowUnavailableError(
        `el run ${run.run_id} guarda artefactos en ${run.artifact_uri}, fuera del proxy de MLflow`,
      );
    }
    return run.artifact_uri.slice(PROXIED_PREFIX.length).replace(/^\/+|\/+$/g, '');
  }

  return {
    async getExperimentIdByName(name) {
      const body = await json<{ experiment: { experiment_id: string } }>(
        `/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(name)}`,
      );
      return body?.experiment.experiment_id ?? null;
    },

    async searchRuns(experimentId) {
      const runs: MlflowRun[] = [];
      let pageToken: string | undefined;
      do {
        const body = await json<{ runs?: MlflowRun[]; next_page_token?: string }>(
          '/api/2.0/mlflow/runs/search',
          {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({
              experiment_ids: [experimentId],
              max_results: PAGE_SIZE,
              order_by: ['attributes.start_time DESC'],
              ...(pageToken ? { page_token: pageToken } : {}),
            }),
          },
        );
        runs.push(...(body?.runs ?? []));
        pageToken = body?.next_page_token || undefined;
      } while (pageToken);
      return runs;
    },

    async getRun(runId) {
      const body = await json<{ run: MlflowRun }>(
        `/api/2.0/mlflow/runs/get?run_id=${encodeURIComponent(runId)}`,
      );
      return body?.run ?? null;
    },

    async getMetricHistory(runId, key) {
      const metrics: MlflowMetric[] = [];
      let pageToken: string | undefined;
      do {
        const query = new URLSearchParams({
          run_id: runId,
          metric_key: key,
          max_results: String(HISTORY_PAGE_SIZE),
          ...(pageToken ? { page_token: pageToken } : {}),
        });
        const body = await json<{ metrics?: MlflowMetric[]; next_page_token?: string }>(
          `/api/2.0/mlflow/metrics/get-history?${query}`,
        );
        metrics.push(...(body?.metrics ?? []));
        pageToken = body?.next_page_token || undefined;
      } while (pageToken);
      return metrics;
    },

    async listArtifacts(run, dir = '') {
      const base = proxiedRoot(run);
      const target = dir ? `${base}/${dir}` : base;
      const body = await json<{ files?: { path: string; is_dir: boolean; file_size?: number }[] }>(
        `/api/2.0/mlflow-artifacts/artifacts?path=${encodeURIComponent(target)}`,
      );
      // El listado del proxy devuelve nombres base; se reconstruye la ruta relativa al run.
      return (body?.files ?? []).map((file) => {
        const name = file.path.split('/').pop() ?? file.path;
        return {
          path: dir ? `${dir}/${name}` : name,
          is_dir: file.is_dir,
          ...(file.file_size === undefined ? {} : { file_size: Number(file.file_size) }),
        };
      });
    },

    async downloadArtifact(run, path) {
      const encoded = path.split('/').map(encodeURIComponent).join('/');
      const res = await request(
        `/api/2.0/mlflow-artifacts/artifacts/${proxiedRoot(run)}/${encoded}`,
      );
      if (!res.ok) {
        await res.body?.cancel();
        throw new MlflowUnavailableError(`MLflow respondió ${res.status} al descargar ${path}`);
      }
      return res;
    },
  };
}
