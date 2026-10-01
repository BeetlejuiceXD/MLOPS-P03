import { createMlflowRestClient, type MlflowReader } from '../data/mlflow/mlflow-rest.client.js';

/** D04-01 — Lector de MLflow para la UI (la UI no importa de `data` directamente). */
export function createMlflowReader(trackingUri: string): MlflowReader {
  return createMlflowRestClient({ baseUrl: trackingUri, timeoutMs: 10_000 });
}
