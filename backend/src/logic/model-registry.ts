/**
 * D04-06 — Reglas puras del registro de modelos (contrato `model_version`, D01-05).
 *
 * - Cada versión vive en `models/p3-cnn-classifier/<semver>/model.pt` del bucket de
 *   modelos (`MODEL_S3_BUCKET`).
 * - Solo es `published` un objeto que existe en el storage con el VersionId registrado y
 *   cuyo SHA-256 (contenido y metadato) y tamaño coinciden con lo declarado al registrar.
 * - `namespace` separa las pruebas locales (`local_test`, MinIO) de la publicación oficial
 *   (`official`, AWS en D06-03); `GET /api/models` solo servirá `official`.
 */
import type { ModelVersion } from './p3.contracts.js';

export const MODEL_OBJECT_NAME = 'model.pt';

export type RegistryNamespace = 'official' | 'local_test';

/** Motivos estables por los que una versión queda `failed` (nunca publicable). */
export type RegistryFailureReason =
  | 'sha256_mismatch'
  | 'size_mismatch'
  | 'object_missing'
  | 'version_id_missing'
  | 'version_mismatch';

/** Fila del registro: el contrato `model_version` más tamaño, motivo de fallo y namespace. */
export interface RegistryEntry {
  namespace: RegistryNamespace;
  semver: string;
  mlflow_run_id: string;
  manifest_hash: string;
  dvc_release: string;
  dvc_release_hash: string;
  s3_bucket: string;
  s3_key: string;
  version_id: string | null;
  sha256: string;
  size_bytes: number;
  status: ModelVersion['status'];
  failure_reason: RegistryFailureReason | null;
  failure_detail: string | null;
  published_at: Date | null;
}

/** Clave del objeto de una versión: `models/p3-cnn-classifier/<semver>/model.pt`. */
export function modelObjectKey(_semver: string): string {
  throw new Error('D04-06: pendiente');
}

export function sha256Hex(_body: Buffer): string {
  throw new Error('D04-06: pendiente');
}

/** Orden ascendente por MAJOR.MINOR.PATCH numérico (no lexicográfico). */
export function compareSemver(_a: string, _b: string): number {
  throw new Error('D04-06: pendiente');
}

/** La fila del registro con la forma exacta del contrato (sin namespace ni extras). */
export function toModelVersion(_entry: RegistryEntry): ModelVersion {
  throw new Error('D04-06: pendiente');
}
