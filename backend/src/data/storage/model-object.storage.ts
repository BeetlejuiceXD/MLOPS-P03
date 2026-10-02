import type * as Minio from 'minio';

/**
 * D04-06 — Adaptador S3-compatible del bucket de modelos (`MODEL_S3_BUCKET`).
 *
 * Sube con el metadato `x-amz-meta-sha256` y lee por VersionId. Funciona igual contra
 * MinIO (local/CI) y AWS S3; este ticket solo lo prueba con MinIO. `head`/`get`
 * devuelven `null` únicamente si el objeto o la versión no existen: un error de red o
 * de credenciales se propaga y nunca se interpreta como "objeto ausente".
 */
export const SHA256_METADATA_KEY = 'sha256';

export interface ModelObjectHead {
  size: number;
  versionId: string | null;
  sha256: string | null;
}

export interface MinioModelStore {
  readonly bucket: string;
  put(key: string, body: Buffer, sha256: string): Promise<{ versionId: string | null }>;
  head(key: string, versionId: string): Promise<ModelObjectHead | null>;
  get(key: string, versionId: string): Promise<Buffer | null>;
}

export function createMinioModelStore(_client: Minio.Client, _bucket: string): MinioModelStore {
  throw new Error('D04-06: pendiente');
}

/** Solo local/CI: crea el bucket si falta y activa versioning (en AWS lo hace Terraform). */
export async function ensureLocalModelBucket(_client: Minio.Client, _bucket: string) {
  throw new Error('D04-06: pendiente');
}

/** Sin versioning no hay VersionId y nada puede publicarse: se rechaza el bucket. */
export async function assertModelBucketVersioned(_client: Minio.Client, _bucket: string) {
  throw new Error('D04-06: pendiente');
}
