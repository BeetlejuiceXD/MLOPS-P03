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

/** Códigos con los que S3/MinIO dicen que el objeto o la versión no existen. */
const MISSING = new Set(['NotFound', 'NoSuchKey', 'NoSuchVersion']);

function isMissing(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code;
  return typeof code === 'string' && MISSING.has(code);
}

async function orNull<T>(operation: () => Promise<T>): Promise<T | null> {
  try {
    return await operation();
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
}

export function createMinioModelStore(client: Minio.Client, bucket: string): MinioModelStore {
  return {
    bucket,
    async put(key, body, sha256) {
      const info = await client.putObject(bucket, key, body, body.length, {
        'Content-Type': 'application/octet-stream',
        'X-Amz-Meta-Sha256': sha256,
      });
      return { versionId: info.versionId ?? null };
    },
    head: (key, versionId) =>
      orNull(async () => {
        const stat = await client.statObject(bucket, key, { versionId });
        const sha256 = stat.metaData?.[SHA256_METADATA_KEY];
        return {
          size: stat.size,
          versionId: stat.versionId ?? null,
          sha256: typeof sha256 === 'string' ? sha256 : null,
        };
      }),
    get: (key, versionId) =>
      orNull(async () => {
        const stream = await client.getObject(bucket, key, { versionId });
        const chunks: Buffer[] = [];
        for await (const chunk of stream) chunks.push(Buffer.from(chunk));
        return Buffer.concat(chunks);
      }),
  };
}

/** Solo local/CI: crea el bucket si falta y activa versioning (en AWS lo hace Terraform). */
export async function ensureLocalModelBucket(client: Minio.Client, bucket: string) {
  if (!(await client.bucketExists(bucket))) await client.makeBucket(bucket);
  await client.setBucketVersioning(bucket, { Status: 'Enabled' });
  await assertModelBucketVersioned(client, bucket);
}

/** Sin versioning no hay VersionId y nada puede publicarse: se rechaza el bucket. */
export async function assertModelBucketVersioned(client: Minio.Client, bucket: string) {
  const config = await client.getBucketVersioning(bucket);
  if (config?.Status !== 'Enabled') {
    throw new Error(`El bucket de modelos ${bucket} no tiene versioning activo`);
  }
}
