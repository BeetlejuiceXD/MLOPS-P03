/**
 * D06-03 — Bucket de modelos en AWS S3 con el SDK oficial (`@aws-sdk/client-s3`).
 *
 * - `createS3ModelStore`: misma interfaz que el store MinIO de D04-06. `put` devuelve el
 *   VersionId; `head`/`get` piden SIEMPRE la versión exacta (nunca el latest). Objeto o
 *   versión inexistente → `null`; cualquier otro error (AccessDenied, red) se propaga.
 * - `inspectModelBucket`: lee, sin cambiarla, la configuración de seguridad del bucket.
 *
 * Credenciales: la cadena estándar del SDK (perfil SSO del principal operacional,
 * MLOPS-S3-MODELS). Nunca en el repo ni en variables versionadas.
 */
import {
  GetBucketEncryptionCommand,
  GetBucketLifecycleConfigurationCommand,
  GetBucketPolicyCommand,
  GetBucketVersioningCommand,
  GetObjectCommand,
  GetPublicAccessBlockCommand,
  HeadObjectCommand,
  PutObjectCommand,
} from '@aws-sdk/client-s3';
import type { ModelObjectStore, StoredObjectHead } from '../../logic/model-registry.service.js';
import type {
  BucketSecurityCheck,
  BucketSecurityReport,
} from '../../logic/official-publication.service.js';

/** Lo único que se usa del `S3Client`: enviar comandos (inyectable en pruebas). */
export interface S3Sender {
  // biome-ignore lint/suspicious/noExplicitAny: cada comando del SDK tiene su propio tipo de salida.
  send(command: any): Promise<any>;
}

const MISSING = new Set(['NotFound', 'NoSuchKey', 'NoSuchVersion']);
const errorName = (error: unknown) => (error as { name?: string } | null)?.name ?? '';

export function createS3ModelStore(client: S3Sender, bucket: string): ModelObjectStore {
  return {
    bucket,
    async put(key, body, sha256) {
      const output = await client.send(
        new PutObjectCommand({
          Bucket: bucket,
          Key: key,
          Body: body,
          ContentType: 'application/octet-stream',
          Metadata: { sha256 },
        }),
      );
      return { versionId: output.VersionId ?? null };
    },
    async head(key, versionId): Promise<StoredObjectHead | null> {
      try {
        const output = await client.send(
          new HeadObjectCommand({ Bucket: bucket, Key: key, VersionId: versionId }),
        );
        return {
          size: output.ContentLength ?? 0,
          versionId: output.VersionId ?? null,
          sha256: output.Metadata?.sha256 ?? null,
        };
      } catch (error) {
        if (MISSING.has(errorName(error))) return null;
        throw error;
      }
    },
    async get(key, versionId) {
      try {
        const output = await client.send(
          new GetObjectCommand({ Bucket: bucket, Key: key, VersionId: versionId }),
        );
        return Buffer.from(await output.Body.transformToByteArray());
      } catch (error) {
        if (MISSING.has(errorName(error))) return null;
        throw error;
      }
    },
  };
}

type Probe = (client: S3Sender, bucket: string) => Promise<BucketSecurityCheck>;

/** Un error al leer una configuración: ausente (motivo propio) o permiso/red (falla igual). */
async function probe(
  name: string,
  read: () => Promise<{ ok: boolean; detail: string }>,
  missing: Record<string, { ok: boolean; detail: string }> = {},
): Promise<BucketSecurityCheck> {
  try {
    return { name, ...(await read()) };
  } catch (error) {
    const known = missing[errorName(error)];
    if (known) return { name, ...known };
    const detail = error instanceof Error ? error.message : String(error);
    return {
      name,
      ok: false,
      detail: `permiso insuficiente o error al leer la configuración (${errorName(error) || 'Error'}): ${detail}`,
    };
  }
}

const versioning: Probe = (client, bucket) =>
  probe('versioning', async () => {
    const { Status } = await client.send(new GetBucketVersioningCommand({ Bucket: bucket }));
    return Status === 'Enabled'
      ? { ok: true, detail: 'Enabled' }
      : { ok: false, detail: Status ? `versioning ${Status}` : 'sin versioning' };
  });

const encryption: Probe = (client, bucket) =>
  probe('encryption', async () => {
    const output = await client.send(new GetBucketEncryptionCommand({ Bucket: bucket }));
    const algorithms = (output.ServerSideEncryptionConfiguration?.Rules ?? []).map(
      (rule: { ApplyServerSideEncryptionByDefault?: { SSEAlgorithm?: string } }) =>
        rule.ApplyServerSideEncryptionByDefault?.SSEAlgorithm,
    );
    return algorithms.length > 0 && algorithms.every((algorithm: unknown) => algorithm === 'AES256')
      ? { ok: true, detail: 'AES256' }
      : { ok: false, detail: `cifrado ${algorithms.join(', ') || 'sin regla'}; se exige AES256` };
  });

const PUBLIC_ACCESS_FLAGS = [
  'BlockPublicAcls',
  'IgnorePublicAcls',
  'BlockPublicPolicy',
  'RestrictPublicBuckets',
] as const;

const publicAccessBlock: Probe = (client, bucket) =>
  probe(
    'public_access_block',
    async () => {
      const output = await client.send(new GetPublicAccessBlockCommand({ Bucket: bucket }));
      const config = output.PublicAccessBlockConfiguration ?? {};
      const off = PUBLIC_ACCESS_FLAGS.filter((flag) => config[flag] !== true);
      return off.length === 0
        ? { ok: true, detail: 'los 4 bloqueos de acceso público activos' }
        : {
            ok: false,
            detail: `bloqueo público incompleto: ${off.map((f) => `${f}=false`).join(', ')}`,
          };
    },
    {
      NoSuchPublicAccessBlockConfiguration: {
        ok: false,
        detail: 'sin bloqueo de acceso público configurado',
      },
    },
  );

interface PolicyStatement {
  Sid?: string;
  Effect?: string;
  Condition?: { Bool?: Record<string, unknown> };
}

const denyInsecureTransport: Probe = (client, bucket) =>
  probe(
    'deny_insecure_transport',
    async () => {
      const output = await client.send(new GetBucketPolicyCommand({ Bucket: bucket }));
      const statements: PolicyStatement[] = JSON.parse(output.Policy ?? '{}').Statement ?? [];
      const denies = statements.some(
        (statement) =>
          statement.Effect === 'Deny' &&
          String(statement.Condition?.Bool?.['aws:SecureTransport']) === 'false',
      );
      return denies
        ? { ok: true, detail: 'DenyInsecureTransport (aws:SecureTransport=false → Deny)' }
        : {
            ok: false,
            detail: 'la política no niega el acceso sin TLS (falta DenyInsecureTransport)',
          };
    },
    { NoSuchBucketPolicy: { ok: false, detail: 'sin política: falta DenyInsecureTransport' } },
  );

interface LifecycleRule {
  ID?: string;
  Status?: string;
  Expiration?: unknown;
  NoncurrentVersionExpiration?: unknown;
}

const lifecycle: Probe = (client, bucket) =>
  probe(
    'lifecycle',
    async () => {
      const output = await client.send(
        new GetBucketLifecycleConfigurationCommand({ Bucket: bucket }),
      );
      const expiring = ((output.Rules ?? []) as LifecycleRule[]).filter(
        (rule) =>
          rule.Status === 'Enabled' && (rule.Expiration || rule.NoncurrentVersionExpiration),
      );
      return expiring.length === 0
        ? { ok: true, detail: 'ninguna regla habilitada expira versiones' }
        : {
            ok: false,
            detail: `reglas que expiran versiones: ${expiring.map((rule) => rule.ID ?? '(sin ID)').join(', ')}`,
          };
    },
    { NoSuchLifecycleConfiguration: { ok: true, detail: 'sin reglas de lifecycle' } },
  );

/** Configuración de seguridad del bucket de modelos, leída sin cambiarla. */
export async function inspectModelBucket(
  client: S3Sender,
  bucket: string,
): Promise<BucketSecurityReport> {
  const checks = [];
  for (const read of [
    versioning,
    encryption,
    publicAccessBlock,
    denyInsecureTransport,
    lifecycle,
  ]) {
    checks.push(await read(client, bucket));
  }
  return { bucket, ok: checks.every((check) => check.ok), checks };
}
