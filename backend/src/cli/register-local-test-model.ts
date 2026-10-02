/**
 * D05-06 — Registra un paquete del modelo en el namespace `local_test` (MinIO del portal)
 * con el servicio de D04-06: register → upload → verify, y lo vuelve a leer por su
 * VersionId exacto. Imprime la evidencia en JSON. No hay opción para `official`: la
 * publicación en AWS es de D06-03.
 *
 * Dentro del contenedor del backend (el archivo se copia antes con `docker compose cp`):
 *
 *   node dist/cli/register-local-test-model.js --file /tmp/model.pt --semver 0.0.1 \
 *     --run-id <32 hex> --manifest-hash <sha256> --release v0.1.1 --release-hash <sha256>
 *
 * Sale con 0 solo si la versión queda `published` y la relectura coincide.
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../logic/model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
} from '../logic/model-registry.service.js';

const FLAGS = {
  '--file': 'file',
  '--semver': 'semver',
  '--run-id': 'mlflow_run_id',
  '--manifest-hash': 'manifest_hash',
  '--release': 'dvc_release',
  '--release-hash': 'dvc_release_hash',
} as const;

type Field = (typeof FLAGS)[keyof typeof FLAGS];
export type RegisterArgs = Record<Field, string>;
export type PackageIdentity = Omit<RegisterArgs, 'file'>;

export function parseRegisterArgs(argv: readonly string[]): RegisterArgs {
  const out: Partial<RegisterArgs> = {};
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i] as keyof typeof FLAGS;
    if (!(flag in FLAGS)) {
      throw new Error(`Argumento desconocido: ${argv[i]} (solo ${Object.keys(FLAGS).join(', ')})`);
    }
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) throw new Error(`Falta el valor de ${flag}`);
    out[FLAGS[flag]] = value;
  }
  const missing = Object.entries(FLAGS).filter(([, field]) => out[field] === undefined);
  if (missing.length > 0) {
    throw new Error(`Faltan argumentos: ${missing.map(([flag]) => flag).join(', ')}`);
  }
  return out as RegisterArgs;
}

/** register → upload → verify en `local_test` y relectura por VersionId exacto. */
export async function registerLocalTestPackage(
  deps: { repo: ModelRegistryRepository; store: ModelObjectStore; now?: () => Date },
  identity: PackageIdentity,
  body: Buffer,
) {
  const registry = createModelRegistryService({ ...deps, namespace: 'local_test' });
  await registry.register({ ...identity, sha256: sha256Hex(body), size_bytes: body.length });
  let outcome = await registry.upload(identity.semver, body);
  if (outcome.failure === null) outcome = await registry.verify(identity.semver);
  const { model, failure } = outcome;
  let readBack: string | null = null;
  if (model.status === 'published' && model.version_id !== null) {
    const stored = await deps.store.get(model.s3_key, model.version_id);
    readBack = stored === null ? null : sha256Hex(stored);
  }
  return {
    namespace: 'local_test' as const,
    ...model,
    size_bytes: body.length,
    failure_reason: failure?.reason ?? null,
    failure_detail: failure?.detail ?? null,
    read_back_sha256: readBack,
  };
}

async function main() {
  const args = parseRegisterArgs(process.argv.slice(2));
  // Importes perezosos: leen la configuración del entorno (MariaDB/MinIO) solo al ejecutar.
  const { env } = await import('../config/env.js');
  const { minioClient } = await import('../data/storage/minio.client.js');
  const { createMinioModelStore, ensureLocalModelBucket } = await import(
    '../data/storage/model-object.storage.js'
  );
  const { mariaDbModelRegistryRepository } = await import('../logic/model-registry.repository.js');
  const { pool } = await import('../data/db/client.js');

  const { file, ...identity } = args;
  const body = await readFile(file);
  await ensureLocalModelBucket(minioClient, env.MODEL_S3_BUCKET);
  try {
    const evidence = await registerLocalTestPackage(
      {
        repo: mariaDbModelRegistryRepository,
        store: createMinioModelStore(minioClient, env.MODEL_S3_BUCKET),
      },
      identity,
      body,
    );
    console.log(JSON.stringify({ file, ...evidence }, null, 2));
    process.exitCode =
      evidence.status === 'published' && evidence.read_back_sha256 === evidence.sha256 ? 0 : 1;
  } finally {
    await pool.end();
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
