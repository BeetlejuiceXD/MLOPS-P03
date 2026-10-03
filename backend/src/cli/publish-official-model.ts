/**
 * D06-03 — Publica el paquete FINAL (D06-02) y su tarjeta en el namespace `official`, en el
 * bucket de modelos de AWS, con el registro de D04-06 (register → upload → verify por
 * VersionId exacto) para los dos objetos. Antes de registrar nada comprueba la seguridad
 * del bucket. Imprime la evidencia en JSON (sin credenciales).
 *
 * Lo corre el principal OPERACIONAL (MLOPS-S3-MODELS) con sus credenciales SSO por la
 * cadena estándar del SDK (p. ej. `AWS_PROFILE`), nunca las de Terraform ni las de DVC, y
 * nunca escritas en el repo:
 *
 *   MODEL_S3_BUCKET=<terraform output bucket_name> AWS_REGION=us-east-1 \
 *   node dist/cli/publish-official-model.js --file model.pt --card model_card.json \
 *     --semver 1.0.0 --run-id <32 hex> --manifest-hash <sha256> --release v0.1.1 \
 *     --release-hash <sha256>
 *
 * Sale con 0 solo si modelo y tarjeta quedan `published` y su relectura por VersionId
 * coincide. No hay opción de namespace: las pruebas locales son de
 * `register-local-test-model`.
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { MODEL_CARD_OBJECT_NAME, sha256Hex } from '../logic/model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
} from '../logic/model-registry.service.js';
import {
  type BucketSecurityReport,
  createOfficialPublication,
} from '../logic/official-publication.service.js';

const FLAGS = {
  '--file': 'file',
  '--card': 'card',
  '--semver': 'semver',
  '--run-id': 'mlflow_run_id',
  '--manifest-hash': 'manifest_hash',
  '--release': 'dvc_release',
  '--release-hash': 'dvc_release_hash',
} as const;

type Field = (typeof FLAGS)[keyof typeof FLAGS];
export type PublishArgs = Record<Field, string>;

export function parsePublishArgs(argv: readonly string[]): PublishArgs {
  const out: Partial<PublishArgs> = {};
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
  return out as PublishArgs;
}

/** El bucket REAL de AWS (output `bucket_name` de terraform/p3-models), nunca el de MinIO. */
export function officialBucketFrom(env: { MODEL_S3_BUCKET?: string }): string {
  const bucket = env.MODEL_S3_BUCKET?.trim();
  if (!bucket) {
    throw new Error(
      'MODEL_S3_BUCKET no está definido: usa el nombre real del bucket (terraform output bucket_name)',
    );
  }
  if (bucket === 'p3-models-local') {
    throw new Error(
      'MODEL_S3_BUCKET=p3-models-local es el bucket de MinIO local: official publica en el bucket de AWS',
    );
  }
  return bucket;
}

export async function publishOfficialPackage(
  deps: {
    modelRepo: ModelRegistryRepository;
    cardRepo: ModelRegistryRepository;
    store: ModelObjectStore;
    inspectBucket: () => Promise<BucketSecurityReport>;
    now?: () => Date;
  },
  args: PublishArgs,
  files: { model: Buffer; card: Buffer },
) {
  const { file: _file, card: _card, ...identity } = args;
  const publication = createOfficialPublication({
    model: createModelRegistryService({
      repo: deps.modelRepo,
      store: deps.store,
      namespace: 'official',
      now: deps.now,
    }),
    card: createModelRegistryService({
      repo: deps.cardRepo,
      store: deps.store,
      namespace: 'official',
      objectName: MODEL_CARD_OBJECT_NAME,
      now: deps.now,
    }),
    store: deps.store,
    inspectBucket: deps.inspectBucket,
  });
  const result = await publication.publish({ identity, model: files.model, card: files.card });
  return {
    namespace: 'official' as const,
    bucket_security: result.bucket_security,
    model: { ...result.model.model, size_bytes: files.model.length },
    model_failure: result.model.failure,
    card: { ...result.card.model, size_bytes: files.card.length },
    card_failure: result.card.failure,
    read_back: result.read_back,
    published: result.published,
  };
}

async function main() {
  const args = parsePublishArgs(process.argv.slice(2));
  const bucket = officialBucketFrom(process.env);
  // Importes perezosos: leen el entorno (MariaDB, SDK de AWS) solo al ejecutar.
  const { S3Client } = await import('@aws-sdk/client-s3');
  const { createS3ModelStore, inspectModelBucket } = await import(
    '../data/storage/s3-model.storage.js'
  );
  const { mariaDbModelRegistryRepository } = await import('../logic/model-registry.repository.js');
  const { mariaDbModelCardRepository } = await import('../logic/model-card.repository.js');
  const { pool } = await import('../data/db/client.js');

  const client = new S3Client({ region: process.env.AWS_REGION ?? 'us-east-1' });
  const [model, card] = await Promise.all([readFile(args.file), readFile(args.card)]);
  try {
    const evidence = await publishOfficialPackage(
      {
        modelRepo: mariaDbModelRegistryRepository,
        cardRepo: mariaDbModelCardRepository,
        store: createS3ModelStore(client, bucket),
        inspectBucket: () => inspectModelBucket(client, bucket),
      },
      args,
      { model, card },
    );
    console.log(JSON.stringify({ file: args.file, card_file: args.card, ...evidence }, null, 2));
    process.exitCode =
      evidence.published &&
      evidence.read_back?.model_sha256 === sha256Hex(model) &&
      evidence.read_back?.card_sha256 === sha256Hex(card)
        ? 0
        : 1;
  } finally {
    await pool.end();
    client.destroy();
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
