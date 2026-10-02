/**
 * D04-06 — Roundtrip REAL del registro de modelos: MinIO (bucket con versioning) +
 * MariaDB (`p3_model_registry`, migración 0008), con reinicio de ambos servicios entre
 * las dos fases. Solo corre con `P3_MODEL_REGISTRY_TEST=1` (job de CI "Jobs
 * persistentes"), contra servicios desechables:
 *
 *   P3_MODEL_REGISTRY_PHASE=write  registra, sube, verifica y provoca las negativas;
 *                                  guarda IDs y hashes en P3_MODEL_REGISTRY_EVIDENCE.
 *   (docker compose restart mariadb minio)
 *   P3_MODEL_REGISTRY_PHASE=check  otro proceso relee registro y objetos por VersionId.
 *
 * Objetos de PRUEBA en el namespace `local_test`, credenciales de MinIO local (nunca de
 * AWS). No acredita ningún modelo final publicado en AWS (D06-03).
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import * as Minio from 'minio';
import { describe, expect, it } from 'vitest';

const enabled = process.env.P3_MODEL_REGISTRY_TEST === '1';
const phase = process.env.P3_MODEL_REGISTRY_PHASE;
const REQUIRED = [
  'DATABASE_URL',
  'MINIO_ENDPOINT',
  'MINIO_PORT',
  'MINIO_ACCESS_KEY',
  'MINIO_SECRET_KEY',
  'MODEL_S3_BUCKET',
  'P3_MODEL_REGISTRY_EVIDENCE',
];
// En CI, saltarse en silencio por falta de configuración pasaría el paso sin probar nada.
if (enabled) {
  const missing = REQUIRED.filter((name) => !process.env[name]);
  if (missing.length > 0 || (phase !== 'write' && phase !== 'check')) {
    throw new Error(
      `P3_MODEL_REGISTRY_TEST=1 exige P3_MODEL_REGISTRY_PHASE=write|check y ${missing.join(', ')}`,
    );
  }
}

const env = (name: string) => process.env[name] as string;
const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');

// Contenido de prueba determinista (no es un checkpoint).
const BODY = Buffer.from(`D04-06 roundtrip local_test ${'0123456789abcdef'.repeat(64)}\n`);
const OTHER = Buffer.from('D04-06 contenido distinto con el mismo metadato\n');

const identity = (semver: string, body: Buffer = BODY) => ({
  semver,
  mlflow_run_id: 'a'.repeat(32),
  manifest_hash: 'd'.repeat(64),
  dvc_release: 'v0.1.1',
  dvc_release_hash: 'e'.repeat(64),
  sha256: sha(body),
  size_bytes: body.length,
});

interface Evidence {
  bucket: string;
  published: { semver: string; key: string; version_id: string; sha256: string; size: number };
  overwrite_version_id: string;
  failed: Record<string, string>;
}

async function setup() {
  const client = new Minio.Client({
    endPoint: env('MINIO_ENDPOINT'),
    port: Number(env('MINIO_PORT')),
    useSSL: false,
    accessKey: env('MINIO_ACCESS_KEY'),
    secretKey: env('MINIO_SECRET_KEY'),
  });
  const storage = await import('../src/data/storage/model-object.storage.js');
  const { mariaDbModelRegistryRepository: repo } = await import(
    '../src/logic/model-registry.repository.js'
  );
  const { createModelRegistryService } = await import('../src/logic/model-registry.service.js');
  const { pool } = await import('../src/data/db/client.js');
  const bucket = env('MODEL_S3_BUCKET');
  const store = storage.createMinioModelStore(client, bucket);
  const service = createModelRegistryService({ repo, store, namespace: 'local_test' });
  return { client, storage, repo, store, service, pool, bucket };
}

describe.skipIf(!enabled || phase !== 'write')('registro de modelos real — escritura', () => {
  it('roundtrip upload/head/get con VersionId y SHA-256; negativas nunca published', async () => {
    const { client, storage, repo, store, service, pool, bucket } = await setup();
    const { ConflictError } = await import('../src/logic/errors.js');
    const { modelsResponseSchema } = await import('../src/logic/p3.contracts.js');
    try {
      // Base desechable: solo se limpia el namespace de prueba.
      await pool.query("DELETE FROM p3_model_registry WHERE namespace = 'local_test'");
      await storage.ensureLocalModelBucket(client, bucket);
      await storage.assertModelBucketVersioned(client, bucket);

      // 1. Positivo: draft → upload → verify → published.
      const draft = await service.register(identity('0.0.1'));
      expect(draft).toMatchObject({ status: 'draft', version_id: null, s3_bucket: bucket });
      await expect(service.register(identity('0.0.1'))).rejects.toBeInstanceOf(ConflictError);

      const uploaded = await service.upload('0.0.1', BODY);
      const versionId = uploaded.model.version_id as string;
      expect(versionId).toEqual(expect.any(String));
      expect(versionId.length).toBeGreaterThan(0);
      expect(versionId).not.toBe('null');

      // head/get directos con el cliente: lo que hay en MinIO es lo registrado.
      const stat = await client.statObject(bucket, uploaded.model.s3_key, { versionId });
      expect(stat.versionId).toBe(versionId);
      expect(stat.size).toBe(BODY.length);
      expect(stat.metaData.sha256).toBe(sha(BODY));
      expect(sha((await store.get(uploaded.model.s3_key, versionId)) as Buffer)).toBe(sha(BODY));

      const verified = await service.verify('0.0.1');
      expect(verified.failure).toBeNull();
      expect(verified.model).toMatchObject({ status: 'published', version_id: versionId });
      expect(verified.model.published_at).toEqual(expect.any(String));

      // Sobrescribir la misma clave crea OTRA versión; la registrada sigue íntegra.
      const overwrite = await client.putObject(bucket, uploaded.model.s3_key, OTHER, OTHER.length);
      expect(overwrite.versionId).toBeTruthy();
      expect(overwrite.versionId).not.toBe(versionId);
      expect(await service.audit('0.0.1')).toEqual({ ok: true });

      // 2. Hash incorrecto al subir (mismo tamaño, un bit distinto): failed y ningún
      //    objeto en su clave.
      const flipped = Buffer.from(BODY);
      flipped[1] = flipped[1] ^ 1;
      await service.register(identity('0.0.2'));
      const wrong = await service.upload('0.0.2', flipped);
      expect(wrong.failure?.reason).toBe('sha256_mismatch');
      expect(wrong.model.status).toBe('failed');
      await expect(client.statObject(bucket, wrong.model.s3_key)).rejects.toMatchObject({
        code: 'NotFound',
      });

      // 3. Objeto ausente: la versión subida se borra antes de verificar.
      await service.register(identity('0.0.3'));
      const gone = await service.upload('0.0.3', BODY);
      await client.removeObject(bucket, gone.model.s3_key, {
        versionId: gone.model.version_id as string,
      });
      expect(await store.head(gone.model.s3_key, gone.model.version_id as string)).toBeNull();
      expect(await store.get(gone.model.s3_key, gone.model.version_id as string)).toBeNull();
      const missing = await service.verify('0.0.3');
      expect(missing.failure?.reason).toBe('object_missing');
      expect(missing.model.status).toBe('failed');

      // 4. Contenido con otro hash bajo el VersionId registrado (metadato "correcto",
      //    bytes distintos, mismo tamaño): solo el SHA-256 del contenido lo detecta.
      const corrupt = Buffer.from(BODY);
      corrupt[0] = corrupt[0] ^ 1;
      await service.register(identity('0.0.4'));
      const forged = await store.put(storageKey('0.0.4'), corrupt, sha(BODY));
      expect(await repo.markUploaded('local_test', '0.0.4', forged.versionId as string)).toBe(true);
      const tampered = await service.verify('0.0.4');
      expect(tampered.failure?.reason).toBe('sha256_mismatch');
      expect(tampered.model.status).toBe('failed');

      // 5. Un VersionId que no existe en el bucket: objeto ausente.
      await service.register(identity('0.0.5'));
      const real = await store.put(storageKey('0.0.5'), BODY, sha(BODY));
      expect(real.versionId).toBeTruthy();
      await client.removeObject(bucket, storageKey('0.0.5'), {
        versionId: real.versionId as string,
      });
      expect(await repo.markUploaded('local_test', '0.0.5', real.versionId as string)).toBe(true);
      expect((await service.verify('0.0.5')).failure?.reason).toBe('object_missing');

      // Ninguna negativa quedó published; published y failed no se reescriben.
      await expect(service.verify('0.0.1')).rejects.toBeInstanceOf(ConflictError);
      await expect(service.upload('0.0.2', BODY)).rejects.toBeInstanceOf(ConflictError);
      const listed = await service.list();
      expect(modelsResponseSchema.safeParse(listed).success).toBe(true);
      expect(listed.models.map((m) => [m.semver, m.status])).toEqual([
        ['0.0.1', 'published'],
        ['0.0.2', 'failed'],
        ['0.0.3', 'failed'],
        ['0.0.4', 'failed'],
        ['0.0.5', 'failed'],
      ]);

      const failed: Record<string, string> = {};
      for (const semver of ['0.0.2', '0.0.3', '0.0.4', '0.0.5']) {
        failed[semver] = (await repo.find('local_test', semver))?.failure_reason as string;
      }
      const evidence: Evidence = {
        bucket,
        published: {
          semver: '0.0.1',
          key: uploaded.model.s3_key,
          version_id: versionId,
          sha256: sha(BODY),
          size: BODY.length,
        },
        overwrite_version_id: overwrite.versionId as string,
        failed,
      };
      fs.writeFileSync(env('P3_MODEL_REGISTRY_EVIDENCE'), JSON.stringify(evidence, null, 2));
    } finally {
      await pool.end();
    }
  });
});

describe.skipIf(!enabled || phase !== 'check')('registro de modelos real — tras reinicio', () => {
  it('otro proceso relee el registro y el objeto por VersionId sin cambios', async () => {
    const evidence = JSON.parse(
      fs.readFileSync(env('P3_MODEL_REGISTRY_EVIDENCE'), 'utf8'),
    ) as Evidence;
    const { client, storage, repo, store, service, pool, bucket } = await setup();
    try {
      expect(bucket).toBe(evidence.bucket);
      await storage.assertModelBucketVersioned(client, bucket);

      const entry = await repo.find('local_test', evidence.published.semver);
      expect(entry).toMatchObject({
        status: 'published',
        version_id: evidence.published.version_id,
        sha256: evidence.published.sha256,
        size_bytes: evidence.published.size,
        s3_key: evidence.published.key,
      });
      expect(entry?.published_at).toBeInstanceOf(Date);

      const head = await store.head(evidence.published.key, evidence.published.version_id);
      expect(head).toEqual({
        size: evidence.published.size,
        versionId: evidence.published.version_id,
        sha256: evidence.published.sha256,
      });
      const body = await store.get(evidence.published.key, evidence.published.version_id);
      expect(sha(body as Buffer)).toBe(evidence.published.sha256);
      expect(await service.audit(evidence.published.semver)).toEqual({ ok: true });

      // La versión que sobrescribió la clave sigue siendo otra y distinta.
      const other = await store.head(evidence.published.key, evidence.overwrite_version_id);
      expect(other?.versionId).toBe(evidence.overwrite_version_id);
      expect(other?.sha256).toBeNull();

      for (const [semver, reason] of Object.entries(evidence.failed)) {
        const failed = await repo.find('local_test', semver);
        expect(failed).toMatchObject({
          status: 'failed',
          failure_reason: reason,
          published_at: null,
        });
      }
      expect((await service.list()).models.filter((m) => m.status === 'published')).toHaveLength(1);
    } finally {
      await pool.end();
    }
  });
});

function storageKey(semver: string) {
  return `models/p3-cnn-classifier/${semver}/model.pt`;
}
