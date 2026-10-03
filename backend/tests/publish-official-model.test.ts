/**
 * D06-03 (preparación) — Script de publicación `official` (`publish-official-model`).
 *
 * Lo corre el principal operacional (MLOPS-S3-MODELS) con sus credenciales SSO, nunca
 * versionadas. Exige el bucket real de AWS en MODEL_S3_BUCKET (no el de MinIO local), el
 * paquete y su tarjeta, y sale con 0 solo si los dos quedan `published`.
 */
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  officialBucketFrom,
  parsePublishArgs,
  publishOfficialPackage,
} from '../src/cli/publish-official-model.js';
import type { RegistryEntry, RegistryNamespace } from '../src/logic/model-registry.js';
import type {
  ModelObjectStore,
  ModelRegistryRepository,
  StoredObjectHead,
} from '../src/logic/model-registry.service.js';

const ARGS = [
  '--file',
  '/tmp/model.pt',
  '--card',
  '/tmp/model_card.json',
  '--semver',
  '1.0.0',
  '--run-id',
  'a'.repeat(32),
  '--manifest-hash',
  'd'.repeat(64),
  '--release',
  'v0.1.1',
  '--release-hash',
  'e'.repeat(64),
];

describe('argumentos', () => {
  it('exige modelo, tarjeta e identidad completa', () => {
    expect(parsePublishArgs(ARGS)).toEqual({
      file: '/tmp/model.pt',
      card: '/tmp/model_card.json',
      semver: '1.0.0',
      mlflow_run_id: 'a'.repeat(32),
      manifest_hash: 'd'.repeat(64),
      dvc_release: 'v0.1.1',
      dvc_release_hash: 'e'.repeat(64),
    });
    expect(() => parsePublishArgs(ARGS.slice(0, 2))).toThrow(/Faltan argumentos.*--card/);
  });

  it('no acepta elegir namespace: este script solo publica official', () => {
    expect(() => parsePublishArgs([...ARGS, '--namespace', 'local_test'])).toThrow(
      /Argumento desconocido: --namespace/,
    );
  });
});

describe('bucket official', () => {
  it.each([
    [undefined, /MODEL_S3_BUCKET/],
    ['', /MODEL_S3_BUCKET/],
    ['p3-models-local', /MinIO local/],
  ])('MODEL_S3_BUCKET=%s → error antes de conectar', (value, message) => {
    expect(() => officialBucketFrom({ MODEL_S3_BUCKET: value })).toThrow(message);
  });

  it('usa el nombre real que dio Terraform', () => {
    expect(officialBucketFrom({ MODEL_S3_BUCKET: 'mlops-p3-models-20261002xyz' })).toBe(
      'mlops-p3-models-20261002xyz',
    );
  });
});

class MemoryStore implements ModelObjectStore {
  readonly bucket = 'mlops-p3-models-test';
  private seq = 0;
  readonly objects = new Map<string, { body: Buffer; sha256: string }>();
  async put(key: string, body: Buffer, sha256: string) {
    const versionId = `v-${++this.seq}`;
    this.objects.set(`${key}#${versionId}`, { body: Buffer.from(body), sha256 });
    return { versionId };
  }
  async head(key: string, versionId: string): Promise<StoredObjectHead | null> {
    const o = this.objects.get(`${key}#${versionId}`);
    return o ? { size: o.body.length, versionId, sha256: o.sha256 } : null;
  }
  async get(key: string, versionId: string) {
    const o = this.objects.get(`${key}#${versionId}`);
    return o ? Buffer.from(o.body) : null;
  }
}

class MemoryRepo implements ModelRegistryRepository {
  readonly rows = new Map<string, RegistryEntry>();
  async insertDraft(entry: RegistryEntry) {
    if (this.rows.has(entry.semver)) return false;
    this.rows.set(entry.semver, { ...entry });
    return true;
  }
  async find(_ns: RegistryNamespace, semver: string) {
    const row = this.rows.get(semver);
    return row ? { ...row } : null;
  }
  async list() {
    return [...this.rows.values()];
  }
  async markUploaded(_ns: RegistryNamespace, semver: string, versionId: string) {
    const row = this.rows.get(semver);
    if (row?.status !== 'draft' || row.version_id !== null) return false;
    row.version_id = versionId;
    return true;
  }
  async markPublished(_ns: RegistryNamespace, semver: string, versionId: string, at: Date) {
    const row = this.rows.get(semver);
    if (row?.status !== 'draft' || row.version_id !== versionId) return false;
    row.status = 'published';
    row.published_at = at;
    return true;
  }
  async markFailed(
    _ns: RegistryNamespace,
    semver: string,
    reason: RegistryEntry['failure_reason'],
    detail: string,
  ) {
    const row = this.rows.get(semver);
    if (row?.status !== 'draft') return false;
    row.status = 'failed';
    row.failure_reason = reason;
    row.failure_detail = detail;
    return true;
  }
}

describe('publishOfficialPackage', () => {
  const model = Buffer.from('modelo FIXTURE');
  const card = Buffer.from(JSON.stringify({ kind: 'final', semver: '1.0.0' }));

  it('publica en namespace official y devuelve la evidencia (bucket, key, VersionId, SHA, tamaño)', async () => {
    const modelRepo = new MemoryRepo();
    const cardRepo = new MemoryRepo();
    const evidence = await publishOfficialPackage(
      {
        modelRepo,
        cardRepo,
        store: new MemoryStore(),
        inspectBucket: async () => ({ bucket: 'mlops-p3-models-test', ok: true, checks: [] }),
        now: () => new Date('2026-10-02T16:00:00.000Z'),
      },
      parsePublishArgs(ARGS),
      { model, card },
    );
    expect(evidence.published).toBe(true);
    expect(evidence.namespace).toBe('official');
    expect(evidence.model).toMatchObject({
      s3_bucket: 'mlops-p3-models-test',
      s3_key: 'models/p3-cnn-classifier/1.0.0/model.pt',
      version_id: 'v-2',
      sha256: createHash('sha256').update(model).digest('hex'),
      size_bytes: model.length,
      status: 'published',
    });
    expect(evidence.card).toMatchObject({
      s3_key: 'models/p3-cnn-classifier/1.0.0/model_card.json',
      version_id: 'v-1',
      size_bytes: card.length,
      status: 'published',
    });
    expect([...modelRepo.rows.values()].every((row) => row.namespace === 'official')).toBe(true);
    expect([...cardRepo.rows.values()].every((row) => row.namespace === 'official')).toBe(true);
  });
});
