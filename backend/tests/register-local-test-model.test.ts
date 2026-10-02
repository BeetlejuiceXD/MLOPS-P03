/**
 * D05-06 (preparación) — Script que registra un paquete en el namespace `local_test`
 * con el servicio de D04-06 (register → upload → verify) y devuelve la evidencia
 * (bucket/key/VersionId/SHA-256/tamaño/estado). Nunca escribe en `official`.
 *
 * Dobles en memoria del repositorio y del bucket; paquete FIXTURE. El paquete real de
 * D05-01 y MinIO/MariaDB reales se evidencian aparte.
 */
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  parseRegisterArgs,
  registerLocalTestPackage,
} from '../src/cli/register-local-test-model.js';
import type { RegistryEntry, RegistryNamespace } from '../src/logic/model-registry.js';
import type {
  ModelObjectStore,
  ModelRegistryRepository,
  StoredObjectHead,
} from '../src/logic/model-registry.service.js';

const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');
const BODY = Buffer.from('paquete FIXTURE de D05-06\n');
const IDENTITY = {
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  manifest_hash: '0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43',
  dvc_release: 'v0.1.1',
  dvc_release_hash: '2e7029bd794da00c9962263bd42f4e4fa94bba25e2350e4cf934505e98f937e8',
};

class Store implements ModelObjectStore {
  readonly bucket = 'p3-models-local';
  objects = new Map<string, { body: Buffer; sha256: string }>();
  versioning = true;
  async put(key: string, body: Buffer, sha256: string) {
    const versionId = this.versioning ? `v${this.objects.size + 1}` : null;
    this.objects.set(`${key}#${versionId}`, { body, sha256 });
    return { versionId };
  }
  async head(key: string, versionId: string): Promise<StoredObjectHead | null> {
    const o = this.objects.get(`${key}#${versionId}`);
    return o ? { size: o.body.length, versionId, sha256: o.sha256 } : null;
  }
  async get(key: string, versionId: string) {
    return this.objects.get(`${key}#${versionId}`)?.body ?? null;
  }
}

class Repo implements ModelRegistryRepository {
  rows = new Map<string, RegistryEntry>();
  async insertDraft(e: RegistryEntry) {
    const id = `${e.namespace}/${e.semver}`;
    if (this.rows.has(id)) return false;
    this.rows.set(id, { ...e });
    return true;
  }
  async find(ns: RegistryNamespace, s: string) {
    const r = this.rows.get(`${ns}/${s}`);
    return r ? { ...r } : null;
  }
  async list(ns: RegistryNamespace) {
    return [...this.rows.values()].filter((r) => r.namespace === ns);
  }
  async markUploaded(ns: RegistryNamespace, s: string, v: string) {
    const r = this.rows.get(`${ns}/${s}`);
    if (r?.status !== 'draft' || r.version_id !== null) return false;
    r.version_id = v;
    return true;
  }
  async markPublished(ns: RegistryNamespace, s: string, v: string, at: Date) {
    const r = this.rows.get(`${ns}/${s}`);
    if (r?.status !== 'draft' || r.version_id !== v) return false;
    r.status = 'published';
    r.published_at = at;
    return true;
  }
  async markFailed(
    ns: RegistryNamespace,
    s: string,
    reason: RegistryEntry['failure_reason'],
    detail: string,
  ) {
    const r = this.rows.get(`${ns}/${s}`);
    if (r?.status !== 'draft') return false;
    r.status = 'failed';
    r.failure_reason = reason;
    r.failure_detail = detail;
    return true;
  }
}

const ARGS = [
  '--file',
  '/tmp/model.pt',
  '--semver',
  '0.0.1',
  '--run-id',
  IDENTITY.mlflow_run_id,
  '--manifest-hash',
  IDENTITY.manifest_hash,
  '--release',
  IDENTITY.dvc_release,
  '--release-hash',
  IDENTITY.dvc_release_hash,
];

describe('argumentos del script', () => {
  it('lee archivo, semver e identidad del paquete', () => {
    expect(parseRegisterArgs(ARGS)).toEqual({
      file: '/tmp/model.pt',
      semver: '0.0.1',
      ...IDENTITY,
    });
  });

  it('falta un argumento → error que lo nombra', () => {
    expect(() => parseRegisterArgs(ARGS.slice(0, -2))).toThrow(/--release-hash/);
  });

  it('no existe opción para escribir en official', () => {
    expect(() => parseRegisterArgs([...ARGS, '--namespace', 'official'])).toThrow(/--namespace/);
  });
});

describe('registro local_test con el servicio de D04-06', () => {
  it('register → upload → verify deja la versión published con su VersionId', async () => {
    const repo = new Repo();
    const store = new Store();
    const evidence = await registerLocalTestPackage(
      { repo, store, now: () => new Date('2026-10-02T05:00:00Z') },
      { semver: '0.0.1', ...IDENTITY },
      BODY,
    );
    expect(evidence).toMatchObject({
      namespace: 'local_test',
      semver: '0.0.1',
      s3_bucket: 'p3-models-local',
      s3_key: 'models/p3-cnn-classifier/0.0.1/model.pt',
      version_id: 'v1',
      sha256: sha(BODY),
      size_bytes: BODY.length,
      status: 'published',
      failure_reason: null,
      read_back_sha256: sha(BODY),
    });
    expect(repo.rows.has('official/0.0.1')).toBe(false);
  });

  it('semver repetido → error de semver inmutable, sin tocar la versión existente', async () => {
    const repo = new Repo();
    const store = new Store();
    const deps = { repo, store };
    await registerLocalTestPackage(deps, { semver: '0.0.1', ...IDENTITY }, BODY);
    await expect(
      registerLocalTestPackage(deps, { semver: '0.0.1', ...IDENTITY }, Buffer.from('otro')),
    ).rejects.toThrow(/inmutable/);
    expect(store.objects.size).toBe(1);
  });

  it('bucket sin versioning → failed con version_id_missing (nunca published)', async () => {
    const store = new Store();
    store.versioning = false;
    const evidence = await registerLocalTestPackage(
      { repo: new Repo(), store },
      { semver: '0.0.1', ...IDENTITY },
      BODY,
    );
    expect(evidence).toMatchObject({ status: 'failed', failure_reason: 'version_id_missing' });
    expect(evidence.read_back_sha256).toBeNull();
  });
});
