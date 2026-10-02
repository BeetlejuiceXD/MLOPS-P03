/**
 * D05-06 (preparación) — Models conectado al registro de D04-06, sin otro registry ni
 * otro adaptador:
 *
 *   GET /models                              → solo `official` (contrato models_response)
 *   GET /models/local-test                   → registro `local_test` con tamaño y motivos
 *   GET /models/local-test/:semver           → versión + integridad comprobada AHORA
 *                                              (audit por VersionId; no cambia el registro)
 *   GET /models/local-test/:semver/object    → bytes de la versión exacta (VersionId),
 *                                              solo si su SHA-256 sigue siendo el registrado
 *
 * Usa el servicio real de D04-06 (`createModelRegistryService`) con dobles en memoria del
 * repositorio y del bucket versionado, y un paquete FIXTURE (bytes de prueba): no acredita
 * el paquete smoke real de D05-01, que se evidencia aparte contra MinIO + MariaDB.
 */
import { createHash } from 'node:crypto';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { RegistryEntry, RegistryNamespace } from '../src/logic/model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
  type StoredObjectHead,
} from '../src/logic/model-registry.service.js';
import { createModelsPortalService } from '../src/logic/models-portal.service.js';
import {
  localTestModelDetailSchema,
  localTestModelsResponseSchema,
  modelsResponseSchema,
} from '../src/logic/p3.contracts.js';
import { createModelsRouter } from '../src/ui/models.routes.js';

const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');
const PACKAGE = Buffer.from('D05-06 paquete FIXTURE: no es el paquete smoke real de D05-01\n');
const NOW = new Date('2026-10-02T05:00:00.000Z');
const IDENTITY = {
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  manifest_hash: '0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43',
  dvc_release: 'v0.1.1',
  dvc_release_hash: '2e7029bd794da00c9962263bd42f4e4fa94bba25e2350e4cf934505e98f937e8',
};

class MemoryStore implements ModelObjectStore {
  readonly bucket = 'p3-models-local';
  broken = false;
  private seq = 0;
  readonly objects = new Map<string, { body: Buffer; sha256: string }>();
  private check() {
    if (this.broken) throw new Error('connect ECONNREFUSED 127.0.0.1:9000');
  }
  async put(key: string, body: Buffer, sha256: string) {
    this.check();
    const versionId = `ver-${++this.seq}`;
    this.objects.set(`${key}#${versionId}`, { body: Buffer.from(body), sha256 });
    return { versionId };
  }
  async head(key: string, versionId: string): Promise<StoredObjectHead | null> {
    this.check();
    const o = this.objects.get(`${key}#${versionId}`);
    return o ? { size: o.body.length, versionId, sha256: o.sha256 } : null;
  }
  async get(key: string, versionId: string) {
    this.check();
    const o = this.objects.get(`${key}#${versionId}`);
    return o ? Buffer.from(o.body) : null;
  }
}

class MemoryRepo implements ModelRegistryRepository {
  readonly rows = new Map<string, RegistryEntry>();
  private id = (ns: RegistryNamespace, semver: string) => `${ns}/${semver}`;
  async insertDraft(entry: RegistryEntry) {
    const id = this.id(entry.namespace, entry.semver);
    if (this.rows.has(id)) return false;
    this.rows.set(id, { ...entry });
    return true;
  }
  async find(ns: RegistryNamespace, semver: string) {
    const row = this.rows.get(this.id(ns, semver));
    return row ? { ...row } : null;
  }
  async list(ns: RegistryNamespace) {
    return [...this.rows.values()].filter((r) => r.namespace === ns).map((r) => ({ ...r }));
  }
  async markUploaded(ns: RegistryNamespace, semver: string, versionId: string) {
    const row = this.rows.get(this.id(ns, semver));
    if (row?.status !== 'draft' || row.version_id !== null) return false;
    row.version_id = versionId;
    return true;
  }
  async markPublished(ns: RegistryNamespace, semver: string, versionId: string, at: Date) {
    const row = this.rows.get(this.id(ns, semver));
    if (row?.status !== 'draft' || row.version_id !== versionId) return false;
    row.status = 'published';
    row.published_at = at;
    return true;
  }
  async markFailed(
    ns: RegistryNamespace,
    semver: string,
    reason: RegistryEntry['failure_reason'],
    detail: string,
  ) {
    const row = this.rows.get(this.id(ns, semver));
    if (row?.status !== 'draft') return false;
    row.status = 'failed';
    row.failure_reason = reason;
    row.failure_detail = detail;
    return true;
  }
}

let store: MemoryStore;
let repo: MemoryRepo;
let server: Server | undefined;
let api = '';

const registry = (namespace: RegistryNamespace) =>
  createModelRegistryService({ repo, store, namespace, now: () => NOW });

/** register → upload → verify con el servicio de D04-06 (lo que hará el script smoke). */
async function publish(namespace: RegistryNamespace, semver: string, body = PACKAGE) {
  const service = registry(namespace);
  await service.register({ semver, ...IDENTITY, sha256: sha(body), size_bytes: body.length });
  await service.upload(semver, body);
  return service.verify(semver);
}

async function getJson(path: string) {
  const res = await fetch(`${api}${path}`);
  return { status: res.status, body: await res.json() };
}

beforeEach(async () => {
  store = new MemoryStore();
  repo = new MemoryRepo();
  const app = express();
  app.use(
    '/models',
    createModelsRouter(createModelsPortalService({ repo, store, now: () => NOW })),
  );
  const started = await new Promise<Server>((resolve) => {
    const listening = app.listen(0, '127.0.0.1', () => resolve(listening));
  });
  server = started;
  api = `http://127.0.0.1:${(started.address() as AddressInfo).port}`;
});

afterEach(async () => {
  await new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve()));
});

describe('GET /models — solo official', () => {
  it('una versión local_test nunca aparece en la lista official', async () => {
    await publish('local_test', '0.0.1');
    const { status, body } = await getJson('/models');
    expect(status).toBe(200);
    expect(modelsResponseSchema.safeParse(body).success).toBe(true);
    expect(body).toEqual({ models: [] });
  });

  it('lista las versiones official del registro con el contrato model_version', async () => {
    await publish('official', '1.0.0');
    await publish('local_test', '0.0.1');
    const { body } = await getJson('/models');
    expect(body.models.map((m: { semver: string }) => m.semver)).toEqual(['1.0.0']);
  });
});

describe('GET /models/local-test — registro de pruebas locales', () => {
  it('muestra namespace, VersionId, SHA-256, tamaño y estado de cada versión', async () => {
    await publish('local_test', '0.0.1');
    await publish('official', '1.0.0');
    const { status, body } = await getJson('/models/local-test');
    expect(status).toBe(200);
    expect(localTestModelsResponseSchema.safeParse(body).success).toBe(true);
    expect(body.namespace).toBe('local_test');
    expect(body.models).toEqual([
      expect.objectContaining({
        namespace: 'local_test',
        semver: '0.0.1',
        s3_bucket: 'p3-models-local',
        s3_key: 'models/p3-cnn-classifier/0.0.1/model.pt',
        version_id: 'ver-1',
        sha256: sha(PACKAGE),
        size_bytes: PACKAGE.length,
        status: 'published',
        failure_reason: null,
      }),
    ]);
  });

  it('una versión fallida muestra su motivo; un borrador queda como draft', async () => {
    const service = registry('local_test');
    await service.register({
      semver: '0.0.2',
      ...IDENTITY,
      sha256: sha(PACKAGE),
      size_bytes: PACKAGE.length,
    });
    await service.upload('0.0.2', Buffer.from('otros bytes'));
    await service.register({
      semver: '0.0.3',
      ...IDENTITY,
      sha256: sha(PACKAGE),
      size_bytes: PACKAGE.length,
    });
    const { body } = await getJson('/models/local-test');
    expect(localTestModelsResponseSchema.safeParse(body).success).toBe(true);
    const bySemver = Object.fromEntries(body.models.map((m: { semver: string }) => [m.semver, m]));
    expect(bySemver['0.0.2']).toMatchObject({ status: 'failed', failure_reason: 'size_mismatch' });
    expect(bySemver['0.0.3']).toMatchObject({ status: 'draft', version_id: null });
  });
});

describe('GET /models/local-test/:semver — integridad comprobada ahora', () => {
  it('versión publicada cuyo objeto sigue intacto → integridad ok', async () => {
    await publish('local_test', '0.0.1');
    const { status, body } = await getJson('/models/local-test/0.0.1');
    expect(status).toBe(200);
    expect(localTestModelDetailSchema.safeParse(body).success).toBe(true);
    expect(body.integrity).toEqual({
      checked_at: NOW.toISOString(),
      ok: true,
      reason: null,
      detail: null,
    });
  });

  it('objeto borrado → object_missing, sin cambiar el registro (sigue published)', async () => {
    await publish('local_test', '0.0.1');
    store.objects.clear();
    const { body } = await getJson('/models/local-test/0.0.1');
    expect(body.integrity).toMatchObject({ ok: false, reason: 'object_missing' });
    expect(body.model.status).toBe('published');
  });

  it('contenido alterado con el mismo tamaño → sha256_mismatch', async () => {
    await publish('local_test', '0.0.1');
    const [key] = [...store.objects.keys()];
    const original = store.objects.get(key as string);
    store.objects.set(key as string, {
      body: Buffer.alloc(PACKAGE.length, 0x41),
      sha256: original?.sha256 as string,
    });
    const { body } = await getJson('/models/local-test/0.0.1');
    expect(body.integrity).toMatchObject({ ok: false, reason: 'sha256_mismatch' });
  });

  it('storage caído → 503 (no se confunde con objeto ausente)', async () => {
    await publish('local_test', '0.0.1');
    store.broken = true;
    const { status, body } = await getJson('/models/local-test/0.0.1');
    expect(status).toBe(503);
    expect(body.error).toMatch(/Storage de modelos no disponible/);
  });

  it('un borrador no se audita (integrity null); semver inexistente → 404; inválido → 400', async () => {
    await registry('local_test').register({
      semver: '0.0.3',
      ...IDENTITY,
      sha256: sha(PACKAGE),
      size_bytes: PACKAGE.length,
    });
    const draft = await getJson('/models/local-test/0.0.3');
    expect(draft.status).toBe(200);
    expect(draft.body.integrity).toBeNull();
    expect((await getJson('/models/local-test/9.9.9')).status).toBe(404);
    expect((await getJson('/models/local-test/no-semver')).status).toBe(400);
  });

  it('una versión official no se lee por la ruta local-test', async () => {
    await publish('official', '1.0.0');
    expect((await getJson('/models/local-test/1.0.0')).status).toBe(404);
  });
});

describe('GET /models/local-test/:semver/object — versión exacta', () => {
  it('devuelve los bytes registrados, leídos por su VersionId', async () => {
    await publish('local_test', '0.0.1');
    await publish('local_test', '0.0.2', Buffer.from('otra versión, otro objeto'));
    const res = await fetch(`${api}/models/local-test/0.0.1/object`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toMatch(/application\/octet-stream/);
    expect(res.headers.get('x-model-version-id')).toBe('ver-1');
    expect(res.headers.get('x-model-sha256')).toBe(sha(PACKAGE));
    expect(Buffer.from(await res.arrayBuffer()).equals(PACKAGE)).toBe(true);
  });

  it('si el contenido ya no coincide con el SHA-256 registrado no se entrega (409)', async () => {
    await publish('local_test', '0.0.1');
    const [key] = [...store.objects.keys()];
    const original = store.objects.get(key as string);
    store.objects.set(key as string, {
      body: Buffer.alloc(PACKAGE.length, 0x41),
      sha256: original?.sha256 as string,
    });
    const { status, body } = await getJson('/models/local-test/0.0.1/object');
    expect(status).toBe(409);
    expect(body.error).toMatch(/sha256_mismatch/);
  });

  it('objeto ausente → 404 object_missing; storage caído → 503', async () => {
    await publish('local_test', '0.0.1');
    store.objects.clear();
    const missing = await getJson('/models/local-test/0.0.1/object');
    expect(missing.status).toBe(404);
    expect(missing.body.error).toMatch(/^object_missing: /);

    await publish('local_test', '0.0.2');
    store.broken = true;
    expect((await getJson('/models/local-test/0.0.2/object')).status).toBe(503);
  });

  it('un borrador o una versión fallida no tienen objeto que entregar (409)', async () => {
    await registry('local_test').register({
      semver: '0.0.3',
      ...IDENTITY,
      sha256: sha(PACKAGE),
      size_bytes: PACKAGE.length,
    });
    const { status, body } = await getJson('/models/local-test/0.0.3/object');
    expect(status).toBe(409);
    expect(body.error).toMatch(/draft/);
  });
});
