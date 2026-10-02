/**
 * D04-06 — Registro de versiones del modelo y adaptador del bucket de modelos.
 *
 * Componentes con objetos de PRUEBA (bytes sintéticos, ningún checkpoint real): el
 * storage y el repositorio en memoria sustituyen a MinIO/MariaDB solo aquí; el recorrido
 * real contra MinIO + MariaDB (con reinicio) está en `model-registry.minio.test.ts`.
 * Nada de esto acredita un modelo final publicado en AWS (D06-03).
 */
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import type * as Minio from 'minio';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  assertModelBucketVersioned,
  createMinioModelStore,
  ensureLocalModelBucket,
} from '../src/data/storage/model-object.storage.js';
import {
  ConflictError,
  NotFoundError,
  ServiceUnavailableError,
  ValidationError,
} from '../src/logic/errors.js';
import {
  compareSemver,
  modelObjectKey,
  type RegistryEntry,
  type RegistryNamespace,
  sha256Hex,
  toModelVersion,
} from '../src/logic/model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
  type RegisterModelInput,
  type StoredObjectHead,
} from '../src/logic/model-registry.service.js';
import { modelsResponseSchema, modelVersionSchema } from '../src/logic/p3.contracts.js';

const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');
const BODY = Buffer.from('D04-06 objeto de prueba: no es un checkpoint real\n');
const NOW = new Date('2026-10-01T20:00:00.000Z');

const input = (semver: string, body: Buffer = BODY): RegisterModelInput => ({
  semver,
  mlflow_run_id: 'a'.repeat(32),
  manifest_hash: 'd'.repeat(64),
  dvc_release: 'v0.1.1',
  dvc_release_hash: 'e'.repeat(64),
  sha256: sha(body),
  size_bytes: body.length,
});

// --- Dobles en memoria ------------------------------------------------------------------

/** Bucket con versioning: cada `put` crea una versión nueva e inmutable. */
class MemoryStore implements ModelObjectStore {
  readonly bucket = 'p3-models-test';
  versioning = true;
  broken = false;
  puts: { key: string; sha256: string }[] = [];
  private seq = 0;
  readonly objects = new Map<string, { body: Buffer; sha256: string }>();

  private check() {
    if (this.broken) throw new Error('connect ECONNREFUSED 127.0.0.1:9000');
  }
  async put(key: string, body: Buffer, sha256: string) {
    this.check();
    this.puts.push({ key, sha256 });
    const versionId = this.versioning ? `ver-${++this.seq}` : null;
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
  // Manipulaciones de prueba sobre la versión guardada.
  remove(key: string, versionId: string) {
    this.objects.delete(`${key}#${versionId}`);
  }
  replace(key: string, versionId: string, body: Buffer, sha256?: string) {
    const o = this.objects.get(`${key}#${versionId}`);
    if (!o) throw new Error('no existe');
    this.objects.set(`${key}#${versionId}`, { body, sha256: sha256 ?? o.sha256 });
  }
}

/** Mismas escrituras condicionales que `p3_model_registry`. */
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
const service = (namespace: RegistryNamespace = 'local_test') =>
  createModelRegistryService({ repo, store, namespace, now: () => NOW });

beforeEach(() => {
  store = new MemoryStore();
  repo = new MemoryRepo();
});

// --- Reglas puras -----------------------------------------------------------------------

describe('reglas del registro', () => {
  it('la clave vive bajo models/p3-cnn-classifier/<semver>/', () => {
    expect(modelObjectKey('1.2.3')).toBe('models/p3-cnn-classifier/1.2.3/model.pt');
  });

  it('sha256Hex es el SHA-256 hex del contenido', () => {
    expect(sha256Hex(BODY)).toBe(sha(BODY));
  });

  it('compara semver numéricamente, no como texto', () => {
    expect(compareSemver('1.10.0', '1.9.0')).toBeGreaterThan(0);
    expect(compareSemver('1.9.0', '1.10.0')).toBeLessThan(0);
    expect(compareSemver('2.0.0', '10.0.0')).toBeLessThan(0);
    expect(compareSemver('1.0.2', '1.0.10')).toBeLessThan(0);
    expect(compareSemver('1.2.3', '1.2.3')).toBe(0);
  });

  it('toModelVersion da exactamente la forma del contrato', () => {
    const entry: RegistryEntry = {
      namespace: 'local_test',
      ...input('1.0.0'),
      s3_bucket: 'p3-models-test',
      s3_key: modelObjectKey('1.0.0'),
      version_id: 'ver-1',
      status: 'published',
      failure_reason: null,
      failure_detail: null,
      published_at: NOW,
    };
    const model = toModelVersion(entry);
    expect(modelVersionSchema.parse(model)).toEqual(model);
    expect(model.published_at).toBe('2026-10-01T20:00:00.000Z');
    expect(Object.keys(model).sort()).toEqual(
      [
        'semver',
        'mlflow_run_id',
        'manifest_hash',
        'dvc_release',
        'dvc_release_hash',
        's3_bucket',
        's3_key',
        'version_id',
        'sha256',
        'status',
        'published_at',
      ].sort(),
    );
  });
});

// --- Registro ---------------------------------------------------------------------------

describe('register', () => {
  it('guarda un draft con la identidad declarada, bucket y clave del semver', async () => {
    const model = await service().register(input('1.0.0'));
    expect(model).toEqual({
      semver: '1.0.0',
      mlflow_run_id: 'a'.repeat(32),
      manifest_hash: 'd'.repeat(64),
      dvc_release: 'v0.1.1',
      dvc_release_hash: 'e'.repeat(64),
      s3_bucket: 'p3-models-test',
      s3_key: 'models/p3-cnn-classifier/1.0.0/model.pt',
      version_id: null,
      sha256: sha(BODY),
      status: 'draft',
      published_at: null,
    });
    const row = await repo.find('local_test', '1.0.0');
    expect(row?.size_bytes).toBe(BODY.length);
    expect(row?.failure_reason).toBeNull();
    expect(store.puts).toHaveLength(0);
  });

  it('el semver es inmutable: registrar dos veces es 409 y no cambia el original', async () => {
    await service().register(input('1.0.0'));
    const other = { ...input('1.0.0'), mlflow_run_id: 'b'.repeat(32) };
    await expect(service().register(other)).rejects.toBeInstanceOf(ConflictError);
    expect((await repo.find('local_test', '1.0.0'))?.mlflow_run_id).toBe('a'.repeat(32));
  });

  it.each([
    ['semver', { semver: '1.0' }],
    ['sha256', { sha256: 'F'.repeat(64) }],
    ['run_id', { mlflow_run_id: 'xyz' }],
    ['manifest_hash', { manifest_hash: 'd'.repeat(63) }],
    ['dvc_release', { dvc_release: '0.1.1' }],
    ['size_bytes 0', { size_bytes: 0 }],
    ['size_bytes no entero', { size_bytes: 1.5 }],
  ])('rechaza identidad inválida (%s) sin escribir', async (_name, patch) => {
    await expect(service().register({ ...input('1.0.0'), ...patch })).rejects.toBeInstanceOf(
      ValidationError,
    );
    expect(repo.rows.size).toBe(0);
  });
});

// --- Roundtrip ----------------------------------------------------------------------------

describe('upload + verify', () => {
  it('objeto íntegro → published con VersionId, published_at y contrato válido', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const uploaded = await svc.upload('1.0.0', BODY);
    expect(uploaded.failure).toBeNull();
    expect(uploaded.model.status).toBe('draft');
    expect(uploaded.model.version_id).toBe('ver-1');
    expect(store.puts).toEqual([{ key: modelObjectKey('1.0.0'), sha256: sha(BODY) }]);

    const verified = await svc.verify('1.0.0');
    expect(verified.failure).toBeNull();
    expect(verified.model).toEqual({
      ...uploaded.model,
      status: 'published',
      published_at: '2026-10-01T20:00:00.000Z',
    });
    expect(modelVersionSchema.safeParse(verified.model).success).toBe(true);
  });

  it('bytes con otro SHA-256 → failed sha256_mismatch y no se sube nada', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const tampered = Buffer.from(BODY);
    tampered[0] = tampered[0] ^ 1;
    const result = await svc.upload('1.0.0', tampered);
    expect(result.failure?.reason).toBe('sha256_mismatch');
    expect(result.model.status).toBe('failed');
    expect(result.model.published_at).toBeNull();
    expect(store.puts).toHaveLength(0);
    await expect(svc.verify('1.0.0')).rejects.toBeInstanceOf(ConflictError);
    expect((await repo.find('local_test', '1.0.0'))?.status).toBe('failed');
  });

  it('bytes con otro tamaño → failed size_mismatch y no se sube nada', async () => {
    const svc = service();
    await svc.register({ ...input('1.0.0'), size_bytes: BODY.length + 1 });
    const result = await svc.upload('1.0.0', BODY);
    expect(result.failure?.reason).toBe('size_mismatch');
    expect(result.model.status).toBe('failed');
    expect(store.puts).toHaveLength(0);
  });

  it('bucket sin versioning (sin VersionId) → failed version_id_missing', async () => {
    store.versioning = false;
    const svc = service();
    await svc.register(input('1.0.0'));
    const result = await svc.upload('1.0.0', BODY);
    expect(result.failure?.reason).toBe('version_id_missing');
    expect(result.model.status).toBe('failed');
    expect(result.model.version_id).toBeNull();
  });

  it('objeto ausente al verificar → failed object_missing', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const { model } = await svc.upload('1.0.0', BODY);
    store.remove(model.s3_key, model.version_id as string);
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('object_missing');
    expect(result.model.status).toBe('failed');
    expect(result.model.published_at).toBeNull();
  });

  it('contenido guardado con otro hash (mismo tamaño y metadato) → failed sha256_mismatch', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const { model } = await svc.upload('1.0.0', BODY);
    const corrupt = Buffer.from(BODY);
    corrupt[corrupt.length - 1] = corrupt[corrupt.length - 1] ^ 1;
    store.replace(model.s3_key, model.version_id as string, corrupt);
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('sha256_mismatch');
    expect(result.model.status).toBe('failed');
  });

  it('metadato sha256 distinto al registrado → failed sha256_mismatch', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const { model } = await svc.upload('1.0.0', BODY);
    store.replace(model.s3_key, model.version_id as string, BODY, 'c'.repeat(64));
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('sha256_mismatch');
  });

  it('tamaño guardado distinto → failed size_mismatch', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const { model } = await svc.upload('1.0.0', BODY);
    store.replace(model.s3_key, model.version_id as string, Buffer.concat([BODY, BODY]));
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('size_mismatch');
  });

  it('head con otro VersionId → failed version_mismatch', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    const original = store.head.bind(store);
    store.head = async (key, versionId) => {
      const head = await original(key, versionId);
      return head && { ...head, versionId: 'otra-version' };
    };
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('version_mismatch');
    expect(result.model.status).toBe('failed');
  });

  it('get sin objeto aunque head responda → failed object_missing', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    store.get = async () => null;
    const result = await svc.verify('1.0.0');
    expect(result.failure?.reason).toBe('object_missing');
  });

  it('error del storage → 503 y la versión sigue en draft (ni failed ni published)', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    store.broken = true;
    await expect(svc.verify('1.0.0')).rejects.toBeInstanceOf(ServiceUnavailableError);
    const row = await repo.find('local_test', '1.0.0');
    expect(row?.status).toBe('draft');
    expect(row?.published_at).toBeNull();

    await svc.register(input('1.1.0'));
    await expect(svc.upload('1.1.0', BODY)).rejects.toBeInstanceOf(ServiceUnavailableError);
    expect((await repo.find('local_test', '1.1.0'))?.status).toBe('draft');
  });

  it('verify antes de subir es 409 y no cambia nada', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await expect(svc.verify('1.0.0')).rejects.toBeInstanceOf(ConflictError);
    expect((await repo.find('local_test', '1.0.0'))?.status).toBe('draft');
  });

  it('semver no registrado es 404', async () => {
    await expect(service().upload('9.9.9', BODY)).rejects.toBeInstanceOf(NotFoundError);
    await expect(service().verify('9.9.9')).rejects.toBeInstanceOf(NotFoundError);
    await expect(service().audit('9.9.9')).rejects.toBeInstanceOf(NotFoundError);
  });

  it('subir dos veces es 409: el VersionId registrado no se reemplaza', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    await expect(svc.upload('1.0.0', BODY)).rejects.toBeInstanceOf(ConflictError);
    expect(store.puts).toHaveLength(1);
    expect((await repo.find('local_test', '1.0.0'))?.version_id).toBe('ver-1');
  });

  it('published y failed son definitivos', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    await svc.verify('1.0.0');
    await expect(svc.verify('1.0.0')).rejects.toBeInstanceOf(ConflictError);
    await expect(svc.upload('1.0.0', BODY)).rejects.toBeInstanceOf(ConflictError);

    await svc.register(input('1.1.0'));
    await svc.upload('1.1.0', Buffer.from('otro contenido'));
    await expect(svc.upload('1.1.0', BODY)).rejects.toBeInstanceOf(ConflictError);
    await expect(svc.verify('1.1.0')).rejects.toBeInstanceOf(ConflictError);
    expect((await repo.find('local_test', '1.1.0'))?.status).toBe('failed');
    expect(store.puts).toHaveLength(1);
  });

  it('si otro proceso cambió la versión entre lectura y escritura, no se publica', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await svc.upload('1.0.0', BODY);
    repo.markPublished = async () => false;
    await expect(svc.verify('1.0.0')).rejects.toBeInstanceOf(ConflictError);
    expect((await repo.find('local_test', '1.0.0'))?.status).toBe('draft');
  });
});

// --- Auditoría ------------------------------------------------------------------------------

describe('audit', () => {
  const published = async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    const { model } = await svc.upload('1.0.0', BODY);
    await svc.verify('1.0.0');
    return { svc, key: model.s3_key, versionId: model.version_id as string };
  };

  it('una versión publicada íntegra sigue ok', async () => {
    const { svc } = await published();
    expect(await svc.audit('1.0.0')).toEqual({ ok: true });
  });

  it('detecta objeto ausente y contenido alterado sin cambiar el registro', async () => {
    const { svc, key, versionId } = await published();
    const corrupt = Buffer.from(BODY);
    corrupt[0] = corrupt[0] ^ 1;
    store.replace(key, versionId, corrupt);
    expect(await svc.audit('1.0.0')).toMatchObject({ ok: false, reason: 'sha256_mismatch' });
    store.remove(key, versionId);
    expect(await svc.audit('1.0.0')).toMatchObject({ ok: false, reason: 'object_missing' });
    expect((await repo.find('local_test', '1.0.0'))?.status).toBe('published');
  });

  it('solo audita versiones publicadas', async () => {
    const svc = service();
    await svc.register(input('1.0.0'));
    await expect(svc.audit('1.0.0')).rejects.toBeInstanceOf(ConflictError);
  });
});

// --- Listado y namespaces ---------------------------------------------------------------------

describe('list', () => {
  it('cumple models_response, ordena por semver numérico y no mezcla namespaces', async () => {
    const test = service('local_test');
    for (const v of ['1.10.0', '1.2.0', '1.9.0']) await test.register(input(v));
    await test.upload('1.9.0', BODY);
    await test.verify('1.9.0');
    await test.upload('1.2.0', Buffer.from('x'));
    await service('official').register(input('3.0.0'));

    const listed = await test.list();
    expect(modelsResponseSchema.safeParse(listed).success).toBe(true);
    expect(listed.models.map((m) => [m.semver, m.status])).toEqual([
      ['1.2.0', 'failed'],
      ['1.9.0', 'published'],
      ['1.10.0', 'draft'],
    ]);
    expect((await service('official').list()).models.map((m) => m.semver)).toEqual(['3.0.0']);
  });

  it('el mismo semver puede existir en local_test y en official sin pisarse', async () => {
    await service('local_test').register(input('1.0.0'));
    await service('official').register({ ...input('1.0.0'), mlflow_run_id: 'b'.repeat(32) });
    expect((await repo.find('local_test', '1.0.0'))?.mlflow_run_id).toBe('a'.repeat(32));
    expect((await repo.find('official', '1.0.0'))?.mlflow_run_id).toBe('b'.repeat(32));
  });
});

// --- Adaptador MinIO (cliente simulado) -----------------------------------------------------------

describe('adaptador MinIO del bucket de modelos', () => {
  const notFound = (code: string) => Object.assign(new Error(code), { code });

  const client = (overrides: Partial<Record<string, unknown>>) =>
    ({
      putObject: async () => ({ etag: 'e', versionId: 'v-1' }),
      statObject: async () => ({
        size: BODY.length,
        etag: 'e',
        lastModified: NOW,
        metaData: { sha256: sha(BODY), 'content-type': 'application/octet-stream' },
        versionId: 'v-1',
      }),
      getObject: async () => Readable.from([BODY.subarray(0, 5), BODY.subarray(5)]),
      ...overrides,
    }) as unknown as Minio.Client;

  it('put escribe el metadato sha256 en el bucket configurado y devuelve el VersionId', async () => {
    const calls: unknown[][] = [];
    const store = createMinioModelStore(
      client({
        putObject: async (...args: unknown[]) => {
          calls.push(args);
          return { etag: 'e', versionId: 'v-9' };
        },
      }),
      'p3-models-local',
    );
    expect(store.bucket).toBe('p3-models-local');
    expect(await store.put('models/p3-cnn-classifier/1.0.0/model.pt', BODY, sha(BODY))).toEqual({
      versionId: 'v-9',
    });
    expect(calls[0]?.[0]).toBe('p3-models-local');
    expect(calls[0]?.[1]).toBe('models/p3-cnn-classifier/1.0.0/model.pt');
    expect(calls[0]?.[3]).toBe(BODY.length);
    expect(calls[0]?.[4]).toMatchObject({ 'X-Amz-Meta-Sha256': sha(BODY) });
  });

  it('put sin VersionId devuelve null (bucket sin versioning)', async () => {
    const store = createMinioModelStore(
      client({ putObject: async () => ({ etag: 'e', versionId: null }) }),
      'b-1',
    );
    expect(await store.put('k', BODY, sha(BODY))).toEqual({ versionId: null });
  });

  it('head y get leen por VersionId y devuelven tamaño, metadato y bytes completos', async () => {
    const seen: unknown[] = [];
    const store = createMinioModelStore(
      client({
        statObject: async (_b: string, _k: string, opts: unknown) => {
          seen.push(opts);
          return {
            size: BODY.length,
            etag: 'e',
            lastModified: NOW,
            metaData: { sha256: sha(BODY) },
            versionId: 'v-1',
          };
        },
        getObject: async (_b: string, _k: string, opts: unknown) => {
          seen.push(opts);
          return Readable.from([BODY.subarray(0, 5), BODY.subarray(5)]);
        },
      }),
      'b-1',
    );
    expect(await store.head('k', 'v-1')).toEqual({
      size: BODY.length,
      versionId: 'v-1',
      sha256: sha(BODY),
    });
    expect((await store.get('k', 'v-1'))?.equals(BODY)).toBe(true);
    expect(seen).toEqual([{ versionId: 'v-1' }, { versionId: 'v-1' }]);
  });

  it('head sin metadato sha256 lo reporta como null', async () => {
    const store = createMinioModelStore(
      client({
        statObject: async () => ({ size: 1, etag: 'e', lastModified: NOW, metaData: {} }),
      }),
      'b-1',
    );
    expect(await store.head('k', 'v-1')).toEqual({ size: 1, versionId: null, sha256: null });
  });

  it.each(['NotFound', 'NoSuchKey', 'NoSuchVersion'])(
    'objeto o versión inexistente (%s) → null',
    async (code) => {
      const store = createMinioModelStore(
        client({
          statObject: async () => {
            throw notFound(code);
          },
          getObject: async () => {
            throw notFound(code);
          },
        }),
        'b-1',
      );
      expect(await store.head('k', 'v-1')).toBeNull();
      expect(await store.get('k', 'v-1')).toBeNull();
    },
  );

  it.each(['AccessDenied', 'ECONNREFUSED', undefined])(
    'otros errores (%s) se propagan: no son "objeto ausente"',
    async (code) => {
      const boom = () => {
        throw Object.assign(new Error('boom'), code ? { code } : {});
      };
      const store = createMinioModelStore(
        client({ statObject: async () => boom(), getObject: async () => boom() }),
        'b-1',
      );
      await expect(store.head('k', 'v-1')).rejects.toThrow('boom');
      await expect(store.get('k', 'v-1')).rejects.toThrow('boom');
    },
  );

  it.each([[{ Status: 'Suspended' }], [{}], [{ Status: 'Disabled' }]])(
    'un bucket sin versioning activo se rechaza (%j)',
    async (config) => {
      const c = client({ getBucketVersioning: async () => config });
      await expect(assertModelBucketVersioned(c, 'b-1')).rejects.toThrow(/versioning/);
    },
  );

  it('un bucket con versioning Enabled se acepta', async () => {
    const c = client({ getBucketVersioning: async () => ({ Status: 'Enabled' }) });
    await expect(assertModelBucketVersioned(c, 'b-1')).resolves.toBeUndefined();
  });

  it('local: crea el bucket que falta y activa versioning antes de aceptarlo', async () => {
    const calls: string[] = [];
    let status: string | undefined;
    const c = client({
      bucketExists: async () => false,
      makeBucket: async (bucket: string) => {
        calls.push(`make:${bucket}`);
      },
      setBucketVersioning: async (bucket: string, config: { Status: string }) => {
        calls.push(`versioning:${bucket}:${config.Status}`);
        status = config.Status;
      },
      getBucketVersioning: async () => ({ Status: status }),
    });
    await ensureLocalModelBucket(c, 'p3-models-local');
    expect(calls).toEqual(['make:p3-models-local', 'versioning:p3-models-local:Enabled']);
  });

  it('local: un bucket existente no se recrea, pero se le activa versioning', async () => {
    const calls: string[] = [];
    const c = client({
      bucketExists: async () => true,
      makeBucket: async () => {
        calls.push('make');
      },
      setBucketVersioning: async () => {
        calls.push('versioning');
      },
      getBucketVersioning: async () => ({ Status: 'Enabled' }),
    });
    await ensureLocalModelBucket(c, 'p3-models-local');
    expect(calls).toEqual(['versioning']);
  });
});
