/**
 * D06-03 (preparación) — Publicación `official` del paquete final y su tarjeta en el bucket
 * de modelos, reutilizando el registro de D04-06 (register → upload → verify) para los dos
 * objetos:
 *
 *   models/p3-cnn-classifier/<semver>/model.pt         (modelo, tabla p3_model_registry)
 *   models/p3-cnn-classifier/<semver>/model_card.json  (tarjeta, tabla p3_model_card)
 *
 * Antes de registrar nada se lee la configuración de seguridad del bucket (versioning,
 * SSE AES256, bloqueo público, DenyInsecureTransport, sin lifecycle que expire versiones);
 * si algo falla o no se puede leer (permiso insuficiente) no se publica. La tarjeta se
 * verifica por VersionId ANTES de subir el modelo: un modelo nunca queda `published` sin su
 * tarjeta verificada.
 *
 * Storage y repositorios en memoria con bytes de FIXTURE: prueban el flujo, no AWS. La
 * publicación real exige el paquete de D06-02, el bucket provisionado y MLOPS-S3-MODELS.
 */
import { createHash } from 'node:crypto';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConflictError, ServiceUnavailableError, ValidationError } from '../src/logic/errors.js';
import {
  MODEL_CARD_OBJECT_NAME,
  type RegistryEntry,
  type RegistryNamespace,
} from '../src/logic/model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
  type StoredObjectHead,
} from '../src/logic/model-registry.service.js';
import { createModelsPortalService } from '../src/logic/models-portal.service.js';
import {
  type BucketSecurityReport,
  createOfficialPublication,
} from '../src/logic/official-publication.service.js';
import { modelsResponseSchema } from '../src/logic/p3.contracts.js';

const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');
const NOW = new Date('2026-10-02T16:00:00.000Z');
const IDENTITY = {
  semver: '1.0.0',
  mlflow_run_id: 'a'.repeat(32),
  manifest_hash: '0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43',
  dvc_release: 'v0.1.1',
  dvc_release_hash: '2e7029bd794da00c9962263bd42f4e4fa94bba25e2350e4cf934505e98f937e8',
};
const MODEL = Buffer.from('D06-03 FIXTURE: bytes de prueba, no el paquete final de D06-02\n');
const CARD = Buffer.from(
  JSON.stringify({
    kind: 'final',
    model: 'p3-cnn-classifier',
    semver: '1.0.0',
    mlflow_run_id: 'a'.repeat(32),
    official_test_metrics: { accuracy: 0.9 },
  }),
);

class MemoryStore implements ModelObjectStore {
  readonly bucket = 'mlops-p3-models-test';
  failPutFor: string | null = null;
  versioned = true;
  private seq = 0;
  readonly objects = new Map<string, { body: Buffer; sha256: string }>();
  readonly puts: string[] = [];
  async put(key: string, body: Buffer, sha256: string) {
    if (this.failPutFor && key.endsWith(this.failPutFor)) {
      throw new Error('AccessDenied: User is not authorized to perform: s3:PutObject');
    }
    this.puts.push(key);
    if (!this.versioned) return { versionId: null };
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

const SECURE: BucketSecurityReport = {
  bucket: 'mlops-p3-models-test',
  ok: true,
  checks: [
    { name: 'versioning', ok: true, detail: 'Enabled' },
    { name: 'encryption', ok: true, detail: 'AES256' },
    { name: 'public_access_block', ok: true, detail: 'los 4 bloqueos activos' },
    { name: 'deny_insecure_transport', ok: true, detail: 'DenyInsecureTransport' },
    { name: 'lifecycle', ok: true, detail: 'sin reglas' },
  ],
};

let store: MemoryStore;
let modelRepo: MemoryRepo;
let cardRepo: MemoryRepo;
let security: BucketSecurityReport;

const modelRegistry = () =>
  createModelRegistryService({ repo: modelRepo, store, namespace: 'official', now: () => NOW });
const cardRegistry = () =>
  createModelRegistryService({
    repo: cardRepo,
    store,
    namespace: 'official',
    objectName: MODEL_CARD_OBJECT_NAME,
    now: () => NOW,
  });
const publication = () =>
  createOfficialPublication({
    model: modelRegistry(),
    card: cardRegistry(),
    store,
    inspectBucket: async () => security,
  });

beforeEach(() => {
  store = new MemoryStore();
  modelRepo = new MemoryRepo();
  cardRepo = new MemoryRepo();
  security = SECURE;
});

describe('publicación official con el bucket seguro', () => {
  it('modelo y tarjeta quedan published por VersionId exacto, con lectura de vuelta', async () => {
    const result = await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    expect(result.published).toBe(true);
    expect(result.bucket_security).toEqual(SECURE);
    expect(result.card.model).toMatchObject({
      s3_bucket: 'mlops-p3-models-test',
      s3_key: 'models/p3-cnn-classifier/1.0.0/model_card.json',
      version_id: 'v-1',
      sha256: sha(CARD),
      status: 'published',
    });
    expect(result.model.model).toMatchObject({
      s3_key: 'models/p3-cnn-classifier/1.0.0/model.pt',
      version_id: 'v-2',
      sha256: sha(MODEL),
      status: 'published',
      published_at: NOW.toISOString(),
      ...IDENTITY,
    });
    expect(result.read_back).toEqual({ model_sha256: sha(MODEL), card_sha256: sha(CARD) });
    // La tarjeta se sube y verifica antes que el modelo.
    expect(store.puts).toEqual([
      'models/p3-cnn-classifier/1.0.0/model_card.json',
      'models/p3-cnn-classifier/1.0.0/model.pt',
    ]);
  });

  it('Models (official) resuelve la versión con su tarjeta', async () => {
    await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    const portal = createModelsPortalService({ repo: modelRepo, store, cardRepo, now: () => NOW });
    const listed = modelsResponseSchema.parse(await portal.official());
    expect(listed.models).toHaveLength(1);
    expect(listed.models[0]?.version_id).toBe('v-2');
    expect(listed.models[0]?.model_card).toEqual({
      s3_key: 'models/p3-cnn-classifier/1.0.0/model_card.json',
      version_id: 'v-1',
      sha256: sha(CARD),
      size_bytes: CARD.length,
      status: 'published',
    });
  });

  it('el semver es inmutable: publicar otra vez el mismo semver → 409 sin tocar lo publicado', async () => {
    await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    await expect(
      publication().publish({ identity: IDENTITY, model: MODEL, card: CARD }),
    ).rejects.toBeInstanceOf(ConflictError);
    expect((await modelRepo.find('official', '1.0.0'))?.status).toBe('published');
    expect(store.puts).toHaveLength(2);
  });
});

describe('el bucket se comprueba antes de registrar nada', () => {
  it.each([
    ['versioning', 'Suspended'],
    ['public_access_block', 'BlockPublicPolicy=false'],
    ['deny_insecure_transport', 'sin DenyInsecureTransport'],
    ['encryption', 'aws:kms'],
    ['lifecycle', 'regla que expira versiones anteriores'],
    ['versioning', 'AccessDenied: permiso insuficiente para s3:GetBucketVersioning'],
  ])('%s: %s → 409, sin registro ni subida', async (name, detail) => {
    security = {
      ...SECURE,
      ok: false,
      checks: SECURE.checks.map((check) =>
        check.name === name ? { name, ok: false, detail } : check,
      ),
    };
    const attempt = publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    await expect(attempt).rejects.toBeInstanceOf(ConflictError);
    await expect(attempt).rejects.toThrow(detail);
    expect(modelRepo.rows.size).toBe(0);
    expect(cardRepo.rows.size).toBe(0);
    expect(store.puts).toEqual([]);
  });
});

describe('la tarjeta y el paquete se contrastan antes de registrar', () => {
  it.each([
    ['tarjeta que no es JSON', Buffer.from('no soy json'), /tarjeta.*JSON/],
    [
      'tarjeta smoke',
      Buffer.from(JSON.stringify({ kind: 'smoke', mlflow_run_id: 'a'.repeat(32) })),
      /smoke/,
    ],
    [
      'tarjeta de otro run',
      Buffer.from(JSON.stringify({ kind: 'final', mlflow_run_id: 'b'.repeat(32) })),
      /otro run/,
    ],
    [
      'tarjeta de otro semver',
      Buffer.from(JSON.stringify({ kind: 'final', semver: '2.0.0' })),
      /semver/,
    ],
  ])('%s → 400 sin registrar nada', async (_name, card, message) => {
    const attempt = publication().publish({ identity: IDENTITY, model: MODEL, card });
    await expect(attempt).rejects.toBeInstanceOf(ValidationError);
    await expect(attempt).rejects.toThrow(message);
    expect(modelRepo.rows.size).toBe(0);
    expect(store.puts).toEqual([]);
  });
});

describe('ningún fallo termina como published', () => {
  it('bucket que no devuelve VersionId → tarjeta y modelo failed', async () => {
    store.versioned = false;
    const result = await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    expect(result.published).toBe(false);
    expect(result.card.failure?.reason).toBe('version_id_missing');
    expect(result.model.model.status).toBe('failed');
    expect(result.model.failure?.reason).toBe('model_card_failed');
    expect(store.puts).toEqual(['models/p3-cnn-classifier/1.0.0/model_card.json']);
  });

  it('tarjeta ausente al verificar → el modelo no se sube y queda failed', async () => {
    const original = store.head.bind(store);
    store.head = async (key, versionId) =>
      key.endsWith('model_card.json') ? null : original(key, versionId);
    const result = await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    expect(result.published).toBe(false);
    expect(result.card.failure?.reason).toBe('object_missing');
    expect(result.model.failure?.reason).toBe('model_card_failed');
    expect(result.model.failure?.detail).toMatch(/model_card\.json/);
    expect(store.puts).not.toContain('models/p3-cnn-classifier/1.0.0/model.pt');
  });

  it('VersionId equivocado del modelo → failed (version_mismatch), aunque la tarjeta esté bien', async () => {
    const original = store.head.bind(store);
    store.head = async (key, versionId) => {
      const head = await original(key, versionId);
      return key.endsWith('model.pt') && head ? { ...head, versionId: 'otra-version' } : head;
    };
    const result = await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    expect(result.published).toBe(false);
    expect(result.card.model.status).toBe('published');
    expect(result.model.failure?.reason).toBe('version_mismatch');
    expect(result.read_back).toBeNull();
  });

  it('contenido del modelo distinto al leer por VersionId → failed (sha256_mismatch)', async () => {
    const original = store.get.bind(store);
    store.get = async (key, versionId) =>
      key.endsWith('model.pt') ? Buffer.from('otros bytes') : original(key, versionId);
    const result = await publication().publish({ identity: IDENTITY, model: MODEL, card: CARD });
    expect(result.published).toBe(false);
    expect(result.model.failure?.reason).toBe('sha256_mismatch');
  });

  it('permiso insuficiente para subir el modelo → 503 y nada published', async () => {
    store.failPutFor = 'model.pt';
    await expect(
      publication().publish({ identity: IDENTITY, model: MODEL, card: CARD }),
    ).rejects.toBeInstanceOf(ServiceUnavailableError);
    expect((await modelRepo.find('official', '1.0.0'))?.status).toBe('draft');
    const portal = createModelsPortalService({ repo: modelRepo, store, cardRepo, now: () => NOW });
    expect(
      (await portal.official()).models.filter((model) => model.status === 'published'),
    ).toEqual([]);
  });
});
