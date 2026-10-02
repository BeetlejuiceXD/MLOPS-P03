/**
 * D04-06 — Registro persistente de versiones del modelo y su objeto en el storage.
 *
 *   register ──▶ draft ──upload──▶ draft + VersionId ──verify──▶ published
 *                  │                       │
 *                  └──────── failed ◀──────┘   (terminal: la versión no se reutiliza)
 *
 * - `register` guarda la identidad declarada (run, manifest, release, SHA-256, tamaño)
 *   como `draft`. El semver es inmutable dentro de su namespace.
 * - `upload` sube los bytes solo si su SHA-256 y tamaño son los declarados (si no, la
 *   versión queda `failed` y no se sube nada) y guarda el VersionId que devuelve el
 *   storage. Sin VersionId (bucket sin versioning) la versión queda `failed`.
 * - `verify` hace head + get del objeto por VersionId: solo si existe, el tamaño, el
 *   metadato `sha256` y el SHA-256 del contenido coinciden pasa a `published`. Objeto
 *   ausente o hash incorrecto → `failed` con su motivo, nunca `published`.
 * - Un error del storage (red, credenciales) no decide nada: 503 y la versión sigue igual.
 * - `audit` vuelve a comprobar una versión publicada (p. ej. tras un reinicio) sin cambiarla.
 *
 * Este ticket lo prueba en `local_test` contra MinIO; la publicación oficial en AWS y su
 * recarga son D06-03/D06-04.
 */
import { z } from 'zod';
import {
  ConflictError,
  NotFoundError,
  ServiceUnavailableError,
  ValidationError,
} from './errors.js';
import {
  compareSemver,
  modelObjectKey,
  type RegistryEntry,
  type RegistryFailureReason,
  type RegistryNamespace,
  sha256Hex,
  toModelVersion,
} from './model-registry.js';
import { type ModelVersion, modelsResponseSchema, modelVersionSchema } from './p3.contracts.js';

type ModelsResponse = z.infer<typeof modelsResponseSchema>;

/** Metadatos del objeto que devuelve `head` (HEAD por VersionId). */
export interface StoredObjectHead {
  size: number;
  versionId: string | null;
  /** Metadato de usuario `x-amz-meta-sha256` escrito al subir. */
  sha256: string | null;
}

/**
 * Storage S3-compatible del bucket de modelos (MinIO local o AWS S3). `head`/`get`
 * devuelven `null` solo si el objeto o la versión no existen; cualquier otro error se
 * propaga (no es evidencia de que falte el objeto).
 */
export interface ModelObjectStore {
  readonly bucket: string;
  put(key: string, body: Buffer, sha256: string): Promise<{ versionId: string | null }>;
  head(key: string, versionId: string): Promise<StoredObjectHead | null>;
  get(key: string, versionId: string): Promise<Buffer | null>;
}

/** Acceso a `p3_model_registry`. Las transiciones son escrituras condicionales. */
export interface ModelRegistryRepository {
  /** `false` (sin escribir) si el semver ya existe en el namespace. */
  insertDraft(entry: RegistryEntry): Promise<boolean>;
  find(namespace: RegistryNamespace, semver: string): Promise<RegistryEntry | null>;
  list(namespace: RegistryNamespace): Promise<RegistryEntry[]>;
  /** Solo `draft` sin VersionId. */
  markUploaded(namespace: RegistryNamespace, semver: string, versionId: string): Promise<boolean>;
  /** Solo `draft` con ese VersionId. */
  markPublished(
    namespace: RegistryNamespace,
    semver: string,
    versionId: string,
    at: Date,
  ): Promise<boolean>;
  /** Solo `draft`. */
  markFailed(
    namespace: RegistryNamespace,
    semver: string,
    reason: RegistryFailureReason,
    detail: string,
  ): Promise<boolean>;
}

export interface RegisterModelInput {
  semver: string;
  mlflow_run_id: string;
  manifest_hash: string;
  dvc_release: string;
  dvc_release_hash: string;
  sha256: string;
  size_bytes: number;
}

export interface RegistryOutcome {
  model: ModelVersion;
  failure: { reason: RegistryFailureReason; detail: string } | null;
}

export type AuditResult =
  | { ok: true }
  | { ok: false; reason: RegistryFailureReason; detail: string };

export interface ModelRegistryService {
  register(input: RegisterModelInput): Promise<ModelVersion>;
  upload(semver: string, body: Buffer): Promise<RegistryOutcome>;
  verify(semver: string): Promise<RegistryOutcome>;
  audit(semver: string): Promise<AuditResult>;
  list(): Promise<ModelsResponse>;
}

type Failure = { reason: RegistryFailureReason; detail: string };

const sizeSchema = z.number().int().positive();

export function createModelRegistryService(deps: {
  repo: ModelRegistryRepository;
  store: ModelObjectStore;
  namespace: RegistryNamespace;
  now?: () => Date;
}): ModelRegistryService {
  const { repo, store, namespace } = deps;
  const now = deps.now ?? (() => new Date());

  /** Un fallo del storage no es evidencia de nada: 503 y el registro no cambia. */
  async function storage<T>(operation: () => Promise<T>): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new ServiceUnavailableError(`Storage de modelos no disponible: ${detail}`);
    }
  }

  async function load(semver: string): Promise<RegistryEntry> {
    const entry = await repo.find(namespace, semver);
    if (!entry) throw new NotFoundError(`La versión ${semver} no está registrada`);
    return entry;
  }

  async function outcome(semver: string, failure: Failure | null): Promise<RegistryOutcome> {
    return { model: toModelVersion(await load(semver)), failure };
  }

  async function fail(entry: RegistryEntry, failure: Failure): Promise<RegistryOutcome> {
    if (!(await repo.markFailed(namespace, entry.semver, failure.reason, failure.detail))) {
      throw new ConflictError(`La versión ${entry.semver} cambió de estado; no se marcó failed`);
    }
    return outcome(entry.semver, failure);
  }

  function requireDraft(entry: RegistryEntry) {
    if (entry.status !== 'draft') {
      throw new ConflictError(`La versión ${entry.semver} ya está ${entry.status}`);
    }
  }

  /** head + get por VersionId: el objeto guardado debe ser exactamente el registrado. */
  async function inspect(entry: RegistryEntry, versionId: string): Promise<Failure | null> {
    const where = `${entry.s3_bucket}/${entry.s3_key}@${versionId}`;
    const head = await storage(() => store.head(entry.s3_key, versionId));
    if (head === null) return { reason: 'object_missing', detail: `No existe ${where}` };
    if (head.versionId !== versionId) {
      return {
        reason: 'version_mismatch',
        detail: `HEAD de ${where} devolvió VersionId ${head.versionId}`,
      };
    }
    if (head.size !== entry.size_bytes) {
      return {
        reason: 'size_mismatch',
        detail: `${where} mide ${head.size} B; registrado ${entry.size_bytes} B`,
      };
    }
    if (head.sha256 !== entry.sha256) {
      return {
        reason: 'sha256_mismatch',
        detail: `Metadato sha256 de ${where} = ${head.sha256}; registrado ${entry.sha256}`,
      };
    }
    const body = await storage(() => store.get(entry.s3_key, versionId));
    if (body === null) return { reason: 'object_missing', detail: `GET de ${where} sin objeto` };
    const actual = sha256Hex(body);
    if (actual !== entry.sha256) {
      return {
        reason: 'sha256_mismatch',
        detail: `SHA-256 del contenido de ${where} = ${actual}; registrado ${entry.sha256}`,
      };
    }
    return null;
  }

  return {
    async register(input) {
      const entry: RegistryEntry = {
        namespace,
        semver: input.semver,
        mlflow_run_id: input.mlflow_run_id,
        manifest_hash: input.manifest_hash,
        dvc_release: input.dvc_release,
        dvc_release_hash: input.dvc_release_hash,
        s3_bucket: store.bucket,
        s3_key: modelObjectKey(input.semver),
        version_id: null,
        sha256: input.sha256,
        size_bytes: input.size_bytes,
        status: 'draft',
        failure_reason: null,
        failure_detail: null,
        published_at: null,
      };
      const contract = modelVersionSchema.safeParse(toModelVersion(entry));
      if (!contract.success) {
        throw new ValidationError(`Versión inválida: ${z.prettifyError(contract.error)}`);
      }
      if (!sizeSchema.safeParse(input.size_bytes).success) {
        throw new ValidationError('size_bytes debe ser un entero positivo');
      }
      if (!(await repo.insertDraft(entry))) {
        throw new ConflictError(`La versión ${input.semver} ya existe: el semver es inmutable`);
      }
      return toModelVersion(entry);
    },

    async upload(semver, body) {
      const entry = await load(semver);
      requireDraft(entry);
      if (entry.version_id !== null) {
        throw new ConflictError(`La versión ${semver} ya se subió (VersionId ${entry.version_id})`);
      }
      if (body.length !== entry.size_bytes) {
        return fail(entry, {
          reason: 'size_mismatch',
          detail: `El objeto mide ${body.length} B; registrado ${entry.size_bytes} B`,
        });
      }
      const actual = sha256Hex(body);
      if (actual !== entry.sha256) {
        return fail(entry, {
          reason: 'sha256_mismatch',
          detail: `SHA-256 del objeto = ${actual}; registrado ${entry.sha256}`,
        });
      }
      const { versionId } = await storage(() => store.put(entry.s3_key, body, entry.sha256));
      if (versionId === null) {
        return fail(entry, {
          reason: 'version_id_missing',
          detail: `${store.bucket} no devolvió VersionId (bucket sin versioning)`,
        });
      }
      if (!(await repo.markUploaded(namespace, semver, versionId))) {
        throw new ConflictError(`La versión ${semver} cambió de estado durante la subida`);
      }
      return outcome(semver, null);
    },

    async verify(semver) {
      const entry = await load(semver);
      requireDraft(entry);
      const versionId = entry.version_id;
      if (versionId === null) {
        throw new ConflictError(`La versión ${semver} todavía no tiene objeto subido`);
      }
      const failure = await inspect(entry, versionId);
      if (failure) return fail(entry, failure);
      if (!(await repo.markPublished(namespace, semver, versionId, now()))) {
        throw new ConflictError(`La versión ${semver} cambió de estado; no se publicó`);
      }
      return outcome(semver, null);
    },

    async audit(semver) {
      const entry = await load(semver);
      if (entry.status !== 'published' || entry.version_id === null) {
        throw new ConflictError(
          `Solo se audita una versión publicada (${semver}: ${entry.status})`,
        );
      }
      const failure = await inspect(entry, entry.version_id);
      return failure ? { ok: false, ...failure } : { ok: true };
    },

    async list() {
      const rows = await repo.list(namespace);
      rows.sort((a, b) => compareSemver(a.semver, b.semver));
      return modelsResponseSchema.parse({ models: rows.map(toModelVersion) });
    },
  };
}
