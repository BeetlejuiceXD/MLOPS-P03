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

import type { z } from 'zod';
import type { RegistryEntry, RegistryFailureReason, RegistryNamespace } from './model-registry.js';
import type { ModelVersion, modelsResponseSchema } from './p3.contracts.js';

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

export function createModelRegistryService(_deps: {
  repo: ModelRegistryRepository;
  store: ModelObjectStore;
  namespace: RegistryNamespace;
  now?: () => Date;
}): ModelRegistryService {
  throw new Error('D04-06: pendiente');
}
