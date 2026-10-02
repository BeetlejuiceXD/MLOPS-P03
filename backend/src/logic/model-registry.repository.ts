// Módulos directos, no `../data/index.js`: el índice también carga el cliente MinIO de
// imágenes (y exige su configuración), que el registro no usa.
import type { P3ModelRegistryRow } from '../data/db/schema.js';
import {
  findModelRegistryRow,
  insertModelRegistryDraft,
  listModelRegistryRows,
  markModelRegistryFailed,
  markModelRegistryPublished,
  markModelRegistryUploaded,
} from '../data/repositories/model-registry.repository.js';
import type { RegistryEntry, RegistryFailureReason } from './model-registry.js';
import type { ModelRegistryRepository } from './model-registry.service.js';

function toEntry(row: P3ModelRegistryRow): RegistryEntry {
  return {
    namespace: row.namespace,
    semver: row.semver,
    mlflow_run_id: row.mlflowRunId,
    manifest_hash: row.manifestHash,
    dvc_release: row.dvcRelease,
    dvc_release_hash: row.dvcReleaseHash,
    s3_bucket: row.s3Bucket,
    s3_key: row.s3Key,
    version_id: row.versionId,
    sha256: row.sha256,
    size_bytes: row.sizeBytes,
    status: row.status,
    failure_reason: row.failureReason as RegistryFailureReason | null,
    failure_detail: row.failureDetail,
    published_at: row.publishedAt,
  };
}

/** Repositorio real del registro de modelos (D04-06): MariaDB `p3_model_registry`. */
export const mariaDbModelRegistryRepository: ModelRegistryRepository = {
  insertDraft: (entry) =>
    insertModelRegistryDraft({
      namespace: entry.namespace,
      semver: entry.semver,
      mlflowRunId: entry.mlflow_run_id,
      manifestHash: entry.manifest_hash,
      dvcRelease: entry.dvc_release,
      dvcReleaseHash: entry.dvc_release_hash,
      s3Bucket: entry.s3_bucket,
      s3Key: entry.s3_key,
      versionId: entry.version_id,
      sha256: entry.sha256,
      sizeBytes: entry.size_bytes,
      status: entry.status,
      failureReason: entry.failure_reason,
      failureDetail: entry.failure_detail,
      publishedAt: entry.published_at,
    }),
  async find(namespace, semver) {
    const row = await findModelRegistryRow(namespace, semver);
    return row ? toEntry(row) : null;
  },
  async list(namespace) {
    return (await listModelRegistryRows(namespace)).map(toEntry);
  },
  markUploaded: markModelRegistryUploaded,
  markPublished: markModelRegistryPublished,
  markFailed: markModelRegistryFailed,
};
