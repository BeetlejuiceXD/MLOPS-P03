// Módulos directos, no `../data/index.js`, como el registro de modelos (D04-06).
import type { P3AnnotationQueueRow, P3InferenceRow } from '../data/db/schema.js';
import {
  findAnnotationQueueRow,
  findInferenceRow,
  insertAnnotationQueueRow,
  insertAnnotationQueueWithNewImage,
  insertInferenceRow,
  listAnnotationQueueRows,
  listInferenceRows,
} from '../data/repositories/inference.repository.js';
import {
  DuplicateQueueItemError,
  type InferenceRepository,
  type QueueRecord,
  type StoredInference,
} from './inference.service.js';
import type { InferenceInput } from './p3.contracts.js';

/** MariaDB guarda `json` como LONGTEXT: según el driver llega como texto o ya parseado. */
const fromJson = <T>(value: unknown): T =>
  (typeof value === 'string' ? JSON.parse(value) : value) as T;

function toStored(row: P3InferenceRow): StoredInference {
  return {
    id: row.id,
    created_at: row.createdAt,
    input: fromJson<InferenceInput>(row.input),
    storage_key: row.storageKey,
    predicted_class: row.predictedClass,
    probabilities: fromJson<Record<string, number>>(row.probabilities),
    model: {
      source: row.modelSource,
      package_id: row.packageId,
      format_version: row.formatVersion,
      model_version: row.modelVersion,
      mlflow_run_id: row.mlflowRunId,
      checkpoint_sha256: row.checkpointSha256,
    },
  };
}

function toQueue(row: P3AnnotationQueueRow): QueueRecord {
  return {
    id: row.id,
    inference_id: row.inferenceId,
    image_id: row.imageId,
    annotation_id: row.annotationId,
    created_at: row.createdAt,
  };
}

/** Repositorio real de Inference (D05-07): MariaDB `p3_inference` + `p3_annotation_queue`. */
export const mariaDbInferenceRepository: InferenceRepository = {
  insert: (record) =>
    insertInferenceRow({
      inputKind: record.input.kind,
      input: record.input,
      inputSha256: record.input.sha256,
      storageKey: record.storage_key,
      predictedClass: record.predicted_class,
      probabilities: record.probabilities,
      modelSource: record.model.source,
      packageId: record.model.package_id,
      formatVersion: record.model.format_version,
      modelVersion: record.model.model_version,
      mlflowRunId: record.model.mlflow_run_id,
      checkpointSha256: record.model.checkpoint_sha256,
      createdAt: record.created_at,
    }),
  async find(id) {
    const row = await findInferenceRow(id);
    return row ? toStored(row) : null;
  },
  async list(limit) {
    return (await listInferenceRows(limit)).map(toStored);
  },
  async findQueueItem(inferenceId) {
    const row = await findAnnotationQueueRow(inferenceId);
    return row ? toQueue(row) : null;
  },
  async insertQueueItem(item) {
    const id = await insertAnnotationQueueRow({
      inferenceId: item.inference_id,
      imageId: item.image_id,
      annotationId: item.annotation_id,
      status: 'pending',
      createdAt: item.created_at,
    });
    if (id === null) throw new DuplicateQueueItemError(item.inference_id);
    return { ...item, id };
  },
  async insertQueueItemWithNewImage(image, item) {
    const created = await insertAnnotationQueueWithNewImage(
      { ...image, status: 'pending' },
      {
        inferenceId: item.inference_id,
        annotationId: item.annotation_id,
        status: 'pending',
        createdAt: item.created_at,
      },
    );
    if (created === null) throw new DuplicateQueueItemError(item.inference_id);
    return { ...item, id: created.id, image_id: created.imageId };
  },
  async listQueue() {
    return (await listAnnotationQueueRows()).map(toQueue);
  },
};
