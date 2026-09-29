/**
 * D02-05 — Adaptador de `data` (MariaDB) a `TrainingJobRepository`.
 * Separado de training-jobs.service.ts para que el servicio se pruebe sin base de datos.
 */
import {
  cancelQueuedTrainingJob,
  findTrainingJobRow,
  insertTrainingJob,
  listTrainingJobLogRows,
  listTrainingJobRows,
  requestTrainingJobCancel,
  type TrainingJobRow,
} from '../data/index.js';
import type { TrainingConfig } from './p3.contracts.js';
import type { TrainingJobRecord, TrainingJobRepository } from './training-jobs.service.js';

function toRecord(row: TrainingJobRow): TrainingJobRecord {
  return {
    id: row.id,
    task: row.task,
    status: row.status,
    datasetVersion: row.datasetVersion,
    manifestHash: row.manifestHash,
    // Se validó contra el contrato al crear el job; toContract() lo vuelve a validar al leer.
    config: row.config as TrainingConfig,
    controlledFailAtEpoch: row.controlledFailAtEpoch,
    progressEpoch: row.progressEpoch,
    totalEpochs: row.totalEpochs,
    mlflowRunId: row.mlflowRunId,
    error: row.error,
    cancelRequested: row.cancelRequested,
    createdAt: row.createdAt,
    startedAt: row.startedAt,
    finishedAt: row.finishedAt,
  };
}

export const mariaDbTrainingJobRepository: TrainingJobRepository = {
  async insert(job) {
    const id = await insertTrainingJob(job);
    const row = await findTrainingJobRow(id);
    if (!row) throw new Error(`El job ${id} no aparece después de insertarlo.`);
    return toRecord(row);
  },
  async list() {
    return (await listTrainingJobRows()).map(toRecord);
  },
  async findById(id) {
    const row = await findTrainingJobRow(id);
    return row ? toRecord(row) : null;
  },
  async listLogs(id) {
    return (await listTrainingJobLogRows(id)).map((row) => ({
      ts: row.ts,
      level: row.level,
      message: row.message,
    }));
  },
  cancelQueued: cancelQueuedTrainingJob,
  requestCancel: requestTrainingJobCancel,
};
