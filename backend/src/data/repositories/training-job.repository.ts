import { and, desc, eq } from 'drizzle-orm';
import { db } from '../db/client.js';
import { type TrainingJobRow, trainingJobLogs, trainingJobs } from '../db/schema.js';

/**
 * D02-05 — Acceso a `training_jobs` y `training_job_logs` (MariaDB).
 * Devuelve filas planas; la capa Logic las valida contra el contrato antes de exponerlas.
 */

export interface TrainingJobInsert {
  task: 'controlled' | 'training';
  datasetVersion: string;
  manifestHash: string;
  config: unknown;
  controlledFailAtEpoch: number | null;
}

/** MariaDB guarda JSON como LONGTEXT: según el driver puede llegar como texto. */
function parseConfig(row: TrainingJobRow): TrainingJobRow {
  return typeof row.config === 'string' ? { ...row, config: JSON.parse(row.config) } : row;
}

export async function insertTrainingJob(job: TrainingJobInsert): Promise<number> {
  const [result] = await db
    .insert(trainingJobs)
    .values({ ...job, status: 'queued' })
    .$returningId();
  if (!result) throw new Error('MariaDB no devolvió el id del job insertado.');
  return result.id;
}

export async function findTrainingJobRow(id: number): Promise<TrainingJobRow | null> {
  const [row] = await db.select().from(trainingJobs).where(eq(trainingJobs.id, id)).limit(1);
  return row ? parseConfig(row) : null;
}

export async function listTrainingJobRows(limit = 200): Promise<TrainingJobRow[]> {
  const rows = await db.select().from(trainingJobs).orderBy(desc(trainingJobs.id)).limit(limit);
  return rows.map(parseConfig);
}

/** Últimas `limit` líneas del job, en orden cronológico. */
export async function listTrainingJobLogRows(jobId: number, limit = 500) {
  const rows = await db
    .select()
    .from(trainingJobLogs)
    .where(eq(trainingJobLogs.jobId, jobId))
    .orderBy(desc(trainingJobLogs.id))
    .limit(limit);
  return rows.reverse();
}

type UpdateResult = [{ affectedRows: number }, unknown];

/** `queued` → `cancelled` en una sola sentencia: si el worker ya lo tomó, no toca nada. */
export async function cancelQueuedTrainingJob(id: number): Promise<boolean> {
  const [result] = (await db
    .update(trainingJobs)
    .set({ status: 'cancelled', finishedAt: new Date() })
    .where(and(eq(trainingJobs.id, id), eq(trainingJobs.status, 'queued')))) as UpdateResult;
  return result.affectedRows === 1;
}

/** Pide cancelar un job en ejecución; el worker la aplica en su siguiente punto de control. */
export async function requestTrainingJobCancel(id: number): Promise<boolean> {
  const [result] = (await db
    .update(trainingJobs)
    .set({ cancelRequested: true })
    .where(and(eq(trainingJobs.id, id), eq(trainingJobs.status, 'running')))) as UpdateResult;
  return result.affectedRows === 1;
}
