/**
 * Runs y jobs SINTÉTICOS de la campaña para los tests de selección (D04-04) y del
 * expediente de campaña (D05-02). Cumplen `experimentRunSchema` y `trainingJobSchema`
 * (contracts/p3), pero no son runs reales de la campaña ni traen datos del frozen test.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { CAMPAIGN_MATRIX, type SelectionReference } from '../src/logic/model-selection.js';
import type { TrainingConfig } from '../src/logic/p3.contracts.js';

const FIXTURES = path.resolve('../contracts/p3/fixtures');
export const fixture = (contract: string, name: string): unknown =>
  JSON.parse(fs.readFileSync(path.join(FIXTURES, contract, `${name}.json`), 'utf8')).payload;

export const IMAGES_MD5 = '22222222222222222222222222222222.dir';
export const ANNOTATIONS_MD5 = '11111111111111111111111111111111.dir';
export const RELEASE_HASH = createHash('sha256')
  .update(`${IMAGES_MD5}:${ANNOTATIONS_MD5}`)
  .digest('hex');
export const MANIFEST_HASH = 'd'.repeat(64);
export const REFERENCE: SelectionReference = {
  dataset_version: 'v0.1.1',
  manifest_hash: MANIFEST_HASH,
  dvc_release_hash: RELEASE_HASH,
};

export type RunStatus = 'RUNNING' | 'SCHEDULED' | 'FINISHED' | 'FAILED' | 'KILLED';

export interface RunOptions {
  id: string; // un carácter hex: run_id = 32 veces ese carácter (o un run_id completo)
  row?: number;
  acc?: number;
  f1?: number;
  loss?: number;
  status?: RunStatus;
  start?: string;
  job?: number;
  checkpoint?: string | null;
  config?: Partial<TrainingConfig>;
  tags?: Record<string, unknown>;
}

export const runId = (id: string) => (id.length === 32 ? id : id.repeat(32));

export function matrixConfig(row: number): TrainingConfig {
  const entry = CAMPAIGN_MATRIX[row - 1];
  if (!entry) throw new Error(`La matriz no tiene fila ${row}`);
  return entry.config;
}

/** Run con 3 épocas; el mejor checkpoint es la época 2 (NO la última), como en la campaña. */
export function makeRun(options: RunOptions): Record<string, unknown> {
  const {
    id,
    row = 1,
    acc = 0.86,
    f1 = 0.85,
    loss = 0.38,
    status = 'FINISHED',
    start = '2026-10-01T10:00:00Z',
    job = 1,
  } = options;
  const base = matrixConfig(row);
  const params = { ...base, ...options.config };
  const active = status === 'RUNNING' || status === 'SCHEDULED';
  const checkpoint =
    options.checkpoint !== undefined
      ? options.checkpoint
      : status === 'FINISHED'
        ? 'e'.repeat(64)
        : null;
  const ineligible = [
    ...(status === 'FINISHED' ? [] : [`estado ${status}: el run no terminó`]),
    ...(checkpoint === null ? ['sin checkpoint_sha256: no hay checkpoint verificado'] : []),
  ];
  return {
    run_id: runId(id),
    experiment_name: 'p3-cnn-classifier',
    status,
    start_time: start,
    end_time: active ? null : '2026-10-01T11:00:00Z',
    params,
    tags: {
      git_commit: 'c'.repeat(40),
      dvc_release: 'v0.1.1',
      dvc_images_md5: IMAGES_MD5,
      dvc_annotations_md5: ANNOTATIONS_MD5,
      dvc_release_hash: RELEASE_HASH,
      manifest_version: 'p3-v0.1.1-s42',
      manifest_hash: MANIFEST_HASH,
      classes: ['cat', 'dog'],
      seed: params.seed,
      job_id: job,
      ...options.tags,
    },
    // Un run FAILED/KILLED puede traer resumen parcial (el contrato lo permite): que
    // tenga métricas no lo vuelve elegible.
    summary: active
      ? null
      : { best_epoch: 2, best_val_accuracy: acc, best_val_macro_f1: f1, best_val_loss: loss },
    history: [
      epoch(1, acc - 0.1, f1 - 0.1, loss + 0.2),
      epoch(2, acc, f1, loss),
      epoch(3, acc - 0.02, f1 - 0.02, loss + 0.05),
    ],
    // D04-01: campos que agrega el adaptador MLflow al contrato de cada run.
    checkpoint_sha256: checkpoint,
    campaign_eligible: ineligible.length === 0,
    ineligible_reasons: ineligible,
  };
}

function epoch(n: number, acc: number, f1: number, loss: number) {
  return {
    epoch: n,
    train_loss: 0.4,
    train_accuracy: 0.8,
    val_loss: loss,
    val_accuracy: acc,
    val_macro_f1: f1,
    learning_rate: 0.001,
  };
}

/** Una corrida por cada fila 1..n de la matriz, todas válidas, con métricas distintas. */
export function campaign(n = 12): Record<string, unknown>[] {
  return CAMPAIGN_MATRIX.slice(0, n).map((entry, i) =>
    makeRun({ id: (i + 1).toString(16), row: entry.row, acc: 0.8 + i * 0.001, job: i + 1 }),
  );
}

const JOB_STATUS: Record<RunStatus, string> = {
  RUNNING: 'running',
  SCHEDULED: 'running',
  FINISHED: 'succeeded',
  FAILED: 'failed',
  KILLED: 'cancelled',
};

/**
 * Job de training (contrato `training_job`) que lanzó `run`: mismo id que `tags.job_id`,
 * la config del run como request y el estado equivalente.
 */
export function jobFor(
  run: Record<string, unknown>,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  const tags = run.tags as { job_id: number; manifest_hash: string };
  const status = JOB_STATUS[run.status as RunStatus];
  const active = status === 'running';
  return {
    id: tags.job_id,
    task: 'training',
    status,
    dataset_version: 'v0.1.1',
    manifest_hash: tags.manifest_hash,
    config: run.params,
    progress: null,
    mlflow_run_id: run.run_id,
    error: status === 'failed' ? 'el worker falló' : null,
    cancel_requested: status === 'cancelled',
    created_at: '2026-10-01T09:59:00Z',
    started_at: '2026-10-01T09:59:30Z',
    finished_at: active ? null : '2026-10-01T11:00:00Z',
    ...overrides,
  };
}

/** Job de training que todavía no creó run (en cola). */
export function queuedJob(id: number, row: number): Record<string, unknown> {
  return {
    id,
    task: 'training',
    status: 'queued',
    dataset_version: 'v0.1.1',
    manifest_hash: MANIFEST_HASH,
    config: matrixConfig(row),
    progress: null,
    mlflow_run_id: null,
    error: null,
    cancel_requested: false,
    created_at: '2026-10-01T12:00:00Z',
    started_at: null,
    finished_at: null,
  };
}

export const jobsFor = (runs: Record<string, unknown>[]) => runs.map((run) => jobFor(run));
