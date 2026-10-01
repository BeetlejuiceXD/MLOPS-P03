/**
 * D04-04 — Selección por validation, estado persistido y bloqueo del test.
 *
 * Todo con runs SINTÉTICOS que cumplen `experimentRunSchema` (contracts/p3): ningún run
 * real de la campaña, ningún dato del frozen test. El candidato que sale de aquí es
 * preparatorio; MODEL SELECTION CLOSED de la campaña es D05-02. El repositorio en memoria
 * sustituye a MariaDB solo en estos tests de componente.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import express from 'express';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConflictError, ServiceUnavailableError, ValidationError } from '../src/logic/errors.js';
import {
  CAMPAIGN_MATRIX,
  campaignRowOf,
  MIN_COMPARABLE_RUNS,
  type SelectionOutcome,
  type SelectionReference,
  selectCandidate,
} from '../src/logic/model-selection.js';
import {
  createModelSelectionService,
  type ModelSelectionRepository,
  runsAdapterPendingSource,
  type SelectionRecord,
} from '../src/logic/model-selection.service.js';
import {
  evaluationResponseSchema,
  experimentRunSchema,
  manifestSummarySchema,
  TRAINING_CONFIG_DEFAULTS,
  type TrainingConfig,
  trainingConfigSchema,
} from '../src/logic/p3.contracts.js';
import { createModelSelectionRouter } from '../src/ui/model-selection.routes.js';

const FIXTURES = path.resolve('../contracts/p3/fixtures');
const fixture = (contract: string, name: string): unknown =>
  JSON.parse(fs.readFileSync(path.join(FIXTURES, contract, `${name}.json`), 'utf8')).payload;

// --- Runs sintéticos -------------------------------------------------------------------

const IMAGES_MD5 = '22222222222222222222222222222222.dir';
const ANNOTATIONS_MD5 = '11111111111111111111111111111111.dir';
const RELEASE_HASH = createHash('sha256').update(`${IMAGES_MD5}:${ANNOTATIONS_MD5}`).digest('hex');
const MANIFEST_HASH = 'd'.repeat(64);
const REFERENCE: SelectionReference = {
  dataset_version: 'v0.1.1',
  manifest_hash: MANIFEST_HASH,
  dvc_release_hash: RELEASE_HASH,
};

interface RunOptions {
  id: string; // un carácter hex: run_id = 32 veces ese carácter (o un run_id completo)
  row?: number;
  acc?: number;
  f1?: number;
  loss?: number;
  status?: 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED';
  start?: string;
  config?: Partial<TrainingConfig>;
  tags?: Record<string, unknown>;
}

const runId = (id: string) => (id.length === 32 ? id : id.repeat(32));

function matrixConfig(row: number): TrainingConfig {
  const entry = CAMPAIGN_MATRIX[row - 1];
  if (!entry) throw new Error(`La matriz no tiene fila ${row}`);
  return entry.config;
}

/** Run con 3 épocas; el mejor checkpoint es la época 2 (NO la última), como en la campaña. */
function makeRun(options: RunOptions): Record<string, unknown> {
  const {
    id,
    row = 1,
    acc = 0.86,
    f1 = 0.85,
    loss = 0.38,
    status = 'FINISHED',
    start = '2026-10-01T10:00:00Z',
  } = options;
  const base = matrixConfig(row);
  const params = { ...base, ...options.config };
  const finished = status !== 'RUNNING';
  return {
    run_id: runId(id),
    experiment_name: 'p3-cnn-classifier',
    status,
    start_time: start,
    end_time: finished ? '2026-10-01T11:00:00Z' : null,
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
      job_id: 1,
      ...options.tags,
    },
    // Un run FAILED/KILLED puede traer resumen parcial (el contrato lo permite): que
    // tenga métricas no lo vuelve elegible.
    summary: finished
      ? { best_epoch: 2, best_val_accuracy: acc, best_val_macro_f1: f1, best_val_loss: loss }
      : null,
    history: [
      epoch(1, acc - 0.1, f1 - 0.1, loss + 0.2),
      epoch(2, acc, f1, loss),
      epoch(3, acc - 0.02, f1 - 0.02, loss + 0.05),
    ],
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
function campaign(n = 12): Record<string, unknown>[] {
  return CAMPAIGN_MATRIX.slice(0, n).map((entry, i) =>
    makeRun({ id: (i + 1).toString(16), row: entry.row, acc: 0.8 + i * 0.001 }),
  );
}

const ids = (outcome: SelectionOutcome) => outcome.ranking.map((run) => run.run_id[0]);
const reasonOf = (outcome: SelectionOutcome, id: string) =>
  outcome.excluded.find((run) => run.run_id === runId(id))?.reason;

// --- Matriz congelada ------------------------------------------------------------------

describe('matriz OFAT congelada (#33)', () => {
  it('tiene 12 configuraciones válidas, distintas, con seeds 7/21/77', () => {
    expect(CAMPAIGN_MATRIX.map((entry) => entry.row)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
    for (const entry of CAMPAIGN_MATRIX) trainingConfigSchema.parse(entry.config);
    const keys = new Set(CAMPAIGN_MATRIX.map((entry) => JSON.stringify(entry.config)));
    expect(keys.size).toBe(12);
    expect(CAMPAIGN_MATRIX.map((entry) => entry.config.seed)).toEqual([
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 21, 77,
    ]);
  });

  it('cada fila cambia respecto al baseline solo los ejes que declara', () => {
    const baseline = matrixConfig(1);
    expect(baseline).toEqual({ ...TRAINING_CONFIG_DEFAULTS, seed: 7 });
    const changed = (config: TrainingConfig) =>
      (Object.keys(config) as (keyof TrainingConfig)[]).filter((k) => config[k] !== baseline[k]);
    expect(CAMPAIGN_MATRIX.map((entry) => changed(entry.config))).toEqual([
      [],
      ['trainable_layers'],
      ['learning_rate'],
      ['optimizer'],
      ['batch_size'],
      ['max_epochs'],
      ['image_size'],
      ['hidden_layers'],
      ['dropout'],
      ['hidden_layers', 'dropout'],
      ['seed'],
      ['seed'],
    ]);
  });

  it('reconoce la fila por la config completa y rechaza cualquier desviación', () => {
    for (const entry of CAMPAIGN_MATRIX) expect(campaignRowOf(entry.config)).toBe(entry.row);
    const baseline = matrixConfig(1);
    // Smoke de D03-04 (max_epochs=10 por defecto) y desvíos en ejes que la matriz no barre.
    expect(campaignRowOf({ ...baseline, max_epochs: 10 })).toBeNull();
    expect(campaignRowOf({ ...baseline, patience: 3 })).toBeNull();
    expect(campaignRowOf({ ...baseline, augmentation: false })).toBeNull();
    expect(campaignRowOf({ ...baseline, seed: 42 })).toBeNull();
    expect(campaignRowOf({ ...baseline, seed: 21, optimizer: 'sgd' })).toBeNull();
  });
});

// --- Ranking ------------------------------------------------------------------------------

describe('ranking solo por validation', () => {
  it('ordena por val_accuracy aunque macro-F1 y loss favorezcan al otro', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'a', row: 1, acc: 0.86, f1: 0.99, loss: 0.01 }),
        makeRun({ id: 'b', row: 2, acc: 0.8601, f1: 0.5, loss: 0.9 }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['b', 'a']);
    expect(outcome.candidate?.run_id).toBe(runId('b'));
  });

  it('accuracy igual a 4 decimales: decide macro-F1, aunque la accuracy cruda sea menor', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'a', row: 1, acc: 0.86004, f1: 0.85, loss: 0.1 }),
        makeRun({ id: 'b', row: 2, acc: 0.86001, f1: 0.8501, loss: 0.9 }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['b', 'a']);
  });

  it('accuracy y macro-F1 iguales a 4 decimales: gana la MENOR val_loss', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'a', row: 1, acc: 0.86, f1: 0.85004, loss: 0.3801 }),
        makeRun({ id: 'b', row: 2, acc: 0.86, f1: 0.85001, loss: 0.38 }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['b', 'a']);
  });

  it('todo igual a 4 decimales: gana el menor run_id', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'b', row: 1, acc: 0.86, f1: 0.85, loss: 0.38001 }),
        makeRun({ id: 'a', row: 2, acc: 0.86, f1: 0.85, loss: 0.38004 }),
        makeRun({ id: 'c', row: 3, acc: 0.86, f1: 0.85, loss: 0.38 }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['a', 'b', 'c']);
  });

  it('usa las métricas del mejor checkpoint (summary), no las de la última época', () => {
    // a: mejor época 0.90, última 0.88. b: 0.89 sostenido. Gana a por su mejor checkpoint.
    const outcome = selectCandidate(
      [makeRun({ id: 'a', row: 1, acc: 0.9 }), makeRun({ id: 'b', row: 2, acc: 0.89 })],
      REFERENCE,
    );
    expect(outcome.candidate).toMatchObject({
      run_id: runId('a'),
      best_epoch: 2,
      val_accuracy: 0.9,
      campaign_row: 1,
    });
  });

  it('el resultado no depende del orden de entrada', () => {
    const runs = campaign();
    const a = selectCandidate(runs, REFERENCE);
    const b = selectCandidate([...runs].reverse(), REFERENCE);
    expect(b.ranking).toEqual(a.ranking);
    expect(b.outcome_hash).toBe(a.outcome_hash);
    expect(a.outcome_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it('campaña completa: candidato = mejor validation, 12 filas, lista para cerrar', () => {
    const outcome = selectCandidate(campaign(), REFERENCE);
    expect(outcome.candidate?.run_id).toBe(runId('c')); // fila 12, acc 0.811
    expect(outcome.campaign_rows).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(outcome.ready_to_close).toBe(true);
    expect(outcome.excluded).toEqual([]);
  });

  it(`con menos de ${MIN_COMPARABLE_RUNS} filas distintas hay candidato pero no se puede cerrar`, () => {
    const nine = selectCandidate(campaign(MIN_COMPARABLE_RUNS - 1), REFERENCE);
    expect(nine.candidate).not.toBeNull();
    expect(nine.ready_to_close).toBe(false);
    expect(selectCandidate(campaign(MIN_COMPARABLE_RUNS), REFERENCE).ready_to_close).toBe(true);
  });

  it('sin runs elegibles no hay candidato', () => {
    const outcome = selectCandidate([makeRun({ id: 'a', status: 'FAILED' })], REFERENCE);
    expect(outcome.candidate).toBeNull();
    expect(outcome.ranking).toEqual([]);
    expect(outcome.ready_to_close).toBe(false);
  });

  it('un run_id repetido es un error de la fuente, no un empate', () => {
    expect(() =>
      selectCandidate([makeRun({ id: 'a', row: 1 }), makeRun({ id: 'a', row: 2 })], REFERENCE),
    ).toThrow(ValidationError);
  });
});

// --- Elegibilidad ---------------------------------------------------------------------

describe('solo runs válidos y comparables', () => {
  const withTestMetric = () => {
    const run = makeRun({ id: '1', row: 2 });
    (run.summary as Record<string, unknown>).test_accuracy = 0.99;
    return run;
  };

  it.each([
    ['en curso', () => makeRun({ id: '1', row: 2, status: 'RUNNING' }), 'not_finished'],
    ['fallido', () => makeRun({ id: '1', row: 2, status: 'FAILED' }), 'not_finished'],
    ['cancelado', () => makeRun({ id: '1', row: 2, status: 'KILLED' }), 'not_finished'],
    [
      'smoke (max_epochs=10, fuera de la matriz)',
      () => makeRun({ id: '1', row: 1, config: { max_epochs: 10 } }),
      'outside_campaign_matrix',
    ],
    [
      'seed fuera de la fila (42 es la del split)',
      () => makeRun({ id: '1', row: 1, config: { seed: 42 } }),
      'outside_campaign_matrix',
    ],
    [
      'otro manifest',
      () => makeRun({ id: '1', row: 2, tags: { manifest_hash: 'e'.repeat(64) } }),
      'manifest_mismatch',
    ],
    [
      'otro release',
      () => makeRun({ id: '1', row: 2, tags: { dvc_release: 'v0.1.0' } }),
      'release_mismatch',
    ],
    [
      'release_hash de otro dataset',
      () => makeRun({ id: '1', row: 2, tags: { dvc_release_hash: 'f'.repeat(64) } }),
      'release_mismatch',
    ],
    [
      'release_hash que no es sha256(images:annotations)',
      () =>
        makeRun({
          id: '1',
          row: 2,
          tags: { dvc_images_md5: '33333333333333333333333333333333.dir' },
        }),
      'provenance_inconsistent',
    ],
    ['con métricas de test', withTestMetric, 'invalid_contract'],
    [
      'FINISHED sin resumen',
      () => ({ ...makeRun({ id: '1', row: 2 }), summary: null }),
      'invalid_contract',
    ],
  ])('excluye un run %s', (_name, build, reason) => {
    const run = build();
    const outcome = selectCandidate([makeRun({ id: 'a', row: 1, acc: 0.5 }), run], REFERENCE);
    expect(ids(outcome)).toEqual(['a']);
    expect(outcome.excluded).toHaveLength(1);
    expect(outcome.excluded[0]?.reason).toBe(reason);
    expect(outcome.excluded[0]?.detail).not.toBe('');
  });

  it('un run fuera de contrato se excluye aunque traiga mejor accuracy', () => {
    const invalid = makeRun({ id: '1', row: 2, acc: 0.99 });
    delete (invalid.tags as Record<string, unknown>).git_commit;
    expect(experimentRunSchema.safeParse(invalid).success).toBe(false);
    const outcome = selectCandidate([makeRun({ id: 'a', row: 1, acc: 0.5 }), invalid], REFERENCE);
    expect(outcome.candidate?.run_id).toBe(runId('a'));
    expect(reasonOf(outcome, '1')).toBe('invalid_contract');
  });

  it('una fila repetida cuenta una vez: la corrida más temprana; las demás se excluyen', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'b', row: 1, acc: 0.95, start: '2026-10-01T12:00:00Z' }),
        makeRun({ id: 'a', row: 1, acc: 0.8, start: '2026-10-01T09:00:00Z' }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['a']);
    expect(reasonOf(outcome, 'b')).toBe('duplicate_campaign_row');
    expect(outcome.campaign_rows).toEqual([1]);
  });

  it('un run fallido no ocupa la fila: la repetición terminada sí cuenta', () => {
    const outcome = selectCandidate(
      [
        makeRun({ id: 'a', row: 1, status: 'FAILED', start: '2026-10-01T09:00:00Z' }),
        makeRun({ id: 'b', row: 1, start: '2026-10-01T12:00:00Z' }),
      ],
      REFERENCE,
    );
    expect(ids(outcome)).toEqual(['b']);
    expect(reasonOf(outcome, 'a')).toBe('not_finished');
  });

  it('los runs sintéticos de este archivo cumplen el contrato real de runs', () => {
    for (const run of campaign()) expect(experimentRunSchema.safeParse(run).success).toBe(true);
  });
});

// --- Servicio: estado persistido y guardas ------------------------------------------------

class InMemorySelectionRepository implements ModelSelectionRepository {
  record: SelectionRecord = { status: 'open', outcome: null, proposedAt: null, closedAt: null };
  writes = 0;

  async read(): Promise<SelectionRecord> {
    return structuredClone(this.record);
  }

  async saveCandidate(outcome: SelectionOutcome, at: Date): Promise<boolean> {
    if (this.record.status === 'closed') return false;
    this.writes += 1;
    this.record = {
      status: 'candidate',
      outcome: structuredClone(outcome),
      proposedAt: at,
      closedAt: null,
    };
    return true;
  }

  async close(outcomeHash: string, at: Date): Promise<boolean> {
    if (this.record.status !== 'candidate' || this.record.outcome?.outcome_hash !== outcomeHash) {
      return false;
    }
    this.writes += 1;
    this.record = { ...this.record, status: 'closed', closedAt: at };
    return true;
  }
}

const frozenManifest = () => fixture('manifest_summary', 'valid-frozen') as Record<string, unknown>;

describe('servicio de selección', () => {
  let repo: InMemorySelectionRepository;
  let runs: Record<string, unknown>[];
  let manifest: Record<string, unknown>;
  let now: Date;
  let listCalls: number;

  const service = () =>
    createModelSelectionService(
      repo,
      {
        list: async () => {
          listCalls += 1;
          return { experiment_name: 'p3-cnn-classifier', runs };
        },
      },
      { manifest: async () => manifest },
      () => now,
    );

  beforeEach(() => {
    repo = new InMemorySelectionRepository();
    runs = campaign();
    manifest = {
      ...frozenManifest(),
      manifest_hash: MANIFEST_HASH,
      dataset_version: 'v0.1.1',
      dvc_release_hash: RELEASE_HASH,
    };
    expect(manifestSummarySchema.safeParse(manifest).success).toBe(true);
    now = new Date('2026-10-01T12:00:00Z');
    listCalls = 0;
  });

  it('empieza abierta, sin candidato, y Evaluation responde bloqueado por contrato', async () => {
    const state = await service().state();
    expect(state).toMatchObject({ status: 'open', candidate: null, closed_at: null });
    const blocked = await service().blockedEvaluation();
    expect(evaluationResponseSchema.parse(blocked)).toMatchObject({
      state: 'blocked',
      reason: 'model_selection_open',
    });
  });

  it('propone el candidato y lo persiste: otra instancia (reinicio) lee lo mismo', async () => {
    const proposed = await service().propose();
    expect(proposed).toMatchObject({
      status: 'candidate',
      ready_to_close: true,
      proposed_at: '2026-10-01T12:00:00.000Z',
      closed_at: null,
    });
    expect(proposed.candidate?.run_id).toBe(runId('c'));
    expect(proposed.reference).toEqual(REFERENCE);
    expect(await service().state()).toEqual(proposed);
  });

  it('un candidato preparatorio NO desbloquea el test', async () => {
    // Por el motivo correcto: no basta que otra comprobación (p. ej. closed_at vacío)
    // también lo rechace.
    const notClosed = /MODEL SELECTION CLOSED no existe/;
    await expect(service().requireClosed()).rejects.toThrow(ConflictError);
    await expect(service().requireClosed()).rejects.toThrow(notClosed);
    await service().propose();
    await expect(service().requireClosed()).rejects.toThrow(ConflictError);
    await expect(service().requireClosed()).rejects.toThrow(notClosed);
    expect(evaluationResponseSchema.parse(await service().blockedEvaluation()).state).toBe(
      'blocked',
    );
  });

  it('se puede re-proponer mientras no esté cerrada (p. ej. llegó un run nuevo)', async () => {
    await service().propose();
    runs = [
      ...runs,
      makeRun({ id: `${'f'.repeat(31)}0`, row: 1, acc: 0.99, start: '2026-10-02T00:00:00Z' }),
    ];
    // Fila 1 repetida y más tardía: no cambia el candidato.
    expect((await service().propose()).candidate?.run_id).toBe(runId('c'));
    expect(repo.writes).toBe(2);
  });

  it('cierra con el candidato confirmado y desbloquea el acceso al test', async () => {
    await service().propose();
    now = new Date('2026-10-01T13:00:00Z');
    const closed = await service().close(runId('c'));
    expect(closed).toMatchObject({ status: 'closed', closed_at: '2026-10-01T13:00:00.000Z' });
    expect(await service().requireClosed()).toEqual({
      candidate_run_id: runId('c'),
      closed_at: '2026-10-01T13:00:00.000Z',
      manifest_hash: MANIFEST_HASH,
    });
    expect(await service().blockedEvaluation()).toBeNull();
  });

  it('después del cierre se rechaza reseleccionar y volver a cerrar', async () => {
    await service().propose();
    await service().close(runId('c'));
    const before = await repo.read();
    const callsAtClose = listCalls;
    runs = [makeRun({ id: 'e', row: 1, acc: 0.999 })];
    await expect(service().propose()).rejects.toThrow(ConflictError);
    await expect(service().close(runId('c'))).rejects.toThrow(ConflictError);
    expect(await repo.read()).toEqual(before);
    // Se rechaza ANTES de recalcular: cerrada, los runs ni se consultan.
    expect(listCalls).toBe(callsAtClose);
  });

  it('sin runs elegibles no propone ni escribe', async () => {
    runs = [
      makeRun({ id: '1', status: 'FAILED' }),
      makeRun({ id: '2', config: { max_epochs: 10 } }),
    ];
    await expect(service().propose()).rejects.toThrow(ConflictError);
    expect(repo.writes).toBe(0);
    expect((await repo.read()).status).toBe('open');
  });

  it('no cierra sin candidato propuesto', async () => {
    await expect(service().close(runId('c'))).rejects.toThrow(ConflictError);
    expect((await repo.read()).status).toBe('open');
  });

  it('no cierra si el run confirmado no es el candidato', async () => {
    await service().propose();
    await expect(service().close(runId('a'))).rejects.toThrow(ConflictError);
    expect((await repo.read()).status).toBe('candidate');
  });

  it.each([undefined, 'abc', 42, 'C'.repeat(32)])(
    'rechaza un run_id inválido (%s)',
    async (value) => {
      await service().propose();
      await expect(service().close(value)).rejects.toThrow(ValidationError);
    },
  );

  it(`no cierra con menos de ${MIN_COMPARABLE_RUNS} configuraciones comparables`, async () => {
    runs = campaign(MIN_COMPARABLE_RUNS - 1);
    const proposed = await service().propose();
    expect(proposed.ready_to_close).toBe(false);
    await expect(service().close(proposed.candidate?.run_id)).rejects.toThrow(ConflictError);
    expect((await repo.read()).status).toBe('candidate');
  });

  it('no cierra si la campaña cambió desde la propuesta (hay que volver a proponer)', async () => {
    await service().propose();
    runs = [...runs.slice(1), makeRun({ id: '1', row: 1, acc: 0.7 })]; // fila 1 con otras métricas
    await expect(service().close(runId('c'))).rejects.toThrow(ConflictError);
    expect((await repo.read()).status).toBe('candidate');
  });

  it('no cierra si otro proceso cambió el estado entre la lectura y la escritura', async () => {
    await service().propose();
    repo.close = async () => false;
    await expect(service().close(runId('c'))).rejects.toThrow(ConflictError);
  });

  it('sin manifest congelado no propone', async () => {
    manifest = { ...manifest, frozen: false };
    await expect(service().propose()).rejects.toThrow(ConflictError);
    expect(repo.writes).toBe(0);
  });

  it.each([
    [
      'manifest caído',
      () => ({ manifest: async () => Promise.reject(new Error('down')) }),
      'manifest',
    ],
    [
      'manifest fuera de contrato',
      () => ({ manifest: async () => ({ frozen: true }) }),
      'manifest',
    ],
    ['runs caídos', () => ({ list: async () => Promise.reject(new Error('down')) }), 'runs'],
    ['runs fuera de contrato', () => ({ list: async () => ({ runs: [] }) }), 'runs'],
  ])('%s: ServiceUnavailableError, sin escribir', async (_name, build, which) => {
    const source = build();
    const svc = createModelSelectionService(
      repo,
      which === 'runs'
        ? (source as { list: () => Promise<unknown> })
        : { list: async () => ({ experiment_name: 'p3-cnn-classifier', runs }) },
      which === 'manifest'
        ? (source as { manifest: () => Promise<unknown> })
        : { manifest: async () => manifest },
      () => now,
    );
    await expect(svc.propose()).rejects.toThrow(ServiceUnavailableError);
    expect(repo.writes).toBe(0);
  });

  it('un run fuera de contrato dentro de la lista se excluye, no tumba la propuesta', async () => {
    const broken = makeRun({ id: '0', row: 1, acc: 0.99 });
    delete (broken.tags as Record<string, unknown>).manifest_hash;
    runs = [...runs.slice(1), broken];
    const proposed = await service().propose();
    expect(proposed.excluded.map((run) => run.reason)).toEqual(['invalid_contract']);
  });
});

// --- API ----------------------------------------------------------------------------------

describe('API de selección y bloqueo de Evaluation', () => {
  let repo: InMemorySelectionRepository;
  let baseUrl: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    repo = new InMemorySelectionRepository();
    const svc = createModelSelectionService(
      repo,
      { list: async () => ({ experiment_name: 'p3-cnn-classifier', runs: campaign() }) },
      {
        manifest: async () => ({
          ...frozenManifest(),
          manifest_hash: MANIFEST_HASH,
          dataset_version: 'v0.1.1',
          dvc_release_hash: RELEASE_HASH,
        }),
      },
      () => new Date('2026-10-01T12:00:00Z'),
    );
    const app = express();
    app.use(express.json());
    app.use(createModelSelectionRouter(svc));
    const server = app.listen(0);
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    close = () => new Promise((resolve) => server.close(() => resolve()));
    return close;
  });

  const call = (method: string, url: string, body?: unknown) =>
    fetch(`${baseUrl}${url}`, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });

  it('GET /evaluation responde blocked (contrato) antes del cierre, aun con candidato', async () => {
    for (const step of ['antes', 'con candidato']) {
      if (step === 'con candidato')
        expect((await call('POST', '/selection/candidate')).status).toBe(200);
      const res = await call('GET', '/evaluation');
      expect(res.status).toBe(200);
      expect(evaluationResponseSchema.parse(await res.json()).state).toBe('blocked');
    }
  });

  it('recorrido completo: estado → candidato → cierre → reselección rechazada', async () => {
    expect(await (await call('GET', '/selection')).json()).toMatchObject({ status: 'open' });

    const proposed = await call('POST', '/selection/candidate');
    expect(proposed.status).toBe(200);
    const { candidate } = await proposed.json();

    expect((await call('POST', '/selection/close', {})).status).toBe(400);
    expect((await call('POST', '/selection/close', { candidate_run_id: runId('a') })).status).toBe(
      409,
    );

    const closed = await call('POST', '/selection/close', { candidate_run_id: candidate.run_id });
    expect(closed.status).toBe(200);
    expect(await closed.json()).toMatchObject({ status: 'closed' });

    expect((await call('POST', '/selection/candidate')).status).toBe(409);
    // Cerrada, Evaluation ya no está bloqueada, pero la evaluación oficial aún no existe (D06-01).
    const evaluation = await call('GET', '/evaluation');
    expect(evaluation.status).toBe(404);
  });

  it('sin el adaptador de runs (D04-01) proponer responde 503 con el motivo, sin escribir', async () => {
    const pendingRepo = new InMemorySelectionRepository();
    const app = express();
    app.use(express.json());
    app.use(
      createModelSelectionRouter(
        createModelSelectionService(pendingRepo, runsAdapterPendingSource, {
          manifest: async () => ({
            ...frozenManifest(),
            manifest_hash: MANIFEST_HASH,
            dataset_version: 'v0.1.1',
            dvc_release_hash: RELEASE_HASH,
          }),
        }),
      ),
    );
    const server = app.listen(0);
    try {
      const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
      const res = await fetch(`${url}/selection/candidate`, { method: 'POST' });
      expect(res.status).toBe(503);
      expect((await res.json()).error).toMatch(/D04-01/);
      expect(pendingRepo.writes).toBe(0);
      const evaluation = await fetch(`${url}/evaluation`);
      expect(evaluationResponseSchema.parse(await evaluation.json()).state).toBe('blocked');
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  });
});
