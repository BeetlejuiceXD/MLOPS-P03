/**
 * D06-05 — Detalle de la evaluación guardada para la página Evaluation.
 *
 * Datos SINTÉTICOS: la evaluación es `tests/fixtures/evaluation-synthetic.json` (salida exacta
 * del productor con 6 crops sintéticos, matriz [[2,1],[1,2]]) y el run cerrado es el fixture
 * compartido `experiment_run_detail/valid-ok`. Prueban el contrato, las guardas y que la API
 * solo lee; ninguna cifra es del frozen test. El cierre con el resultado official es D06-01.
 */
import fs from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import express from 'express';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConflictError, NotFoundError, ServiceUnavailableError } from '../src/logic/errors.js';
import {
  accuracyTarget,
  createEvaluationService,
  type EvaluationNamespace,
  type EvaluationRepository,
  type EvaluationRunSource,
  evaluationExamples,
  type StoredEvaluation,
} from '../src/logic/evaluation.service.js';
import type { SelectionOutcome } from '../src/logic/model-selection.js';
import {
  createModelSelectionService,
  type ModelSelectionRepository,
  runsAdapterPendingSource,
  type SelectionRecord,
} from '../src/logic/model-selection.service.js';
import {
  type EvaluationPredictions,
  type ExperimentRunDetail,
  evaluationDetailsSchema,
  evaluationPredictionsSchema,
} from '../src/logic/p3.contracts.js';
import { createEvaluationRouter } from '../src/ui/evaluation.routes.js';

const read = (file: string) => JSON.parse(fs.readFileSync(path.resolve(file), 'utf8'));
const SYNTHETIC: StoredEvaluation = read('tests/fixtures/evaluation-synthetic.json');
const RUN: ExperimentRunDetail = read(
  '../contracts/p3/fixtures/experiment_run_detail/valid-ok.json',
).payload;
const CANDIDATE = 'a'.repeat(32);
const MANIFEST_HASH = 'd'.repeat(64);
const CLOSED_AT = new Date('2026-10-01T13:00:00.456Z');

const outcome = (): SelectionOutcome => ({
  reference: {
    dataset_version: 'v0.1.1',
    manifest_hash: MANIFEST_HASH,
    dvc_release_hash: 'e'.repeat(64),
  },
  ranking: [],
  excluded: [],
  candidate: {
    run_id: CANDIDATE,
    campaign_row: 1,
    start_time: '2026-10-01T10:00:00Z',
    best_epoch: 2,
    val_accuracy: 0.86,
    val_macro_f1: 0.85,
    val_loss: 0.38,
  },
  campaign_rows: [1],
  ready_to_close: true,
  outcome_hash: '1'.repeat(64),
});

class SelectionRepo implements ModelSelectionRepository {
  record: SelectionRecord = { status: 'open', outcome: null, proposedAt: null, closedAt: null };
  async read() {
    return structuredClone(this.record);
  }
  async saveCandidate() {
    return false;
  }
  async close() {
    return false;
  }
  set(status: 'candidate' | 'closed') {
    this.record = {
      status,
      outcome: outcome(),
      proposedAt: CLOSED_AT,
      closedAt: status === 'closed' ? CLOSED_AT : null,
    };
  }
}

/** Solo `read`: la interfaz no permite escribir. Cuenta lecturas y guarda lo leído. */
class EvaluationRepo implements EvaluationRepository {
  rows = new Map<EvaluationNamespace, StoredEvaluation>();
  reads = 0;
  async read(namespace: EvaluationNamespace) {
    this.reads += 1;
    const row = this.rows.get(namespace);
    return row ? structuredClone(row) : null;
  }
}

class RunSource implements EvaluationRunSource {
  calls: string[] = [];
  detail: ExperimentRunDetail = structuredClone(RUN);
  failure: Error | null = null;
  async getRun(runId: string) {
    this.calls.push(runId);
    if (this.failure) throw this.failure;
    return structuredClone(this.detail);
  }
}

const asNamespace = (namespace: EvaluationNamespace): StoredEvaluation => {
  const row = structuredClone(SYNTHETIC);
  (row.evaluation as { namespace: string }).namespace = namespace;
  (row.predictions as EvaluationPredictions).namespace = namespace;
  return row;
};

let selection: SelectionRepo;
let evaluations: EvaluationRepo;
let runs: RunSource;

const guard = () =>
  createModelSelectionService(selection, runsAdapterPendingSource, {
    manifest: async () => {
      throw new Error('la evaluación no debe leer el manifest');
    },
  });
const service = (namespace: EvaluationNamespace = 'official', source: RunSource | null = runs) =>
  createEvaluationService(evaluations, guard(), namespace, source ?? undefined);

beforeEach(() => {
  selection = new SelectionRepo();
  evaluations = new EvaluationRepo();
  runs = new RunSource();
});

describe('detalle de la evaluación (lo que muestra Evaluation)', () => {
  beforeEach(() => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('official'));
  });

  it('procedencia completa: run y checkpoint de MLflow, manifest y split de lo guardado', async () => {
    const details = await service().details();
    expect(evaluationDetailsSchema.parse(details)).toEqual(details);
    expect(details).toMatchObject({ namespace: 'official', final: true });
    expect(details.provenance).toEqual({
      candidate_run_id: CANDIDATE,
      campaign_row: 1,
      best_epoch: 2,
      checkpoint_sha256: 'f'.repeat(64),
      git_commit: 'c'.repeat(40),
      dataset_version: 'v0.1.1',
      dvc_release_hash: 'e'.repeat(64),
      manifest_hash: MANIFEST_HASH,
      test_split_hash: '898c9b54504a279db6e8ce4a86037a0a05b4be37942924398d539af86188dcc8',
      closed_at: '2026-10-01T13:00:00.456Z',
      evaluated_at: '2026-10-01T14:00:00.000Z',
      n_test: 6,
    });
    expect(runs.calls).toEqual([CANDIDATE]);
  });

  it('umbral con los aciertos de la matriz guardada: 4/6 no llega a 0.85', async () => {
    expect((await service().details()).target).toEqual({
      accuracy: 0.85,
      correct: 4,
      n_test: 6,
      met: false,
    });
  });

  it('ejemplos: aciertos y errores reales de la exportación, por crop_id', async () => {
    const { examples } = await service().details();
    const exported = evaluationPredictionsSchema.parse(SYNTHETIC.predictions).predictions;
    expect(examples.n_correct + examples.n_errors).toBe(exported.length);
    for (const example of [...examples.correct, ...examples.errors]) {
      const source = exported.find((sample) => sample.crop_id === example.crop_id);
      expect(source).toBeDefined();
      expect(example).toEqual({
        crop_id: source?.crop_id,
        true_class: source?.true_class,
        predicted_class: source?.predicted_class,
        confidence: source?.probabilities[example.predicted_class],
      });
    }
    expect(examples.errors.every((e) => e.true_class !== e.predicted_class)).toBe(true);
    expect(examples.correct.every((e) => e.true_class === e.predicted_class)).toBe(true);
  });

  it('synthetic: se sirve con su namespace y nunca como final', async () => {
    evaluations.rows.set('synthetic', asNamespace('synthetic'));
    expect(await service('synthetic').details()).toMatchObject({
      namespace: 'synthetic',
      final: false,
    });
  });
});

describe('cálculos sin redondeo', () => {
  it('umbral con enteros: 85/100 cumple, 84/100 y 1699/2000 (0.8495) no', () => {
    expect(accuracyTarget(85, 100).met).toBe(true);
    expect(accuracyTarget(84, 100).met).toBe(false);
    expect(accuracyTarget(1699, 2000).met).toBe(false);
    expect(accuracyTarget(1700, 2000).met).toBe(true);
    expect(accuracyTarget(57, 67).met).toBe(true); // 0.8507…
    expect(accuracyTarget(56, 66).met).toBe(false); // 0.8484…
  });

  it('ejemplos: orden por confianza descendente y, si empatan, por crop_id', () => {
    const exported = structuredClone(evaluationPredictionsSchema.parse(SYNTHETIC.predictions));
    const p = (cat: number) => ({ cat, dog: Number((1 - cat).toFixed(2)) });
    exported.predictions = [
      { crop_id: 1, true_class: 'cat', predicted_class: 'cat', probabilities: p(0.6) },
      { crop_id: 2, true_class: 'cat', predicted_class: 'cat', probabilities: p(0.9) },
      { crop_id: 3, true_class: 'cat', predicted_class: 'cat', probabilities: p(0.9) },
      { crop_id: 4, true_class: 'dog', predicted_class: 'cat', probabilities: p(0.7) },
      { crop_id: 5, true_class: 'dog', predicted_class: 'cat', probabilities: p(0.95) },
    ];
    const examples = evaluationExamples(exported);
    expect(examples.correct.map((e) => e.crop_id)).toEqual([2, 3, 1]);
    expect(examples.errors.map((e) => e.crop_id)).toEqual([5, 4]);
  });

  it('ejemplos: mayor confianza primero, empate por crop_id, con tope por tipo', () => {
    const exported = evaluationPredictionsSchema.parse(SYNTHETIC.predictions);
    const examples = evaluationExamples(exported, 1);
    expect(examples.correct).toHaveLength(1);
    expect(examples.errors).toHaveLength(1);
    const best = (kind: 'correct' | 'errors') =>
      Math.max(
        ...exported.predictions
          .filter((s) => (s.true_class === s.predicted_class) === (kind === 'correct'))
          .map((s) => s.probabilities[s.predicted_class] ?? 0),
      );
    expect(examples.correct[0]?.confidence).toBe(best('correct'));
    expect(examples.errors[0]?.confidence).toBe(best('errors'));
  });
});

describe('guardas y negativos: nada del test antes de tiempo, nada incoherente', () => {
  it.each(['open', 'candidate'] as const)(
    'selección %s: 409 sin leer la evaluación ni MLflow',
    async (status) => {
      if (status === 'candidate') selection.set('candidate');
      evaluations.rows.set('official', asNamespace('official'));
      await expect(service().details()).rejects.toThrow(ConflictError);
      expect(evaluations.reads).toBe(0);
      expect(runs.calls).toEqual([]);
    },
  );

  it('cerrada sin resultado: 404, no un detalle vacío', async () => {
    selection.set('closed');
    await expect(service().details()).rejects.toThrow(NotFoundError);
    expect(runs.calls).toEqual([]);
  });

  it('un resultado synthetic guardado como official no se sirve (503)', async () => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('synthetic'));
    await expect(service().details()).rejects.toThrow(ServiceUnavailableError);
  });

  it('predicción incompleta (falta un crop): 503', async () => {
    selection.set('closed');
    const row = asNamespace('official');
    (row.predictions as EvaluationPredictions).predictions.pop();
    evaluations.rows.set('official', row);
    await expect(service().details()).rejects.toThrow(/evaluation_predictions/);
  });

  it('el run de MLflow tiene otro manifest: 503 con el motivo', async () => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('official'));
    runs.detail.run.tags.manifest_hash = '9'.repeat(64);
    await expect(service().details()).rejects.toThrow(/otro manifest/);
  });

  it('el run no es un training elegible de la campaña (p. ej. local_test/smoke): 503', async () => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('official'));
    runs.detail.run.params = { ...runs.detail.run.params, learning_rate: 0.123 };
    await expect(service().details()).rejects.toThrow(/no es un training elegible/);
  });

  it('MLflow caído o sin conectar: 503 con el motivo, sin inventar procedencia', async () => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('official'));
    runs.failure = new Error('ECONNREFUSED mlflow:5000');
    const failure = service().details();
    await expect(failure).rejects.toThrow(ServiceUnavailableError);
    await expect(service().details()).rejects.toThrow(
      `No se pudo leer el run cerrado ${CANDIDATE}: ECONNREFUSED`,
    );
    await expect(service('official', null).details()).rejects.toThrow(/MLflow no está conectado/);
  });
});

describe('HTTP y no reejecución', () => {
  async function serve() {
    const app = express();
    app.use(createEvaluationRouter(service()));
    const server = app.listen(0);
    const { port } = server.address() as AddressInfo;
    return { url: `http://127.0.0.1:${port}`, close: () => server.close() };
  }

  it('GET /evaluation/details: 409 antes del cierre, 200 con el contrato después', async () => {
    const http = await serve();
    try {
      expect((await fetch(`${http.url}/evaluation/details`)).status).toBe(409);
      selection.set('closed');
      expect((await fetch(`${http.url}/evaluation/details`)).status).toBe(404);
      evaluations.rows.set('official', asNamespace('official'));
      const response = await fetch(`${http.url}/evaluation/details`);
      expect(response.status).toBe(200);
      expect(evaluationDetailsSchema.parse(await response.json()).provenance.n_test).toBe(6);
    } finally {
      http.close();
    }
  });

  it('refrescar la página solo lee: lo guardado no cambia y nada vuelve a evaluar', async () => {
    selection.set('closed');
    evaluations.rows.set('official', asNamespace('official'));
    const before = structuredClone(evaluations.rows.get('official'));
    const http = await serve();
    try {
      const paths = ['/evaluation', '/evaluation/details', '/evaluation/predictions'];
      const first = await Promise.all(
        paths.map((p) => fetch(`${http.url}${p}`).then((r) => r.json())),
      );
      for (let refresh = 0; refresh < 3; refresh += 1) {
        const again = await Promise.all(
          paths.map((p) => fetch(`${http.url}${p}`).then((r) => r.json())),
        );
        expect(again).toEqual(first);
      }
      // Cada GET es una lectura: 4 recorridos × 3 endpoints. El repositorio no tiene escritura.
      expect(evaluations.reads).toBe(12);
      expect(Object.getOwnPropertyNames(EvaluationRepo.prototype)).toEqual(['constructor', 'read']);
      expect(evaluations.rows.get('official')).toEqual(before);
    } finally {
      http.close();
    }
  });
});
