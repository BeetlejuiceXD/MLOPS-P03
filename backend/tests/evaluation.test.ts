/**
 * D04-05 — API de evaluación y exportación de predicciones por muestra.
 *
 * La evaluación guardada es `tests/fixtures/evaluation-synthetic.json`: la salida EXACTA
 * del productor Python (`app/evaluation/producer.py`, comprobado en
 * `app/tests/test_evaluation_producer.py`) con 6 crops SINTÉTICOS, matriz [[2,1],[1,2]].
 * Ningún dato del frozen test. Las guardas son las REALES de D04-04
 * (`createModelSelectionService`) sobre un repositorio en memoria; MariaDB real es
 * `evaluation.mariadb.test.ts` (job de CI "Jobs persistentes").
 */
import fs from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import express from 'express';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConflictError, NotFoundError, ServiceUnavailableError } from '../src/logic/errors.js';
import {
  confusionRows,
  createEvaluationService,
  type EvaluationNamespace,
  type EvaluationRepository,
  predictionsToCsv,
  type StoredEvaluation,
  testSplitHash,
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
  evaluationPredictionsSchema,
  evaluationResponseSchema,
} from '../src/logic/p3.contracts.js';
import { createEvaluationRouter } from '../src/ui/evaluation.routes.js';

const SYNTHETIC: StoredEvaluation = JSON.parse(
  fs.readFileSync(path.resolve('tests/fixtures/evaluation-synthetic.json'), 'utf8'),
);
const CANDIDATE = 'a'.repeat(32);
const MANIFEST_HASH = 'd'.repeat(64);
const CLOSED_AT = new Date('2026-10-01T13:00:00.456Z');

const outcome = (candidate = CANDIDATE, manifestHash = MANIFEST_HASH): SelectionOutcome => ({
  reference: {
    dataset_version: 'v0.1.1',
    manifest_hash: manifestHash,
    dvc_release_hash: 'e'.repeat(64),
  },
  ranking: [],
  excluded: [],
  candidate: {
    run_id: candidate,
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
  setCandidate() {
    this.record = {
      status: 'candidate',
      outcome: outcome(),
      proposedAt: CLOSED_AT,
      closedAt: null,
    };
  }
  setClosed(candidate = CANDIDATE, manifestHash = MANIFEST_HASH, closedAt = CLOSED_AT) {
    this.record = {
      status: 'closed',
      outcome: outcome(candidate, manifestHash),
      proposedAt: CLOSED_AT,
      closedAt,
    };
  }
}

/** Repositorio en memoria por namespace; cuenta lecturas para probar que la guarda va antes. */
class EvaluationRepo implements EvaluationRepository {
  rows = new Map<EvaluationNamespace, StoredEvaluation>();
  reads = 0;
  async read(namespace: EvaluationNamespace) {
    this.reads += 1;
    const row = this.rows.get(namespace);
    return row ? structuredClone(row) : null;
  }
}

const synthetic = (): StoredEvaluation => structuredClone(SYNTHETIC);
/** La misma exportación declarada como oficial (para probar la ruta oficial). */
const official = (): StoredEvaluation => {
  const row = synthetic();
  (row.predictions as EvaluationPredictions).namespace = 'official';
  return row;
};

function first<T>(items: T[]): T {
  const item = items[0];
  if (item === undefined) throw new Error('lista vacía');
  return item;
}

let selectionRepo: SelectionRepo;
let evaluationRepo: EvaluationRepo;

const guard = () =>
  createModelSelectionService(selectionRepo, runsAdapterPendingSource, {
    manifest: async () => {
      throw new Error('la evaluación no debe leer el manifest');
    },
  });
const service = (namespace: EvaluationNamespace = 'official') =>
  createEvaluationService(evaluationRepo, guard(), namespace);

beforeEach(() => {
  selectionRepo = new SelectionRepo();
  evaluationRepo = new EvaluationRepo();
});

// --- Cálculos y exportación ------------------------------------------------------------

describe('exportación por muestra', () => {
  it('el fixture del productor cumple los dos contratos', () => {
    expect(evaluationResponseSchema.parse(SYNTHETIC.evaluation).state).toBe('ready');
    expect(evaluationPredictionsSchema.parse(SYNTHETIC.predictions).namespace).toBe('synthetic');
  });

  it('la matriz reconstruida desde las predicciones es la reportada', () => {
    const predictions = evaluationPredictionsSchema.parse(SYNTHETIC.predictions);
    expect(confusionRows(predictions, ['cat', 'dog'])).toEqual([
      [2, 1],
      [1, 2],
    ]);
    // Filas = real, columnas = predicha, en el orden de las etiquetas pedidas.
    const skewed = structuredClone(predictions);
    skewed.predictions[0] = { ...first(skewed.predictions), true_class: 'dog' };
    expect(confusionRows(skewed, ['cat', 'dog'])).toEqual([
      [1, 1],
      [2, 2],
    ]);
    expect(confusionRows(skewed, ['dog', 'cat'])).toEqual([
      [2, 2],
      [1, 1],
    ]);
  });

  it('test_split_hash = sha256 del JSON canónico de los crop_id ordenados (igual que Python)', () => {
    expect(testSplitHash([30, 3, 21, 7, 20, 12])).toBe(
      '898c9b54504a279db6e8ce4a86037a0a05b4be37942924398d539af86188dcc8',
    );
  });

  it('CSV: una fila por muestra con verdad, predicción, probabilidades y hashes', () => {
    const csv = predictionsToCsv(evaluationPredictionsSchema.parse(SYNTHETIC.predictions));
    const lines = csv.trimEnd().split('\n');
    expect(lines[0]).toBe(
      'namespace,candidate_run_id,manifest_hash,test_split_hash,evaluated_at,' +
        'crop_id,true_class,predicted_class,p_cat,p_dog',
    );
    const meta = [
      'synthetic',
      CANDIDATE,
      MANIFEST_HASH,
      '898c9b54504a279db6e8ce4a86037a0a05b4be37942924398d539af86188dcc8',
      '2026-10-01T14:00:00.000Z',
    ].join(',');
    expect(lines.slice(1)).toEqual([
      `${meta},3,cat,cat,0.9,0.1`,
      `${meta},7,cat,cat,0.8,0.2`,
      `${meta},12,cat,dog,0.3,0.7`,
      `${meta},20,dog,dog,0.25,0.75`,
      `${meta},21,dog,cat,0.6,0.4`,
      `${meta},30,dog,dog,0.1,0.9`,
    ]);
    expect(csv.endsWith('\n')).toBe(true);
  });
});

// --- Guardas de D04-04 -------------------------------------------------------------------

describe('guardas: nada del test antes de MODEL SELECTION CLOSED', () => {
  it.each(['open', 'candidate'] as const)(
    'selección %s: evaluation blocked y predictions 409, sin leer resultados',
    async (status) => {
      if (status === 'candidate') selectionRepo.setCandidate();
      evaluationRepo.rows.set('official', official());
      const svc = service();
      const evaluation = await svc.evaluation();
      expect(evaluationResponseSchema.parse(evaluation).state).toBe('blocked');
      await expect(svc.predictions()).rejects.toBeInstanceOf(ConflictError);
      await expect(svc.predictions()).rejects.toThrow(/MODEL SELECTION CLOSED/);
      expect(evaluationRepo.reads).toBe(0);
    },
  );

  it('cerrada sin evaluación guardada: 404 en ambas', async () => {
    selectionRepo.setClosed();
    await expect(service().evaluation()).rejects.toBeInstanceOf(NotFoundError);
    await expect(service().predictions()).rejects.toBeInstanceOf(NotFoundError);
  });

  it('cerrada con evaluación oficial coherente: ready y exportación idénticas a lo guardado', async () => {
    selectionRepo.setClosed();
    evaluationRepo.rows.set('official', official());
    expect(await service().evaluation()).toEqual(SYNTHETIC.evaluation);
    expect(await service().predictions()).toEqual(official().predictions);
  });
});

describe('aislamiento del namespace sintético', () => {
  it('una evaluación sintética nunca se sirve como oficial', async () => {
    selectionRepo.setClosed();
    evaluationRepo.rows.set('synthetic', synthetic());
    await expect(service('official').evaluation()).rejects.toBeInstanceOf(NotFoundError);
    await expect(service('official').predictions()).rejects.toBeInstanceOf(NotFoundError);
    // El servicio del namespace sintético sí la ve (recorrido de prueba).
    expect(await service('synthetic').evaluation()).toEqual(SYNTHETIC.evaluation);
    expect((await service('synthetic').predictions()).namespace).toBe('synthetic');
  });

  it('una fila oficial que declara namespace synthetic se rechaza (503)', async () => {
    selectionRepo.setClosed();
    evaluationRepo.rows.set('official', synthetic());
    await expect(service().evaluation()).rejects.toThrow(/namespace/);
  });

  it('el namespace sintético también exige la selección cerrada', async () => {
    evaluationRepo.rows.set('synthetic', synthetic());
    expect((await service('synthetic').evaluation()).state).toBe('blocked');
    await expect(service('synthetic').predictions()).rejects.toBeInstanceOf(ConflictError);
  });
});

// --- Datos incompatibles: no se sirve nada incoherente -----------------------------------

describe('datos guardados incompatibles → 503 con el motivo', () => {
  const tampered: [string, RegExp, (row: StoredEvaluation) => void, (() => void)?][] = [
    [
      'candidato distinto al de MODEL SELECTION CLOSED',
      /candidato/,
      () => undefined,
      () => selectionRepo.setClosed('b'.repeat(32)),
    ],
    [
      'manifest distinto al de la selección',
      /manifest/,
      () => undefined,
      () => selectionRepo.setClosed(CANDIDATE, 'e'.repeat(64)),
    ],
    [
      'closed_at distinto al del cierre',
      /closed_at/,
      () => undefined,
      () => selectionRepo.setClosed(CANDIDATE, MANIFEST_HASH, new Date('2026-10-01T12:00:00Z')),
    ],
    [
      'evaluación de otro candidato (exportación correcta)',
      /candidato/,
      (row) => {
        (row.evaluation as { selection: { candidate_run_id: string } }).selection.candidate_run_id =
          'b'.repeat(32);
      },
    ],
    [
      'evaluación de otro manifest (exportación correcta)',
      /manifest/,
      (row) => {
        (row.evaluation as { manifest_hash: string }).manifest_hash = 'e'.repeat(64);
      },
    ],
    [
      'clases de la exportación en otro orden que la matriz',
      /clases/,
      (row) => {
        const p = row.predictions as EvaluationPredictions;
        p.classes = ['dog', 'cat'];
      },
    ],
    [
      'exportación de otro candidato',
      /candidato/,
      (row) => {
        (row.predictions as EvaluationPredictions).candidate_run_id = 'b'.repeat(32);
      },
    ],
    [
      'exportación de otro manifest',
      /manifest/,
      (row) => {
        (row.predictions as EvaluationPredictions).manifest_hash = 'e'.repeat(64);
      },
    ],
    [
      'evaluated_at distinto entre evaluación y exportación',
      /evaluated_at/,
      (row) => {
        (row.predictions as EvaluationPredictions).evaluated_at = '2026-10-01T15:00:00.000Z';
      },
    ],
    [
      'test_split_hash que no corresponde a los crop_id',
      /test_split_hash/,
      (row) => {
        (row.predictions as EvaluationPredictions).test_split_hash = '0'.repeat(64);
      },
    ],
    [
      'una predicción cambiada: la matriz ya no coincide',
      /matriz/,
      (row) => {
        const p = row.predictions as EvaluationPredictions;
        p.predictions[0] = {
          ...first(p.predictions),
          predicted_class: 'dog',
          probabilities: { cat: 0.1, dog: 0.9 },
        };
      },
    ],
    [
      'evaluación que no cumple el contrato',
      /evaluation_response/,
      (row) => {
        (row.evaluation as { n_test: number }).n_test = 7;
      },
    ],
    [
      'evaluación guardada en estado blocked',
      /evaluation_response/,
      (row) => {
        row.evaluation = { state: 'blocked', reason: 'model_selection_open', detail: 'x' };
      },
    ],
    [
      'exportación que no cumple el contrato',
      /evaluation_predictions/,
      (row) => {
        (row.predictions as EvaluationPredictions).n_test = 7;
      },
    ],
  ];

  it.each(tampered)('%s', async (_name, reason, mutate, setup) => {
    selectionRepo.setClosed();
    setup?.();
    const row = official();
    mutate(row);
    evaluationRepo.rows.set('official', row);
    for (const call of [() => service().evaluation(), () => service().predictions()]) {
      const error = await call().catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ServiceUnavailableError);
      expect((error as Error).message).toMatch(reason);
    }
  });
});

// --- API HTTP ------------------------------------------------------------------------------

describe('API de evaluación', () => {
  let baseUrl: string;

  beforeEach(() => {
    const app = express();
    app.use(createEvaluationRouter(service()));
    const server = app.listen(0);
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    return () => new Promise<void>((resolve) => server.close(() => resolve()));
  });

  const get = (url: string) => fetch(`${baseUrl}${url}`);

  it('antes del cierre: GET /evaluation 200 blocked y /evaluation/predictions 409', async () => {
    evaluationRepo.rows.set('official', official());
    const evaluation = await get('/evaluation');
    expect(evaluation.status).toBe(200);
    expect(evaluationResponseSchema.parse(await evaluation.json()).state).toBe('blocked');
    for (const url of ['/evaluation/predictions', '/evaluation/predictions?format=csv']) {
      const res = await get(url);
      expect(res.status).toBe(409);
      expect((await res.json()).error).toMatch(/MODEL SELECTION CLOSED/);
    }
    expect(evaluationRepo.reads).toBe(0);
  });

  it('cerrada sin evaluación oficial: 404', async () => {
    selectionRepo.setClosed();
    expect((await get('/evaluation')).status).toBe(404);
    expect((await get('/evaluation/predictions')).status).toBe(404);
  });

  it('recorrido cerrado: ready, exportación JSON y CSV descargable', async () => {
    selectionRepo.setClosed();
    evaluationRepo.rows.set('official', official());

    const evaluation = await get('/evaluation');
    expect(evaluation.status).toBe(200);
    expect(await evaluation.json()).toEqual(SYNTHETIC.evaluation);

    const json = await get('/evaluation/predictions');
    expect(json.status).toBe(200);
    const exported = evaluationPredictionsSchema.parse(await json.json());
    expect(exported).toEqual(official().predictions);

    const csv = await get('/evaluation/predictions?format=csv');
    expect(csv.status).toBe(200);
    expect(csv.headers.get('content-type')).toMatch(/^text\/csv/);
    expect(csv.headers.get('content-disposition') ?? '').toMatch(
      /attachment; filename="p3-evaluation-predictions-official\.csv"/,
    );
    expect(await csv.text()).toBe(predictionsToCsv(exported));
  });

  it('formato desconocido: 400 sin tocar la evaluación', async () => {
    selectionRepo.setClosed();
    evaluationRepo.rows.set('official', official());
    const res = await get('/evaluation/predictions?format=xml');
    expect(res.status).toBe(400);
    expect(evaluationRepo.reads).toBe(0);
  });

  it('evaluación guardada incoherente: 503 con el motivo', async () => {
    selectionRepo.setClosed('b'.repeat(32));
    evaluationRepo.rows.set('official', official());
    const res = await get('/evaluation');
    expect(res.status).toBe(503);
    expect((await res.json()).error).toMatch(/candidato/);
  });
});
