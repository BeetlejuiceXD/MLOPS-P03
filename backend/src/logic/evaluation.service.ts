/**
 * D04-05 — API de evaluación y exportación por muestra.
 *
 * La evaluación la calcula y guarda el productor Python (`app/evaluation`, motor de
 * D03-05) en `p3_evaluation`; aquí solo se LEE, siempre detrás de las guardas de D04-04:
 *
 * - `evaluation()`: `blocked` mientras no exista MODEL SELECTION CLOSED; cerrada, la
 *   evaluación `ready` del namespace del servicio, o `pending` (D05-05) si todavía no
 *   existe: resultado ausente, distinto del bloqueo y de un fallo (503).
 * - `predictions()`: 409 antes del cierre (no se lee nada), 404 sin evaluación.
 *
 * Antes de servir, lo guardado se contrasta con el cierre persistido y entre sí: mismo
 * candidato, `closed_at` y manifest; mismo `evaluated_at` y orden de clases;
 * `test_split_hash` de los crop_id; y la matriz reconstruida desde las predicciones igual
 * a la reportada (que, con los dos contratos, implica el mismo `n_test`). Cualquier diferencia es 503 con el motivo: no se sirve una evaluación
 * incoherente.
 *
 * `server.ts` monta SOLO el namespace `official`; `synthetic` (recorridos de prueba con
 * predicciones conocidas) existe para los tests y nunca se presenta como oficial.
 */
import { createHash } from 'node:crypto';
import { NotFoundError, ServiceUnavailableError } from './errors.js';
import type { ClosedSelection, ModelSelectionService } from './model-selection.service.js';
import {
  type EvaluationPredictions,
  type EvaluationResponse,
  type evaluationNamespaces,
  evaluationPredictionsSchema,
  evaluationResponseSchema,
} from './p3.contracts.js';

export type EvaluationNamespace = (typeof evaluationNamespaces)[number];

/** Fila de `p3_evaluation`: los dos JSON tal como los guardó el productor. */
export interface StoredEvaluation {
  evaluation: unknown;
  predictions: unknown;
}

/** Acceso a `p3_evaluation`. La implementación real vive en `data`. */
export interface EvaluationRepository {
  read(namespace: EvaluationNamespace): Promise<StoredEvaluation | null>;
}

/** Las guardas reales de D04-04 (`createModelSelectionService`). */
export type EvaluationGuard = Pick<ModelSelectionService, 'requireClosed' | 'blockedEvaluation'>;

export interface EvaluationService {
  evaluation(): Promise<EvaluationResponse>;
  predictions(): Promise<EvaluationPredictions>;
}

type ReadyEvaluation = Extract<EvaluationResponse, { state: 'ready' }>;
type PendingEvaluation = Extract<EvaluationResponse, { state: 'pending' }>;

function missingDetail(namespace: EvaluationNamespace): string {
  return namespace === 'official'
    ? 'La selección está cerrada, pero la evaluación oficial del frozen test aún no existe.'
    : `No hay evaluación guardada en el namespace ${namespace}.`;
}

/** D05-05 — Cerrada sin resultado: identidad de la selección cerrada, nada del test. */
function pendingEvaluation(
  closed: ClosedSelection,
  namespace: EvaluationNamespace,
): PendingEvaluation {
  return {
    state: 'pending',
    namespace,
    reason: 'evaluation_missing',
    selection: {
      candidate_run_id: closed.candidate_run_id,
      metric: 'val_accuracy',
      closed_at: closed.closed_at,
    },
    manifest_hash: closed.manifest_hash,
    detail: missingDetail(namespace),
  };
}

/** Igual que `frozen_test_split_hash` (Python): sha256 del JSON de los crop_id ordenados. */
export function testSplitHash(cropIds: readonly number[]): string {
  const canonical = JSON.stringify([...cropIds].sort((a, b) => a - b));
  return createHash('sha256').update(canonical).digest('hex');
}

/** Matriz desde la exportación: filas = clase real, columnas = predicha, en `labels`. */
export function confusionRows(
  predictions: EvaluationPredictions,
  labels: readonly string[],
): number[][] {
  const rows = labels.map(() => labels.map(() => 0));
  for (const sample of predictions.predictions) {
    const row = rows[labels.indexOf(sample.true_class)];
    const column = labels.indexOf(sample.predicted_class);
    if (row && column >= 0) row[column] = (row[column] ?? 0) + 1;
  }
  return rows;
}

const CSV_META = [
  'namespace',
  'candidate_run_id',
  'manifest_hash',
  'test_split_hash',
  'evaluated_at',
] as const;

/** Una fila por muestra; cada fila lleva sus hashes para ser trazable por sí sola. */
export function predictionsToCsv(predictions: EvaluationPredictions): string {
  const header = [
    ...CSV_META,
    'crop_id',
    'true_class',
    'predicted_class',
    ...predictions.classes.map((name) => `p_${name}`),
  ];
  const meta = CSV_META.map((field) => predictions[field]);
  const lines = predictions.predictions.map((sample) =>
    [
      ...meta,
      sample.crop_id,
      sample.true_class,
      sample.predicted_class,
      ...predictions.classes.map((name) => sample.probabilities[name]),
    ].join(','),
  );
  return `${[header.join(','), ...lines].join('\n')}\n`;
}

function coherenceProblems(
  evaluation: ReadyEvaluation,
  predictions: EvaluationPredictions,
  closed: ClosedSelection,
  namespace: EvaluationNamespace,
): string[] {
  const problems: string[] = [];
  const sameInstant = (a: string, b: string) => Date.parse(a) === Date.parse(b);
  // D05-05: una evaluación de prueba nunca se sirve con otro namespace (p. ej. oficial).
  if (evaluation.namespace !== namespace) {
    problems.push(`la evaluación declara namespace ${evaluation.namespace}, no ${namespace}`);
  }
  if (predictions.namespace !== namespace) {
    problems.push(`la exportación declara namespace ${predictions.namespace}, no ${namespace}`);
  }
  if (
    evaluation.selection.candidate_run_id !== closed.candidate_run_id ||
    predictions.candidate_run_id !== closed.candidate_run_id
  ) {
    problems.push(`el candidato no es el de MODEL SELECTION CLOSED (${closed.candidate_run_id})`);
  }
  if (!sameInstant(evaluation.selection.closed_at, closed.closed_at)) {
    problems.push(`closed_at no es el del cierre persistido (${closed.closed_at})`);
  }
  if (
    evaluation.manifest_hash !== closed.manifest_hash ||
    predictions.manifest_hash !== closed.manifest_hash
  ) {
    problems.push(`el manifest no es el de la selección cerrada (${closed.manifest_hash})`);
  }
  if (!sameInstant(evaluation.evaluated_at, predictions.evaluated_at)) {
    problems.push('evaluated_at distinto entre la evaluación y la exportación');
  }
  const labels = evaluation.confusion_matrix.labels;
  if (labels.join() !== predictions.classes.join()) {
    problems.push('clases distintas entre la evaluación y la exportación');
  }
  const ids = predictions.predictions.map((sample) => sample.crop_id);
  if (testSplitHash(ids) !== predictions.test_split_hash) {
    problems.push('test_split_hash no corresponde a los crop_id exportados');
  }
  const rebuilt = confusionRows(predictions, labels);
  if (JSON.stringify(rebuilt) !== JSON.stringify(evaluation.confusion_matrix.rows)) {
    problems.push('la matriz reconstruida desde las predicciones no es la reportada');
  }
  return problems;
}

export function createEvaluationService(
  repo: EvaluationRepository,
  guard: EvaluationGuard,
  namespace: EvaluationNamespace,
): EvaluationService {
  async function load(): Promise<
    | { evaluation: ReadyEvaluation; predictions: EvaluationPredictions }
    | { evaluation: PendingEvaluation; predictions: null }
  > {
    // Primero la guarda: antes del cierre no se lee ningún resultado del test.
    const closed = await guard.requireClosed();
    const stored = await repo.read(namespace);
    if (!stored) return { evaluation: pendingEvaluation(closed, namespace), predictions: null };
    const evaluation = evaluationResponseSchema.safeParse(stored.evaluation);
    if (!evaluation.success || evaluation.data.state !== 'ready') {
      throw new ServiceUnavailableError(
        'La evaluación guardada no cumple evaluation_response (ready).',
      );
    }
    const predictions = evaluationPredictionsSchema.safeParse(stored.predictions);
    if (!predictions.success) {
      throw new ServiceUnavailableError(
        'La exportación guardada no cumple evaluation_predictions.',
      );
    }
    const problems = coherenceProblems(evaluation.data, predictions.data, closed, namespace);
    if (problems.length > 0) {
      throw new ServiceUnavailableError(`Evaluación guardada incoherente: ${problems.join('; ')}.`);
    }
    return { evaluation: evaluation.data, predictions: predictions.data };
  }

  return {
    async evaluation() {
      const blocked = await guard.blockedEvaluation();
      if (blocked) return blocked;
      return (await load()).evaluation;
    },

    async predictions() {
      const { predictions } = await load();
      if (!predictions) throw new NotFoundError(missingDetail(namespace));
      return predictions;
    },
  };
}
