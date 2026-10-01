/**
 * D04-05 — API de evaluación y exportación por muestra (stub Red).
 */
import type { ModelSelectionService } from './model-selection.service.js';
import type {
  EvaluationPredictions,
  EvaluationResponse,
  evaluationNamespaces,
} from './p3.contracts.js';

export type EvaluationNamespace = (typeof evaluationNamespaces)[number];

export interface StoredEvaluation {
  evaluation: unknown;
  predictions: unknown;
}

export interface EvaluationRepository {
  read(namespace: EvaluationNamespace): Promise<StoredEvaluation | null>;
}

export type EvaluationGuard = Pick<ModelSelectionService, 'requireClosed' | 'blockedEvaluation'>;

export interface EvaluationService {
  evaluation(): Promise<EvaluationResponse>;
  predictions(): Promise<EvaluationPredictions>;
}

export function testSplitHash(_cropIds: readonly number[]): string {
  throw new Error('not implemented');
}

export function confusionRows(
  _predictions: EvaluationPredictions,
  _labels: readonly string[],
): number[][] {
  throw new Error('not implemented');
}

export function predictionsToCsv(_predictions: EvaluationPredictions): string {
  throw new Error('not implemented');
}

export function createEvaluationService(
  _repo: EvaluationRepository,
  _guard: EvaluationGuard,
  _namespace: EvaluationNamespace,
): EvaluationService {
  return {
    async evaluation() {
      throw new Error('not implemented');
    },
    async predictions() {
      throw new Error('not implemented');
    },
  };
}
