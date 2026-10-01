/**
 * D04-04 — Estado persistido de la selección y guardas del test (stub Red).
 */

import type { SelectionOutcome } from './model-selection.js';
import type { EvaluationResponse } from './p3.contracts.js';

export type SelectionStatus = 'open' | 'candidate' | 'closed';

export interface SelectionRecord {
  status: SelectionStatus;
  outcome: SelectionOutcome | null;
  proposedAt: Date | null;
  closedAt: Date | null;
}

/** Acceso a `p3_model_selection`. La implementación real vive en `data`. */
export interface ModelSelectionRepository {
  read(): Promise<SelectionRecord>;
  /** Guarda el candidato preparatorio; `false` (sin escribir) si la selección ya está cerrada. */
  saveCandidate(outcome: SelectionOutcome, at: Date): Promise<boolean>;
  /** `candidate` → `closed` en una sola sentencia, solo si el resultado guardado es `outcomeHash`. */
  close(outcomeHash: string, at: Date): Promise<boolean>;
}

/** Runs de MLflow con el contrato `experiment_runs_response` (adaptador de D04-01). */
export interface CampaignRunsSource {
  list(): Promise<unknown>;
}

/** Manifest oficial con el contrato `manifest_summary` (D03-01/D03-03). */
export interface FrozenManifestSource {
  manifest(): Promise<unknown>;
}

export interface SelectionState {
  status: SelectionStatus;
  candidate: SelectionOutcome['candidate'];
  ranking: SelectionOutcome['ranking'];
  excluded: SelectionOutcome['excluded'];
  campaign_rows: number[];
  min_comparable_runs: number;
  ready_to_close: boolean;
  reference: SelectionOutcome['reference'] | null;
  outcome_hash: string | null;
  proposed_at: string | null;
  closed_at: string | null;
}

export interface ClosedSelection {
  candidate_run_id: string;
  closed_at: string;
  manifest_hash: string;
}

export interface ModelSelectionService {
  state(): Promise<SelectionState>;
  propose(): Promise<SelectionState>;
  close(candidateRunId: unknown): Promise<SelectionState>;
  /** Guarda de todo acceso a resultados de test: ConflictError mientras no esté cerrada. */
  requireClosed(): Promise<ClosedSelection>;
  /** `evaluation_response` bloqueado mientras la selección no esté cerrada; `null` si lo está. */
  blockedEvaluation(): Promise<EvaluationResponse | null>;
}

export function createModelSelectionService(
  _repo: ModelSelectionRepository,
  _runs: CampaignRunsSource,
  _manifest: FrozenManifestSource,
  _clock: () => Date = () => new Date(),
): ModelSelectionService {
  const red = async (): Promise<never> => {
    throw new Error('NotImplemented');
  };
  return {
    state: red,
    propose: red,
    close: red,
    requireClosed: red,
    blockedEvaluation: red,
  };
}
