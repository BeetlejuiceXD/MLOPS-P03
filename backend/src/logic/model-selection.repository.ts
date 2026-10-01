import {
  closeModelSelection,
  readModelSelection,
  saveModelSelectionCandidate,
} from '../data/index.js';
import type { SelectionOutcome } from './model-selection.js';
import type { ModelSelectionRepository } from './model-selection.service.js';

/** MariaDB guarda JSON como LONGTEXT: según el driver puede llegar como texto. */
function parseOutcome(value: unknown): SelectionOutcome | null {
  if (value === null || value === undefined) return null;
  return (typeof value === 'string' ? JSON.parse(value) : value) as SelectionOutcome;
}

/** Repositorio real de la selección (D04-04): MariaDB `p3_model_selection`. */
export const mariaDbModelSelectionRepository: ModelSelectionRepository = {
  async read() {
    const row = await readModelSelection();
    if (!row) throw new Error('Falta el registro de p3_model_selection (migración 0006).');
    return {
      status: row.status,
      outcome: parseOutcome(row.outcome),
      proposedAt: row.proposedAt,
      closedAt: row.closedAt,
    };
  },
  saveCandidate: (outcome, at) => saveModelSelectionCandidate(outcome, outcome.outcome_hash, at),
  close: (outcomeHash, at) => closeModelSelection(outcomeHash, at),
};
