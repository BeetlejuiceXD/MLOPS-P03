/**
 * D05-03 (preparación) — Contrato compartido `selection_state` para `GET /api/selection`
 * (D04-04). El portal lo valida antes de mostrar el candidato en Experiments; aquí se
 * comprueba que lo que devuelve el servicio real de selección cumple ese contrato en sus
 * tres estados. No cambia la lógica de selección de D04-04.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  createModelSelectionService,
  type ModelSelectionRepository,
  runsAdapterPendingSource,
  type SelectionRecord,
} from '../src/logic/model-selection.service.js';
import { selectionStateSchema } from '../src/logic/p3.contracts.js';

const fixture = (name: string) =>
  JSON.parse(
    fs.readFileSync(
      path.resolve('../contracts/p3/fixtures/selection_state', `${name}.json`),
      'utf8',
    ),
  ).payload;

function stateFrom(record: SelectionRecord) {
  const repo: ModelSelectionRepository = {
    read: async () => record,
    saveCandidate: async () => false,
    close: async () => false,
  };
  return createModelSelectionService(repo, runsAdapterPendingSource, {
    manifest: async () => {
      throw new Error('no se usa');
    },
  }).state();
}

/** El registro persistido que produciría cada fixture (outcome + fechas). */
function recordOf(payload: ReturnType<typeof fixture>): SelectionRecord {
  return {
    status: payload.status,
    outcome:
      payload.outcome_hash === null
        ? null
        : {
            reference: payload.reference,
            ranking: payload.ranking,
            excluded: payload.excluded,
            candidate: payload.candidate,
            campaign_rows: payload.campaign_rows,
            ready_to_close: payload.ready_to_close,
            outcome_hash: payload.outcome_hash,
          },
    proposedAt: payload.proposed_at ? new Date(payload.proposed_at) : null,
    closedAt: payload.closed_at ? new Date(payload.closed_at) : null,
  };
}

describe('GET /selection cumple el contrato selection_state', () => {
  it.each(['valid-open', 'valid-candidate', 'valid-closed', 'valid-candidate-not-ready'])(
    '%s: el estado del servicio de D04-04 es exactamente el del contrato',
    async (name) => {
      const payload = fixture(name);
      const state = await stateFrom(recordOf(payload));
      expect(selectionStateSchema.safeParse(state).success).toBe(true);
      expect(state).toEqual(payload);
    },
  );
});
