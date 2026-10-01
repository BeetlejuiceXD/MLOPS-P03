/**
 * D04-04 — Estado persistido de la selección y guardas del frozen test.
 *
 * Máquina de estados de un solo registro (`p3_model_selection`):
 *
 *   open ──propose──▶ candidate ──propose──▶ candidate ──close──▶ closed
 *
 * - `propose` recalcula el ranking con los runs actuales (solo validation) y guarda un
 *   candidato PREPARATORIO; se puede repetir mientras la selección no esté cerrada.
 * - `close` exige confirmar el run_id del candidato, al menos MIN_COMPARABLE_RUNS
 *   configuraciones de la matriz y que la campaña no haya cambiado desde la propuesta
 *   (se recalcula y se compara `outcome_hash`). La escritura es condicional: si otro
 *   proceso cambió el estado, no se cierra.
 * - `closed` es definitivo: no se re-propone ni se vuelve a cerrar.
 * - `requireClosed` es la guarda de cualquier acceso a resultados del test, y
 *   `blockedEvaluation` la respuesta `blocked` de `GET /api/evaluation` mientras tanto.
 *
 * Este ticket prueba la transición con datos sintéticos; el cierre de la campaña
 * oficial (MODEL SELECTION CLOSED) es de D05-02.
 */
import { z } from 'zod';
import { ConflictError, ServiceUnavailableError, ValidationError } from './errors.js';
import {
  MIN_COMPARABLE_RUNS,
  type SelectionOutcome,
  type SelectionReference,
  selectCandidate,
} from './model-selection.js';
import {
  type EvaluationResponse,
  evaluationResponseSchema,
  manifestSummarySchema,
  P3_EXPERIMENT,
} from './p3.contracts.js';

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

/**
 * Fuente por defecto hasta que exista el adaptador MLflow de D04-01: la API responde 503
 * con el motivo en vez de inventar runs o dar una selección vacía como válida.
 */
export const runsAdapterPendingSource: CampaignRunsSource = {
  async list() {
    throw new Error('el adaptador de runs de MLflow (D04-01) aún no está integrado');
  },
};

const runIdSchema = z.string().regex(/^[0-9a-f]{32}$/);

// El sobre de la respuesta se valida aquí; cada run se valida (y excluye) por separado
// en `selectCandidate`, para que un run roto no tumbe ni contamine la selección.
const runsEnvelopeSchema = z.object({
  experiment_name: z.literal(P3_EXPERIMENT),
  runs: z.array(z.unknown()),
});

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

function toState(record: SelectionRecord): SelectionState {
  const { outcome } = record;
  return {
    status: record.status,
    candidate: outcome?.candidate ?? null,
    ranking: outcome?.ranking ?? [],
    excluded: outcome?.excluded ?? [],
    campaign_rows: outcome?.campaign_rows ?? [],
    min_comparable_runs: MIN_COMPARABLE_RUNS,
    ready_to_close: outcome?.ready_to_close ?? false,
    reference: outcome?.reference ?? null,
    outcome_hash: outcome?.outcome_hash ?? null,
    proposed_at: record.proposedAt?.toISOString() ?? null,
    closed_at: record.closedAt?.toISOString() ?? null,
  };
}

export function createModelSelectionService(
  repo: ModelSelectionRepository,
  runs: CampaignRunsSource,
  manifest: FrozenManifestSource,
  clock: () => Date = () => new Date(),
): ModelSelectionService {
  async function loadReference(): Promise<SelectionReference> {
    let raw: unknown;
    try {
      raw = await manifest.manifest();
    } catch (error) {
      throw new ServiceUnavailableError(`Manifest oficial no disponible: ${errorText(error)}`);
    }
    const parsed = manifestSummarySchema.safeParse(raw);
    if (!parsed.success) {
      throw new ServiceUnavailableError('El manifest oficial no cumple su contrato.');
    }
    if (!parsed.data.frozen) {
      throw new ConflictError(
        'El manifest P3 no está congelado (D03-01): no hay selección posible.',
      );
    }
    return {
      dataset_version: parsed.data.dataset_version,
      manifest_hash: parsed.data.manifest_hash,
      dvc_release_hash: parsed.data.dvc_release_hash,
    };
  }

  async function currentOutcome(): Promise<SelectionOutcome> {
    const reference = await loadReference();
    let raw: unknown;
    try {
      raw = await runs.list();
    } catch (error) {
      throw new ServiceUnavailableError(`Runs de MLflow no disponibles: ${errorText(error)}`);
    }
    const envelope = runsEnvelopeSchema.safeParse(raw);
    if (!envelope.success) {
      throw new ServiceUnavailableError('La lista de runs no cumple experiment_runs_response.');
    }
    return selectCandidate(envelope.data.runs, reference);
  }

  return {
    async state() {
      return toState(await repo.read());
    },

    async propose() {
      if ((await repo.read()).status === 'closed') {
        throw new ConflictError('La selección ya está cerrada: no se vuelve a seleccionar.');
      }
      const outcome = await currentOutcome();
      if (outcome.candidate === null) {
        throw new ConflictError(
          `Ningún run elegible para la selección (${outcome.excluded.length} excluidos).`,
        );
      }
      if (!(await repo.saveCandidate(outcome, clock()))) {
        throw new ConflictError('La selección se cerró mientras se proponía el candidato.');
      }
      return toState(await repo.read());
    },

    async close(candidateRunId) {
      const runId = runIdSchema.safeParse(candidateRunId);
      if (!runId.success) {
        throw new ValidationError('candidate_run_id debe ser un run_id de MLflow (32 hex).');
      }
      const record = await repo.read();
      if (record.status === 'closed') {
        throw new ConflictError('La selección ya está cerrada.');
      }
      const proposed = record.outcome;
      if (record.status !== 'candidate' || proposed === null || proposed.candidate === null) {
        throw new ConflictError('No hay candidato propuesto que cerrar.');
      }
      if (proposed.candidate.run_id !== runId.data) {
        throw new ConflictError(
          `El run confirmado no es el candidato propuesto (${proposed.candidate.run_id}).`,
        );
      }
      if (!proposed.ready_to_close) {
        throw new ConflictError(
          `Hacen falta al menos ${MIN_COMPARABLE_RUNS} configuraciones comparables ` +
            `(hay ${proposed.campaign_rows.length}).`,
        );
      }
      const current = await currentOutcome();
      if (current.outcome_hash !== proposed.outcome_hash) {
        throw new ConflictError('La campaña cambió desde la propuesta: vuelve a proponer.');
      }
      if (!(await repo.close(proposed.outcome_hash, clock()))) {
        throw new ConflictError('El estado de la selección cambió durante el cierre.');
      }
      return toState(await repo.read());
    },

    async requireClosed() {
      const { status, outcome, closedAt } = await repo.read();
      if (status !== 'closed') {
        throw new ConflictError(
          'MODEL SELECTION CLOSED no existe: los resultados del test siguen bloqueados.',
        );
      }
      if (!outcome?.candidate || closedAt === null) {
        throw new ConflictError('Registro de cierre incompleto: el test sigue bloqueado.');
      }
      return {
        candidate_run_id: outcome.candidate.run_id,
        closed_at: closedAt.toISOString(),
        manifest_hash: outcome.reference.manifest_hash,
      };
    },

    async blockedEvaluation() {
      if ((await repo.read()).status === 'closed') return null;
      return evaluationResponseSchema.parse({
        state: 'blocked',
        reason: 'model_selection_open',
        detail:
          'MODEL SELECTION CLOSED no existe todavía: el frozen test solo se evalúa ' +
          'después del cierre de la selección por validation.',
      });
    },
  };
}
