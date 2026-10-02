/**
 * D05-02 — Expediente de campaña: concilia request → job → run de cada intento con su fila
 * OFAT (#33) y explica, fila por fila, el representante y el ranking validation-only.
 *
 * Función pura. NO decide nada nuevo: el ranking, el representante cronológico de cada
 * fila y el `outcome_hash` salen de `selectCandidate` (D04-04), así que el expediente y
 * el servicio de selección nunca pueden discrepar. Lo que agrega:
 *
 * - Todos los intentos, aunque no tengan run (job fallido o en cola) o el run no se pueda
 *   leer, con su rol (`representative`, `retry`, `pending`, `excluded`) y su motivo.
 * - La conciliación de cada intento: el run declara el job que lo lanzó, entrenó la config
 *   que pidió el job (request), con su manifest y release, y los estados son coherentes.
 * - La identidad completa del candidato (run, job, commit, config, checkpoint, procedencia).
 * - Qué impide cerrar el conteo: menos de MIN_COMPARABLE_RUNS filas aceptadas, intentos
 *   de la matriz pendientes o intentos que no concilian.
 *
 * Ninguna métrica decide qué intento representa una fila, y nada de esto lee el frozen
 * test: el candidato sigue siendo una propuesta hasta el acto de cierre (D05-08).
 */
import {
  CAMPAIGN_MATRIX,
  campaignRowOf,
  type ExclusionReason,
  MIN_COMPARABLE_RUNS,
  type RankedRun,
  type ExcludedRun as SelectionExcludedRun,
  type SelectionReference,
  selectCandidate,
} from './model-selection.js';
import {
  type ExcludedRun as AdapterExcludedRun,
  type ExperimentRun,
  excludedRunSchema,
  experimentRunSchema,
  type TrainingConfig,
  type TrainingJob,
  trainingJobSchema,
} from './p3.contracts.js';

export type AttemptRole = 'representative' | 'retry' | 'pending' | 'excluded';
export type AttemptReason = ExclusionReason | 'job_failed' | 'job_cancelled' | 'adapter_excluded';
export type RowStatus = 'accepted' | 'pending' | 'missing';

export interface CampaignAttempt {
  job_id: number | null;
  job_status: TrainingJob['status'] | null;
  run_id: string | null;
  run_status: ExperimentRun['status'] | null;
  start_time: string | null;
  role: AttemptRole;
  reason: AttemptReason | null;
  detail: string | null;
  checkpoint_sha256: string | null;
  git_commit: string | null;
  /** Por qué el intento no concilia request → job → run. Cualquiera bloquea el cierre. */
  problems: string[];
}

export interface CampaignRowReport {
  row: number;
  change: string;
  config: TrainingConfig;
  status: RowStatus;
  representative: string | null;
  attempts: CampaignAttempt[];
}

export interface CandidateIdentity {
  run_id: string;
  campaign_row: number;
  job_id: number;
  git_commit: string;
  start_time: string;
  config: TrainingConfig;
  checkpoint_sha256: string | null;
  dataset_version: string;
  dvc_release_hash: string;
  dvc_images_md5: string;
  dvc_annotations_md5: string;
  manifest_version: string;
  manifest_hash: string;
  classes: string[];
  best_epoch: number;
  val_accuracy: number;
  val_macro_f1: number;
  val_loss: number;
}

export interface CampaignDossier {
  reference: SelectionReference;
  rows: CampaignRowReport[];
  /** Intentos que no son de ninguna fila (smoke, auxiliares, jobs o runs ilegibles). */
  unattributed: CampaignAttempt[];
  ranking: RankedRun[];
  candidate: CandidateIdentity | null;
  accepted_rows: number[];
  min_comparable_runs: number;
  close_blockers: string[];
  ready_to_close: boolean;
  outcome_hash: string;
}

/** Runs con el contrato `experiment_runs_response` (D04-01): elegibles y excluidos. */
export interface CampaignRunsInput {
  runs: readonly unknown[];
  excluded?: readonly unknown[];
}

const ACTIVE_JOB = new Set<TrainingJob['status']>(['queued', 'running']);
const ACTIVE_RUN = new Set<ExperimentRun['status']>(['RUNNING', 'SCHEDULED']);

/** Estado del run que corresponde a cada estado del job (un job en cola aún no tiene run). */
const RUN_STATUS_FOR_JOB: Record<TrainingJob['status'], readonly ExperimentRun['status'][]> = {
  queued: [],
  running: ['RUNNING', 'SCHEDULED'],
  succeeded: ['FINISHED'],
  failed: ['FAILED'],
  cancelled: ['KILLED', 'FAILED'],
};

function rawField(raw: unknown, field: string): unknown {
  return (raw as Record<string, unknown> | null)?.[field];
}

const rawRunId = (raw: unknown) => {
  const value = rawField(raw, 'run_id');
  return typeof value === 'string' ? value : null;
};

const rawJobId = (raw: unknown) => {
  const value = rawField(raw, 'id');
  return typeof value === 'number' ? value : null;
};

function sameConfig(a: TrainingConfig, b: TrainingConfig): boolean {
  const keys = Object.keys(a) as (keyof TrainingConfig)[];
  return keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key]);
}

/** Problemas de conciliación entre el job (request) y el run que dice haber lanzado. */
function conciliate(job: TrainingJob, run: ExperimentRun): string[] {
  const problems: string[] = [];
  if (run.tags.job_id !== job.id) {
    problems.push(`el run ${run.run_id} declara job_id ${run.tags.job_id}, no ${job.id}`);
  }
  if (!sameConfig(job.config, run.params)) {
    problems.push(`la config del run ${run.run_id} no es el request del job ${job.id}`);
  }
  if (job.manifest_hash !== run.tags.manifest_hash) {
    problems.push(`el manifest del job ${job.id} no es el del run ${run.run_id}`);
  }
  if (job.dataset_version !== run.tags.dvc_release) {
    problems.push(`el release del job ${job.id} no es el del run ${run.run_id}`);
  }
  return problems;
}

function statusProblem(job: TrainingJob, runStatus: ExperimentRun['status']): string[] {
  return RUN_STATUS_FOR_JOB[job.status].includes(runStatus)
    ? []
    : [`estado del job ${job.id} (${job.status}) ≠ estado del run (${runStatus})`];
}

interface Located {
  row: number | null;
  attempt: CampaignAttempt;
  created_at: string | null;
}

function compareText(a: string | null, b: string | null): number {
  const x = a ?? '~';
  const y = b ?? '~';
  if (x === y) return 0;
  return x < y ? -1 : 1;
}

/** Cronológico: start_time del run (o creación del job sin run), luego run_id y job_id. */
function chronological(a: Located, b: Located): number {
  return (
    compareText(a.attempt.start_time ?? a.created_at, b.attempt.start_time ?? b.created_at) ||
    compareText(a.attempt.run_id, b.attempt.run_id) ||
    (a.attempt.job_id ?? Number.MAX_SAFE_INTEGER) - (b.attempt.job_id ?? Number.MAX_SAFE_INTEGER)
  );
}

export function buildCampaignDossier(
  reference: SelectionReference,
  input: CampaignRunsInput,
  jobs: readonly unknown[],
): CampaignDossier {
  const outcome = selectCandidate(input.runs, reference);
  const representatives = new Set(outcome.ranking.map((run) => run.run_id));
  const exclusions = new Map<string, SelectionExcludedRun>();
  for (const entry of outcome.excluded) {
    if (entry.run_id !== null) exclusions.set(entry.run_id, entry);
  }

  const runs = new Map<string, ExperimentRun>();
  const listedRuns = new Set<string>();
  for (const raw of input.runs) {
    const id = rawRunId(raw);
    if (id !== null) listedRuns.add(id);
    const parsed = experimentRunSchema.safeParse(raw);
    if (parsed.success) runs.set(parsed.data.run_id, parsed.data);
  }
  const adapterExcluded = new Map<string, AdapterExcludedRun>();
  const located: Located[] = [];
  for (const raw of input.excluded ?? []) {
    const parsed = excludedRunSchema.safeParse(raw);
    if (parsed.success) {
      adapterExcluded.set(parsed.data.run_id, parsed.data);
      listedRuns.add(parsed.data.run_id);
    } else {
      located.push({
        row: null,
        created_at: null,
        attempt: {
          ...emptyAttempt(),
          run_id: rawRunId(raw),
          reason: 'invalid_contract',
          detail: 'excluido por D04-01, pero fuera del contrato excluded_run',
          problems: ['el run excluido por D04-01 no cumple su contrato'],
        },
      });
    }
  }

  /** Rol y motivo de un run según la decisión de `selectCandidate`. */
  function roleOf(runId: string): Pick<CampaignAttempt, 'role' | 'reason' | 'detail'> | null {
    if (representatives.has(runId)) return { role: 'representative', reason: null, detail: null };
    const excluded = exclusions.get(runId);
    if (excluded) {
      return {
        role: excluded.reason === 'duplicate_campaign_row' ? 'retry' : 'excluded',
        reason: excluded.reason,
        detail: excluded.detail,
      };
    }
    const adapter = adapterExcluded.get(runId);
    if (adapter) {
      return { role: 'excluded', reason: 'adapter_excluded', detail: adapter.reasons.join('; ') };
    }
    return null;
  }

  const claimed = new Set<string>();
  const jobIds = new Map<number, number>();
  for (const raw of jobs) {
    const id = rawJobId(raw);
    if (id !== null) jobIds.set(id, (jobIds.get(id) ?? 0) + 1);
  }

  for (const raw of jobs) {
    const parsed = trainingJobSchema.safeParse(raw);
    if (!parsed.success) {
      located.push({
        row: null,
        created_at: null,
        attempt: {
          ...emptyAttempt(),
          job_id: rawJobId(raw),
          reason: 'invalid_contract',
          detail: parsed.error.issues
            .slice(0, 3)
            .map((issue) => `${issue.path.map(String).join('.') || '(raíz)'}: ${issue.message}`)
            .join('; '),
          problems: ['el job no cumple el contrato training_job'],
        },
      });
      continue;
    }
    const job = parsed.data;
    if (job.task !== 'training') continue; // controlled (D02-05) no es campaña.

    const runId = job.mlflow_run_id;
    if (runId !== null) claimed.add(runId);
    const run = runId === null ? undefined : runs.get(runId);
    const adapter = runId === null ? undefined : adapterExcluded.get(runId);
    const runStatus = run?.status ?? adapter?.status ?? null;

    const problems: string[] = [];
    if ((jobIds.get(job.id) ?? 0) > 1) problems.push(`job_id ${job.id} repetido en la fuente`);
    if (runId !== null && !listedRuns.has(runId)) {
      problems.push(`el run ${runId} del job ${job.id} no está en MLflow`);
    }
    if (run) problems.push(...conciliate(job, run));
    if (runStatus !== null) problems.push(...statusProblem(job, runStatus));

    const active = ACTIVE_JOB.has(job.status) || (runStatus !== null && ACTIVE_RUN.has(runStatus));
    const decided = runId === null ? null : roleOf(runId);
    let verdict: Pick<CampaignAttempt, 'role' | 'reason' | 'detail'>;
    if (decided?.role === 'representative') verdict = decided;
    else if (active) verdict = { role: 'pending', reason: null, detail: null };
    else if (decided) verdict = decided;
    else if (job.status === 'failed') {
      verdict = { role: 'excluded', reason: 'job_failed', detail: job.error };
    } else if (job.status === 'cancelled') {
      verdict = { role: 'excluded', reason: 'job_cancelled', detail: 'cancelado sin run' };
    } else verdict = { role: 'excluded', reason: null, detail: 'sin run en MLflow' };

    located.push({
      // Lo que se entrenó manda: la fila sale de la config del run si se pudo leer.
      row: campaignRowOf(run?.params ?? job.config),
      created_at: job.created_at,
      attempt: {
        job_id: job.id,
        job_status: job.status,
        run_id: runId,
        run_status: runStatus,
        start_time: run?.start_time ?? adapter?.start_time ?? null,
        ...verdict,
        checkpoint_sha256: run?.checkpoint_sha256 ?? null,
        git_commit: run?.tags.git_commit ?? null,
        problems,
      },
    });
  }

  // Runs que ningún job registra: se listan, y si son de training no concilian.
  for (const raw of input.runs) {
    const runId = rawRunId(raw);
    if (runId === null || claimed.has(runId)) continue;
    const run = runs.get(runId);
    located.push({
      row: run ? campaignRowOf(run.params) : null,
      created_at: null,
      attempt: {
        job_id: null,
        job_status: null,
        run_id: runId,
        run_status: run?.status ?? null,
        start_time: run?.start_time ?? null,
        ...(roleOf(runId) ?? { role: 'excluded', reason: null, detail: null }),
        ...(run && ACTIVE_RUN.has(run.status) ? { role: 'pending' as const } : {}),
        checkpoint_sha256: run?.checkpoint_sha256 ?? null,
        git_commit: run?.tags.git_commit ?? null,
        problems: [
          run
            ? `ningún job registra el run ${runId} (declara job_id ${run.tags.job_id})`
            : `ningún job registra el run ${runId}`,
        ],
      },
    });
  }
  for (const [runId, adapter] of adapterExcluded) {
    if (claimed.has(runId)) continue;
    located.push({
      row: null,
      created_at: null,
      attempt: {
        ...emptyAttempt(),
        run_id: runId,
        run_status: adapter.status,
        start_time: adapter.start_time,
        reason: 'adapter_excluded',
        detail: adapter.reasons.join('; '),
      },
    });
  }

  located.sort(chronological);
  const rows: CampaignRowReport[] = CAMPAIGN_MATRIX.map((entry) => {
    const attempts = located.filter((item) => item.row === entry.row).map((item) => item.attempt);
    const representative =
      outcome.ranking.find((run) => run.campaign_row === entry.row)?.run_id ?? null;
    const status: RowStatus = attempts.some((attempt) => attempt.role === 'pending')
      ? 'pending'
      : representative !== null
        ? 'accepted'
        : 'missing';
    return {
      row: entry.row,
      change: entry.change,
      config: entry.config,
      status,
      representative,
      attempts,
    };
  });
  const unattributed = located.filter((item) => item.row === null).map((item) => item.attempt);

  const acceptedRows = rows.filter((row) => row.status === 'accepted').map((row) => row.row);
  const pendingRows = rows.filter((row) => row.status === 'pending').map((row) => row.row);
  const unreconciled = [...rows.flatMap((row) => row.attempts), ...unattributed].filter(
    (attempt) => attempt.problems.length > 0,
  );
  const closeBlockers: string[] = [];
  if (acceptedRows.length < MIN_COMPARABLE_RUNS) {
    closeBlockers.push(
      `${acceptedRows.length} filas aceptadas; hacen falta al menos ${MIN_COMPARABLE_RUNS}.`,
    );
  }
  if (pendingRows.length > 0) {
    closeBlockers.push(
      `Intentos pendientes en las filas ${pendingRows.join(', ')}: no se cierra el conteo con intentos en curso.`,
    );
  }
  if (unreconciled.length > 0) {
    closeBlockers.push(
      `${unreconciled.length} intento(s) no concilian request → job → run: ${unreconciled
        .flatMap((attempt) => attempt.problems)
        .slice(0, 3)
        .join('; ')}.`,
    );
  }

  return {
    reference,
    rows,
    unattributed,
    ranking: outcome.ranking,
    candidate: outcome.candidate ? identityOf(outcome.candidate, runs) : null,
    accepted_rows: acceptedRows,
    min_comparable_runs: MIN_COMPARABLE_RUNS,
    close_blockers: closeBlockers,
    ready_to_close: outcome.ready_to_close && closeBlockers.length === 0,
    outcome_hash: outcome.outcome_hash,
  };
}

function emptyAttempt(): CampaignAttempt {
  return {
    job_id: null,
    job_status: null,
    run_id: null,
    run_status: null,
    start_time: null,
    role: 'excluded',
    reason: null,
    detail: null,
    checkpoint_sha256: null,
    git_commit: null,
    problems: [],
  };
}

function identityOf(
  candidate: RankedRun,
  runs: ReadonlyMap<string, ExperimentRun>,
): CandidateIdentity | null {
  const run = runs.get(candidate.run_id);
  if (!run) return null;
  const { tags } = run;
  return {
    run_id: run.run_id,
    campaign_row: candidate.campaign_row,
    job_id: tags.job_id,
    git_commit: tags.git_commit,
    start_time: run.start_time,
    config: run.params,
    checkpoint_sha256: run.checkpoint_sha256,
    dataset_version: tags.dvc_release,
    dvc_release_hash: tags.dvc_release_hash,
    dvc_images_md5: tags.dvc_images_md5,
    dvc_annotations_md5: tags.dvc_annotations_md5,
    manifest_version: tags.manifest_version,
    manifest_hash: tags.manifest_hash,
    classes: [...tags.classes],
    best_epoch: candidate.best_epoch,
    val_accuracy: candidate.val_accuracy,
    val_macro_f1: candidate.val_macro_f1,
    val_loss: candidate.val_loss,
  };
}
