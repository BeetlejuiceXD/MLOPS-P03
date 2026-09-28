/**
 * D02-05 — Jobs de entrenamiento persistentes (capa Logic).
 *
 * La API solo valida y encola: el trabajo lo ejecuta `trainer-worker` fuera del
 * request HTTP, tomando los jobs `queued` de MariaDB. Toda respuesta pasa por el
 * contrato `training_job` (contracts/p3): un registro incoherente no sale como dato.
 */
import { ConflictError, NotFoundError, ValidationError } from './errors.js';
import {
  type CreateTrainingJobRequest,
  createTrainingJobRequestSchema,
  jobLogsSchema,
  manifestSummarySchema,
  releasesResponseSchema,
  type TrainingConfig,
  type TrainingJob,
  trainingJobSchema,
  type trainingJobStatuses,
  type trainingJobTasks,
} from './p3.contracts.js';

type JobStatus = (typeof trainingJobStatuses)[number];
type JobTask = (typeof trainingJobTasks)[number];

export interface NewTrainingJob {
  task: JobTask;
  datasetVersion: string;
  manifestHash: string;
  config: TrainingConfig;
  controlledFailAtEpoch: number | null;
}

export interface TrainingJobRecord extends NewTrainingJob {
  id: number;
  status: JobStatus;
  progressEpoch: number | null;
  totalEpochs: number | null;
  mlflowRunId: string | null;
  error: string | null;
  cancelRequested: boolean;
  createdAt: Date;
  startedAt: Date | null;
  finishedAt: Date | null;
}

export interface TrainingJobLogRecord {
  ts: Date;
  level: 'info' | 'warning' | 'error';
  message: string;
}

/** Acceso a `training_jobs`/`training_job_logs`. La implementación real vive en `data`. */
export interface TrainingJobRepository {
  insert(job: NewTrainingJob): Promise<TrainingJobRecord>;
  list(): Promise<TrainingJobRecord[]>;
  findById(id: number): Promise<TrainingJobRecord | null>;
  listLogs(id: number): Promise<TrainingJobLogRecord[]>;
  /** `queued` → `cancelled` en una sola sentencia; `false` si ya no estaba en cola. */
  cancelQueued(id: number): Promise<boolean>;
  /** Marca `cancel_requested` si sigue `running`; el worker aplica la cancelación. */
  requestCancel(id: number): Promise<boolean>;
}

export type Eligibility = { eligible: true } | { eligible: false; reason: string };

/** Decide si un `task=training` puede encolarse (release elegible + manifest oficial). */
export interface TrainingEligibilityGate {
  check(datasetVersion: string, manifestHash: string): Promise<Eligibility>;
}

export interface OfficialSources {
  releases(): Promise<unknown>;
  manifest(): Promise<unknown>;
}

/**
 * Compuerta de training real a partir de las fuentes oficiales (resolver de releases y
 * manifest 70/20/10). Cualquier fuente caída o fuera de contrato cuenta como no elegible.
 */
export function createEligibilityGate(sources: OfficialSources): TrainingEligibilityGate {
  return {
    async check(datasetVersion, manifestHash) {
      let releases: unknown;
      let manifest: unknown;
      try {
        [releases, manifest] = await Promise.all([sources.releases(), sources.manifest()]);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        return { eligible: false, reason: `Fuentes oficiales no disponibles: ${detail}` };
      }
      const parsedReleases = releasesResponseSchema.safeParse(releases);
      const parsedManifest = manifestSummarySchema.safeParse(manifest);
      if (!parsedReleases.success || !parsedManifest.success) {
        return { eligible: false, reason: 'Las fuentes oficiales no cumplen su contrato.' };
      }
      const approved = parsedReleases.data.approved.some(
        (release) => release.dataset_version === datasetVersion,
      );
      if (!approved) {
        return { eligible: false, reason: `El release ${datasetVersion} no está aprobado.` };
      }
      const official = parsedManifest.data;
      if (!official.frozen) {
        return { eligible: false, reason: 'El manifest 70/20/10 oficial no está congelado.' };
      }
      if (official.dataset_version !== datasetVersion || official.manifest_hash !== manifestHash) {
        return {
          eligible: false,
          reason: 'release o hash no coinciden con el manifest oficial congelado.',
        };
      }
      return { eligible: true };
    },
  };
}

/**
 * Compuerta por defecto hasta que la API exponga el resolver y el manifest oficial
 * (congelación en D03-01, integración en D03-03): el training real no se habilita.
 */
export const officialSourcesUnavailableGate: TrainingEligibilityGate = {
  async check() {
    return {
      eligible: false,
      reason:
        'El release y el manifest oficiales aún no están disponibles en la API (D03-01/D03-03); ' +
        'solo se aceptan tareas controladas.',
    };
  },
};

function formatIssues(issues: { path: PropertyKey[]; message: string }[]): string {
  return issues
    .map((issue) => `${issue.path.map(String).join('.') || '(raíz)'}: ${issue.message}`)
    .join('; ');
}

function toContract(record: TrainingJobRecord): TrainingJob {
  const dto = {
    id: record.id,
    task: record.task,
    status: record.status,
    dataset_version: record.datasetVersion,
    manifest_hash: record.manifestHash,
    config: record.config,
    progress:
      record.progressEpoch === null || record.totalEpochs === null
        ? null
        : { epoch: record.progressEpoch, total_epochs: record.totalEpochs },
    mlflow_run_id: record.mlflowRunId,
    error: record.error,
    cancel_requested: record.cancelRequested,
    created_at: record.createdAt.toISOString(),
    started_at: record.startedAt?.toISOString() ?? null,
    finished_at: record.finishedAt?.toISOString() ?? null,
  };
  const parsed = trainingJobSchema.safeParse(dto);
  if (!parsed.success) {
    // Error 500: el registro persistido es incoherente; no se entrega como dato válido.
    throw new Error(
      `El job ${record.id} no cumple el contrato training_job: ${formatIssues(parsed.error.issues)}`,
    );
  }
  return parsed.data;
}

export function createTrainingJobsService(
  repository: TrainingJobRepository,
  gate: TrainingEligibilityGate,
) {
  async function mustFind(id: number): Promise<TrainingJobRecord> {
    const record = await repository.findById(id);
    if (!record) throw new NotFoundError(`No existe el job ${id}.`);
    return record;
  }

  return {
    async create(body: unknown): Promise<TrainingJob> {
      const parsed = createTrainingJobRequestSchema.safeParse(body);
      if (!parsed.success) {
        throw new ValidationError(`Job inválido: ${formatIssues(parsed.error.issues)}`);
      }
      const request: CreateTrainingJobRequest = parsed.data;
      if (request.task === 'training') {
        const eligibility = await gate.check(request.dataset_version, request.manifest_hash);
        if (!eligibility.eligible) {
          throw new ConflictError(`No se puede encolar un training real: ${eligibility.reason}`);
        }
      }
      const record = await repository.insert({
        task: request.task,
        datasetVersion: request.dataset_version,
        manifestHash: request.manifest_hash,
        config: request.config,
        controlledFailAtEpoch: request.controlled?.fail_at_epoch ?? null,
      });
      return toContract(record);
    },

    async list(): Promise<{ jobs: TrainingJob[] }> {
      return { jobs: (await repository.list()).map(toContract) };
    },

    async get(id: number): Promise<TrainingJob> {
      return toContract(await mustFind(id));
    },

    async logs(id: number) {
      await mustFind(id);
      const lines = (await repository.listLogs(id)).map((line) => ({
        ts: line.ts.toISOString(),
        level: line.level,
        message: line.message,
      }));
      return jobLogsSchema.parse({ job_id: id, lines });
    },

    async cancel(id: number): Promise<TrainingJob> {
      const record = await mustFind(id);
      if (record.status === 'queued' && (await repository.cancelQueued(id))) {
        return toContract(await mustFind(id));
      }
      // Se relee: el worker pudo tomar el job entre la lectura y la cancelación.
      let current = await mustFind(id);
      if (current.status === 'running' && (await repository.requestCancel(id))) {
        return toContract(await mustFind(id));
      }
      current = await mustFind(id);
      throw new ConflictError(`El job ${id} ya terminó (${current.status}); no se puede cancelar.`);
    },
  };
}

export type TrainingJobsService = ReturnType<typeof createTrainingJobsService>;
