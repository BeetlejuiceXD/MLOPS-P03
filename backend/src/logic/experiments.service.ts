/**
 * D04-01 — Adaptador MLflow → portal (`GET /api/experiments/runs`, detalle y artefactos).
 *
 * Lee los runs reales del experimento `p3-cnn-classifier` y los traduce al contrato
 * `experiment_runs_response` (#33 / D01-05) sin inventar nada:
 *
 * - Solo los runs `p3.run_kind=training` pueden ir a `runs`. Los auxiliares
 *   (`controlled_task` de D02-05, `short_run_instrumentation` de D02-06,
 *   `persistence_check` de D02-01) y los que no declaran tipo van a `excluded` con su motivo.
 * - Provenance, params y curvas salen tal cual de MLflow. Si falta un tag de trazabilidad,
 *   una época o una métrica, el run va a `excluded`: nunca se rellena con el valor oficial
 *   ni se interpola una curva.
 * - Elegible para campaña = training FINISHED con resumen y `checkpoint_sha256`. FINISHED
 *   solo no basta. Que el run pertenezca a la matriz OFAT de la campaña es de D04-03/D04-04.
 * - MLflow caído o respondiendo mal → `ServiceUnavailableError` (503 con el motivo), nunca
 *   un listado vacío.
 */
import {
  type MlflowMetric,
  type MlflowReader,
  type MlflowRun,
  MlflowUnavailableError,
} from '../data/mlflow/mlflow-rest.client.js';
import {
  ConflictError,
  NotFoundError,
  ServiceUnavailableError,
  ValidationError,
} from './errors.js';
import {
  type ArtifactEntry,
  type ExcludedRun,
  type ExperimentRun,
  type ExperimentRunDetail,
  experimentRunSchema,
  P3_EXPERIMENT,
} from './p3.contracts.js';

export const RUN_KIND_TAG = 'p3.run_kind';
export const TRAINING_RUN_KIND = 'training';
const EPOCH_KEYS = [
  'train_loss',
  'train_accuracy',
  'val_loss',
  'val_accuracy',
  'val_macro_f1',
  'learning_rate',
] as const;
const SUMMARY_KEYS = [
  'best_epoch',
  'best_val_accuracy',
  'best_val_macro_f1',
  'best_val_loss',
] as const;
const PROVENANCE_TAGS = [
  'git_commit',
  'dvc_release',
  'dvc_images_md5',
  'dvc_annotations_md5',
  'dvc_release_hash',
  'manifest_version',
  'manifest_hash',
  'classes',
  'seed',
  'job_id',
] as const;
// `classes` también se registra como param (runner.start_training); su valor vive en tags.
const NON_CONFIG_PARAMS = new Set(['classes']);
const RUN_ID = /^[0-9a-f]{32}$/;
const MAX_ARTIFACT_DEPTH = 8;

type Classified = { kind: 'run'; run: ExperimentRun } | { kind: 'excluded'; excluded: ExcludedRun };

export interface ArtifactDownload {
  body: ReadableStream<Uint8Array> | null;
  size: number | null;
  filename: string;
}

export interface ExperimentsService {
  listRuns(): Promise<{
    experiment_name: typeof P3_EXPERIMENT;
    runs: ExperimentRun[];
    excluded: ExcludedRun[];
  }>;
  getRun(runId: string): Promise<ExperimentRunDetail>;
  downloadArtifact(runId: string, path: string): Promise<ArtifactDownload>;
}

const toIso = (ms: number | undefined) =>
  ms === undefined ? null : new Date(Number(ms)).toISOString();

/** Los params de MLflow son texto (`str(value)` de Python): se tipan sin reinterpretar. */
function typedParam(value: string): unknown {
  if (value === 'True' || value === 'true') return true;
  if (value === 'False' || value === 'false') return false;
  if (/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(value)) return Number(value);
  return value;
}

const asRecord = (pairs: { key: string; value: string }[] = []) =>
  Object.fromEntries(pairs.map(({ key, value }) => [key, value]));

const STATUS = new Set(['RUNNING', 'FINISHED', 'FAILED', 'KILLED']);

function excludedFrom(run: MlflowRun, kind: string | null, reasons: string[]): Classified {
  const status = STATUS.has(run.info.status) ? run.info.status : 'FAILED';
  const extra = STATUS.has(run.info.status)
    ? []
    : [`estado desconocido de MLflow: ${run.info.status}`];
  return {
    kind: 'excluded',
    excluded: {
      run_id: run.info.run_id,
      run_kind: kind,
      status: status as ExcludedRun['status'],
      start_time: toIso(run.info.start_time) as string,
      reasons: [...reasons, ...extra],
    },
  };
}

/** Curvas por época tal cual: cada época 1..N con las seis métricas, o un motivo. */
function buildHistory(series: Record<string, MlflowMetric[]>) {
  const reasons: string[] = [];
  const byKey = new Map<string, Map<number, number>>();
  const steps = new Set<number>();
  for (const key of EPOCH_KEYS) {
    const values = new Map<number, number>();
    // Si una época se registró dos veces, vale la última escritura (mayor timestamp).
    const ordered = [...(series[key] ?? [])].sort(
      (a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0),
    );
    for (const metric of ordered) {
      values.set(Number(metric.step), Number(metric.value));
      steps.add(Number(metric.step));
    }
    byKey.set(key, values);
  }
  const epochs = [...steps].sort((a, b) => a - b);
  const last = epochs.at(-1) ?? 0;
  if (epochs.length > 0 && (epochs[0] !== 1 || epochs.length !== last)) {
    reasons.push(
      `historia incompleta: épocas registradas ${epochs.join(', ')} (se esperaba 1..${last})`,
    );
  }
  const history = [];
  for (let epoch = 1; epoch <= last; epoch += 1) {
    const row: Record<string, number> = { epoch };
    for (const key of EPOCH_KEYS) {
      const value = byKey.get(key)?.get(epoch);
      if (value === undefined) {
        reasons.push(`historia incompleta: época ${epoch} sin ${key}`);
      } else {
        row[key] = value;
      }
    }
    history.push(row);
  }
  return { history, reasons };
}

export function createExperimentsService(mlflow: MlflowReader): ExperimentsService {
  async function guarded<T>(work: () => Promise<T>): Promise<T> {
    try {
      return await work();
    } catch (error) {
      if (error instanceof MlflowUnavailableError) {
        throw new ServiceUnavailableError(`mlflow_unavailable: ${error.message}`);
      }
      throw error;
    }
  }

  async function classify(run: MlflowRun): Promise<Classified> {
    const tags = asRecord(run.data.tags);
    const kind = tags[RUN_KIND_TAG] || null;
    if (kind === null) {
      return excludedFrom(run, null, ['sin tag p3.run_kind: no se puede saber qué es']);
    }
    if (kind !== TRAINING_RUN_KIND) {
      return excludedFrom(run, kind, [`run auxiliar (${kind}): no es de campaña`]);
    }

    const reasons: string[] = [];
    for (const name of PROVENANCE_TAGS) {
      if (!tags[name])
        reasons.push(`tags.${name}: falta en MLflow (no se rellena con el valor oficial)`);
    }

    const params = Object.fromEntries(
      (run.data.params ?? [])
        .filter(({ key }) => !NON_CONFIG_PARAMS.has(key))
        .map(({ key, value }) => [key, typedParam(value)]),
    );

    const series = Object.fromEntries(
      await Promise.all(
        EPOCH_KEYS.map(
          async (key) => [key, await mlflow.getMetricHistory(run.info.run_id, key)] as const,
        ),
      ),
    );
    const { history, reasons: historyReasons } = buildHistory(series);
    reasons.push(...historyReasons);

    const latest = new Map((run.data.metrics ?? []).map((m) => [m.key, Number(m.value)]));
    const present = SUMMARY_KEYS.filter((key) => latest.has(key));
    if (present.length > 0 && present.length < SUMMARY_KEYS.length) {
      const missing = SUMMARY_KEYS.filter((key) => !latest.has(key));
      reasons.push(`resumen incompleto: faltan ${missing.join(', ')}`);
    }
    if (reasons.length > 0) return excludedFrom(run, kind, reasons);

    const summary =
      present.length === SUMMARY_KEYS.length
        ? {
            best_epoch: latest.get('best_epoch'),
            best_val_accuracy: latest.get('best_val_accuracy'),
            best_val_macro_f1: latest.get('best_val_macro_f1'),
            best_val_loss: latest.get('best_val_loss'),
          }
        : null;
    const checkpoint = tags.checkpoint_sha256 || null;
    const status = run.info.status;

    const ineligible: string[] = [];
    if (status !== 'FINISHED') ineligible.push(`estado ${status}: el run no terminó`);
    else if (summary === null) ineligible.push('FINISHED sin resumen de mejores métricas');
    if (checkpoint === null) ineligible.push('sin checkpoint_sha256: no hay checkpoint verificado');

    const candidate = {
      run_id: run.info.run_id,
      experiment_name: P3_EXPERIMENT,
      status,
      start_time: toIso(run.info.start_time),
      end_time: status === 'RUNNING' ? null : toIso(run.info.end_time),
      params,
      tags: {
        ...Object.fromEntries(PROVENANCE_TAGS.map((name) => [name, tags[name]])),
        classes: String(tags.classes).split(','),
        seed: typedParam(String(tags.seed)),
        job_id: typedParam(String(tags.job_id)),
      },
      summary,
      history,
      checkpoint_sha256: checkpoint,
      campaign_eligible: ineligible.length === 0,
      ineligible_reasons: ineligible,
    };
    const parsed = experimentRunSchema.safeParse(candidate);
    if (!parsed.success) {
      return excludedFrom(
        run,
        kind,
        parsed.error.issues.map((issue) => `${issue.path.join('.') || 'run'}: ${issue.message}`),
      );
    }
    return { kind: 'run', run: parsed.data };
  }

  /** Run del experimento P3 por id; 400 si el id no es de MLflow, 404 si no es de P3. */
  async function findRun(runId: string): Promise<MlflowRun> {
    if (!RUN_ID.test(runId)) throw new ValidationError('run_id de MLflow inválido (32 hex)');
    const experimentId = await mlflow.getExperimentIdByName(P3_EXPERIMENT);
    const run = experimentId === null ? null : await mlflow.getRun(runId);
    if (!run || run.info.experiment_id !== experimentId || run.info.lifecycle_stage === 'deleted') {
      throw new NotFoundError(`El run ${runId} no existe en ${P3_EXPERIMENT}`);
    }
    return run;
  }

  async function presentable(run: MlflowRun): Promise<ExperimentRun> {
    const classified = await classify(run);
    if (classified.kind === 'excluded') {
      throw new ConflictError(
        `El run ${run.info.run_id} no se presenta como corrida (${classified.excluded.run_kind ?? 'sin tipo'}): ${classified.excluded.reasons.join('; ')}`,
      );
    }
    return classified.run;
  }

  async function listTree(run: MlflowRun, dir = '', depth = 0): Promise<ArtifactEntry[]> {
    const entries = await mlflow.listArtifacts(run.info, dir);
    const out: ArtifactEntry[] = [];
    for (const entry of entries) {
      out.push({
        path: entry.path,
        is_dir: entry.is_dir,
        size_bytes: entry.is_dir ? null : (entry.file_size ?? null),
      });
      if (entry.is_dir && depth < MAX_ARTIFACT_DEPTH) {
        out.push(...(await listTree(run, entry.path, depth + 1)));
      }
    }
    return out;
  }

  return {
    listRuns: () =>
      guarded(async () => {
        const experimentId = await mlflow.getExperimentIdByName(P3_EXPERIMENT);
        const all = experimentId === null ? [] : await mlflow.searchRuns(experimentId);
        const classified = await Promise.all(
          all.filter((run) => run.info.lifecycle_stage !== 'deleted').map(classify),
        );
        const newestFirst = <T extends { start_time: string; run_id: string }>(a: T, b: T) =>
          b.start_time.localeCompare(a.start_time) || a.run_id.localeCompare(b.run_id);
        return {
          experiment_name: P3_EXPERIMENT,
          runs: classified.flatMap((c) => (c.kind === 'run' ? [c.run] : [])).sort(newestFirst),
          excluded: classified
            .flatMap((c) => (c.kind === 'excluded' ? [c.excluded] : []))
            .sort(newestFirst),
        };
      }),

    getRun: (runId) =>
      guarded(async () => {
        const run = await findRun(runId);
        const presented = await presentable(run);
        const artifacts = (await listTree(run)).sort((a, b) => a.path.localeCompare(b.path));
        return { run: presented, artifacts };
      }),

    downloadArtifact: (runId, path) =>
      guarded(async () => {
        const parts = path.split('/');
        if (
          !path ||
          path.startsWith('/') ||
          parts.some((p) => p === '' || p === '.' || p === '..')
        ) {
          throw new ValidationError('Ruta de artefacto inválida: relativa al run, sin . ni ..');
        }
        const run = await findRun(runId);
        await presentable(run);
        const dir = parts.slice(0, -1).join('/');
        const entry = (await mlflow.listArtifacts(run.info, dir)).find((e) => e.path === path);
        if (!entry || entry.is_dir) {
          throw new NotFoundError(`artifact_missing: ${path} no existe en el run ${runId}`);
        }
        const res = await mlflow.downloadArtifact(run.info, path);
        return {
          body: res.body,
          size: entry.file_size ?? null,
          filename: parts.at(-1) as string,
        };
      }),
  };
}
